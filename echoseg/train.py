"""Train a multi-label segmentation model.

    python -m echoseg.train --config configs/base.yaml --set data.context=2 run_name=unet_r34_ctx2
"""
import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import config_hash, load_config
from .data import FrameDataset, list_patients, load_sequences, split_patients, to_multilabel
from .model import DiceBCELoss, build_model


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def auto_pos_weight(sequences, labels, cap=10.0):
    """sqrt(negative / positive pixel ratio) per label, so rare labels weigh more in BCE."""
    pos = np.zeros(len(labels))
    total = 0
    for seq in sequences[:200]:
        target = to_multilabel(seq.masks, labels)
        pos += target.sum((0, 2, 3))
        total += seq.masks.size
    return np.clip(np.sqrt((total - pos) / np.maximum(pos, 1)), 1.0, cap).tolist()


@torch.no_grad()
def validate(model, loader, device, amp, threshold):
    model.eval()
    inter = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with torch.autocast(device.type, enabled=amp):
            pred = torch.sigmoid(model(x).float()) > threshold
        inter = inter + (pred & (y > 0.5)).sum((0, 2, 3))
        total = total + pred.sum((0, 2, 3)) + y.sum((0, 2, 3))
    return (2 * inter / total.clamp(min=1)).tolist()


def run(cfg, runs_dir="runs"):
    seed_everything(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = cfg["train"]["amp"] and device.type == "cuda"
    run_dir = Path(runs_dir) / cfg["run_name"]
    run_dir.mkdir(parents=True, exist_ok=True)

    data, labels = cfg["data"], cfg["labels"]
    patients = list_patients(data["processed_dir"])[:data["max_patients"]]
    split = split_patients(patients, data["val_fraction"], data["test_fraction"], cfg["seed"])
    (run_dir / "split.json").write_text(json.dumps(split, indent=1))
    (run_dir / "config.json").write_text(json.dumps(cfg, indent=1))

    train_seqs = load_sequences(data["processed_dir"], split["train"])
    val_seqs = load_sequences(data["processed_dir"], split["val"])
    train_set = FrameDataset(train_seqs, labels, data["context"], data["frame_step"], augment=True)
    val_set = FrameDataset(val_seqs, labels, data["context"])
    tc = cfg["train"]
    loader_args = dict(batch_size=tc["batch_size"], num_workers=tc["num_workers"],
                       pin_memory=device.type == "cuda")
    train_loader = DataLoader(train_set, shuffle=True, drop_last=True, **loader_args)
    val_loader = DataLoader(val_set, **loader_args)
    print(f"{device} | train {len(split['train'])} patients / {len(train_set)} frames | "
          f"val {len(split['val'])} patients / {len(val_set)} frames")

    model = build_model(cfg).to(device)
    pos_weight = auto_pos_weight(train_seqs, labels) if tc["pos_weight"] == "auto" else tc["pos_weight"]
    criterion = DiceBCELoss(tc["bce_weight"], pos_weight).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=tc["lr"], weight_decay=tc["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=tc["epochs"] * len(train_loader))
    scaler = torch.amp.GradScaler(device.type, enabled=amp)

    start_epoch, best = 0, -1.0
    last_path = run_dir / "last.pt"
    if tc["resume"] and last_path.exists():
        ckpt = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        scaler.load_state_dict(ckpt["scaler"])
        start_epoch, best = ckpt["epoch"] + 1, ckpt["best_dice"]
        print(f"resumed from epoch {start_epoch}")

    log_path = run_dir / "log.csv"
    if start_epoch == 0:
        with log_path.open("w", newline="") as f:
            csv.writer(f).writerow(["epoch", "train_loss", "val_mean_dice", *labels, "seconds"])

    for epoch in range(start_epoch, tc["epochs"]):
        model.train()
        t0, losses = time.time(), []
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device.type, enabled=amp):
                loss = criterion(model(x), y)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            losses.append(loss.item())

        dice = validate(model, val_loader, device, amp, cfg["eval"]["threshold"])
        mean_dice = float(np.mean(dice))
        seconds = time.time() - t0
        with log_path.open("a", newline="") as f:
            csv.writer(f).writerow([epoch, np.mean(losses), mean_dice, *dice, round(seconds)])
        print(f"epoch {epoch} loss {np.mean(losses):.4f} val dice {mean_dice:.4f} ({seconds:.0f}s)")

        ckpt = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
                "epoch": epoch, "best_dice": max(best, mean_dice), "cfg": cfg,
                "config_hash": config_hash(cfg), "split": split}
        torch.save(ckpt, last_path)
        if mean_dice > best:
            best = mean_dice
            torch.save(ckpt, run_dir / "best.pt")
    return run_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    args = parser.parse_args()
    run(load_config(args.config, args.set))
