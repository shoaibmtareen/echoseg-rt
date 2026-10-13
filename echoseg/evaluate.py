"""Evaluate a checkpoint on its held-out test patients.

    python -m echoseg.evaluate --checkpoint runs/unet_r34_ctx0/best.pt
"""
import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from .data import load_sequences, sequence_tensor, to_multilabel
from .metrics import area_jitter, dice_iou, failure_events, presence_events, ratio
from .model import load_checkpoint


@torch.no_grad()
def predict_sequence(model, seq, cfg, device, batch_size=32):
    """Boolean predictions (T, L, H, W) for every frame of a sequence."""
    x = sequence_tensor(seq, cfg["data"]["context"])
    amp = cfg["train"]["amp"] and device.type == "cuda"
    out = []
    for i in range(0, len(x), batch_size):
        with torch.autocast(device.type, enabled=amp):
            logits = model(x[i:i + batch_size].to(device))
        out.append((torch.sigmoid(logits.float()) > cfg["eval"]["threshold"]).cpu().numpy())
    return np.concatenate(out)


def evaluate(checkpoint, split="test"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ckpt = load_checkpoint(checkpoint, device)
    cfg, ec = ckpt["cfg"], ckpt["cfg"]["eval"]
    labels = list(cfg["labels"])
    # The split saved in the checkpoint is used, so evaluation can never touch training patients.
    patients = ckpt["split"][split]
    sequences = load_sequences(cfg["data"]["processed_dir"], patients)

    acc = {name: defaultdict(float) for name in labels}
    dice_all = {name: [] for name in labels}
    iou_all = {name: [] for name in labels}
    jitter = {name: {"pred": [], "gt": []} for name in labels}
    by_quality = defaultdict(list)
    minutes = n_frames = 0
    for seq in sequences:
        pred = predict_sequence(model, seq, cfg, device)
        gt = to_multilabel(seq.masks, cfg["labels"])
        dice, iou = dice_iou(pred, gt)
        pred_area, gt_area = pred.sum((-1, -2)), gt.sum((-1, -2))
        pred_present, gt_present = pred_area >= ec["min_area"], gt_area >= ec["min_area"]
        minutes += len(seq.frames) / seq.fps / 60
        n_frames += len(seq.frames)
        by_quality[seq.quality].append(np.nanmean(dice))
        for k, name in enumerate(labels):
            a = acc[name]
            dice_all[name].extend(dice[:, k])
            iou_all[name].extend(iou[:, k])
            a["tp"] += np.sum(pred_present[:, k] & gt_present[:, k])
            a["fp"] += np.sum(pred_present[:, k] & ~gt_present[:, k])
            a["fn"] += np.sum(~pred_present[:, k] & gt_present[:, k])
            for key, value in presence_events(gt_present[:, k], pred_present[:, k]).items():
                a[key] += value
            a["failures"] += failure_events(dice[:, k], ec["fail_dice"], ec["fail_min_frames"])
            jitter[name]["pred"].append(area_jitter(pred_area[:, k]))
            jitter[name]["gt"].append(area_jitter(gt_area[:, k]))

    per_label = {}
    for name in labels:
        a = acc[name]
        per_label[name] = {
            "dice": float(np.nanmean(dice_all[name])),
            "iou": float(np.nanmean(iou_all[name])),
            "frame_precision": ratio(a["tp"], a["tp"] + a["fp"]),
            "frame_recall": ratio(a["tp"], a["tp"] + a["fn"]),
            "gt_events": int(a["gt_events"]),
            "event_recall": ratio(a["detected"], a["gt_events"]),
            "false_alarms": int(a["false_alarms"]),
            "false_alarms_per_min": ratio(a["false_alarms"], minutes),
            "sustained_failures": int(a["failures"]),
            "sustained_failures_per_sequence": ratio(a["failures"], len(sequences)),
            "area_jitter_pred": float(np.nanmean(jitter[name]["pred"])),
            "area_jitter_gt": float(np.nanmean(jitter[name]["gt"])),
        }
    return {
        "run_name": cfg["run_name"],
        "config_hash": ckpt["config_hash"],
        "checkpoint_epoch": ckpt["epoch"],
        "split": split,
        "patients": len(patients),
        "sequences": len(sequences),
        "frames": n_frames,
        "video_minutes": minutes,
        "mean_dice": float(np.mean([m["dice"] for m in per_label.values()])),
        "mean_iou": float(np.mean([m["iou"] for m in per_label.values()])),
        "dice_by_image_quality": {q: float(np.mean(v)) for q, v in sorted(by_quality.items())},
        "per_label": per_label,
    }


def to_markdown(report):
    def fmt(v):
        return "n/a" if isinstance(v, float) and math.isnan(v) else f"{v:.3f}" if isinstance(v, float) else str(v)

    columns = ["dice", "iou", "frame_precision", "frame_recall", "gt_events", "event_recall",
               "false_alarms_per_min", "sustained_failures_per_sequence", "area_jitter_pred", "area_jitter_gt"]
    lines = [
        f"### {report['run_name']} ({report['split']} split)",
        "",
        f"{report['patients']} patients, {report['sequences']} sequences, {report['frames']} frames, "
        f"{report['video_minutes']:.2f} min of video. Config `{report['config_hash']}`, "
        f"epoch {report['checkpoint_epoch']}.",
        "",
        f"**Mean Dice {report['mean_dice']:.3f}, mean IoU {report['mean_iou']:.3f}**",
        "",
        "| label | " + " | ".join(columns) + " |",
        "|---|" + "---|" * len(columns),
    ]
    for name, m in report["per_label"].items():
        lines.append(f"| {name} | " + " | ".join(fmt(m[c]) for c in columns) + " |")
    quality = ", ".join(f"{q}: {v:.3f}" for q, v in report["dice_by_image_quality"].items())
    lines += ["", f"Mean Dice by image quality: {quality}"]
    return "\n".join(lines) + "\n"


def _json_safe(value):
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    return None if isinstance(value, float) and math.isnan(value) else value


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test")
    args = parser.parse_args()
    result = evaluate(args.checkpoint, args.split)
    out_dir = Path(args.checkpoint).parent
    (out_dir / f"report_{args.split}.json").write_text(json.dumps(_json_safe(result), indent=1))
    (out_dir / f"report_{args.split}.md").write_text(to_markdown(result))
    print(to_markdown(result))
