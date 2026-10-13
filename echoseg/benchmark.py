"""Measure inference latency across precision, memory format, compilation and batch size.

    python -m echoseg.benchmark --checkpoint runs/unet_r34_ctx0/best.pt
"""
import argparse
import time
from pathlib import Path

import torch

from .config import load_config
from .model import build_model, load_checkpoint


@torch.no_grad()
def time_model(model, x, amp, iters, warmup=10):
    """Mean milliseconds per forward pass."""
    device = x.device

    def step():
        with torch.autocast(device.type, enabled=amp):
            model(x)

    for _ in range(warmup):
        step()
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        step()
    if device.type == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000


def benchmark(model, cfg, device, batch_sizes=(1, 4, 8), iters=50):
    size, channels = cfg["data"]["image_size"], cfg["data"]["context"] + 1
    cuda = device.type == "cuda"
    variants = [("fp32", False, False, False), ("fp32 + channels_last", False, True, False)]
    if cuda:
        variants += [("mixed precision", True, False, False),
                     ("mixed precision + channels_last", True, True, False),
                     ("mixed precision + channels_last + torch.compile", True, True, True)]
    rows = []
    for name, amp, channels_last, compiled in variants:
        net = model.to(memory_format=torch.channels_last if channels_last else torch.contiguous_format)
        if compiled:
            net = torch.compile(net)
        for batch in batch_sizes:
            x = torch.rand(batch, channels, size, size, device=device)
            if channels_last:
                x = x.to(memory_format=torch.channels_last)
            try:
                ms = time_model(net, x, amp, iters)
            except Exception as error:  # torch.compile needs a compiler toolchain
                print(f"skipped {name}: {type(error).__name__}")
                break
            rows.append({"variant": name, "batch": batch, "ms_per_batch": ms,
                         "ms_per_frame": ms / batch, "fps": 1000 * batch / ms})
    return rows


def to_markdown(rows, device_name, size):
    lines = [
        f"Inference benchmark on {device_name}, input {size}x{size}.",
        "",
        "| variant | batch | ms / batch | ms / frame | frames / s | max source fps at stride 2 |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(f"| {r['variant']} | {r['batch']} | {r['ms_per_batch']:.1f} | "
                     f"{r['ms_per_frame']:.2f} | {r['fps']:.0f} | {2 * r['fps']:.0f} |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint")
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--iters", type=int, default=50)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.checkpoint:
        model, ckpt = load_checkpoint(args.checkpoint, device)
        cfg, out = ckpt["cfg"], Path(args.checkpoint).parent / "benchmark.md"
    else:
        cfg = load_config(args.config, ["model.encoder_weights=null"])
        model, out = build_model(cfg).to(device).eval(), Path("benchmark.md")
    device_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"
    table = to_markdown(benchmark(model, cfg, device, iters=args.iters), device_name,
                        cfg["data"]["image_size"])
    out.write_text(table)
    print(table)
