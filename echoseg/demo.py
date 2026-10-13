"""Render a ground-truth vs prediction overlay GIF for one test sequence.

    python -m echoseg.demo --checkpoint runs/unet_r34_ctx0/best.pt --out assets/demo.gif
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from .data import load_sequences, to_multilabel
from .evaluate import predict_sequence
from .model import load_checkpoint

COLORS = [(66, 220, 120), (255, 170, 40), (80, 170, 255), (240, 90, 200), (250, 240, 90)]


def overlay(frame, masks, title, scale=2):
    """Draw each label as a coloured contour, which stays readable where labels overlap."""
    size = frame.shape[0] * scale
    image = cv2.cvtColor(cv2.resize(frame, (size, size)), cv2.COLOR_GRAY2RGB)
    for k, mask in enumerate(masks):
        big = cv2.resize(mask.astype(np.uint8), (size, size), interpolation=cv2.INTER_NEAREST)
        contours, _ = cv2.findContours(big, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(image, contours, -1, COLORS[k % len(COLORS)], 2)
    cv2.putText(image, title, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return image


def legend(labels, width):
    bar = np.zeros((26, width, 3), np.uint8)
    x = 8
    for k, name in enumerate(labels):
        cv2.rectangle(bar, (x, 8), (x + 12, 20), COLORS[k % len(COLORS)], -1)
        cv2.putText(bar, name, (x + 18, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        x += 30 + 9 * len(name)
    return bar


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", default="assets/demo.gif")
    parser.add_argument("--index", type=int, default=0, help="which test sequence to render")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ckpt = load_checkpoint(args.checkpoint, device)
    cfg = ckpt["cfg"]
    seq = load_sequences(cfg["data"]["processed_dir"], ckpt["split"]["test"])[args.index]
    pred = predict_sequence(model, seq, cfg, device)
    gt = to_multilabel(seq.masks, cfg["labels"])
    images = []
    for t, frame in enumerate(seq.frames):
        pair = np.hstack([overlay(frame, gt[t], "clinician annotation"),
                          overlay(frame, pred[t], "model prediction")])
        images.append(Image.fromarray(np.vstack([pair, legend(cfg["labels"], pair.shape[1])])))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    # Played forward then backward at reduced speed so the short clip loops smoothly.
    images += images[-2:0:-1]
    images[0].save(args.out, save_all=True, append_images=images[1:], duration=100, loop=0)
    print(f"wrote {args.out} from {seq.name}")
