"""CAMUS preprocessing, patient-level splits and frame datasets.

Processed format: one ``<patient>_<view>.npz`` per ultrasound sequence holding
``frames`` (T, H, W) uint8, ``masks`` (T, H, W) uint8 class ids, ``fps`` and ``quality``.
"""
import argparse
from pathlib import Path
from typing import NamedTuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

DEFAULT_FPS = 50.0


class Sequence(NamedTuple):
    name: str
    patient: str
    frames: np.ndarray
    masks: np.ndarray
    fps: float
    quality: str


def patient_of(path):
    return Path(path).stem.split("_")[0]


def _read_cfg(path):
    info = {}
    if path.exists():
        for line in path.read_text().splitlines():
            if ":" in line:
                key, value = line.split(":", 1)
                info[key.strip()] = value.strip()
    return info


def preprocess_camus(raw_dir, out_dir, size=256):
    """Convert CAMUS half-sequence NIfTI files into resized npz sequences."""
    import nibabel as nib

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for img_path in sorted(Path(raw_dir).rglob("*_half_sequence.nii.gz")):
        gt_path = img_path.with_name(img_path.name.replace("_half_sequence", "_half_sequence_gt"))
        if not gt_path.exists():
            continue
        patient, view = img_path.name.split("_")[:2]
        # NIfTI arrays are (W, H, T); move to (T, H, W).
        frames = np.asanyarray(nib.load(img_path).dataobj).transpose(2, 1, 0)
        masks = np.asanyarray(nib.load(gt_path).dataobj).transpose(2, 1, 0)
        frames = np.stack([cv2.resize(f, (size, size), interpolation=cv2.INTER_AREA) for f in frames])
        masks = np.stack([cv2.resize(m, (size, size), interpolation=cv2.INTER_NEAREST) for m in masks])
        info = _read_cfg(img_path.parent / f"Info_{view}.cfg")
        np.savez_compressed(
            out_dir / f"{patient}_{view}.npz",
            frames=np.clip(frames, 0, 255).astype(np.uint8),
            masks=masks.astype(np.uint8),
            fps=float(info.get("FrameRate") or DEFAULT_FPS),
            quality=info.get("ImageQuality", "unknown"),
        )
        written += 1
    return written


def make_synthetic(out_dir, n_patients=8, n_frames=12, size=64, seed=0):
    """Write fake echo-like sequences (pulsating ring plus an intermittent blob) for tests."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:size, :size]
    for p in range(n_patients):
        for view in ("2CH", "4CH"):
            cx, cy = rng.uniform(0.4, 0.6, 2) * size
            radius = rng.uniform(0.14, 0.2) * size
            gone = rng.integers(0, n_frames)  # the blob is out of view for a few frames
            frames, masks = [], []
            for t in range(n_frames):
                r = radius * (1 + 0.15 * np.sin(2 * np.pi * t / n_frames))
                dist = np.hypot(xx - cx, yy - cy)
                mask = np.zeros((size, size), np.uint8)
                mask[dist < r + 0.06 * size] = 2
                mask[dist < r] = 1
                if not gone <= t < gone + 3:
                    blob = np.hypot(xx - cx, yy - cy - 2.2 * radius) < 0.7 * radius
                    mask[blob & (mask == 0)] = 3
                image = np.choose(mask, [30, 60, 170, 90]) * rng.gamma(4, 0.25, mask.shape)
                frames.append(np.clip(image, 0, 255).astype(np.uint8))
                masks.append(mask)
            np.savez_compressed(
                out_dir / f"patient{p:04d}_{view}.npz",
                frames=np.stack(frames), masks=np.stack(masks), fps=DEFAULT_FPS, quality="Good",
            )


def list_patients(processed_dir):
    return sorted({patient_of(p) for p in Path(processed_dir).glob("*.npz")})


def split_patients(patients, val_fraction, test_fraction, seed):
    """Deterministic patient-level split, so no patient contributes frames to two splits."""
    ids = sorted(set(patients))
    order = np.random.default_rng(seed).permutation(len(ids))
    n_test = max(1, round(len(ids) * test_fraction))
    n_val = max(1, round(len(ids) * val_fraction))
    shuffled = [ids[i] for i in order]
    return {
        "test": sorted(shuffled[:n_test]),
        "val": sorted(shuffled[n_test:n_test + n_val]),
        "train": sorted(shuffled[n_test + n_val:]),
    }


def load_sequences(processed_dir, patients):
    wanted = set(patients)
    sequences = []
    for path in sorted(Path(processed_dir).glob("*.npz")):
        if patient_of(path) in wanted:
            with np.load(path) as z:
                sequences.append(Sequence(path.stem, patient_of(path), z["frames"], z["masks"],
                                          float(z["fps"]), str(z["quality"])))
    return sequences


def to_multilabel(masks, labels):
    """Class-id masks (..., H, W) -> boolean multi-label masks (..., L, H, W)."""
    return np.stack([np.isin(masks, ids) for ids in labels.values()], axis=-3)


def stack_context(frames, t, context):
    """Frame t with its ``context`` previous frames as channels; the first frame is repeated."""
    idx = np.clip(np.arange(t - context, t + 1), 0, len(frames) - 1)
    return frames[idx]


class FrameDataset(Dataset):
    def __init__(self, sequences, labels, context=0, frame_step=1, augment=False):
        self.sequences = sequences
        self.labels = labels
        self.context = context
        self.augment = augment
        self.index = [(s, t) for s, seq in enumerate(sequences)
                      for t in range(0, len(seq.frames), frame_step)]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        s, t = self.index[i]
        seq = self.sequences[s]
        x = torch.from_numpy(stack_context(seq.frames, t, self.context)).float() / 255
        y = torch.from_numpy(to_multilabel(seq.masks[t], self.labels)).float()
        if self.augment:
            # Ultrasound gain and contrast vary by machine and operator.
            gamma = float(np.random.uniform(0.7, 1.5))
            gain = float(np.random.uniform(0.8, 1.2))
            x = (x.pow(gamma) * gain).clamp(0, 1)
        return x, y


def sequence_tensor(seq, context):
    """All frames of a sequence as model input (T, context + 1, H, W)."""
    x = np.stack([stack_context(seq.frames, t, context) for t in range(len(seq.frames))])
    return torch.from_numpy(x).float() / 255


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    pre = sub.add_parser("preprocess")
    pre.add_argument("--raw", required=True)
    pre.add_argument("--out", default="data/processed")
    pre.add_argument("--size", type=int, default=256)
    syn = sub.add_parser("synthetic")
    syn.add_argument("--out", default="data/synthetic")
    args = parser.parse_args()
    if args.command == "preprocess":
        print(f"wrote {preprocess_camus(args.raw, args.out, args.size)} sequences to {args.out}")
    else:
        make_synthetic(args.out)
