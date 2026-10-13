"""Per-label, frame-level and event-level segmentation metrics (numpy)."""
import numpy as np


def dice_iou(pred, gt):
    """Boolean (T, L, H, W) arrays -> Dice and IoU of shape (T, L); NaN where both are empty."""
    inter = (pred & gt).sum((-1, -2))
    p, g = pred.sum((-1, -2)), gt.sum((-1, -2))
    with np.errstate(invalid="ignore", divide="ignore"):
        dice = np.where(p + g > 0, 2 * inter / (p + g), np.nan)
        iou = np.where(p + g > 0, inter / (p + g - inter), np.nan)
    return dice, iou


def runs(flags):
    """Half-open (start, end) index ranges of consecutive True values."""
    out, start = [], None
    for i, flag in enumerate(flags):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            out.append((start, i))
            start = None
    if start is not None:
        out.append((start, len(flags)))
    return out


def presence_events(gt_present, pred_present):
    """An event is one continuous appearance of a label across frames.

    A ground-truth event is detected if the label is predicted in any of its frames.
    A false alarm is a predicted appearance that overlaps no ground-truth event.
    """
    gt_runs, pred_runs = runs(gt_present), runs(pred_present)
    detected = sum(bool(np.any(pred_present[a:b])) for a, b in gt_runs)
    false_alarms = sum(not np.any(gt_present[a:b]) for a, b in pred_runs)
    return {"gt_events": len(gt_runs), "detected": detected, "false_alarms": false_alarms}


def failure_events(dice, threshold, min_frames):
    """Count runs of at least ``min_frames`` consecutive frames with Dice below threshold."""
    failed = np.nan_to_num(dice, nan=1.0) < threshold
    return sum(b - a >= min_frames for a, b in runs(failed))


def area_jitter(area):
    """Mean frame-to-frame area change relative to mean area; higher means more flicker."""
    area = np.asarray(area, dtype=float)
    if len(area) < 2 or area.mean() == 0:
        return float("nan")
    return float(np.abs(np.diff(area)).mean() / area.mean())


def ratio(num, den):
    return float(num / den) if den else float("nan")
