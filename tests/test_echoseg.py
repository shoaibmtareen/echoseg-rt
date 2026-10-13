import json

import numpy as np
import pytest
import torch

from echoseg import data, metrics
from echoseg.config import load_config
from echoseg.evaluate import evaluate, to_markdown
from echoseg.model import DiceBCELoss
from echoseg.train import run

LABELS = {"lv_endo": [1], "myocardium": [2], "lv_epi": [1, 2], "left_atrium": [3]}


@pytest.fixture(scope="module")
def processed(tmp_path_factory):
    out = tmp_path_factory.mktemp("processed")
    data.make_synthetic(out, n_patients=8, n_frames=8, size=64)
    return out


def test_split_has_no_patient_leakage():
    patients = [f"patient{i:04d}" for i in range(100)]
    split = data.split_patients(patients, 0.1, 0.2, seed=42)
    train, val, test = (set(split[k]) for k in ("train", "val", "test"))
    assert not (train & val or train & test or val & test)
    assert train | val | test == set(patients)
    assert (len(val), len(test)) == (10, 20)
    assert split == data.split_patients(patients[::-1], 0.1, 0.2, seed=42)
    assert split != data.split_patients(patients, 0.1, 0.2, seed=7)


def test_both_views_of_a_patient_stay_in_one_split(processed):
    split = data.split_patients(data.list_patients(processed), 0.25, 0.25, seed=0)
    seen = {}
    for name, patients in split.items():
        for seq in data.load_sequences(processed, patients):
            assert seen.setdefault(seq.patient, name) == name
    assert len(seen) == 8


def test_preprocess_reorients_nifti_to_time_first(tmp_path):
    nib = pytest.importorskip("nibabel")
    folder = tmp_path / "raw" / "patient0001"
    folder.mkdir(parents=True)
    frames = np.random.default_rng(0).uniform(0, 255, (40, 60, 5)).astype(np.float32)  # (W, H, T)
    masks = np.zeros((40, 60, 5), np.float32)
    masks[:20, :, 3] = 2
    nib.save(nib.Nifti1Image(frames, np.eye(4)), folder / "patient0001_2CH_half_sequence.nii.gz")
    nib.save(nib.Nifti1Image(masks, np.eye(4)), folder / "patient0001_2CH_half_sequence_gt.nii.gz")
    (folder / "Info_2CH.cfg").write_text("FrameRate: 48.4\nImageQuality: Good\n")
    assert data.preprocess_camus(tmp_path / "raw", tmp_path / "out", size=32) == 1
    seq = data.load_sequences(tmp_path / "out", ["patient0001"])[0]
    assert seq.frames.shape == seq.masks.shape == (5, 32, 32)
    assert (seq.fps, seq.quality) == (48.4, "Good")
    assert set(np.unique(seq.masks[3])) == {0, 2} and not seq.masks[0].any()
    assert (seq.masks[3][:, :16] == 2).all()  # the W axis ends up as image columns


def test_dataset_shapes_context_and_overlapping_labels(processed):
    seqs = data.load_sequences(processed, data.list_patients(processed)[:2])
    x, y = data.FrameDataset(seqs, LABELS, context=2)[0]
    assert x.shape == (3, 64, 64) and y.shape == (4, 64, 64)
    assert torch.equal(x[0], x[2])  # first frame has no history, so it is repeated
    endo, myo, epi = y[0].bool(), y[1].bool(), y[2].bool()
    assert torch.equal(epi, endo | myo) and endo.any() and myo.any()
    assert len(data.FrameDataset(seqs, LABELS, frame_step=2)) == len(seqs) * 4


def test_dice_and_iou():
    gt = np.zeros((2, 1, 4, 4), bool)
    pred = np.zeros_like(gt)
    gt[0, 0, :2] = True       # 8 pixels
    pred[0, 0, 1:3] = True    # 8 pixels, 4 shared
    dice, iou = metrics.dice_iou(pred, gt)
    assert dice[0, 0] == pytest.approx(0.5) and iou[0, 0] == pytest.approx(1 / 3)
    assert np.isnan(dice[1, 0])  # empty in both


def test_event_metrics():
    gt = np.array([0, 1, 1, 0, 0, 1, 1, 1, 0, 0], bool)
    pred = np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 1], bool)
    assert metrics.runs(gt) == [(1, 3), (5, 8)]
    assert metrics.presence_events(gt, pred) == {"gt_events": 2, "detected": 1, "false_alarms": 1}
    dice = np.array([0.9, 0.5, 0.4, 0.9, 0.2, 0.1, 0.3, np.nan, 0.6])
    assert metrics.failure_events(dice, threshold=0.7, min_frames=3) == 1
    assert metrics.area_jitter([10, 10, 10]) == 0


def test_loss_is_lower_for_better_predictions():
    target = torch.zeros(2, 4, 16, 16)
    target[:, :, 4:12, 4:12] = 1
    criterion = DiceBCELoss(pos_weight=[1.0, 2.0, 1.0, 3.0])
    good = criterion((target * 2 - 1) * 8, target)
    assert good < criterion(torch.zeros_like(target), target) < criterion(-(target * 2 - 1) * 8, target)
    assert torch.isfinite(criterion((target * 2 - 1).half() * 8, target))


def test_train_evaluate_end_to_end(processed, tmp_path):
    cfg = load_config("configs/base.yaml", [
        f"data.processed_dir={processed.as_posix()}", "data.image_size=64", "data.context=1",
        "data.val_fraction=0.25", "data.test_fraction=0.25", "model.encoder=resnet18",
        "model.encoder_weights=null", "train.epochs=2", "train.batch_size=8",
        "train.num_workers=0", "train.pos_weight=auto", "run_name=smoke",
    ])
    run_dir = run(cfg, runs_dir=tmp_path)
    split = json.loads((run_dir / "split.json").read_text())
    assert (run_dir / "best.pt").exists() and len((run_dir / "log.csv").read_text().splitlines()) == 3

    report = evaluate(run_dir / "best.pt")
    assert report["patients"] == len(split["test"]) == 2 and report["sequences"] == 4
    assert set(report["per_label"]) == set(LABELS)
    assert 0 <= report["mean_dice"] <= 1
    # The synthetic atrium leaves the view for a few frames, so some sequences hold two events.
    assert report["per_label"]["left_atrium"]["gt_events"] > report["per_label"]["lv_endo"]["gt_events"] == 4
    assert "| lv_epi |" in to_markdown(report)

    cfg["train"].update(epochs=3, resume=True)
    run(cfg, runs_dir=tmp_path)
    assert len((run_dir / "log.csv").read_text().splitlines()) == 4


def test_config_rejects_unknown_override():
    with pytest.raises(KeyError):
        load_config("configs/base.yaml", ["train.epocs=3"])
