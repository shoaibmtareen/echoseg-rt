"""Model construction, loss and checkpoint loading."""
import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F
from torch import nn


def build_model(cfg):
    return smp.create_model(
        cfg["model"]["arch"],
        encoder_name=cfg["model"]["encoder"],
        encoder_weights=cfg["model"]["encoder_weights"],
        in_channels=cfg["data"]["context"] + 1,
        classes=len(cfg["labels"]),
    )


def load_checkpoint(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ckpt["cfg"]
    cfg["model"]["encoder_weights"] = None  # weights come from the checkpoint
    model = build_model(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    return model.eval(), ckpt


class DiceBCELoss(nn.Module):
    """Per-label soft Dice plus BCE. Dice keeps small and thin labels from being ignored."""

    def __init__(self, bce_weight=0.5, pos_weight=None):
        super().__init__()
        self.bce_weight = bce_weight
        self.register_buffer("pos_weight", None if pos_weight is None
                             else torch.as_tensor(pos_weight, dtype=torch.float32).view(-1, 1, 1))

    def forward(self, logits, target):
        logits = logits.float()  # pixel sums overflow in half precision
        bce = F.binary_cross_entropy_with_logits(logits, target, pos_weight=self.pos_weight)
        prob = torch.sigmoid(logits)
        inter = (prob * target).sum((0, 2, 3))
        total = prob.sum((0, 2, 3)) + target.sum((0, 2, 3))
        dice = 1 - ((2 * inter + 1) / (total + 1)).mean()
        return self.bce_weight * bce + (1 - self.bce_weight) * dice
