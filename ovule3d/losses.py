from __future__ import annotations

import torch
from torch.nn import functional as F


def topology_loss(logits: torch.Tensor, target: torch.Tensor, weights: torch.Tensor):
    valid = target != 255
    if not valid.any():
        raise ValueError("Sample has no valid annotated voxels")
    ce = F.cross_entropy(logits.float(), target, weight=weights, ignore_index=255)
    safe = target.clamp(0, logits.shape[1] - 1)
    truth = F.one_hot(safe, num_classes=logits.shape[1]).movedim(-1, 1).float()
    mask = valid[:, None]
    truth = truth * mask
    probability = logits.float().softmax(1) * mask
    axes = (0, 2, 3, 4)
    numerator = 2 * (truth * probability).sum(axes)
    denominator = (truth + probability).sum(axes)
    active = truth.sum(axes) > 0
    dice = ((numerator + 1e-6) / (denominator + 1e-6))[active]
    return ce + (1 - dice.mean()), {"ce": float(ce.detach()), "soft_dice": float(dice.mean().detach())}


def confusion_matrix(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    predicted = logits.argmax(1)
    valid = target != 255
    return torch.bincount(target[valid] * 3 + predicted[valid], minlength=9).reshape(3, 3).to(torch.float64)


def confusion_metrics(confusion: torch.Tensor) -> dict:
    confusion = confusion.detach().cpu().double()
    denominator = confusion.sum(0) + confusion.sum(1)
    dice = (2 * confusion.diag() / denominator.clamp_min(1)).tolist()
    return {"dice_bg": dice[0], "dice_interior": dice[1], "dice_wall": dice[2],
            "macro_dice_interior_wall": (dice[1] + dice[2]) / 2,
            "confusion": confusion.to(torch.int64).tolist()}
