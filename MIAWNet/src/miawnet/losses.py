import torch.nn.functional as F
from torch import Tensor, nn


class SegmentationLoss(nn.Module):
    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, logits: Tensor, target: Tensor) -> dict[str, Tensor]:
        if logits.shape != target.shape:
            raise ValueError(f"Mask shapes differ: {logits.shape} and {target.shape}")
        logits, target = logits.float(), target.float()
        bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        bce = bce.flatten(1).mean(dim=1)
        probability = logits.sigmoid().flatten(1)
        truth = target.flatten(1)
        overlap = (probability * truth).sum(dim=1)
        dice = 1 - (2 * overlap + self.eps) / (probability.sum(dim=1) + truth.sum(dim=1) + self.eps)
        return {"loss": (bce + dice).mean(), "bce": bce.mean(), "dice": dice.mean()}
