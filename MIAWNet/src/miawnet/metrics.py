import torch
import torch.nn.functional as F
from torch import Tensor


class SegmentationMetrics:
    def __init__(self, threshold: float = 0.5) -> None:
        if not 0 < threshold < 1:
            raise ValueError("Probability threshold must be in (0, 1)")
        self.threshold = threshold
        self.intersection = 0
        self.union = 0
        self.iou_sum = 0.0
        self.count = 0
        self.hits = {0.5: 0, 0.7: 0, 0.9: 0}

    @torch.no_grad()
    def update(self, logits: Tensor, targets: list[Tensor] | Tensor) -> None:
        if len(logits) != len(targets):
            raise ValueError("Prediction and target batch sizes differ")
        for prediction, target in zip(logits, targets):
            target = target.to(prediction.device).bool().reshape(*target.shape[-2:])
            prediction = F.interpolate(
                prediction[None].float(), size=target.shape, mode="bilinear", align_corners=False
            )[0, 0]
            prediction = prediction.sigmoid() >= self.threshold
            intersection = int((prediction & target).sum())
            union = int((prediction | target).sum())
            iou = intersection / union if union else 1.0
            self.intersection += intersection
            self.union += union
            self.iou_sum += iou
            self.count += 1
            for cutoff in self.hits:
                self.hits[cutoff] += int(iou > cutoff)

    def compute(self) -> dict[str, float]:
        if not self.count:
            raise ValueError("No samples were evaluated")
        return {
            "oIoU": 100 * self.intersection / self.union if self.union else 100.0,
            "mIoU": 100 * self.iou_sum / self.count,
            **{f"P@{cutoff:.1f}": 100 * hits / self.count for cutoff, hits in self.hits.items()},
        }
