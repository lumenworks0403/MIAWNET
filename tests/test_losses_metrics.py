import math

import pytest
import torch

from miawnet.losses import SegmentationLoss
from miawnet.metrics import SegmentationMetrics


def test_loss_is_per_sample_bce_plus_dice():
    logits = torch.zeros(2, 1, 2, 2, requires_grad=True)
    target = torch.tensor([[[[1, 0], [0, 0]]], [[[1, 1], [1, 0]]]], dtype=torch.float32)
    eps = 1e-6
    values = SegmentationLoss(eps)(logits, target)
    expected_dice = ((1 - (1 + eps) / (3 + eps)) + (1 - (3 + eps) / (5 + eps))) / 2
    assert values["bce"].item() == pytest.approx(math.log(2))
    assert values["dice"].item() == pytest.approx(expected_dice)
    assert values["loss"].item() == pytest.approx(math.log(2) + expected_dice)
    values["loss"].backward()
    assert torch.isfinite(logits.grad).all()


def test_loss_handles_extreme_logits():
    logits = torch.tensor([[[[-1000.0, 1000.0]]]], requires_grad=True)
    target = torch.tensor([[[[1.0, 0.0]]]])
    loss = SegmentationLoss()(logits, target)["loss"]
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(logits.grad).all()


def test_metrics_aggregate_pixels_and_strict_precision():
    target = torch.tensor([[[[1, 1], [1, 1]]], [[[1, 1], [0, 0]]]])
    prediction = torch.tensor([[[[1, 1], [1, 1]]], [[[1, 0], [0, 0]]]])
    logits = prediction.float() * 20 - 10
    metric = SegmentationMetrics()
    metric.update(logits[:1], target[:1])
    metric.update(logits[1:], target[1:])
    scores = metric.compute()
    assert scores["oIoU"] == pytest.approx(100 * 5 / 6)
    assert scores["mIoU"] == pytest.approx(75)
    assert scores["P@0.5"] == 50
    assert scores["P@0.7"] == 50
    assert scores["P@0.9"] == 50


def test_metrics_resize_logits_before_thresholding():
    metrics = SegmentationMetrics()
    targets = [torch.ones(1, 7, 9), torch.zeros(1, 3, 5)]
    metrics.update(torch.tensor([[[[10.0]]], [[[-10.0]]]]), targets)
    assert metrics.compute()["mIoU"] == 100


def test_metrics_need_samples():
    with pytest.raises(ValueError, match="No samples"):
        SegmentationMetrics().compute()
