import math

import pytest
import torch
import torch.nn.functional as F
from conftest import tiny_model

from miawnet.config import ModelConfig
from miawnet.losses import SegmentationLoss
from miawnet.models.modules import AFW, MFIE, CrossModalAttention


@pytest.mark.parametrize("mfie,afw", [(False, False), (True, False), (False, True), (True, True)])
def test_ablation_backward(mfie, afw):
    config = ModelConfig(pretrained=False, feature_dim=8, decoder_dim=8, use_mfie=mfie, use_afw=afw)
    model = tiny_model(config)
    image = torch.randn(2, 3, 32, 32)
    tokens = torch.randint(0, 32, (2, 6))
    logits = model(image, tokens, torch.ones_like(tokens))
    assert logits.shape == (2, 1, 32, 32)
    target = torch.zeros_like(logits)
    target[:, :, 4:12, 7:15] = 1
    SegmentationLoss()(logits, target)["loss"].backward()
    parameters = dict(model.named_parameters())
    for name, parameter in parameters.items():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name


def test_cross_modal_padding_is_ignored():
    layer = CrossModalAttention(8, 16, 8).eval()
    visual = torch.randn(2, 8, 5, 7)
    text = torch.randn(2, 6, 16)
    mask = torch.tensor([[1, 1, 1, 0, 0, 0], [1, 1, 0, 0, 0, 0]])
    changed = text.clone()
    changed[~mask.bool()] = torch.randn_like(changed[~mask.bool()]) * 100
    torch.testing.assert_close(layer(visual, text, mask), layer(visual, changed, mask))


def test_mfie_matches_explicit_equations():
    torch.manual_seed(7)
    layer = MFIE(4, pool_size=2).double()
    features = torch.randn(2, 4, 6, 8, dtype=torch.float64)
    q, k, v = layer.qkv(layer.input_projection(features)).chunk(3, dim=1)
    query = q.flatten(2).transpose(1, 2)
    key = F.avg_pool2d(k, 2).flatten(2)
    value = F.avg_pool2d(v, 2).flatten(2).transpose(1, 2)
    global_features = (query @ key / math.sqrt(4)).softmax(dim=-1) @ value
    global_features = global_features.transpose(1, 2).reshape_as(features)
    local_weights = layer.local_weights(layer.local_q(q) * layer.local_k(k))
    local_features = (local_weights / math.sqrt(4)).tanh() * layer.local_v(v)
    expected = layer.output_projection(torch.cat([local_features, global_features], dim=1))
    torch.testing.assert_close(layer(features), expected)


def test_afw_allows_independent_signed_weights():
    layer = AFW(4)
    with torch.no_grad():
        for parameter in layer.parameters():
            parameter.zero_()
        layer.original_gate[2].bias.fill_(-1)
        layer.enhanced_gate[2].bias.fill_(0.5)
    original, enhanced = torch.ones(1, 4, 3, 3), torch.full((1, 4, 3, 3), 2.0)
    expected = math.tanh(-1) * original + math.tanh(0.5) * enhanced
    torch.testing.assert_close(layer(original, enhanced), expected)


def test_empty_attention_mask_is_rejected():
    model = tiny_model()
    with pytest.raises(ValueError, match="non-empty"):
        model(
            torch.randn(1, 3, 32, 32),
            torch.ones(1, 5, dtype=torch.long),
            torch.zeros(1, 5, dtype=torch.long),
        )
