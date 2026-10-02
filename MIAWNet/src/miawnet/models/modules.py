import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class CrossModalAttention(nn.Module):
    def __init__(self, visual_dim: int, text_dim: int, dim: int) -> None:
        super().__init__()
        self.query = nn.Conv2d(visual_dim, dim, 1)
        self.key = nn.Linear(text_dim, dim)
        self.value = nn.Linear(text_dim, dim)
        self.projection = nn.Conv2d(dim, dim, 1)

    def forward(self, visual: Tensor, text: Tensor, attention_mask: Tensor) -> Tensor:
        batch, _, height, width = visual.shape
        query = self.query(visual).flatten(2).transpose(1, 2).unsqueeze(1)
        key = self.key(text).unsqueeze(1)
        value = self.value(text).unsqueeze(1)
        valid = attention_mask[:, None, None, :].bool()
        aligned = F.scaled_dot_product_attention(query, key, value, attn_mask=valid)
        aligned = aligned.squeeze(1).transpose(1, 2).reshape(batch, -1, height, width)
        return self.projection(aligned)


class MFIE(nn.Module):
    def __init__(self, dim: int, pool_size: int = 4, local_kernel: int = 3) -> None:
        super().__init__()
        self.pool_size = pool_size
        self.scale = math.sqrt(dim)
        self.input_projection = nn.Conv2d(dim, dim, 1)
        self.qkv = nn.Conv2d(dim, dim * 3, 1)
        self.local_q = nn.Conv2d(dim, dim, local_kernel, padding=local_kernel // 2, groups=dim)
        self.local_k = nn.Conv2d(dim, dim, local_kernel, padding=local_kernel // 2, groups=dim)
        self.local_v = nn.Conv2d(dim, dim, local_kernel, padding=local_kernel // 2, groups=dim)
        self.local_weights = nn.Sequential(
            nn.Conv2d(dim, dim, 1), nn.SiLU(), nn.Conv2d(dim, dim, 1)
        )
        self.output_projection = nn.Conv2d(dim * 2, dim, 1)

    def forward(self, features: Tensor) -> Tensor:
        batch, dim, height, width = features.shape
        query, key, value = self.qkv(self.input_projection(features)).chunk(3, dim=1)
        kernel = min(self.pool_size, height, width)
        pooled_key = F.avg_pool2d(key, kernel, stride=kernel)
        pooled_value = F.avg_pool2d(value, kernel, stride=kernel)
        global_features = F.scaled_dot_product_attention(
            query.flatten(2).transpose(1, 2).unsqueeze(1),
            pooled_key.flatten(2).transpose(1, 2).unsqueeze(1),
            pooled_value.flatten(2).transpose(1, 2).unsqueeze(1),
        )
        global_features = global_features.squeeze(1).transpose(1, 2)
        global_features = global_features.reshape(batch, dim, height, width)
        weights = self.local_weights(self.local_q(query) * self.local_k(key))
        local_features = torch.tanh(weights / self.scale) * self.local_v(value)
        return self.output_projection(torch.cat([local_features, global_features], dim=1))


class AFW(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.original_gate = self._gate(dim)
        self.enhanced_gate = self._gate(dim)

    @staticmethod
    def _gate(dim: int) -> nn.Sequential:
        return nn.Sequential(nn.Conv2d(dim, dim, 1), nn.GELU(), nn.Conv2d(dim, dim, 1), nn.Tanh())

    def forward(self, original: Tensor, enhanced: Tensor) -> Tensor:
        return self.original_gate(original) * original + self.enhanced_gate(enhanced) * enhanced


class MultiScaleDecoder(nn.Module):
    def __init__(self, visual_dims: tuple[int, ...], feature_dim: int, dim: int) -> None:
        super().__init__()
        self.projections = nn.ModuleList(
            [nn.Conv2d(channels + feature_dim, dim, 1) for channels in visual_dims]
        )
        groups = math.gcd(dim, 32)
        self.head = nn.Sequential(
            nn.Conv2d(dim * len(visual_dims), dim, 1, bias=False),
            nn.GroupNorm(groups, dim),
            nn.GELU(),
            nn.Conv2d(dim, dim, 3, padding=1, bias=False),
            nn.GroupNorm(groups, dim),
            nn.GELU(),
            nn.Conv2d(dim, 1, 1),
        )

    def forward(
        self, visual: list[Tensor], semantic: list[Tensor], size: tuple[int, int]
    ) -> Tensor:
        target_size = visual[0].shape[-2:]
        stages = []
        for projection, image_features, text_features in zip(self.projections, visual, semantic):
            stage = projection(torch.cat([image_features, text_features], dim=1))
            stages.append(
                F.interpolate(stage, size=target_size, mode="bilinear", align_corners=False)
            )
        logits = self.head(torch.cat(stages, dim=1))
        return F.interpolate(logits, size=size, mode="bilinear", align_corners=False)
