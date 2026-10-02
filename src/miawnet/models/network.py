from torch import Tensor, nn
from torchvision.models import Swin_B_Weights, swin_b
from transformers import BertConfig, BertModel

from ..config import ModelConfig
from .modules import AFW, MFIE, CrossModalAttention, MultiScaleDecoder


class SwinBackbone(nn.Module):
    channels = (128, 256, 512, 1024)

    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        weights = Swin_B_Weights.IMAGENET1K_V1 if pretrained else None
        self.features = swin_b(weights=weights).features

    def forward(self, image: Tensor) -> list[Tensor]:
        outputs = []
        for index, layer in enumerate(self.features):
            image = layer(image)
            if index in (1, 3, 5, 7):
                outputs.append(image.permute(0, 3, 1, 2).contiguous())
        return outputs


class MIAWNet(nn.Module):
    def __init__(
        self,
        config: ModelConfig,
        *,
        initialize_pretrained: bool | None = None,
        visual_encoder: nn.Module | None = None,
        text_encoder: nn.Module | None = None,
    ) -> None:
        super().__init__()
        pretrained = config.pretrained if initialize_pretrained is None else initialize_pretrained
        self.visual_encoder = (
            visual_encoder if visual_encoder is not None else SwinBackbone(pretrained)
        )
        if text_encoder is not None:
            self.text_encoder = text_encoder
        elif pretrained:
            self.text_encoder = BertModel.from_pretrained(
                config.bert_name,
                add_pooling_layer=False,
                local_files_only=config.local_files_only,
            )
        else:
            self.text_encoder = BertModel(BertConfig(), add_pooling_layer=False)
        visual_dims = self.visual_encoder.channels
        text_dim = self.text_encoder.config.hidden_size
        self.alignment = nn.ModuleList(
            [
                CrossModalAttention(channels, text_dim, config.feature_dim)
                for channels in visual_dims
            ]
        )
        self.interaction = nn.ModuleList(
            [
                MFIE(config.feature_dim, pool, config.local_kernel)
                if config.use_mfie
                else nn.Identity()
                for pool in config.pool_sizes
            ]
        )
        self.weighting = (
            nn.ModuleList([AFW(config.feature_dim) for _ in visual_dims])
            if config.use_afw
            else None
        )
        self.decoder = MultiScaleDecoder(visual_dims, config.feature_dim, config.decoder_dim)

    def forward(self, image: Tensor, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError("Expected RGB images with shape [B, 3, H, W]")
        if input_ids.ndim != 2 or input_ids.shape != attention_mask.shape:
            raise ValueError("Token IDs and attention masks must have shape [B, L]")
        if input_ids.shape[0] != image.shape[0] or not attention_mask.bool().any(dim=1).all():
            raise ValueError("Each image needs a non-empty token sequence")
        visual = self.visual_encoder(image)
        text = self.text_encoder(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state
        semantic = []
        for index, (alignment, interaction, image_features) in enumerate(
            zip(self.alignment, self.interaction, visual)
        ):
            original = alignment(image_features, text, attention_mask)
            enhanced = interaction(original)
            fused = (
                self.weighting[index](original, enhanced)
                if self.weighting is not None
                else enhanced
            )
            semantic.append(fused)
        return self.decoder(visual, semantic, image.shape[-2:])


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
