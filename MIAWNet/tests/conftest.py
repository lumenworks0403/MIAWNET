import torch
from torch import nn
from transformers import BertConfig, BertModel

from miawnet.config import ModelConfig
from miawnet.models import MIAWNet

torch.set_num_threads(2)


class TinyVision(nn.Module):
    channels = (8, 16, 24, 32)

    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                nn.Conv2d(source, target, 3, stride=2, padding=1)
                for source, target in zip((3, *self.channels[:-1]), self.channels)
            ]
        )

    def forward(self, image):
        outputs = []
        for layer in self.layers:
            image = layer(image)
            outputs.append(image)
        return outputs


def tiny_model(config=None, **kwargs):
    config = config or ModelConfig(pretrained=False, feature_dim=8, decoder_dim=8)
    text = BertModel(
        BertConfig(
            vocab_size=32,
            hidden_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=32,
        ),
        add_pooling_layer=False,
    )
    return MIAWNet(config, visual_encoder=TinyVision(), text_encoder=text)


class TinyTokenizer:
    def __call__(self, texts, **kwargs):
        size = kwargs["max_length"]
        return {
            "input_ids": torch.ones(len(texts), size, dtype=torch.long),
            "attention_mask": torch.ones(len(texts), size, dtype=torch.long),
        }
