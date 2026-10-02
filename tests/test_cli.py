import json
import sys

import numpy as np
import torch
from conftest import TinyVision, tiny_model
from PIL import Image
from transformers import BertTokenizerFast

from miawnet import engine, evaluate, predict
from miawnet.config import Config, ModelConfig
from miawnet.models import MIAWNet


def create_offline_checkpoint(tmp_path, monkeypatch):
    bert_dir = tmp_path / "bert"
    bert_dir.mkdir()
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "the", "white", "car"]
    vocab += [f"word{index}" for index in range(24)]
    vocab_path = bert_dir / "vocab.txt"
    vocab_path.write_text("\n".join(vocab), encoding="utf-8")
    BertTokenizerFast(vocab_file=str(vocab_path)).save_pretrained(bert_dir)
    config = Config(
        model=ModelConfig(
            pretrained=False,
            feature_dim=8,
            decoder_dim=8,
            bert_name=str(bert_dir),
            local_files_only=True,
        )
    )
    config.data.image_size = 32
    config.data.root = str(tmp_path)
    model = tiny_model(config.model).eval()
    checkpoint = tmp_path / "model.pt"
    engine.save_checkpoint(
        checkpoint,
        {
            "config": config.to_dict(),
            "model": model.state_dict(),
            "bert_config": model.text_encoder.config.to_dict(),
        },
    )

    def build(config, initialize_pretrained=None, text_encoder=None):
        return MIAWNet(config, visual_encoder=TinyVision(), text_encoder=text_encoder)

    monkeypatch.setattr(engine, "MIAWNet", build)
    image_path = tmp_path / "scene.png"
    Image.fromarray(np.full((19, 27, 3), 128, dtype=np.uint8)).save(image_path)
    return checkpoint, image_path, model


def test_checkpoint_load_uses_saved_bert_configuration(tmp_path, monkeypatch):
    checkpoint, _, original = create_offline_checkpoint(tmp_path, monkeypatch)
    restored, _ = engine.load_model(checkpoint, torch.device("cpu"))
    image = torch.randn(1, 3, 32, 32)
    ids = torch.tensor([[2, 5, 7, 3]])
    with torch.no_grad():
        torch.testing.assert_close(
            original(image, ids, torch.ones_like(ids)),
            restored(image, ids, torch.ones_like(ids)),
            rtol=0,
            atol=0,
        )


def test_prediction_and_evaluation_commands_work_offline(tmp_path, monkeypatch):
    checkpoint, image_path, _ = create_offline_checkpoint(tmp_path, monkeypatch)
    mask_path, overlay_path = tmp_path / "prediction.png", tmp_path / "overlay.png"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "predict",
            "--checkpoint",
            str(checkpoint),
            "--image",
            str(image_path),
            "--text",
            "the white car",
            "--output",
            str(mask_path),
            "--overlay",
            str(overlay_path),
            "--device",
            "cpu",
        ],
    )
    predict.main()
    with Image.open(mask_path) as mask:
        assert mask.size == (27, 19)
        assert set(np.unique(mask).tolist()).issubset({0, 255})
    with Image.open(overlay_path) as overlay:
        assert overlay.size == (27, 19)
    manifest = tmp_path / "val.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "id": "sample",
                "image": image_path.name,
                "mask": mask_path.name,
                "text": "the white car",
            }
        )
        + "\n"
    )
    result_path = tmp_path / "metrics.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate",
            "--checkpoint",
            str(checkpoint),
            "--manifest",
            str(manifest),
            "--output",
            str(result_path),
            "--workers",
            "0",
            "--device",
            "cpu",
        ],
    )
    evaluate.main()
    scores = json.loads(result_path.read_text())
    assert scores["samples"] == 1
    assert scores["resolution"] == "original"
    assert 0 <= scores["mIoU"] <= 100
