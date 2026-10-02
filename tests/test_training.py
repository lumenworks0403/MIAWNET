import random
import sys

import numpy as np
import pytest
import torch
import yaml
from conftest import tiny_model
from torch.utils.data import DataLoader

from miawnet import train
from miawnet.config import from_dict, load_config
from miawnet.data import collate_samples
from miawnet.engine import (
    cosine_lr,
    load_checkpoint,
    restore_rng,
    rng_state,
    save_checkpoint,
    seed_everything,
)


def test_cosine_endpoints():
    assert cosine_lr(0, 50, 3e-5, 1e-6) == pytest.approx(3e-5)
    assert cosine_lr(49, 50, 3e-5, 1e-6) == pytest.approx(1e-6)


def test_configuration_rejects_unknown_fields():
    with pytest.raises(ValueError, match="Unknown"):
        from_dict({"typo": {}})
    with pytest.raises(TypeError):
        from_dict({"model": {"unknown": 1}})


def test_checkpoint_preserves_random_states(tmp_path):
    seed_everything(19)
    path = tmp_path / "state.pt"
    save_checkpoint(path, {"rng": rng_state(), "value": torch.tensor([1.0])})
    expected = (random.random(), np.random.rand(), torch.rand(3))
    seed_everything(81)
    checkpoint = load_checkpoint(path)
    restore_rng(checkpoint["rng"])
    actual = (random.random(), np.random.rand(), torch.rand(3))
    assert actual[:2] == expected[:2]
    torch.testing.assert_close(actual[2], expected[2])
    assert not path.with_suffix(".tmp").exists()


def synthetic_loader(config, manifest, shuffle=False, seed=None):
    generator = torch.Generator().manual_seed(5)
    samples = []
    for index in range(3):
        target = torch.zeros(1, 32, 32)
        target[:, 5:12, 8:16] = 1
        samples.append(
            {
                "id": str(index),
                "image": torch.randn(3, 32, 32, generator=generator),
                "target": target,
                "native_target": target,
                "input_ids": torch.tensor([1, 2, 3, 4]),
                "attention_mask": torch.ones(4, dtype=torch.long),
            }
        )
    return DataLoader(
        samples,
        batch_size=2,
        shuffle=shuffle,
        collate_fn=collate_samples,
        generator=torch.Generator().manual_seed(config.training.seed),
    )


def test_epoch_resume_matches_uninterrupted_training(tmp_path, monkeypatch):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "model": {"feature_dim": 8, "decoder_dim": 8, "pretrained": False},
                "data": {"image_size": 32},
                "training": {"epochs": 3, "batch_size": 2},
                "runtime": {"workers": 0, "device": "cpu"},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(train, "MIAWNet", tiny_model)
    monkeypatch.setattr(train, "make_loader", synthetic_loader)
    full, resumed = tmp_path / "full", tmp_path / "resumed"

    def run(output, checkpoint=None):
        argv = ["train", "--config", str(config_path), "--output", str(output)]
        if checkpoint:
            argv += ["--resume", str(checkpoint)]
        monkeypatch.setattr(sys, "argv", argv)
        train.main()

    run(full)
    real_train_epoch = train.train_epoch

    def interrupt_epoch(*args, **kwargs):
        if args[-1] == 2:
            raise InterruptedError("test interruption")
        return real_train_epoch(*args, **kwargs)

    monkeypatch.setattr(train, "train_epoch", interrupt_epoch)
    with pytest.raises(InterruptedError):
        run(resumed)
    assert load_checkpoint(resumed / "last.pt")["epoch"] == 0
    monkeypatch.setattr(train, "train_epoch", real_train_epoch)
    run(resumed, resumed / "last.pt")
    expected, actual = load_checkpoint(full / "last.pt"), load_checkpoint(resumed / "last.pt")
    for name in expected["model"]:
        torch.testing.assert_close(expected["model"][name], actual["model"][name], rtol=0, atol=0)
    assert expected["validation"] == actual["validation"]
    assert load_config(config_path).training.epochs == 3
