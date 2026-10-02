import json

import numpy as np
import pytest
from conftest import TinyTokenizer
from PIL import Image

from miawnet.check_data import audit_splits
from miawnet.data import ReferringDataset, collate_samples, read_manifest


def write_sample(root, split="train", video="video1", image_name="image.png"):
    Image.fromarray(np.full((13, 21, 3), 128, dtype=np.uint8)).save(root / image_name)
    mask = np.zeros((13, 21), dtype=np.uint8)
    mask[3:7, 8:12] = 255
    Image.fromarray(mask).save(root / f"{split}.png")
    row = {
        "id": split,
        "image": image_name,
        "mask": f"{split}.png",
        "text": "the white car",
        "video_id": video,
    }
    manifest = root / f"{split}.jsonl"
    manifest.write_text(json.dumps(row) + "\n", encoding="utf-8")
    return manifest


def test_dataset_keeps_binary_and_native_masks(tmp_path):
    path = write_sample(tmp_path)
    dataset = ReferringDataset(tmp_path, path, TinyTokenizer(), image_size=32)
    sample = dataset[0]
    assert sample["image"].shape == (3, 32, 32)
    assert sample["target"].shape == (1, 32, 32)
    assert sample["native_target"].shape == (1, 13, 21)
    assert set(sample["target"].unique().tolist()) == {0.0, 1.0}
    batch = collate_samples([sample, sample])
    assert batch["image"].shape == (2, 3, 32, 32)
    assert len(batch["native_target"]) == 2


def test_duplicate_ids_are_rejected(tmp_path):
    path = write_sample(tmp_path)
    path.write_text(path.read_text() * 2)
    with pytest.raises(ValueError, match="Duplicate"):
        read_manifest(path)


def test_video_leakage_is_rejected(tmp_path):
    write_sample(tmp_path)
    write_sample(tmp_path, "val", image_name="other.png")
    with pytest.raises(ValueError, match="Video sequences"):
        audit_splits(tmp_path, ["train.jsonl", "val.jsonl"], require_video_ids=True)


def test_binary_mask_audit(tmp_path):
    write_sample(tmp_path)
    Image.fromarray(np.full((13, 21), 17, dtype=np.uint8)).save(tmp_path / "train.png")
    with pytest.raises(ValueError, match="binary"):
        audit_splits(tmp_path, ["train.jsonl"])
