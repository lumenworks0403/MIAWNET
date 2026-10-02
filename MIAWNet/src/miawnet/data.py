import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def image_tensor(image: Image.Image, size: int) -> Tensor:
    image = image.convert("RGB").resize((size, size), Image.Resampling.BILINEAR)
    values = torch.from_numpy(np.asarray(image, dtype=np.float32).copy()).permute(2, 0, 1) / 255
    return (values - values.new_tensor(MEAN)[:, None, None]) / values.new_tensor(STD)[:, None, None]


def mask_tensor(mask: Image.Image) -> Tensor:
    return torch.from_numpy((np.asarray(mask.convert("L")) > 0).astype(np.float32)).unsqueeze(0)


def read_manifest(path: str | Path) -> list[dict]:
    rows = []
    seen = set()
    with open(path, encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            required = {"id", "image", "mask", "text"}
            if not isinstance(row, dict) or not required.issubset(row):
                raise ValueError(f"{path}:{line_number}: expected fields {sorted(required)}")
            if any(not isinstance(row[key], str) or not row[key].strip() for key in required):
                raise ValueError(f"{path}:{line_number}: required fields must be non-empty strings")
            if row["id"] in seen:
                raise ValueError(f"Duplicate sample ID: {row['id']}")
            seen.add(row["id"])
            rows.append(row)
    if not rows:
        raise ValueError(f"Empty manifest: {path}")
    return rows


class ReferringDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        manifest: str | Path,
        tokenizer: Any,
        image_size: int = 480,
        max_text_length: int = 64,
        evaluation_resolution: str = "original",
    ) -> None:
        self.root = Path(root)
        manifest = Path(manifest)
        self.rows = read_manifest(manifest if manifest.is_absolute() else self.root / manifest)
        self.image_size = image_size
        self.evaluation_resolution = evaluation_resolution
        self.tokens = tokenizer(
            [row["text"] for row in self.rows],
            padding="max_length",
            truncation=True,
            max_length=max_text_length,
            return_tensors="pt",
        )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        row = self.rows[index]
        with Image.open(self.root / row["image"]) as source:
            original_size = source.size
            image = image_tensor(source, self.image_size)
        with Image.open(self.root / row["mask"]) as source:
            if source.size != original_size:
                raise ValueError(f"Image and mask sizes differ for {row['id']}")
            source = source.convert("L")
            resized = source.resize((self.image_size, self.image_size), Image.Resampling.NEAREST)
            target = mask_tensor(resized)
            native_target = (
                mask_tensor(source) if self.evaluation_resolution == "original" else target
            )
        return {
            "id": row["id"],
            "image": image,
            "target": target,
            "native_target": native_target,
            "input_ids": self.tokens["input_ids"][index],
            "attention_mask": self.tokens["attention_mask"][index],
        }


def collate_samples(samples: list[dict]) -> dict:
    keys = ("image", "target", "input_ids", "attention_mask")
    batch = {key: torch.stack([sample[key] for sample in samples]) for key in keys}
    batch["id"] = [sample["id"] for sample in samples]
    batch["native_target"] = [sample["native_target"] for sample in samples]
    return batch
