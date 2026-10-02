import math
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertConfig, BertModel, BertTokenizerFast

from .config import Config, from_dict
from .data import ReferringDataset, collate_samples
from .losses import SegmentationLoss
from .metrics import SegmentationMetrics
from .models import MIAWNet


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id: int) -> None:
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)


def make_scaler(enabled: bool):
    if hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda", enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def get_device(name: str) -> torch.device:
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; use --device cpu for a CPU check")
    return device


def make_loader(
    config: Config, manifest: str, shuffle: bool = False, seed: int | None = None
) -> DataLoader:
    tokenizer = BertTokenizerFast.from_pretrained(
        config.model.bert_name,
        local_files_only=config.model.local_files_only,
    )
    dataset = ReferringDataset(
        config.data.root,
        manifest,
        tokenizer,
        config.data.image_size,
        config.data.max_text_length,
        config.data.evaluation_resolution,
    )
    generator = torch.Generator().manual_seed(config.training.seed if seed is None else seed)
    return DataLoader(
        dataset,
        batch_size=config.training.batch_size,
        shuffle=shuffle,
        num_workers=config.runtime.workers,
        collate_fn=collate_samples,
        pin_memory=config.runtime.device.startswith("cuda"),
        worker_init_fn=seed_worker,
        generator=generator,
        drop_last=False,
    )


def move_batch(batch: dict, device: torch.device) -> dict:
    keys = ("image", "target", "input_ids", "attention_mask")
    return {key: batch[key].to(device, non_blocking=True) for key in keys}


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: SegmentationLoss,
    device: torch.device,
    scaler: torch.cuda.amp.GradScaler,
    epoch: int,
) -> dict[str, float]:
    model.train()
    totals = {"loss": 0.0, "bce": 0.0, "dice": 0.0}
    count = 0
    progress = tqdm(loader, desc=f"Train {epoch}")
    for batch in progress:
        moved = move_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=scaler.is_enabled()):
            logits = model(moved["image"], moved["input_ids"], moved["attention_mask"])
            losses = criterion(logits, moved["target"])
        if not torch.isfinite(losses["loss"]):
            raise FloatingPointError(f"Non-finite loss for samples {batch['id']}")
        scaler.scale(losses["loss"]).backward()
        scaler.step(optimizer)
        scaler.update()
        size = moved["image"].shape[0]
        count += size
        for key in totals:
            totals[key] += float(losses[key].detach()) * size
        progress.set_postfix(loss=totals["loss"] / count)
    return {key: value / count for key, value in totals.items()}


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: SegmentationLoss | None = None,
    threshold: float = 0.5,
) -> dict:
    model.eval()
    metrics = SegmentationMetrics(threshold)
    loss_sum, count = 0.0, 0
    for batch in tqdm(loader, desc="Evaluate"):
        moved = move_batch(batch, device)
        logits = model(moved["image"], moved["input_ids"], moved["attention_mask"])
        metrics.update(logits, batch["native_target"])
        if criterion is not None:
            size = moved["image"].shape[0]
            loss_sum += float(criterion(logits, moved["target"])["loss"]) * size
            count += size
    result = metrics.compute()
    if criterion is not None:
        result["loss"] = loss_sum / count
    result["samples"] = metrics.count
    return result


def cosine_lr(epoch: int, epochs: int, start: float, end: float) -> float:
    return end + (start - end) * (1 + math.cos(math.pi * epoch / (epochs - 1))) / 2


def rng_state() -> dict:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


def save_checkpoint(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def load_checkpoint(path: str | Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


def load_model(path: str | Path, device: torch.device) -> tuple[nn.Module, Config]:
    checkpoint = load_checkpoint(path)
    config = from_dict(checkpoint["config"])
    text_encoder = BertModel(
        BertConfig.from_dict(checkpoint["bert_config"]), add_pooling_layer=False
    )
    model = MIAWNet(config.model, initialize_pretrained=False, text_encoder=text_encoder).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    return model, config
