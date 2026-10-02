import argparse
import json
from pathlib import Path

import torch
import yaml
from transformers import BertConfig, BertModel

from .config import load_config
from .engine import (
    cosine_lr,
    evaluate,
    get_device,
    load_checkpoint,
    make_loader,
    make_scaler,
    restore_rng,
    rng_state,
    save_checkpoint,
    seed_everything,
    train_epoch,
)
from .losses import SegmentationLoss
from .models import MIAWNet
from .models.network import parameter_count


def main() -> None:
    parser = argparse.ArgumentParser(description="Train MIAWNet")
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--device")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    config = load_config(args.config)
    if args.device:
        config.runtime.device = args.device
    if args.output:
        config.runtime.output = str(args.output)
    device = get_device(config.runtime.device)
    seed_everything(config.training.seed)
    output = Path(config.runtime.output)
    output.mkdir(parents=True, exist_ok=True)
    if not args.resume and (output / "last.pt").exists():
        raise FileExistsError("Output contains a checkpoint; use --resume or another --output")
    checkpoint = load_checkpoint(args.resume) if args.resume else None
    if checkpoint:
        for section in ("model", "data", "training"):
            if checkpoint["config"][section] != config.to_dict()[section]:
                raise ValueError(f"Resume configuration differs in {section}")
    text_encoder = None
    if checkpoint:
        text_encoder = BertModel(
            BertConfig.from_dict(checkpoint["bert_config"]), add_pooling_layer=False
        )
    model = MIAWNet(
        config.model, initialize_pretrained=False if checkpoint else None, text_encoder=text_encoder
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.lr,
        weight_decay=config.training.weight_decay,
        betas=tuple(config.training.betas),
        eps=config.training.eps,
    )
    scaler = make_scaler(enabled=config.training.amp and device.type == "cuda")
    criterion = SegmentationLoss(config.training.dice_eps)
    start, best = 0, -1.0
    if checkpoint:
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint["scaler"])
        start, best = checkpoint["epoch"] + 1, checkpoint["best_miou"]
    train_loader = make_loader(config, config.data.train, shuffle=True)
    val_loader = make_loader(config, config.data.val)
    if checkpoint:
        restore_rng(checkpoint["rng"])
    (output / "config.yaml").write_text(
        yaml.safe_dump(config.to_dict(), sort_keys=False), encoding="utf-8"
    )
    print(f"Trainable parameters: {parameter_count(model):,}")
    for epoch in range(start, config.training.epochs):
        train_loader.generator.manual_seed(config.training.seed + epoch)
        lr = cosine_lr(epoch, config.training.epochs, config.training.lr, config.training.min_lr)
        for group in optimizer.param_groups:
            group["lr"] = lr
        train_scores = train_epoch(
            model, train_loader, optimizer, criterion, device, scaler, epoch + 1
        )
        val_scores = evaluate(model, val_loader, device, criterion)
        is_best = val_scores["mIoU"] > best
        best = max(best, val_scores["mIoU"])
        state = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "best_miou": best,
            "config": config.to_dict(),
            "rng": rng_state(),
            "validation": val_scores,
            "bert_config": model.text_encoder.config.to_dict(),
        }
        save_checkpoint(output / "last.pt", state)
        if is_best:
            save_checkpoint(output / "best.pt", state)
        record = {"epoch": epoch + 1, "lr": lr, "train": train_scores, "validation": val_scores}
        with (output / "metrics.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record) + "\n")
        print(json.dumps(record))


if __name__ == "__main__":
    main()
