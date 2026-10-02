import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import BertTokenizerFast

from .data import image_tensor
from .engine import get_device, load_model


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description="Segment a target described by text")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overlay", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()
    if not args.text.strip() or not 0 < args.threshold < 1:
        raise ValueError("Provide non-empty text and a threshold in (0, 1)")
    device = get_device(args.device)
    model, config = load_model(args.checkpoint, device)
    tokenizer = BertTokenizerFast.from_pretrained(
        config.model.bert_name,
        local_files_only=config.model.local_files_only,
    )
    tokens = tokenizer(
        args.text, truncation=True, max_length=config.data.max_text_length, return_tensors="pt"
    )
    with Image.open(args.image) as source:
        source = source.convert("RGB")
        image = image_tensor(source, config.data.image_size).unsqueeze(0).to(device)
        logits = model(image, tokens["input_ids"].to(device), tokens["attention_mask"].to(device))
        logits = F.interpolate(
            logits.float(), size=(source.height, source.width), mode="bilinear", align_corners=False
        )
        mask = (logits.sigmoid()[0, 0] >= args.threshold).cpu().numpy()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(mask.astype(np.uint8) * 255).save(args.output)
        if args.overlay:
            values = np.asarray(source).copy()
            color = np.array([255, 80, 80], dtype=np.float32)
            values[mask] = (0.55 * values[mask] + 0.45 * color).astype(np.uint8)
            args.overlay.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(values).save(args.overlay)


if __name__ == "__main__":
    main()
