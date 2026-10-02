import argparse
import json
from pathlib import Path

from .engine import evaluate, get_device, load_model, make_loader


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate an MIAWNet checkpoint")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--root")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--resolution", choices=["original", "resized"])
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    device = get_device(args.device)
    model, config = load_model(args.checkpoint, device)
    config.runtime.device, config.runtime.workers = args.device, args.workers
    if args.root:
        config.data.root = args.root
    if args.batch_size:
        config.training.batch_size = args.batch_size
    if args.resolution:
        config.data.evaluation_resolution = args.resolution
    config.validate()
    result = evaluate(model, make_loader(config, args.manifest), device, threshold=args.threshold)
    result.update(
        manifest=args.manifest,
        resolution=config.data.evaluation_resolution,
        threshold=args.threshold,
    )
    text = json.dumps(result, indent=2)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
