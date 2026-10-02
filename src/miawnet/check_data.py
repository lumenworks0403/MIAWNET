import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from .data import read_manifest


def audit_splits(root: Path, manifests: list[str], require_video_ids: bool = False) -> dict:
    report = {}
    used_images, used_videos = set(), set()
    for manifest in manifests:
        rows = read_manifest(root / manifest)
        images, videos = set(), set()
        missing_video_ids = 0
        for row in rows:
            with Image.open(root / row["image"]) as image, Image.open(root / row["mask"]) as mask:
                if image.size != mask.size:
                    raise ValueError(f"Image/mask size mismatch: {row['id']}")
                if mask.mode not in {"1", "L", "I", "I;16", "P"}:
                    raise ValueError(f"Expected a single-channel target mask: {row['mask']}")
                values = set(np.unique(np.asarray(mask.convert("L"))).tolist())
                if not values.issubset({0, 1, 255}):
                    raise ValueError(f"Expected a binary target mask: {row['mask']}")
                if values.issubset({0}):
                    raise ValueError(f"Empty target mask: {row['id']}")
            images.add(str((root / row["image"]).resolve()))
            if row.get("video_id") is not None:
                videos.add(str(row["video_id"]))
            else:
                missing_video_ids += 1
        if used_images & images:
            raise ValueError(f"Images shared across splits: {manifest}")
        if used_videos & videos:
            raise ValueError(f"Video sequences shared across splits: {manifest}")
        if require_video_ids and missing_video_ids:
            raise ValueError(f"Missing video_id for {missing_video_ids} samples in {manifest}")
        used_images.update(images)
        used_videos.update(videos)
        report[manifest] = {
            "triplets": len(rows),
            "images": len(images),
            "videos": len(videos),
            "missing_video_ids": missing_video_ids,
        }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Check RIS masks and split leakage")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--manifests", nargs="+", required=True)
    parser.add_argument("--require-video-ids", action="store_true")
    args = parser.parse_args()
    print(json.dumps(audit_splits(args.root, args.manifests, args.require_video_ids), indent=2))


if __name__ == "__main__":
    main()
