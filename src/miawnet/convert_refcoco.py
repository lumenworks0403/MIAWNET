import argparse
import json
import pickle
from pathlib import Path

import numpy as np
from PIL import Image
from pycocotools.coco import COCO
from tqdm import tqdm


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert official RefCOCO annotations to JSONL")
    parser.add_argument(
        "--refs", type=Path, required=True, help="Trusted official refs(split).p file"
    )
    parser.add_argument("--instances", type=Path, required=True, help="Official instances.json")
    parser.add_argument("--images", type=Path, required=True, help="COCO train2014 directory")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with args.refs.open("rb") as stream:
        refs = pickle.load(stream)
    coco = COCO(str(args.instances))
    masks_dir = args.output / "masks"
    annotations_dir = args.output / "annotations"
    masks_dir.mkdir(parents=True, exist_ok=True)
    annotations_dir.mkdir(parents=True, exist_ok=True)
    splits = {}
    written_masks = set()
    for ref in tqdm(refs, desc="Convert"):
        split = ref["split"]
        if split not in {"train", "val", "test", "testA", "testB"}:
            raise ValueError(f"Unexpected official split: {split}")
        annotation = coco.anns[ref["ann_id"]]
        image_info = coco.imgs[ref["image_id"]]
        image_path = args.images / image_info["file_name"]
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        mask_name = f"{ref['ann_id']}.png"
        if mask_name not in written_masks:
            mask = coco.annToMask(annotation)
            Image.fromarray(mask.astype(np.uint8) * 255).save(masks_dir / mask_name)
            written_masks.add(mask_name)
        for sentence in ref["sentences"]:
            row = {
                "id": f"{ref['ref_id']}_{sentence['sent_id']}",
                "image": str(image_path.resolve()),
                "mask": f"masks/{mask_name}",
                "text": sentence["raw"],
                "image_id": ref["image_id"],
                "ref_id": ref["ref_id"],
                "ann_id": ref["ann_id"],
            }
            splits.setdefault(split, []).append(row)
    for split, rows in splits.items():
        path = annotations_dir / f"{split}.jsonl"
        with path.open("w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"{split}: {len(rows)} triplets -> {path}")


if __name__ == "__main__":
    main()
