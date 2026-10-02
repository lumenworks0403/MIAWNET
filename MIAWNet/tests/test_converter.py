import json
import pickle
import sys

import numpy as np
from PIL import Image

from miawnet.convert_refcoco import main
from miawnet.data import read_manifest


def test_converter_preserves_expressions_and_splits(tmp_path, monkeypatch):
    images = tmp_path / "images"
    images.mkdir()
    Image.fromarray(np.zeros((20, 30, 3), dtype=np.uint8)).save(images / "frame.jpg")
    annotations = {
        "images": [{"id": 1, "file_name": "frame.jpg", "height": 20, "width": 30}],
        "categories": [{"id": 1, "name": "car"}],
        "annotations": [
            {
                "id": 10,
                "image_id": 1,
                "category_id": 1,
                "segmentation": [[3, 4, 12, 4, 12, 13, 3, 13]],
                "area": 81,
                "bbox": [3, 4, 9, 9],
                "iscrowd": 0,
            }
        ],
    }
    instances = tmp_path / "instances.json"
    instances.write_text(json.dumps(annotations), encoding="utf-8")
    refs = tmp_path / "refs.p"
    with refs.open("wb") as stream:
        pickle.dump(
            [
                {
                    "ref_id": 5,
                    "ann_id": 10,
                    "image_id": 1,
                    "split": "testA",
                    "sentences": [
                        {"sent_id": 11, "raw": "the car"},
                        {"sent_id": 12, "raw": "the vehicle on the left"},
                    ],
                }
            ],
            stream,
        )
    output = tmp_path / "converted"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "convert",
            "--refs",
            str(refs),
            "--instances",
            str(instances),
            "--images",
            str(images),
            "--output",
            str(output),
        ],
    )
    main()
    rows = read_manifest(output / "annotations/testA.jsonl")
    assert [row["id"] for row in rows] == ["5_11", "5_12"]
    assert [row["text"] for row in rows] == ["the car", "the vehicle on the left"]
    assert not (output / "annotations/train.jsonl").exists()
    with Image.open(output / rows[0]["mask"]) as mask:
        values = np.asarray(mask)
        assert values.shape == (20, 30)
        assert set(np.unique(values).tolist()) == {0, 255}
        assert values[8, 8] == 255
        assert values[0, 0] == 0
