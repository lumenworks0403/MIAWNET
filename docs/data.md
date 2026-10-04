# Data preparation

## Download and convert the published archive

Download **[DroneRIS.zip](https://drive.google.com/file/d/1bgE3XX192pZ4l6NpVN2WDiv4olWkADgK/view?usp=sharing)** and extract it into `downloads/DroneRIS/`:

```text
downloads/DroneRIS/
├── images/
│   └── ris_lad/             # 00001.jpg ... 02082.jpg
└── ris_lad/
    ├── instances.json
    └── refs(unc).p
```

The published archive has 10,851 triplets and 2,082 images, selected from RIS-LAD by category and decoded-mask-size strata. Its 58 visual groups are assigned wholly to train, val or test, with 7,596 / 1,085 / 2,170 triplets respectively. Source text and mask pixels are preserved; image names and IDs are renumbered consistently.

After installing the repository and its annotation-conversion dependencies, run the existing converter from the repository root:

```bash
python tools/convert_refcoco.py \
  --refs "downloads/DroneRIS/ris_lad/refs(unc).p" \
  --instances downloads/DroneRIS/ris_lad/instances.json \
  --images downloads/DroneRIS/images/ris_lad \
  --output data/droneris
```

The converter supports the archive's RefCOCO-style structure and preserves the `train`, `val` and `test` fields. It writes target-mask PNGs and `annotations/train.jsonl`, `annotations/val.jsonl`, and `annotations/test.jsonl`. Image paths point to the extracted original images.

Check this converted release:

```bash
python tools/check_data.py --root data/droneris \
  --manifests annotations/train.jsonl annotations/val.jsonl annotations/test.jsonl
```

Then use the existing training and evaluation commands in the README. Keep the extracted images in place because converted image paths are absolute.

The archive's SHA-256 is `f36bdeab39b227cd3f20df96e89dd99bb0d51e042947f3b621ac7ff5bf5228e1`. Original data credit and usage terms are provided by [RIS-LAD](https://github.com/AHideoKuzeA/RIS-LAD-A-Benchmark-and-Model-for-Referring-Low-Altitude-Drone-Image-Segmentation).

## DroneRIS

The manuscript describes 10,851 image-expression-mask triplets from CO-Drone videos, with approximately 7:1:2 train/validation/test proportions. Source frames are cropped into 1080 × 1080 patches. The eight categories are people, car, motor, bicycle, tricycle, truck, bus and boat.

For benchmark files with original source-video metadata, use their original split lists and preserve each source video's identifier in `video_id`, then use `miawnet-check-data --require-video-ids` across all three manifests. The published archive above uses its saved visual-group split assignment. The ratio alone does not specify sample membership.

Annotations are binary masks for the referred instance, paired with the verified referring expression. Multiple expressions for an instance are separate records and may share an image/mask. Each record must have a unique string `id`. This repository consumes verified annotations; the semi-automatic annotation pipeline is not a substitute for the benchmark files.

Create one UTF-8 JSONL manifest per split. The schema is:

| Field | Required | Meaning |
| --- | --- | --- |
| `id` | Yes | Unique image-expression pair identifier |
| `image` | Yes | RGB image path |
| `mask` | Yes | Single-channel binary target mask path |
| `text` | Yes | Referring expression |
| `video_id` | For DroneRIS auditing | Source sequence identifier |

## RefCOCO and RefCOCO+

Obtain the official annotations for the selected dataset and split convention, and the corresponding COCO train2014 images. Keep the source `refs(unc).p` and `instances.json` together for each dataset. Use the same split convention for comparisons.

```bash
miawnet-convert-refcoco \
  --refs "downloads/refcoco/refs(unc).p" \
  --instances downloads/refcoco/instances.json \
  --images downloads/coco/train2014 \
  --output data/refcoco
```

Repeat with RefCOCO+ source paths and `--output data/refcoco_plus`. The converter preserves the annotated train/val/testA/testB membership, writes one record per expression, and obtains each target mask from its COCO instance annotation. Source image paths are stored as absolute paths so image files need not be copied. When moving the dataset to another machine, rerun conversion or update the image paths.

Only use a trusted annotation pickle. JSONL output contains no pickle objects and is read without deserialization of executable objects.

## Resolution

Training resizes RGB images directly to 480 × 480 and uses nearest-neighbor interpolation for masks. No random augmentation is applied. Validation loss is calculated on the resized mask. Segmentation metrics use the native mask by default, resizing logits before sigmoid and thresholding. The alternative `resized` mode must be reported explicitly because it can change benchmark scores.

## Validation

`miawnet-check-data` verifies record fields, unique IDs within each split, readable files, matching image/mask sizes, binary foreground values, non-empty target masks, and cross-split image/video overlap. Without sequence identifiers, image separation can be checked but video separation cannot be established.

