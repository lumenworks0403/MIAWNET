# Data preparation

## DroneRIS

The manuscript describes 10,851 image-expression-mask triplets from CO-Drone videos, with approximately 7:1:2 train/validation/test proportions. Source frames are cropped into 1080 × 1080 patches. The eight categories are people, car, motor, bicycle, tricycle, truck, bus and boat.

Use the original released split lists if available. The ratio alone is insufficient to reconstruct exact sample membership. Different crops or frames from the same source video must remain in one split. Preserve each source video's identifier in `video_id`, then use `miawnet-check-data --require-video-ids` across all three manifests.

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

