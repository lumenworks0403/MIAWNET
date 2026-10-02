# MIAWNet

PyTorch implementation of **MIAWNet: Multi-scale Interaction and Adaptive Weighting for Referring Image Segmentation in Low-altitude UAV Imagery**.

The model combines a Swin-B image encoder, a BERT text encoder, spatial cross-modal attention, Multi-scale Feature Interaction and Extraction (MFIE), and Adaptive Feature Weighting (AFW). The final mask is supervised by equally weighted binary cross-entropy and Dice losses.

This repository implements the method described in the manuscript. Dataset files and trained checkpoints are not included. The reported benchmark scores require the original data splits and experimental checkpoints; they are not results obtained from this release. Implementation choices for details omitted from the manuscript are listed in [docs/implementation.md](implementation.md).

## Installation

Use Python 3.10 on Linux for the manuscript environment. Install the PyTorch build that matches your CUDA setup, then install the remaining packages:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -e ".[dev,refcoco]"
```

The manuscript uses PyTorch 2.0.0, torchvision 0.15.1, Ubuntu 22.04, and an RTX 4090. Python 3.12 requires a newer compatible PyTorch/torchvision pair; do not use `requirements.txt` with that interpreter. On Windows, activate the environment with `.venv\Scripts\Activate.ps1`.

Swin-B ImageNet-1K weights and `bert-base-uncased` are downloaded on first training use. For offline training, cache the torchvision weights, point `model.bert_name` to a local BERT directory containing both model and tokenizer files, and set `model.local_files_only: true`.

## Data

Each JSONL record describes one referring expression and its target instance mask:

```json
{"id":"video03_frame001_car02","image":"images/video03/frame001.jpg","mask":"masks/video03/frame001_car02.png","text":"The white car next to the bus.","video_id":"video03"}
```

Paths are relative to `data.root`; absolute paths are also accepted. Masks must contain only the referred target, with background 0 and foreground 1 or 255. Images and masks must have the same native dimensions. See [docs/data.md](data.md) for DroneRIS and RefCOCO preparation.

```text
data/droneris/
  images/
  masks/
  annotations/
    train.jsonl
    val.jsonl
    test.jsonl
```

Check annotations and sequence separation before training:

```bash
miawnet-check-data --root data/droneris \
  --manifests annotations/train.jsonl annotations/val.jsonl annotations/test.jsonl \
  --require-video-ids
```

## Training

```bash
miawnet-train --config configs/droneris.yaml
miawnet-train --config configs/refcoco.yaml
miawnet-train --config configs/refcoco_plus.yaml
```

The default recipe uses 480 × 480 images, batch size 8, 50 epochs, AdamW with weight decay 0.05, and cosine learning rates from 3e-5 to 1e-6. No data augmentation or auxiliary loss is applied. Both encoders are optimized with the rest of the network. Mixed precision is disabled by default and can be enabled with `training.amp: true`.

Training writes `config.yaml`, `metrics.jsonl`, `last.pt`, and `best.pt` to the run directory. `best.pt` is selected by validation mIoU. Resume from a completed epoch:

```bash
miawnet-train --config configs/droneris.yaml --resume runs/droneris/last.pt
```

Model, optimizer, gradient scaler, and random states are restored. Model, data, and training configuration must match. Runtime paths, worker count, and device may change. Use a separate output directory for a new experiment.

## Evaluation

```bash
miawnet-evaluate --checkpoint runs/droneris/best.pt \
  --manifest annotations/test.jsonl --output runs/droneris/test.json

miawnet-evaluate --checkpoint runs/refcoco/best.pt \
  --manifest annotations/testA.jsonl --output runs/refcoco/testA.json

miawnet-evaluate --checkpoint runs/refcoco/best.pt \
  --manifest annotations/testB.jsonl --output runs/refcoco/testB.json
```

Metrics are reported as percentages: oIoU, mIoU, P@0.5, P@0.7, and P@0.9. oIoU aggregates intersections and unions across all image-expression pairs. P@X uses `IoU > X`. The probability threshold is 0.5. By default, logits are resized to the original mask resolution before thresholding. Use `--resolution resized` to evaluate at the training resolution, and record the setting with any reported result.

## Prediction

```bash
miawnet-predict --checkpoint runs/droneris/best.pt \
  --image scene.jpg --text "The white car next to the bus." \
  --output runs/demo/mask.png --overlay runs/demo/overlay.png
```

Prediction saves a binary PNG mask at the original image resolution. The optional overlay highlights the predicted target. Only load checkpoints produced by a trusted source.

## Ablations

```bash
miawnet-train --config configs/ablations/baseline.yaml
miawnet-train --config configs/ablations/mfie.yaml
miawnet-train --config configs/ablations/afw.yaml
miawnet-train --config configs/droneris.yaml --output runs/ablations/full
```

The AFW-only variant passes the same aligned feature map into two independently parameterized gates. This convention is documented in [docs/implementation.md](implementation.md). All four configurations use the same data and training defaults.

## Checks

```bash
ruff check src tests
ruff format --check src tests
pytest -q
```

Tests use synthetic images and small injected encoders; they do not download pretrained weights or measure benchmark performance. They cover feature gradients, padding masks, loss and metric definitions, data validation, official-format annotation conversion, and checkpoint restoration.

The encoders use the public [torchvision Swin API](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.swin_b.html) and [Transformers BERT API](https://huggingface.co/docs/transformers/v4.38.2/model_doc/bert).
