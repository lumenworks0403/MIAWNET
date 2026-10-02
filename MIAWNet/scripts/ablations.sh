#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
python train.py --config configs/ablations/baseline.yaml "$@"
python train.py --config configs/ablations/mfie.yaml "$@"
python train.py --config configs/ablations/afw.yaml "$@"
python train.py --config configs/droneris.yaml --output runs/ablations/full "$@"
