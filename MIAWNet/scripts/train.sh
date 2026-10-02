#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
config="${1:-configs/droneris.yaml}"
if [[ $# -gt 0 ]]; then shift; fi
python train.py --config "$config" "$@"
