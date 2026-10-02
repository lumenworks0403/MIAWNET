#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
checkpoint="${1:-runs/droneris/best.pt}"
manifest="${2:-annotations/test.jsonl}"
if [[ $# -gt 0 ]]; then shift; fi
if [[ $# -gt 0 ]]; then shift; fi
python evaluate.py --checkpoint "$checkpoint" --manifest "$manifest" "$@"
