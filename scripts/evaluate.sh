#!/usr/bin/env bash
set -euo pipefail

BACKBONE="${1:-chatglm3}"
DATASET="${2:-mosei}"
DATA_ROOT="${3:-./data}"
CHECKPOINT="${4:-./outputs/${DATASET}_${BACKBONE}/best_adapter.pt}"
OUTPUT_DIR="${5:-./outputs/${DATASET}_${BACKBONE}_eval}"

tha evaluate \
  --backbone "${BACKBONE}" \
  --dataset "${DATASET}" \
  --data-root "${DATA_ROOT}" \
  --checkpoint "${CHECKPOINT}" \
  --output-dir "${OUTPUT_DIR}"
