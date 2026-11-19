#!/usr/bin/env bash
set -euo pipefail

BACKBONE="${1:-chatglm3}"
DATASET="${2:-mosei}"
DATA_ROOT="${3:-./data}"
OUTPUT_DIR="${4:-./outputs/${DATASET}_${BACKBONE}}"

COMMAND=(
  tha train
  --backbone "${BACKBONE}"
  --dataset "${DATASET}"
  --data-root "${DATA_ROOT}"
  --output-dir "${OUTPUT_DIR}"
)
if [[ $# -ge 5 ]]; then
  COMMAND+=(--seed "$5")
fi
"${COMMAND[@]}"
