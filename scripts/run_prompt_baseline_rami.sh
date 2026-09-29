#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo "Usage: $0 METHOD XLSR_DIR RAMI_PROTOCOL_ROOT OUTPUT_DIR" >&2
  echo "METHOD: oisoprompt | singleprompt | kaprompt | smope | rainbow" >&2
  exit 2
fi

METHOD="$1"
XLSR_DIR="$2"
PROTOCOL_ROOT="$3"
OUTPUT_DIR="$4"

case "$METHOD" in
  oisoprompt|singleprompt|kaprompt|smope|rainbow) ;;
  *) echo "Unsupported adapted prompt method: $METHOD" >&2; exit 2 ;;
esac

python -u scripts/train_rfprompt.py \
  --method "$METHOD" \
  --xlsr "$XLSR_DIR" \
  --protocol_root "$PROTOCOL_ROOT" \
  --task_layout b \
  --backbone_mode frozen \
  --epochs 50 \
  --batch_size 32 \
  --num_workers 8 \
  --eval_num_workers 4 \
  --seed 2026 \
  --lr 1e-4 \
  --min_lr 1e-6 \
  --dev_interval 1 \
  --output_dir "$OUTPUT_DIR"
