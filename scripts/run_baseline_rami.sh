#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo "Usage: $0 METHOD SSL_DIR RAMI_PROTOCOL_ROOT OUTPUT_DIR" >&2
  echo "METHOD: sequential | ewc | lwf | owm | rawm | rwm | rego" >&2
  exit 2
fi

METHOD="$1"
SSL_DIR="$2"
PROTOCOL_ROOT="$3"
OUTPUT_DIR="$4"

case "$METHOD" in
  sequential|ewc|lwf|owm|rawm|rwm|rego) ;;
  *) echo "Unsupported baseline: $METHOD" >&2; exit 2 ;;
esac

python -u scripts/train_rfprompt.py \
  --xlsr "$SSL_DIR" \
  --ssl_backbone xlsr \
  --protocol_root "$PROTOCOL_ROOT" \
  --task_layout b \
  --method "$METHOD" \
  --backbone_mode frozen \
  --epochs 50 \
  --batch_size 32 \
  --num_workers 8 \
  --eval_num_workers 4 \
  --seed 2026 \
  --lr 1e-4 \
  --min_lr 1e-6 \
  --dev_interval 1 \
  --checkpoint_interval 10 \
  --max_tasks 4 \
  --output_dir "$OUTPUT_DIR"
