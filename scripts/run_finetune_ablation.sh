#!/usr/bin/env bash
# Pretrained (Doc3D) vs from-scratch training on DeepScores, equal data/steps/resolution.
# Usage: scripts/run_finetune_ablation.sh NAME TRAIN_DIR EPOCHS LR SEED [INIT]
set -euo pipefail
cd "$(dirname "$0")/.."
NAME=$1 TRAIN=$2 EPOCHS=$3 LR=$4 SEED=$5 INIT=${6:-}
OUT=outputs/deepscores_finetune/$NAME
mkdir -p "$OUT"
ARGS=(--data "$TRAIN" --val-data data/prepared/deepscores_pilot/val --geometry-only
      --epochs "$EPOCHS" --patience 0 --bs 8 --size 384 512 --lr "$LR" --seed "$SEED" --out "$OUT/best.pt")
[[ -n "$INIT" ]] && ARGS+=(--init "$INIT")
.venv/bin/python scripts/train.py "${ARGS[@]}"
