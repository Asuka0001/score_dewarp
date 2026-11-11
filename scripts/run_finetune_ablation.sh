#!/usr/bin/env bash
# Pretrained (Doc3D) vs from-scratch training on DeepScores, equal data/steps/resolution.
# Usage: [SIZE="W H"] [CONTEXT=none|attn] [COORD=1] scripts/run_finetune_ablation.sh NAME TRAIN_DIR EPOCHS LR SEED [INIT]
# Note: the plain U-Net collapses to a constant prediction when trained from scratch at
# 384x512. Use COORD=1 CONTEXT=attn (recommended) or a smaller SIZE; see README section 10.
set -euo pipefail
cd "$(dirname "$0")/.."
NAME=$1 TRAIN=$2 EPOCHS=$3 LR=$4 SEED=$5 INIT=${6:-}
read -r W H <<< "${SIZE:-384 512}"
OUT=outputs/deepscores_finetune/$NAME
mkdir -p "$OUT"
ARGS=(--data "$TRAIN" --val-data data/prepared/deepscores_pilot/val --geometry-only
      --epochs "$EPOCHS" --patience 0 --bs 8 --size "$W" "$H" --lr "$LR" --seed "$SEED"
      --context "${CONTEXT:-none}" --out "$OUT/best.pt")
[[ "${COORD:-0}" == 1 ]] && ARGS+=(--coord-input)
[[ -n "$INIT" ]] && ARGS+=(--init "$INIT")
.venv/bin/python scripts/train.py "${ARGS[@]}"
