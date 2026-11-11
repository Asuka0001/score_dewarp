#!/usr/bin/env bash
# Phase 1 ablation with the fixed architecture (CoordConv + bottleneck attention, 384x512):
# Doc3D-pretrained vs from-scratch on deepscores_pilot, 60 epochs each, then validation eval.
# The test split is NOT evaluated here; pick models on val first (see README section 10).
set -uo pipefail
cd "$(dirname "$0")/.."
P=data/prepared/deepscores_pilot/train
PRE=outputs/doc3d_pretrain/best.pt
ROOT=outputs/deepscores_finetune
LOGS=$ROOT/logs
mkdir -p "$LOGS"
export COORD=1 CONTEXT=attn SIZE="384 512"

run() {  # NAME LR [INIT]
  local name=$1 lr=$2 init=${3:-}
  if [[ -f $ROOT/$name/best.report.json ]]; then
    echo "skip train $name (done)"
  else
    echo "train $name $(date +%H:%M)"
    scripts/run_finetune_ablation.sh "$name" "$P" 60 "$lr" 7 $init > "$LOGS/$name.log" 2>&1 \
      || { echo "FAILED train $name"; return; }
  fi
  if [[ -f $ROOT/$name/eval_val/summary.json ]]; then
    echo "skip eval $name (done)"
  else
    echo "eval $name $(date +%H:%M)"
    .venv/bin/python scripts/evaluate_geometry.py --data data/prepared/deepscores_pilot/val \
      --checkpoint "$ROOT/$name/best.pt" --out "$ROOT/$name/eval_val" > "$LOGS/$name.eval_val.log" 2>&1 \
      || echo "FAILED eval $name"
  fi
  echo "finished $name $(date +%H:%M)"
}

run scratch_ca_lr1e-3_s7 1e-3
run ft_ca_lr1e-4_s7      1e-4 "$PRE"
run ft_ca_lr3e-4_s7      3e-4 "$PRE"
run scratch_ca_lr3e-4_s7 3e-4
run ft_ca_lr1e-3_s7      1e-3 "$PRE"
echo PHASE1_CA_DONE
