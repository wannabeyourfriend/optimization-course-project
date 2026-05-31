#!/usr/bin/env bash
# DDP correctness gate (needs >=2 visible GPUs + torchrun).
#
# Verifies that 2-rank DistributedDPAdaptive reproduces the single-GPU DP
# mechanism on the offline tiny-synthetic dry-run:
#   - epsilon_spent matches  (accounting is GPU-count independent)
#   - phi matches            (DP-noise variance for DP-AdamBC; world_size factor)
#   - sigma=0 (eps=inf) DDP run completes without hanging
#
# Usage (on a multi-GPU box, repo root):
#   PYTHONPATH=src:third_party/opacus bash tests/test_ddp_equivalence.sh
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export WANDB_MODE=disabled
PYBIN="${PYBIN:-.venv/bin/python}"
TORCHRUN="${TORCHRUN_BIN:-.venv/bin/torchrun}"
export PYTHONPATH="src:third_party/opacus:${PYTHONPATH:-}"

num() { grep -oE "\"$2\": *[0-9.eE+-]+" "$1" | tail -1 | grep -oE "[0-9.eE+-]+$"; }
approx_eq() {  # a b tol
  awk -v a="$1" -v b="$2" -v t="$3" 'BEGIN{d=a-b; if(d<0)d=-d; exit !(d<=t)}'
}

S=/tmp/ddpeq_single.json M=/tmp/ddpeq_multi.json I=/tmp/ddpeq_inf.json
echo "[1/3] single-GPU eps=8 dp-adambc"
CUDA_VISIBLE_DEVICES=0 "$PYBIN" src/train.py --dry-run --epsilon 8 \
  --optimizer dp-adambc --seed 0 >"$S" 2>/dev/null || { echo "FAIL: single rc=$?"; exit 1; }

echo "[2/3] 2-rank DDP eps=8 dp-adambc"
CUDA_VISIBLE_DEVICES=0,1 timeout 300 "$TORCHRUN" --nproc_per_node=2 \
  --rdzv-backend=c10d --rdzv-endpoint=127.0.0.1:29591 \
  src/train.py --dry-run --epsilon 8 --optimizer dp-adambc --seed 0 --ddp \
  >"$M" 2>/dev/null || { echo "FAIL: ddp rc=$?"; exit 1; }

echo "[3/3] 2-rank DDP eps=inf sigma=0 dp-adamw-bc"
CUDA_VISIBLE_DEVICES=0,1 timeout 300 "$TORCHRUN" --nproc_per_node=2 \
  --rdzv-backend=c10d --rdzv-endpoint=127.0.0.1:29592 \
  src/train.py --dry-run --epsilon inf --optimizer dp-adamw-bc --seed 0 --ddp \
  >"$I" 2>/dev/null || { echo "FAIL: ddp-inf rc=$?"; exit 1; }

es=$(num "$S" epsilon_spent); em=$(num "$M" epsilon_spent)
ps=$(num "$S" phi);           pm=$(num "$M" phi)
echo "epsilon_spent: single=$es ddp=$em"
echo "phi:           single=$ps ddp=$pm"

fail=0
approx_eq "$es" "$em" 1e-9 || { echo "FAIL: epsilon_spent mismatch"; fail=1; }
approx_eq "$ps" "$pm" 1e-12 || { echo "FAIL: phi mismatch (world_size factor?)"; fail=1; }
[ "$fail" -eq 0 ] && echo "PASS test_ddp_equivalence (epsilon_spent + phi match)" || exit 1
