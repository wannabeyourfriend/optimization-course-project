#!/usr/bin/env bash
# Phase 1 baseline round: RoBERTa-base / SST-2, DP-Adam @ eps {inf,8}, seeds
# {0,1,2} on zhou-2. Correctness gate before the full sweep. Thin wrapper that
# generates a round-id and delegates to experiments/sweep.sh.
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=experiments/_lib.sh
source "$DIR/../_lib.sh"
export PYTHONPATH="$REPO_ROOT/src"

CONFIG="$REPO_ROOT/configs/roberta_base_sst2.yaml"

# Arm = full fine-tune (default) or LoRA, taken from passthrough flags; encode it
# in the round-id so the two arms' result files and dashboard rows stay distinct.
ARM="full"
for a in "$@"; do [ "$a" = "--lora" ] && ARM="lora"; done
ROUND="$(gen_round_id "phase1-${ARM}")"

log "launching Phase 1 baseline round (${ARM}): $ROUND"

exec "$DIR/../sweep.sh" \
  --round "$ROUND" \
  --config "$CONFIG" \
  --cluster zhou-2 \
  --optimizers dp-adam \
  --eps inf,8 \
  --seeds 0,1,2 \
  "$@"
