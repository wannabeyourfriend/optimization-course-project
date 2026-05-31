#!/usr/bin/env bash
# Create a W&B sweep from a YAML config and launch one agent per GPU (detached).
# Each agent pulls trials from the sweep controller until the grid is exhausted,
# so N GPUs give N-way parallelism with no manual wave packing.
#
# Usage: experiments/wandb_sweep.sh <sweep_config.yaml> [num_gpus] [gpu_base]
#   WANDB_API_KEY must be exported; HF_ENDPOINT defaults to the mirror.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=experiments/_lib.sh
source "$HERE/_lib.sh"

CONFIG="${1:?usage: wandb_sweep.sh <config.yaml> [num_gpus] [gpu_base]}"
NGPU="${2:-8}"
GPU_BASE="${3:-0}"
ENTITY="${WANDB_ENTITY:-ziw178-uc-san-diego}"

[ -f "$CONFIG" ] || die "sweep config not found: $CONFIG"
mkdir -p "$REPO_ROOT/logs"

# Create the sweep via the Python API (returns just the sweep id — cleaner than
# scraping the CLI output).
log "creating W&B sweep from $CONFIG (project=$WANDB_PROJECT entity=$ENTITY)"
SWEEP_ID="$("$PYBIN" - "$CONFIG" "$WANDB_PROJECT" "$ENTITY" <<'PY'
import sys, yaml, wandb
cfg = yaml.safe_load(open(sys.argv[1]))
print(wandb.sweep(cfg, project=sys.argv[2], entity=sys.argv[3]))
PY
)"
[ -n "$SWEEP_ID" ] || die "failed to create sweep"
SWEEP_PATH="${ENTITY}/${WANDB_PROJECT}/${SWEEP_ID}"
log "sweep created: $SWEEP_PATH"
log "view: https://wandb.ai/${ENTITY}/${WANDB_PROJECT}/sweeps/${SWEEP_ID}"

# Launch one agent per GPU.
for (( i=0; i<NGPU; i++ )); do
  gpu=$(( GPU_BASE + i ))
  log "launching agent on GPU $gpu"
  CUDA_VISIBLE_DEVICES="$gpu" \
  PYTHONPATH="$REPO_ROOT/src" \
  HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}" \
  WANDB_PROJECT="$WANDB_PROJECT" \
  PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}" \
    setsid nohup "$PYBIN" -m wandb agent "$SWEEP_PATH" \
      >"$REPO_ROOT/logs/sweep_agent_${gpu}.log" 2>&1 < /dev/null &
done
log "launched $NGPU agents for $SWEEP_PATH; logs in logs/sweep_agent_*.log"
echo "$SWEEP_PATH"
