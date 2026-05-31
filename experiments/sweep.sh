#!/usr/bin/env bash
# Round launcher: expands (optimizer x eps x seed) into train.py runs, packs
# them across cluster GPUs per the model-size allocation table (single-GPU via
# CUDA_VISIBLE_DEVICES, DDP via torchrun), runs in waves up to capacity, and
# enforces a 5h-per-run wall-clock guard. Writes logs/<round>/SUMMARY.txt.
set -euo pipefail

SWEEP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=experiments/_lib.sh
source "$SWEEP_DIR/_lib.sh"

usage() {
  cat >&2 <<'EOF'
usage: sweep.sh --round <id> --config <yaml> --cluster {zhou-1,zhou-2}
                --optimizers <csv> --eps <csv> --seeds <csv>
                [--steps N] [--lora] [--force]
EOF
  exit 2
}

ROUND="" CONFIG="" CLUSTER="" OPTIMIZERS="" EPS="" SEEDS="" STEPS="" FORCE=0 LORA=0
BATCH="" MICRO="" GPU_BASE=0 NGPU=""
while [ $# -gt 0 ]; do
  case "$1" in
    --round)       ROUND="$2"; shift 2 ;;
    --config)      CONFIG="$2"; shift 2 ;;
    --cluster)     CLUSTER="$2"; shift 2 ;;
    --optimizers)  OPTIMIZERS="$2"; shift 2 ;;
    --eps)         EPS="$2"; shift 2 ;;
    --seeds)       SEEDS="$2"; shift 2 ;;
    --steps)       STEPS="$2"; shift 2 ;;
    --batch-size)  BATCH="$2"; shift 2 ;;
    --micro-batch) MICRO="$2"; shift 2 ;;
    --gpu-base)    GPU_BASE="$2"; shift 2 ;;  # first GPU index to use (partitioning)
    --gpus)        NGPU="$2"; shift 2 ;;      # how many GPUs from gpu-base to use
    --lora)        LORA=1; shift ;;
    --force)       FORCE=1; shift ;;
    -h|--help)     usage ;;
    *) die "unknown arg: $1" ;;
  esac
done

[ -n "$ROUND" ]       || { warn "missing --round"; usage; }
[ -n "$CONFIG" ]      || { warn "missing --config"; usage; }
[ -n "$CLUSTER" ]     || { warn "missing --cluster"; usage; }
[ -n "$OPTIMIZERS" ]  || { warn "missing --optimizers"; usage; }
[ -n "$EPS" ]         || { warn "missing --eps"; usage; }
[ -n "$SEEDS" ]       || { warn "missing --seeds"; usage; }
[ -f "$CONFIG" ]      || die "config not found: $CONFIG"

# Cluster GPU capacity (mirrors scripts/hosts.env).
case "$CLUSTER" in
  zhou-1) TOTAL_GPUS=4 ;;
  zhou-2) TOTAL_GPUS=8 ;;
  *) die "unknown cluster: $CLUSTER (expected zhou-1|zhou-2)" ;;
esac
# Restrict to a GPU sub-range so several sweeps can share the box (partitioning).
[ -n "$NGPU" ] || NGPU="$TOTAL_GPUS"
if [ "$(( GPU_BASE + NGPU ))" -gt "$TOTAL_GPUS" ]; then
  die "gpu-base($GPU_BASE)+gpus($NGPU) exceeds $CLUSTER capacity ($TOTAL_GPUS)"
fi
TOTAL_GPUS="$NGPU"

# Derive everything model-dependent from the config.
MODEL="$(yaml_get "$CONFIG" model)"
TASK="$(yaml_get "$CONFIG" task)"
[ -n "$MODEL" ] || die "config $CONFIG has no 'model:' key"
[ -n "$TASK" ]  || die "config $CONFIG has no 'task:' key"
[ -n "$STEPS" ] || STEPS="$(yaml_get "$CONFIG" steps 1000)"

GPN="$(gpus_per_run "$MODEL")"            # GPUs per run (1|2|4)
# LoRA freezes the base model, so it fits on a single GPU even for the large
# models; DDP is only needed for memory-bound full fine-tuning. Force 1 GPU/run.
[ "$LORA" -eq 1 ] && GPN=1
CAP="$(parallel_runs "$TOTAL_GPUS" "$GPN")"  # max concurrent runs this wave
PSS="$(per_step_seconds "$MODEL")"
# LoRA freezes the base model -> per-step is ~LORA_SPEEDUP x cheaper. DDP over GPN
# GPUs also scales throughput ~GPN x. Adjust the per-step estimate for the guard.
if [ "$LORA" -eq 1 ]; then
  PSS="$(awk -v p="$PSS" -v s="${LORA_SPEEDUP:-15}" 'BEGIN{printf "%.4f", p/s}')"
fi
if [ "$GPN" -gt 1 ]; then
  PSS="$(awk -v p="$PSS" -v g="$GPN" 'BEGIN{printf "%.4f", p/g}')"
fi

log "round=$ROUND cluster=$CLUSTER model=$MODEL task=$TASK"
log "gpus/run=$GPN  total_gpus=$TOTAL_GPUS  parallel_cap=$CAP  steps=$STEPS"

# --- 5h guard -------------------------------------------------------------
# Projected wall per run = steps * per-step-seconds (coarse). 5h = 18000s.
WALL_S="$(awk -v s="$STEPS" -v p="$PSS" 'BEGIN{printf "%d", s*p}')"
WALL_H="$(awk -v w="$WALL_S" 'BEGIN{printf "%.2f", w/3600.0}')"
log "projected wall/run ~= ${WALL_H}h (${WALL_S}s)"
if [ "$WALL_S" -gt 18000 ]; then
  warn "projected wall/run ${WALL_H}h exceeds the 5h round budget"
  warn "cut --steps, shrink the eps/seed slice, or add DDP GPUs"
  if [ "$FORCE" -ne 1 ]; then
    die "refusing to launch a >5h run (pass --force to override)"
  fi
  warn "--force set: proceeding despite >5h projection"
fi

# --- expand the grid into a run list --------------------------------------
IFS=',' read -r -a OPT_ARR <<< "$OPTIMIZERS"
IFS=',' read -r -a EPS_ARR <<< "$EPS"
IFS=',' read -r -a SEED_ARR <<< "$SEEDS"

RUNS=()   # each element: "opt|eps|seed"
for opt in "${OPT_ARR[@]}"; do
  for eps in "${EPS_ARR[@]}"; do
    for seed in "${SEED_ARR[@]}"; do
      RUNS+=("$opt|$eps|$seed")
    done
  done
done
N=${#RUNS[@]}
log "expanded grid: $N runs ($N/$CAP per wave)"

SUMMARY_DIR="$REPO_ROOT/logs/$ROUND"
mkdir -p "$SUMMARY_DIR"
SUMMARY="$SUMMARY_DIR/SUMMARY.txt"
{
  echo "round_id : $ROUND"
  echo "cluster  : $CLUSTER (${TOTAL_GPUS} gpus)"
  echo "config   : $CONFIG"
  echo "model    : $MODEL    task: $TASK"
  echo "gpus/run : $GPN    parallel_cap: $CAP    steps: $STEPS"
  echo "wall/run : ~${WALL_H}h"
  echo "runs     : $N"
  echo "started  : $(date)"
  echo "--------------------------------------------------------------"
} > "$SUMMARY"

# launch_one <gpu-base-index> <opt> <eps> <seed>
# Single-GPU runs pin CUDA_VISIBLE_DEVICES; DDP runs use torchrun over the
# GPN consecutive GPUs starting at <gpu-base-index>.
launch_one() {
  local base="$1" opt="$2" eps="$3" seed="$4"
  local arm="full"; [ "$LORA" -eq 1 ] && arm="lora"
  local name="${MODEL}__${TASK}__${arm}__${opt}__eps${eps}__s${seed}"
  local devs="" g
  for (( g=0; g<GPN; g++ )); do
    devs="${devs:+$devs,}$(( base + g ))"
  done

  local common=( --model "$MODEL" --task "$TASK" --optimizer "$opt"
                 --epsilon "$eps" --seed "$seed" --steps "$STEPS"
                 --config "$CONFIG" --round-id "$ROUND"
                 --wandb-project "$WANDB_PROJECT" --out "$REPO_ROOT/results/" )
  [ "$LORA" -eq 1 ] && common+=( --lora )
  [ -n "$BATCH" ] && common+=( --batch-size "$BATCH" )
  [ -n "$MICRO" ] && common+=( --micro-batch "$MICRO" )

  if [ "$GPN" -gt 1 ]; then
    # Each concurrent DDP run in a wave needs its own c10d rendezvous port,
    # else parallel torchrun jobs collide on the default 29500. Derive a stable
    # per-slot port from the GPU base index (29500 + base).
    local port=$(( 29500 + base ))
    CUDA_VISIBLE_DEVICES="$devs" \
      run_logged "$ROUND" "$name" -- \
        "$TORCHRUN_BIN" --nproc_per_node="$GPN" \
          --rdzv-backend=c10d --rdzv-endpoint="127.0.0.1:${port}" \
          "$REPO_ROOT/src/train.py" \
          "${common[@]}" --ddp &
  else
    CUDA_VISIBLE_DEVICES="$devs" \
      run_logged "$ROUND" "$name" -- \
        "$PYBIN" "$REPO_ROOT/src/train.py" "${common[@]}" &
  fi
  echo "$name dev=$devs pid=$!" >> "$SUMMARY"
}

# --- wave packing ---------------------------------------------------------
WAVE=0
i=0
while [ "$i" -lt "$N" ]; do
  WAVE=$(( WAVE + 1 ))
  log "=== wave $WAVE ==="
  echo "[wave $WAVE]" >> "$SUMMARY"
  slot=0
  pids=()
  while [ "$slot" -lt "$CAP" ] && [ "$i" -lt "$N" ]; do
    IFS='|' read -r opt eps seed <<< "${RUNS[$i]}"
    base=$(( GPU_BASE + slot * GPN ))
    launch_one "$base" "$opt" "$eps" "$seed"
    pids+=("$!")
    slot=$(( slot + 1 ))
    i=$(( i + 1 ))
  done
  # Wait for the whole wave; record any failures but keep the round going.
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      warn "a run in wave $WAVE exited non-zero (pid $pid)"
      echo "  FAILED pid=$pid" >> "$SUMMARY"
    fi
  done
  log "wave $WAVE complete"
done

{
  echo "--------------------------------------------------------------"
  echo "finished : $(date)"
} >> "$SUMMARY"
log "round $ROUND complete; summary: $SUMMARY"
