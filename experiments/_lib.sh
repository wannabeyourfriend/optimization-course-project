#!/usr/bin/env bash
# Shared helpers for DP-optimizer experiment launchers: wandb env, round-id
# generation, tiny YAML readers, GPU sizing, and logged command execution.
set -euo pipefail

# Repo root = parent of the experiments/ dir this file lives in.
LIB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$LIB_DIR/.." && pwd)"
export PYTHONPATH="$REPO_ROOT/src"

# The venv is not activated under non-interactive ssh, and on the remote
# (Ubuntu 18.04) bare `python` is Python 2.7. Always invoke the venv explicitly.
PYBIN="${PYBIN:-$REPO_ROOT/.venv/bin/python}"
TORCHRUN_BIN="${TORCHRUN_BIN:-$REPO_ROOT/.venv/bin/torchrun}"

# ---------------------------------------------------------------------------
# wandb environment
# ---------------------------------------------------------------------------
# Default project; callers may override via WANDB_PROJECT before sourcing.
export WANDB_PROJECT="${WANDB_PROJECT:-dp-optimizer-finetuning}"
# Never block a training run waiting on a wandb prompt.
export WANDB_SILENT="${WANDB_SILENT:-true}"

# ---------------------------------------------------------------------------
# HuggingFace hub endpoint
# ---------------------------------------------------------------------------
# The GPU boxes sit behind a network that resets TLS to huggingface.co, so
# default to the hf-mirror.com mirror. Override by exporting HF_ENDPOINT before
# launching (e.g. HF_ENDPOINT=https://huggingface.co on an unfiltered host).
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

# Opacus per-sample grads spike memory; expandable_segments curbs fragmentation
# OOMs on the 24GB 3090s.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# ---------------------------------------------------------------------------
# logging
# ---------------------------------------------------------------------------
log()  { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*" >&2; }
die()  { printf '[%s] ERROR: %s\n' "$(date +%H:%M:%S)" "$*" >&2; exit 1; }
warn() { printf '[%s] WARNING: %s\n' "$(date +%H:%M:%S)" "$*" >&2; }

# gen_round_id [prefix] -> "<prefix>-YYYYmmdd-HHMMSS" (prefix defaults to "r").
gen_round_id() {
  local prefix="${1:-r}"
  printf '%s-%s' "$prefix" "$(date +%Y%m%d-%H%M%S)"
}

# run_logged <round_id> <name> -- <cmd...>
# Tees combined stdout/stderr of <cmd...> to logs/<round_id>/<name>.log and
# preserves the command's exit status (pipefail makes tee not mask it).
run_logged() {
  local round_id="$1" name="$2"
  shift 2
  [ "${1:-}" = "--" ] && shift
  local dir="$REPO_ROOT/logs/$round_id"
  mkdir -p "$dir"
  log "run[$name]: $*"
  "$@" 2>&1 | tee "$dir/$name.log"
}

# ---------------------------------------------------------------------------
# minimal YAML reader (flat "key: value" only; good enough for our configs)
# ---------------------------------------------------------------------------
# yaml_get <file> <key> [default] -> prints value (quotes/comment stripped).
yaml_get() {
  local file="$1" key="$2" default="${3:-}" val
  val="$(grep -E "^[[:space:]]*${key}[[:space:]]*:" "$file" 2>/dev/null \
        | head -n1 \
        | sed -E "s/^[[:space:]]*${key}[[:space:]]*:[[:space:]]*//; s/[[:space:]]*#.*$//; s/^[\"']//; s/[\"']\$//" )"
  if [ -z "$val" ]; then printf '%s' "$default"; else printf '%s' "$val"; fi
}

# ---------------------------------------------------------------------------
# GPU-allocation table (by model size) + 5h timing constants
# ---------------------------------------------------------------------------
# gpus_per_run <model-alias> -> 1 | 2 | 4
gpus_per_run() {
  case "$1" in
    roberta-base|gpt2)            echo 1 ;;
    roberta-large|gpt2-large)     echo 2 ;;
    qwen2.5-1.5b|qwen2.5-3b)      echo 4 ;;
    *) die "unknown model alias for GPU sizing: $1" ;;
  esac
}

# per_step_seconds <model-alias> -> coarse FULL fine-tune wall per optimizer step
# (measured on a 3090: roberta-base ~75s at batch 1024 due to Opacus per-sample
# grads). Used only for the 5h guard; sweep.sh divides by LORA_SPEEDUP for LoRA.
per_step_seconds() {
  case "$1" in
    roberta-base|gpt2)            echo 75.0 ;;
    roberta-large|gpt2-large)     echo 280.0 ;;
    qwen2.5-1.5b)                 echo 120.0 ;;
    qwen2.5-3b)                   echo 200.0 ;;
    *) die "unknown model alias for timing: $1" ;;
  esac
}

# Approx LoRA speedup over full fine-tune (frozen base -> no big per-sample grads).
LORA_SPEEDUP=15

# parallel_runs <total-gpus> <gpus-per-run> -> floor(total/per), at least 1.
parallel_runs() {
  local total="$1" per="$2" n
  n=$(( total / per ))
  [ "$n" -lt 1 ] && n=1
  echo "$n"
}
