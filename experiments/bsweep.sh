#!/usr/bin/env bash
# BATCH-SIZE ABLATION (money experiment for H3) — roberta-large/MNLI, DP-LoRA r16, C=0.1.
# Shared LR=1e-3 for all Adam arms (tuned to the dp-adam BASELINE => conservative, biases
# AGAINST a BC win). Gate proved phi_over_vhat(measured)=phi/(vhat_true+phi) ~= 1.0 at
# B=64/eps3 (noise dominates, vhat_true~0); sweep B to trace the 1->0 crossover at 0.5.
# Fixed micro-budget per run (=> ~constant wall-clock across B). B and the xi-floor are
# encoded into round-id because train.py's W&B name + results JSON key on (round,opt,eps,seed)
# only — without this the batch sizes / xi variants would overwrite each other.
cd /home/2025user/zhou/dp-llm-ft || exit 1
set -a; source .env 2>/dev/null; set +a            # WANDB_API_KEY (falls back to ~/.netrc)
export HF_ENDPOINT=https://hf-mirror.com PYTHONPATH=src WANDB_PROJECT=dp-optimizer-finetuning
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True WANDB_MODE=online
mkdir -p logs results
ROUND=bsweep; LR=1e-3; C=0.1; MICRO=8; MICRO_BUDGET=4800
MAXJ=6; GPUS=(1 1 1 3 3 3)
xival(){ case "$1" in 8) echo 1e-8;; 6) echo 1e-6;; esac; }   # xitag -> xi

RUNS=()
# CORE money plot: eps=3, dp-adam vs dp-adambc, xi=1e-8, 6 batch sizes, 3 seeds.
for b in 16 32 64 128 256 512; do
  for o in dp-adam dp-adambc; do for s in 0 1 2; do RUNS+=("$o|3|$b|8|$s"); done; done
done
# FLOOR-ROBUSTNESS: dp-adambc with a LARGER floor (xi=1e-6, paper-style gamma') in the
# crossover region, where vhat_true~0 can make the naive 1e-8 floor blow up the step.
for b in 32 64 128; do for s in 0 1 2; do RUNS+=("dp-adambc|3|$b|6|$s"); done; done
# CONTROL A (Phi=0 falsifier): eps=inf, sigma=0 -> BC must match Adam (no spurious gain).
for b in 16 64 512; do for o in dp-adam dp-adambc; do RUNS+=("$o|inf|$b|8|0"); done; done
# CONTROL B (floor-only): xi-floored Adam (NO phi-subtraction) with a BINDING floor (1e-6)
# at the crossover -> isolates "does flooring alone help?" from phi-subtraction.
for s in 0 1 2; do RUNS+=("dp-adam-xi|3|64|6|$s"); done

echo "=== BSWEEP START $(date +%F_%H:%M:%S): ${#RUNS[@]} runs, $MAXJ lanes ==="
j=0
for run in "${RUNS[@]}"; do
  IFS='|' read -r o e b xt s <<< "$run"
  xi=$(xival "$xt"); S=$(( MICRO_BUDGET * MICRO / b )); EV=$(( S / 3 )); [ "$EV" -lt 20 ] && EV=20
  g=${GPUS[$(( j % ${#GPUS[@]} ))]}
  rid="${ROUND}-b${b}f${xt}"
  echo "  launch $o eps=$e B=$b xi=$xi s=$s -> GPU$g steps=$S ev=$EV ($(date +%H:%M:%S))"
  CUDA_VISIBLE_DEVICES="$g" nohup .venv/bin/python -u src/train.py \
    --model roberta-large --task mnli --lora --lora-r 16 \
    --optimizer "$o" --epsilon "$e" --batch-size "$b" --micro-batch "$MICRO" \
    --lr "$LR" --max-grad-norm "$C" --steps "$S" --eval-every "$EV" \
    --seed "$s" --xi "$xi" --round-id "$rid" \
    --wandb-project dp-optimizer-finetuning --out results/ \
    > "logs/${ROUND}_${o}_eps${e}_b${b}f${xt}_s${s}.log" 2>&1 &
  j=$((j+1))
  while [ "$(jobs -rp | wc -l)" -ge "$MAXJ" ]; do wait -n; done
done
wait
echo "=== BSWEEP DONE $(date +%F_%H:%M:%S) ==="
