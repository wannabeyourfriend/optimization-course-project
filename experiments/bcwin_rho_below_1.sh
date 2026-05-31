#!/usr/bin/env bash
# POSITIVE-RESULT ATTEMPT: at B=4096/eps=8 on Qwen/E2E the model LEARNS and rho=0.955<1
# (v_true~2e-11 now measurable). Test whether DP-AdamBC, with a floor xi small enough to
# actually de-bias (not fully clamp), beats DP-Adam -- the regime the rho-diagnostic predicts.
# Sweep xi in {1e-12,1e-11,1e-10}; dp-adam baseline gets 3 seeds. Track clamp_frac (<1 => BC acts).
cd /home/2025user/zhou/dp-llm-ft || exit 1
set -a; source .env 2>/dev/null; set +a
export HF_ENDPOINT=https://hf-mirror.com PYTHONPATH=src WANDB_PROJECT=dp-optimizer-finetuning
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True WANDB_MODE=online
mkdir -p logs results
LR=1e-3; C=0.1; B=4096; MICRO=16; STEPS=80; EPS=8
MAXJ=6; GPUS=(1 1 1 3 3 3); j=0
RUNS=()
for s in 0 1 2; do RUNS+=("dp-adam|adam|1e-8|$s"); done       # baseline, 3 seeds
RUNS+=("dp-adambc|x12|1e-12|0")                                # BC, floor sweep (1 seed to scan)
RUNS+=("dp-adambc|x11|1e-11|0")
RUNS+=("dp-adambc|x10|1e-10|0")
echo "=== BCWIN START $(date +%F_%H:%M:%S): ${#RUNS[@]} runs (B=$B eps=$EPS) ==="
for run in "${RUNS[@]}"; do
  IFS='|' read -r o tag xi s <<< "$run"
  EV=$(( STEPS/2 )); g=${GPUS[$(( j % ${#GPUS[@]} ))]}; rid="bcwin-${tag}"
  echo "  $o xi=$xi s=$s -> GPU$g"
  CUDA_VISIBLE_DEVICES="$g" nohup .venv/bin/python -u src/train.py \
    --model qwen2.5-1.5b --task e2e --lora --lora-r 16 \
    --optimizer "$o" --epsilon "$EPS" --batch-size "$B" --micro-batch "$MICRO" \
    --lr "$LR" --max-grad-norm "$C" --steps "$STEPS" --eval-every "$EV" \
    --seed "$s" --xi "$xi" --round-id "$rid" \
    --wandb-project dp-optimizer-finetuning --out results/ \
    > "logs/bcwin_${o}_${tag}_s${s}.log" 2>&1 &
  j=$((j+1)); while [ "$(jobs -rp | wc -l)" -ge "$MAXJ" ]; do wait -n; done
done
wait; echo "=== BCWIN DONE $(date +%F_%H:%M:%S) ==="
