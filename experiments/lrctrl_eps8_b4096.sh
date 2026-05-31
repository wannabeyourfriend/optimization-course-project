#!/usr/bin/env bash
# CRITICAL CONTROL for the rho<1 positive: is DP-AdamBC's win genuine per-coordinate de-biasing,
# or just an effective-LR increase (BC removes ~95% of v_hat -> ~4.7x larger step)? Sweep dp-adam's
# LR at B=4096/eps=8 and compare to dp-adambc-x11 (LR=1e-3, =56.36). If NO dp-adam LR matches
# dp-adambc -> BC helps beyond LR tuning (STRONG positive). If a tuned dp-adam matches -> BC is a
# step-size effect (cycle-15 synthesis holds even at rho<1). LAUNCH WHEN zhou-1 FREE.
cd /home/2025user/zhou/dp-llm-ft || exit 1
set -a; source .env 2>/dev/null; set +a
export HF_ENDPOINT=https://hf-mirror.com PYTHONPATH=src WANDB_PROJECT=dp-optimizer-finetuning
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True WANDB_MODE=online
mkdir -p logs results
C=0.1; B=4096; MICRO=16; STEPS=80; EPS=8
MAXJ=4; GPUS=(1 1 3 3); j=0
RUNS=()
# dp-adam LR sweep (the control), 2 seeds; plus dp-adambc-x11 anchor (more seeds via bcconfirm)
for lr in 2e-3 3e-3 5e-3 1e-2; do for s in 0 1; do RUNS+=("dp-adam|$lr|1e-8|$s"); done; done
echo "=== LRCTRL START $(date +%F_%H:%M:%S): ${#RUNS[@]} runs ==="
for run in "${RUNS[@]}"; do
  IFS='|' read -r o lr xi s <<< "$run"
  g=${GPUS[$(( j % ${#GPUS[@]} ))]}; rid="lrctrl-lr${lr}"
  echo "  $o lr=$lr s=$s -> GPU$g"
  CUDA_VISIBLE_DEVICES="$g" nohup .venv/bin/python -u src/train.py \
    --model qwen2.5-1.5b --task e2e --lora --lora-r 16 \
    --optimizer "$o" --epsilon "$EPS" --batch-size "$B" --micro-batch "$MICRO" \
    --lr "$lr" --max-grad-norm "$C" --steps "$STEPS" --eval-every 40 \
    --seed "$s" --xi "$xi" --round-id "$rid" \
    --wandb-project dp-optimizer-finetuning --out results/ \
    > "logs/lrctrl_lr${lr}_s${s}.log" 2>&1 &
  j=$((j+1)); while [ "$(jobs -rp | wc -l)" -ge "$MAXJ" ]; do wait -n; done
done
wait; echo "=== LRCTRL DONE $(date +%F_%H:%M:%S) ==="
