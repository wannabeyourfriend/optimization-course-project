#!/usr/bin/env bash
# CONFIRM the positive result: add seeds 1,2 of the winning floor xi=1e-11 (and one x10 seed)
# at B=4096/eps=8 so dp-adambc gets 3 seeds vs the existing 3 dp-adam seeds. Full step-80.
cd /home/2025user/zhou/dp-llm-ft || exit 1
set -a; source .env 2>/dev/null; set +a
export HF_ENDPOINT=https://hf-mirror.com PYTHONPATH=src WANDB_PROJECT=dp-optimizer-finetuning
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True WANDB_MODE=online
mkdir -p logs results
LR=1e-3; C=0.1; B=4096; MICRO=16; STEPS=80; EPS=8
# (xi-tag|xi|seed|gpu)
RUNS=("x11|1e-11|1|1" "x11|1e-11|2|1" "x10|1e-10|1|3" "x10|1e-10|2|3")
echo "=== BCCONFIRM START $(date +%F_%H:%M:%S) ==="
for run in "${RUNS[@]}"; do
  IFS='|' read -r tag xi s g <<< "$run"
  CUDA_VISIBLE_DEVICES="$g" nohup .venv/bin/python -u src/train.py \
    --model qwen2.5-1.5b --task e2e --lora --lora-r 16 \
    --optimizer dp-adambc --epsilon "$EPS" --batch-size "$B" --micro-batch "$MICRO" \
    --lr "$LR" --max-grad-norm "$C" --steps "$STEPS" --eval-every 40 \
    --seed "$s" --xi "$xi" --round-id "bcwin-${tag}" \
    --wandb-project dp-optimizer-finetuning --out results/ \
    > "logs/bcconfirm_${tag}_s${s}.log" 2>&1 &
  echo "  launched dp-adambc xi=$xi s=$s -> GPU$g"
done
wait; echo "=== BCCONFIRM DONE $(date +%F_%H:%M:%S) ==="
