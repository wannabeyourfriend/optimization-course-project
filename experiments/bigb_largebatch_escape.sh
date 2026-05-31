#!/usr/bin/env bash
# LARGE-BATCH ESCAPE TEST: does a literature-scale batch (B>=2048) drop the noise share
# below 1 and let roberta-large/MNLI DP-LoRA actually learn? DP needs big batches to average
# out per-step noise (Li et al. 2022 use B~6144). Solo per GPU for speed; micro=32.
cd /home/2025user/zhou/dp-llm-ft || exit 1
set -a; source .env 2>/dev/null; set +a
export HF_ENDPOINT=https://hf-mirror.com PYTHONPATH=src WANDB_PROJECT=dp-optimizer-finetuning
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True WANDB_MODE=online
mkdir -p logs results
ROUND=bigb; LR=1e-3; C=0.1; MICRO=32; STEPS=250
# (batch|eps|gpu) — solo per GPU
RUNS=("2048|8|1" "2048|inf|3")
echo "=== BIGB START $(date +%F_%H:%M:%S) ==="
for run in "${RUNS[@]}"; do
  IFS='|' read -r b e g <<< "$run"
  EV=$(( STEPS/3 )); rid="${ROUND}-b${b}e${e}"
  echo "  dp-adam B=$b eps=$e -> GPU$g"
  CUDA_VISIBLE_DEVICES="$g" nohup .venv/bin/python -u src/train.py \
    --model roberta-large --task mnli --lora --lora-r 16 \
    --optimizer dp-adam --epsilon "$e" --batch-size "$b" --micro-batch "$MICRO" \
    --lr "$LR" --max-grad-norm "$C" --steps "$STEPS" --eval-every "$EV" \
    --seed 0 --round-id "$rid" --wandb-project dp-optimizer-finetuning --out results/ \
    > "logs/${ROUND}_b${b}e${e}.log" 2>&1 &
done
wait; echo "=== BIGB DONE $(date +%F_%H:%M:%S) ==="
