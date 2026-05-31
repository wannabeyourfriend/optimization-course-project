#!/usr/bin/env bash
# EPS-SWEEP: map pov(eps) and learning(eps) on roberta-large/MNLI DP-LoRA (r16, C=0.1).
# Finds the escape threshold where Phi/(v_true+Phi) falls below 1 and loss drops below chance.
# B=256 fixed (fast), micro=32, dp-adam. This locates the regime (if any) where BC can matter.
cd /home/2025user/zhou/dp-llm-ft || exit 1
set -a; source .env 2>/dev/null; set +a
export HF_ENDPOINT=https://hf-mirror.com PYTHONPATH=src WANDB_PROJECT=dp-optimizer-finetuning
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True WANDB_MODE=online
mkdir -p logs results
ROUND=epsweep; LR=1e-3; C=0.1; B=256; MICRO=32; STEPS=350
MAXJ=6; GPUS=(1 1 1 3 3 3); j=0
RUNS=(8 16 32 64 128 inf)
echo "=== EPSWEEP START $(date +%F_%H:%M:%S): ${#RUNS[@]} eps @ B=$B ==="
for e in "${RUNS[@]}"; do
  EV=$(( STEPS/3 )); g=${GPUS[$(( j % ${#GPUS[@]} ))]}; rid="${ROUND}-e${e}"
  echo "  dp-adam eps=$e -> GPU$g ($(date +%H:%M:%S))"
  CUDA_VISIBLE_DEVICES="$g" nohup .venv/bin/python -u src/train.py \
    --model roberta-large --task mnli --lora --lora-r 16 \
    --optimizer dp-adam --epsilon "$e" --batch-size "$B" --micro-batch "$MICRO" \
    --lr "$LR" --max-grad-norm "$C" --steps "$STEPS" --eval-every "$EV" \
    --seed 0 --round-id "$rid" --wandb-project dp-optimizer-finetuning --out results/ \
    > "logs/${ROUND}_e${e}.log" 2>&1 &
  j=$((j+1)); while [ "$(jobs -rp | wc -l)" -ge "$MAXJ" ]; do wait -n; done
done
wait; echo "=== EPSWEEP DONE $(date +%F_%H:%M:%S) ==="
