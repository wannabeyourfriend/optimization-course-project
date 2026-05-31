#!/usr/bin/env bash
# LEARNING-FRONTIER PROBE: find where roberta-large/MNLI DP-LoRA actually trains above
# chance (loss << ln3=1.0986). Vary batch x eps, dp-adam only, micro=32 for A100 throughput,
# short. Read final loss + diag/phi_over_vhat per cell to pick the regime for the BC study.
cd /home/2025user/zhou/dp-llm-ft || exit 1
set -a; source .env 2>/dev/null; set +a
export HF_ENDPOINT=https://hf-mirror.com PYTHONPATH=src WANDB_PROJECT=dp-optimizer-finetuning
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True WANDB_MODE=online
mkdir -p logs results
ROUND=probe; LR=1e-3; C=0.1; MICRO=32; STEPS=250
MAXJ=4; GPUS=(1 1 3 3); j=0
RUNS=()
for b in 256 512 1024; do for e in 3 8; do RUNS+=("$b|$e"); done; done
# looser clip variant at the most promising mid batch (C cancels in pov but test learning speed)
RUNS+=("512|inf")
echo "=== PROBE START $(date +%F_%H:%M:%S): ${#RUNS[@]} runs ==="
for run in "${RUNS[@]}"; do
  IFS='|' read -r b e <<< "$run"
  EV=$(( STEPS/2 )); g=${GPUS[$(( j % ${#GPUS[@]} ))]}; rid="${ROUND}-b${b}e${e}"
  echo "  probe dp-adam B=$b eps=$e -> GPU$g ($(date +%H:%M:%S))"
  CUDA_VISIBLE_DEVICES="$g" nohup .venv/bin/python -u src/train.py \
    --model roberta-large --task mnli --lora --lora-r 16 \
    --optimizer dp-adam --epsilon "$e" --batch-size "$b" --micro-batch "$MICRO" \
    --lr "$LR" --max-grad-norm "$C" --steps "$STEPS" --eval-every "$EV" \
    --seed 0 --round-id "$rid" --wandb-project dp-optimizer-finetuning --out results/ \
    > "logs/${ROUND}_b${b}e${e}.log" 2>&1 &
  j=$((j+1)); while [ "$(jobs -rp | wc -l)" -ge "$MAXJ" ]; do wait -n; done
done
wait; echo "=== PROBE DONE $(date +%F_%H:%M:%S) ==="
