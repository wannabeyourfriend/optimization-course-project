#!/usr/bin/env bash
# MONEY-PLOT batch sweep at fixed eps=8: trace BC gain vs rho as batch grows (rho 1 -> <1).
# dp-adam vs dp-adambc(xi=1e-11, the winning floor) at B in {512,1024,2048} (B=4096 already done
# in bcwin). 2 seeds each. Measures diag/phi_over_vhat + eval BLEU per cell. LAUNCH WHEN zhou-1 FREE.
cd /home/2025user/zhou/dp-llm-ft || exit 1
set -a; source .env 2>/dev/null; set +a
export HF_ENDPOINT=https://hf-mirror.com PYTHONPATH=src WANDB_PROJECT=dp-optimizer-finetuning
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True WANDB_MODE=online
mkdir -p logs results
LR=1e-3; C=0.1; MICRO=16; STEPS=80; EPS=8
MAXJ=4; GPUS=(1 1 3 3); j=0
RUNS=()
for b in 512 1024 2048; do
  for s in 0 1; do
    RUNS+=("dp-adam|$b|1e-8|$s")
    RUNS+=("dp-adambc|$b|1e-11|$s")
  done
done
echo "=== RATEBATCH START $(date +%F_%H:%M:%S): ${#RUNS[@]} runs ==="
for run in "${RUNS[@]}"; do
  IFS='|' read -r o b xi s <<< "$run"
  g=${GPUS[$(( j % ${#GPUS[@]} ))]}; rid="ratebatch-b${b}"
  echo "  $o B=$b xi=$xi s=$s -> GPU$g"
  CUDA_VISIBLE_DEVICES="$g" nohup .venv/bin/python -u src/train.py \
    --model qwen2.5-1.5b --task e2e --lora --lora-r 16 \
    --optimizer "$o" --epsilon "$EPS" --batch-size "$b" --micro-batch "$MICRO" \
    --lr "$LR" --max-grad-norm "$C" --steps "$STEPS" --eval-every 40 \
    --seed "$s" --xi "$xi" --round-id "$rid" \
    --wandb-project dp-optimizer-finetuning --out results/ \
    > "logs/ratebatch_${o}_b${b}_s${s}.log" 2>&1 &
  j=$((j+1)); while [ "$(jobs -rp | wc -l)" -ge "$MAXJ" ]; do wait -n; done
done
wait; echo "=== RATEBATCH DONE $(date +%F_%H:%M:%S) ==="
