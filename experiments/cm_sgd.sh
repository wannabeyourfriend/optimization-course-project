#!/usr/bin/env bash
# DP-CorrSGD CAPSTONE: correlated noise on PLAIN SGD (momentum 0, no v_hat denominator) -- the
# textbook DP-FTRL/MF matched workload (theta_T = prefix-sum of g+w). Removes the two confounds of
# the Adam-based dp-corrmom (momentum-window mismatch + denominator poisoning). Plain SGD needs a
# MUCH larger LR than Adam (no per-coord normalization), so sweep LR x lambda. INTERNAL test:
# lambda=0.95 vs lambda=0 at matched LR -> does correlated noise help the matched workload? (NOTE:
# SGD<Adam absolutely on E2E, so this tests the MECHANISM, not necessarily beating DP-Adam.)
cd /home/2025user/zhou/dp-llm-ft || exit 1
set -a; source .env 2>/dev/null; set +a
export HF_ENDPOINT=https://hf-mirror.com PYTHONPATH=src WANDB_PROJECT=dp-optimizer-finetuning
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True WANDB_MODE=online
mkdir -p logs results
EPS=8; C=0.1; B=256; MICRO=16; STEPS=120; EV=40; SEED=0
GPUS=(0 1 2 3); MAXJ=4; j=0
launch() { # lr lam rid
  local lr="$1" lam="$2" rid="$3"
  local g=${GPUS[$(( j % ${#GPUS[@]} ))]}
  echo "  $rid (lr=$lr lam=$lam) -> GPU$g ($(date +%H:%M:%S))"
  CUDA_VISIBLE_DEVICES="$g" nohup .venv/bin/python -u src/train.py \
    --model qwen2.5-1.5b --task e2e --lora --lora-r 16 \
    --optimizer dp-corrsgd --epsilon "$EPS" --batch-size "$B" --micro-batch "$MICRO" \
    --lr "$lr" --lambda-corr "$lam" --max-grad-norm "$C" --steps "$STEPS" --eval-every "$EV" \
    --seed "$SEED" --round-id "$rid" --wandb-project dp-optimizer-finetuning --out results/ \
    > "logs/${rid}.log" 2>&1 &
  j=$((j+1)); while [ "$(jobs -rp | wc -l)" -ge "$MAXJ" ]; do wait -n; done
}
echo "=== CM-SGD START $(date +%F_%H:%M:%S) ==="
for lr in 1 10 100; do
  launch "$lr" 0    "cmsgd-lr${lr}-l0"
  launch "$lr" 0.95 "cmsgd-lr${lr}-l0.95"
done
wait; echo "=== CM-SGD DONE $(date +%F_%H:%M:%S) ==="
