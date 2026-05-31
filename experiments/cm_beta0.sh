#!/usr/bin/env bash
# DP-CorrMom MATCHED-WORKLOAD test: beta1=0 (NO momentum), so the parameter trajectory IS the
# gradient prefix-sum the bidiagonal strategy is designed for. The cmqe runs showed lambda>0 HURTS
# on Adam-momentum (54 vs 55.7 for lambda=0) -- hypothesis: momentum's short EMA window mismatches
# the full-prefix-sum cancellation while paying the full kappa noise inflation. Here, with beta1=0,
# correlated noise should provide the averaging momentum otherwise would. Clean test: cmb0-l0
# (beta1=0, lambda=0 = plain noisy DP-SGD, should learn POORLY at rho~=1) vs cmb0-l{0.9,0.95,0.99}.
# If lambda>0 >> lambda=0 at beta1=0, correlated noise structurally replaces momentum -> positive.
cd /home/2025user/zhou/dp-llm-ft || exit 1
set -a; source .env 2>/dev/null; set +a
export HF_ENDPOINT=https://hf-mirror.com PYTHONPATH=src WANDB_PROJECT=dp-optimizer-finetuning
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True WANDB_MODE=online
mkdir -p logs results
EPS=8; C=0.1; B=256; MICRO=16; STEPS=120; EV=40; LR=2e-3; SEED=0; BETA1=0
GPUS=(1 3); MAXJ=2; j=0
launch() { # lam rid
  local lam="$1" rid="$2"
  local g=${GPUS[$(( j % ${#GPUS[@]} ))]}
  echo "  $rid (lam=$lam beta1=$BETA1) -> GPU$g ($(date +%H:%M:%S))"
  CUDA_VISIBLE_DEVICES="$g" nohup .venv/bin/python -u src/train.py \
    --model qwen2.5-1.5b --task e2e --lora --lora-r 16 \
    --optimizer dp-corrmom --epsilon "$EPS" --batch-size "$B" --micro-batch "$MICRO" \
    --lr "$LR" --beta1 "$BETA1" --lambda-corr "$lam" --max-grad-norm "$C" --steps "$STEPS" \
    --eval-every "$EV" --seed "$SEED" --round-id "$rid" \
    --wandb-project dp-optimizer-finetuning --out results/ \
    > "logs/${rid}.log" 2>&1 &
  j=$((j+1)); while [ "$(jobs -rp | wc -l)" -ge "$MAXJ" ]; do wait -n; done
}
echo "=== CM-BETA0 START $(date +%F_%H:%M:%S) eps=$EPS beta1=0 ==="
launch 0    cmb0-l0-eps8
launch 0.9  cmb0-l0.9-eps8
launch 0.95 cmb0-l0.95-eps8
launch 0.99 cmb0-l0.99-eps8
wait; echo "=== CM-BETA0 DONE $(date +%F_%H:%M:%S) ==="
