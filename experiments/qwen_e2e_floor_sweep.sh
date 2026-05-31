#!/usr/bin/env bash
# QWEN-E2E FLOOR SWEEP: the working 1.5B DP-FT setting (E2E learns: BLEU~56 vs SGD~20.7),
# affordable (B=512/150 steps finished before). Tests the floor dichotomy that governs DP-AdamBC
# in the noise-saturated regime (pov~1): small floor xi over-subtracts (v_hat-phi -> ~0 -> blow
# up), large floor kills adaptivity (-> scaled momentum-SGD). dp-adam baseline + dp-adambc at
# xi in {1e-8,1e-7,1e-6} + dp-adam-xi(1e-6) floor-only control, eps=3, 3 seeds. xi in round-id
# (W&B name/JSON key on round,opt,eps,seed only).
cd /home/2025user/zhou/dp-llm-ft || exit 1
set -a; source .env 2>/dev/null; set +a
export HF_ENDPOINT=https://hf-mirror.com PYTHONPATH=src WANDB_PROJECT=dp-optimizer-finetuning
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True WANDB_MODE=online
mkdir -p logs results
ROUND=qe2efl; LR=1e-3; C=0.1; B=512; MICRO=16; STEPS=150; EPS=3
MAXJ=6; GPUS=(1 1 1 3 3 3); j=0
xival(){ case "$1" in 8) echo 1e-8;; 7) echo 1e-7;; 6) echo 1e-6;; esac; }
RUNS=()
for s in 0 1 2; do
  RUNS+=("dp-adam|8|$s")                                  # baseline (xi irrelevant)
  for xt in 8 7 6; do RUNS+=("dp-adambc|$xt|$s"); done    # BC at 3 floors
  RUNS+=("dp-adam-xi|6|$s")                               # floor-only control (binding 1e-6)
done
echo "=== QE2EFL START $(date +%F_%H:%M:%S): ${#RUNS[@]} runs ==="
for run in "${RUNS[@]}"; do
  IFS='|' read -r o xt s <<< "$run"; xi=$(xival "$xt")
  g=${GPUS[$(( j % ${#GPUS[@]} ))]}; rid="${ROUND}-x${xt}"
  echo "  $o xi=$xi s=$s -> GPU$g"
  CUDA_VISIBLE_DEVICES="$g" nohup .venv/bin/python -u src/train.py \
    --model qwen2.5-1.5b --task e2e --lora --lora-r 16 \
    --optimizer "$o" --epsilon "$EPS" --batch-size "$B" --micro-batch "$MICRO" \
    --lr "$LR" --max-grad-norm "$C" --steps "$STEPS" --eval-every 50 \
    --seed "$s" --xi "$xi" --round-id "$rid" \
    --wandb-project dp-optimizer-finetuning --out results/ \
    > "logs/${ROUND}_${o}_x${xt}_s${s}.log" 2>&1 &
  j=$((j+1)); while [ "$(jobs -rp | wc -l)" -ge "$MAXJ" ]; do wait -n; done
done
wait; echo "=== QE2EFL DONE $(date +%F_%H:%M:%S) ==="
