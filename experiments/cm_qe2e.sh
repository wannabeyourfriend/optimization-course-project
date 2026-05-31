#!/usr/bin/env bash
# DP-CorrMom headline probe on the LEARNING setting (Qwen2.5-1.5B/E2E, BLEU~56; roberta/MNLI is
# noise-swamped at chance so the lever can't show there). Tests whether anti-correlated noise on
# the first-moment/prefix-sum path beats its OWN unamplified control (lambda=0 = DP-SGD-momentum)
# at matched eps/LR/batch -- a mechanism-attributable win (lambda changes the NOISE, not the step).
# dp-adam = amplified reference (shows the Poisson-amplification gap dp-corrmom forgoes, route A).
# Single-participation (make_corr_loader RAISES if steps*B > n_train -> safe, no silent privacy break).
cd /home/2025user/zhou/dp-llm-ft || exit 1
set -a; source .env 2>/dev/null; set +a
export HF_ENDPOINT=https://hf-mirror.com PYTHONPATH=src WANDB_PROJECT=dp-optimizer-finetuning
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True WANDB_MODE=online
mkdir -p logs results
EPS=8; C=0.1; B=256; MICRO=16; STEPS=120; EV=40; LR=2e-3; SEED=0
GPUS=(1 3); MAXJ=2; j=0
launch() { # opt lam rid
  local opt="$1" lam="$2" rid="$3"
  local g=${GPUS[$(( j % ${#GPUS[@]} ))]}
  echo "  $rid (opt=$opt lam=$lam) -> GPU$g ($(date +%H:%M:%S))"
  CUDA_VISIBLE_DEVICES="$g" nohup .venv/bin/python -u src/train.py \
    --model qwen2.5-1.5b --task e2e --lora --lora-r 16 \
    --optimizer "$opt" --epsilon "$EPS" --batch-size "$B" --micro-batch "$MICRO" \
    --lr "$LR" --lambda-corr "$lam" --max-grad-norm "$C" --steps "$STEPS" --eval-every "$EV" \
    --seed "$SEED" --round-id "$rid" --wandb-project dp-optimizer-finetuning --out results/ \
    > "logs/${rid}.log" 2>&1 &
  j=$((j+1)); while [ "$(jobs -rp | wc -l)" -ge "$MAXJ" ]; do wait -n; done
}
echo "=== CM-QE2E START $(date +%F_%H:%M:%S) eps=$EPS B=$B steps=$STEPS ==="
launch dp-corrmom 0    cmqe-l0-eps8
launch dp-corrmom 0.9  cmqe-l0.9-eps8
launch dp-corrmom 0.95 cmqe-l0.95-eps8
launch dp-adam    0    cmqe-adam-eps8
wait; echo "=== CM-QE2E DONE $(date +%F_%H:%M:%S) ==="
