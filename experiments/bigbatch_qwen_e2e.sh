#!/usr/bin/env bash
# ROBUSTNESS: does a large batch on the LEARNING setting (Qwen/E2E) drop the noise share
# rho below 1? Analytic answer: no (v_true~=0 already), but confirm empirically at B=4096
# (8x the qe2efl batch), eps=8 (loosest -> smallest Phi -> best chance). dp-adam, measure rho.
cd /home/2025user/zhou/dp-llm-ft || exit 1
set -a; source .env 2>/dev/null; set +a
export HF_ENDPOINT=https://hf-mirror.com PYTHONPATH=src WANDB_PROJECT=dp-optimizer-finetuning
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True WANDB_MODE=online
mkdir -p logs results
LR=1e-3; C=0.1; MICRO=16; STEPS=50
CUDA_VISIBLE_DEVICES=1 nohup .venv/bin/python -u src/train.py \
  --model qwen2.5-1.5b --task e2e --lora --lora-r 16 \
  --optimizer dp-adam --epsilon 8 --batch-size 4096 --micro-batch "$MICRO" \
  --lr "$LR" --max-grad-norm "$C" --steps "$STEPS" --eval-every 25 \
  --seed 0 --round-id bigbatch-b4096e8 \
  --wandb-project dp-optimizer-finetuning --out results/ \
  > logs/bigbatch_b4096e8.log 2>&1 &
echo "launched pid=$!"
