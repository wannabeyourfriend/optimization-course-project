#!/usr/bin/env bash
# E0: first-moment lever validation on roberta-large/MNLI DP-LoRA (r16, C=0.1), eps=3.
# Q: does an extra low-pass on m_hat (dp-adam-lp) beat a TUNED-LR dp-adam? lowpass is
# privacy-free post-processing (eps unchanged). Controls: dp-adam LR curve {5e-4,1e-3,2e-3}
# (any lp gain must beat the BEST of these, else it is just an effective-LR knob like BC);
# dp-sgd = momentum-SGD ablation. 1 seed probe; expand to eps=1 + 3 seeds if it moves.
cd /home/2025user/zhou/dp-llm-ft || exit 1
set -a; source .env 2>/dev/null; set +a
export HF_ENDPOINT=https://hf-mirror.com PYTHONPATH=src WANDB_PROJECT=dp-optimizer-finetuning
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True WANDB_MODE=online
mkdir -p logs results
EPS=3; C=0.1; B=256; MICRO=32; STEPS=350; EV=116; SEED=0
GPUS=(0 2); MAXJ=2; j=0
launch() { # opt lr lp rid
  local opt="$1" lr="$2" lp="$3" rid="$4"
  local g=${GPUS[$(( j % ${#GPUS[@]} ))]}
  echo "  $rid (opt=$opt lr=$lr lp=$lp) -> GPU$g ($(date +%H:%M:%S))"
  CUDA_VISIBLE_DEVICES="$g" nohup .venv/bin/python -u src/train.py \
    --model roberta-large --task mnli --lora --lora-r 16 \
    --optimizer "$opt" --epsilon "$EPS" --batch-size "$B" --micro-batch "$MICRO" \
    --lr "$lr" --lowpass-beta "$lp" --max-grad-norm "$C" --steps "$STEPS" --eval-every "$EV" \
    --seed "$SEED" --round-id "$rid" --wandb-project dp-optimizer-finetuning --out results/ \
    > "logs/${rid}.log" 2>&1 &
  j=$((j+1)); while [ "$(jobs -rp | wc -l)" -ge "$MAXJ" ]; do wait -n; done
}
echo "=== E0 START $(date +%F_%H:%M:%S) eps=$EPS ==="
launch dp-adam     5e-4 0    e0-adam-lr5e-4
launch dp-adam     1e-3 0    e0-adam-lr1e-3
launch dp-adam     2e-3 0    e0-adam-lr2e-3
launch dp-adam-lp  1e-3 0.9  e0-lp0.9-lr1e-3
launch dp-adam-lp  1e-3 0.95 e0-lp0.95-lr1e-3
launch dp-sgd      1e-3 0    e0-sgd-lr1e-3
wait; echo "=== E0 DONE $(date +%F_%H:%M:%S) ==="
