#!/usr/bin/env python3
"""Entry point for a W&B hyperparameter sweep agent.

`wandb agent` launches this once per trial. wandb.init() picks up the trial's
config from the sweep controller; we translate it into train.py's argv and run
one DP-finetuning job (with the full per-step dynamics logged to the trial run).

Sweep config lives in configs/sweep_*.yaml. Launch with experiments/wandb_sweep.sh.
"""
import sys

import wandb

from train import parse_args, train


def main():
    run = wandb.init()  # sweep controller injects the trial hyperparameters
    c = dict(wandb.config)

    argv = [
        "--model", str(c["model"]),
        "--task", str(c["task"]),
        "--optimizer", str(c["optimizer"]),
        "--epsilon", str(c["epsilon"]),
        "--batch-size", str(c.get("batch_size", 2048)),
        "--micro-batch", str(c.get("micro_batch", 64)),
        "--steps", str(c.get("steps", 300)),
        "--seed", str(c.get("seed", 0)),
        "--max-len", str(c.get("max_len", 128)),
        "--eval-every", str(c.get("eval_every", 25)),  # dense eval for dynamics
        "--round-id", str(c.get("round_id", "wandb-sweep")),
    ]
    if c.get("lora"):
        argv += ["--lora", "--lora-r", str(c.get("lora_r", 16))]

    args = parse_args(argv)
    train(args, run=run)


if __name__ == "__main__":
    sys.exit(main())
