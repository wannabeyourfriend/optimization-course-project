# Optimization Methods for AI Course Project

<img width="1672" height="941" alt="42c9a90a-5a77-4868-84c1-1c836fa97144" src="https://github.com/user-attachments/assets/a54f5e49-0ad9-4c36-8f44-eeed8f25fa2a" />

### An Optimization View of DP LLM Fine-tuning: *When Does Bias Correction Help, and Can the Optimizer Be Improved?*

This project studies, from an optimization viewpoint, the reference method **DP-AdamBC**
(Tang et al., AAAI 2024) — DP-Adam with the analytic DP-noise variance bias `Φ = (σ·C/B)²`
subtracted from Adam's second moment `v̂` — and asks **when it actually helps in differentially
private LLM fine-tuning, and whether a principled optimizer change can beat a well-tuned, privacy-amplified DP-Adam.**

**Main results** (full derivations + experiments in [`report/course-report.pdf`](report/course-report.pdf)):

1. **A diagnostic `ρ = Φ/v̂ ∈ [0,1]`** tells you *a priori* whether bias correction can help. Its
   useful regime is `ρ ≈ ½`; but in the standard DP-LoRA recipe **`ρ` saturates at ≈1**, so `v̂` is
   an inert noise floor and **learning rides the first moment `m̂`** (a gradient prefix-sum).
2. At `ρ≈1`, **bias correction collapses to momentum-SGD** and at `ρ<1` it is only an
   **effective learning-rate** knob — settled by four controls (floor sweep, floor-only control, an
   LR sweep, and a Muon/orthogonalization probe), two models, three seeds.
3. We then build a **positive method** — **DP-CorrMom / DP-CorrSGD**: anti-correlated noise
   `w_t = z_t − λ·z_{t-1}` on the first-moment / prefix-sum path (à la DP-FTRL / matrix
   factorization, with a corrected privacy sensitivity `κ = √((1−λ^{2T})/(1−λ²))`). It **does not
   beat a tuned, amplified DP-Adam** in this regime — a *unified negative* explained by a
   **signal-ceiling**: at `ρ≈1` the model is signal-limited, not variance-limited, so further noise
   reduction cannot help.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate     # or: uv venv
pip install -r requirements.txt                       # includes opacus==1.6.0
export PYTHONPATH=src
```

## Optimizer variants (`--optimizer`)

| name | update | what it tests |
|---|---|---|
| `dp-sgd` | SGD(+momentum) | baseline |
| `dp-adam` | Adam | the strong, amplified baseline (the bar) |
| `dp-adambc` | Adam, `v̂ − Φ` | **bias correction** (the reference method) |
| `dp-adam-xi` | Adam, floor only | control: floor without Φ-subtraction |
| `dp-adam-lp` | Adam + low-pass on `m̂` | first-moment denoising (privacy-free) |
| `dp-corrmom` | Adam + correlated noise | correlated noise on the momentum path |
| `dp-corrsgd` | plain SGD + correlated noise | correlated noise on the *matched* prefix-sum workload |

The correlated optimizers use **single-participation, unamplified** accounting: the loader uses
`steps·batch` distinct examples (it *raises* otherwise, so privacy is never silently violated) and
σ is inflated by `corr_sensitivity(λ, steps)`. See [`src/dp_optim/dp_adaptive.py`](src/dp_optim/dp_adaptive.py).

## Reproduce

**Step 1 — correctness/privacy unit tests (fast, CPU, no download):**
```bash
PYTHONPATH=src python tests/test_dp_adaptive.py

PYTHONPATH=src python tests/test_dp_corrmom.py

PYTHONPATH=src python src/train.py 
   --dry-run \ 
   --task e2e \ 
   --optimizer dp-corrmom \
   --epsilon 8 \
   --lambda-corr 0.8
```

**Step 2 — experiments (GPU; each script → one result in the report):**

| script | result |
|---|---|
| `experiments/qwen_e2e_floor_sweep.sh` | bias correction → momentum-SGD collapse (BC vs floor-only) |
| `experiments/lrctrl_eps8_b4096.sh`    | LR control: tuned DP-Adam plateau, inverted-U |
| `experiments/epsweep.sh`, `bsweep.sh` | `ρ` saturation across ε / batch |
| `experiments/e0_lever.sh`             | first-moment low-pass (negative) |
| `experiments/cm_qe2e.sh` · `cm_beta0.sh` · `cm_sgd.sh` | DP-CorrMom on Adam / β1=0 / matched SGD (all negative) |

Example single run:
```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python src/train.py \
  --model qwen2.5-1.5b \
  --task e2e --lora --lora-r 16 \
  --optimizer dp-adambc \
  --epsilon 3 \
  --batch-size 512 \
  --micro-batch 16 \
  --lr 1e-3 \
  --max-grad-norm 0.1 \
  --steps 150 \
  --eval-every 50 \
  --seed 0 \
  --out results/
```

## Repository layout

```
src/dp_optim/   DPAdaptive + DPCorrelatedOptimizer/Adaptive + make_dp_optimizer + corr_sensitivity
src/train.py    single training entry point (DP-LoRA, Opacus, PRV accounting, single-participation path)
tests/          privacy/correctness unit tests
experiments/    one bash script per result (W&B-tracked)
scripts/        plotting + dashboard
report/         the course report (course-report.tex + proofs + course-report.pdf)
```

## Main Reference of this project

@misc{tang2023dpadambcdpadamactuallydpsgd,
      title={DP-AdamBC: Your DP-Adam Is Actually DP-SGD (Unless You Apply Bias Correction)}, 
      author={Qiaoyue Tang and Frederick Shpilevskiy and Mathias Lécuyer},
      year={2023},
      eprint={2312.14334},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2312.14334}, 
}

Models: RoBERTa-large/MNLI and Qwen2.5-1.5B/E2E-NLG.

## Citation


