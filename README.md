# An optimization view of differentially private LLM fine-tuning.

> When does DP-AdamBC help, and can a better optimizer beat well-tuned DP-Adam?

<img width="100%" alt="Project overview figure" src="https://github.com/user-attachments/assets/a54f5e49-0ad9-4c36-8f44-eeed8f25fa2a" />

## Overview

This repository studies **differentially private LLM fine-tuning** through the lens of optimization.

The starting point is **DP-AdamBC**: a bias-corrected variant of DP-Adam that subtracts the analytic DP-noise variance

$$
\Phi = (\sigma C / B)^2
$$

from Adam's second-moment estimate $\hat v$. The project asks two questions:

1. **When does second-moment bias correction actually help in DP fine-tuning?**
2. **Can a principled correlated-noise optimizer outperform a well-tuned, privacy-amplified DP-Adam baseline?**

---

## Method Summary

We study the reference optimizer **DP-AdamBC** and introduce a correlated-noise optimizer family:

* **DP-CorrMom**: Adam-style optimization with anti-correlated noise on the first-moment path.
* **DP-CorrSGD**: matched SGD-style optimization with correlated noise on the prefix-sum workload.

The correlated noise takes the form

$$
w_t = z_t - \lambda z_{t-1},
$$

with corrected privacy sensitivity

$$
\kappa = \sqrt{\frac{1-\lambda^{2T}}{1-\lambda^2}}.
$$

This connects optimizer behavior to DP-FTRL / matrix-factorization mechanisms for private prefix sums.

---

## Installation

```bash
python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
export PYTHONPATH=src
```

The project uses `opacus==1.6.0` for DP training and accounting.

---

## Optimizer Variants

Use the `--optimizer` flag to choose an optimizer.

| Name         | Update rule                             | Purpose                                             |
| ------------ | --------------------------------------- | --------------------------------------------------- |
| `dp-sgd`     | SGD with optional momentum              | Basic DP baseline                                   |
| `dp-adam`    | DP-Adam                                 | Strong privacy-amplified baseline                   |
| `dp-adambc`  | DP-Adam with $\hat v - \Phi$            | Bias-correction reference method                    |
| `dp-adam-xi` | DP-Adam with floor only                 | Control: floor without $\Phi$-subtraction           |
| `dp-adam-lp` | DP-Adam with low-pass filtered $\hat m$ | First-moment denoising control                      |
| `dp-corrmom` | Adam-style update with correlated noise | Correlated noise on the momentum path               |
| `dp-corrsgd` | SGD-style update with correlated noise  | Correlated noise on the matched prefix-sum workload |

The correlated optimizers use **single-participation, unamplified accounting**. The dataloader requires `steps × batch_size` distinct examples and raises an error otherwise, preventing silent privacy violations.

Relevant implementation:

```text
src/dp_optim/dp_adaptive.py
```

---

## Reproduction

### Step 1: Unit Tests

Fast CPU tests with no dataset download:

```bash
PYTHONPATH=src python tests/test_dp_adaptive.py
PYTHONPATH=src python tests/test_dp_corrmom.py
```

Dry-run privacy / correctness check:

```bash
PYTHONPATH=src python src/train.py \
  --dry-run               \
  --task e2e              \
  --optimizer dp-corrmom  \
  --epsilon 8             \
  --lambda-corr 0.8
```

### Step 2: Experiments

Each script corresponds to one result discussed in the project.

| Script                                | Result                                                               |
| ------------------------------------- | -------------------------------------------------------------------- |
| `experiments/qwen_e2e_floor_sweep.sh` | Bias correction versus floor-only control; DP-AdamBC collapse        |
| `experiments/lrctrl_eps8_b4096.sh`    | Learning-rate control; tuned DP-Adam plateau and inverted-U behavior |
| `experiments/epsweep.sh`              | $\rho$-saturation across privacy budgets                             |
| `experiments/bsweep.sh`               | $\rho$-saturation across batch sizes                                 |
| `experiments/e0_lever.sh`             | First-moment low-pass denoising control                              |
| `experiments/cm_qe2e.sh`              | DP-CorrMom on Adam-style optimization                                |
| `experiments/cm_beta0.sh`             | DP-CorrMom with $\beta_1 = 0$                                        |
| `experiments/cm_sgd.sh`               | DP-CorrSGD on the matched SGD workload                               |

Example single run:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python src/train.py \
  --model qwen2.5-1.5b  \
  --task e2e            \
  --lora                \
  --lora-r 16           \ 
  --optimizer dp-adambc \
  --epsilon 3           \
  --batch-size 512      \
  --micro-batch 16      \
  --lr 1e-3             \ 
  --max-grad-norm 0.1   \
  --steps 150           \
  --eval-every 50       \
  --seed 0              \
  --out results/
```

---

## Models and Tasks

The experiments focus on two DP fine-tuning settings:

| Model         | Task    | Purpose                      |
| ------------- | ------- | ---------------------------- |
| RoBERTa-large | MNLI    | Classification setting       |
| Qwen2.5-1.5B  | E2E-NLG | Generation / DP-LoRA setting |

---

## Main Reference

This project is motivated by DP-AdamBC:

```bibtex
@misc{tang2023dpadambc,
  title         = {DP-AdamBC: Your DP-Adam Is Actually DP-SGD (Unless You Apply Bias Correction)},
  author        = {Tang, Qiaoyue and Shpilevskiy, Frederick and Lécuyer, Mathias},
  year          = {2023},
  eprint        = {2312.14334},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  url           = {https://arxiv.org/abs/2312.14334}
}
```

---

## Citation

If you use this repository or build on the project, please cite:

```bibtex
@misc{wang2026optimizationviewdpllm,
  title        = {An Optimization View of DP LLM Fine-tuning: When Does Bias Correction Help, and Can the Optimizer Be Improved?},
  author       = {Wang, Zixuan},
  year         = {2026},
  note         = {Optimization Methods for AI course project},
  howpublished = {\url{https://github.com/wannabeyourfriend/optimization-course-project}}
}
```

Please also cite the directly relevant prior work:

```bibtex
@misc{tang2023dpadambc,
  title         = {DP-AdamBC: Your DP-Adam Is Actually DP-SGD (Unless You Apply Bias Correction)},
  author        = {Tang, Qiaoyue and Shpilevskiy, Frederick and Lécuyer, Mathias},
  year          = {2023},
  eprint        = {2312.14334},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  url           = {https://arxiv.org/abs/2312.14334}
}

@article{yousefpour2021opacus,
  title   = {Opacus: User-Friendly Differential Privacy Library in PyTorch},
  author  = {Yousefpour, Ashkan and Shilov, Igor and Sablayrolles, Alexandre and Testuggine, Davide and Prasad, Karthik and Malek, Mani and Nguyen, John and Ghosh, Sayan and Bharadwaj, Akash},
  journal = {arXiv preprint arXiv:2109.12298},
  year    = {2021},
  url     = {https://arxiv.org/abs/2109.12298}
}
```
