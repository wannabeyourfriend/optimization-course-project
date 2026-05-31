#!/usr/bin/env python3
"""DP-CorrMom lambda-sweep figure: anti-correlated noise (lambda>0) never beats lambda=0 across
Adam-momentum, beta1=0, and the matched plain-SGD workload, and all stay below an amplified DP-Adam.
Qwen2.5-1.5B / E2E, eps=8, B=256, 1 seed (proxy BLEU). Data from results/ (cmqe-*, cmb0-*, cmsgd-*)."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# (lambda, BLEU) per variant, 1 seed
ADAM = ([0.0, 0.9, 0.95], [55.67, 54.35, 54.10])          # dp-corrmom, beta1=0.9 (Adam momentum)
BETA0 = ([0.0, 0.9, 0.95, 0.99], [56.01, 54.35, 54.26, 52.19])  # dp-corrmom, beta1=0
SGD = ([0.0, 0.95], [55.70, 55.35])                        # dp-corrsgd, plain SGD lr=10 (matched workload)
DPADAM_AMP = 56.22                                          # amplified DP-Adam reference (eps=6.40)

fig, ax = plt.subplots(figsize=(6.2, 4.2))
ax.axhline(DPADAM_AMP, ls="--", color="black", lw=1.4,
           label="amplified DP-Adam (ref, 56.2)")
for (lams, bleu), name, m, c in [
    (ADAM, r"dp-corrmom ($\beta_1{=}0.9$, Adam)", "o", "#1f77b4"),
    (BETA0, r"dp-corrmom ($\beta_1{=}0$)", "s", "#d62728"),
    (SGD, r"dp-corrsgd (plain SGD, matched)", "^", "#2ca02c"),
]:
    ax.plot(lams, bleu, marker=m, color=c, lw=1.8, ms=7, label=name)

ax.set_xlabel(r"anti-correlation strength $\lambda$")
ax.set_ylabel("E2E proxy BLEU (step 120)")
ax.set_title(r"Correlated noise ($\lambda{>}0$) never helps: the first-moment path"
             "\nis signal-limited, not variance-limited", fontsize=10.5)
ax.set_xlim(-0.03, 1.0)
ax.grid(alpha=0.3)
ax.legend(fontsize=8.5, loc="lower left")
ax.annotate(r"$\lambda{=}0$ controls", xy=(0.0, 55.9), xytext=(0.18, 56.0),
            fontsize=8, arrowprops=dict(arrowstyle="->", lw=0.7, color="gray"))
fig.tight_layout()

for d in ("paper/latex/figures", "docs/figures"):
    os.makedirs(d, exist_ok=True)
    fig.savefig(os.path.join(d, "corrmom_lambda.png"), dpi=150, bbox_inches="tight")
print("wrote corrmom_lambda.png to paper/latex/figures and docs/figures")
