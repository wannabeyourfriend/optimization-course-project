#!/usr/bin/env python3
"""Floor-dichotomy money plot for DP-AdamBC at pov~1 (Qwen-1.5B/E2E, eps=3).

Pulls the qe2efl round from W&B and plots utility (proxy BLEU) vs the xi-floor for
dp-adambc, against the dp-adam baseline and the dp-adam-xi (floor-only) control, with
clamp_frac annotated. The story: tiny floor -> BC ~= Adam; growing floor -> over-clamping ->
adaptivity collapses toward scaled-SGD. BC never beats Adam in the noise-saturated regime.

Usage: WANDB_API_KEY=... python scripts/plot_floor_dichotomy.py
Writes docs/figures/floor_dichotomy.png and data/qe2efl_summary.csv.
"""
import csv
import os
from collections import defaultdict
import statistics as st

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import wandb

ENT, PROJ = "ziw178-uc-san-diego", "dp-optimizer-finetuning"
XI = {"8": 1e-8, "7": 1e-7, "6": 1e-6}


def main():
    api = wandb.Api(timeout=30)
    runs = [r for r in api.runs(f"{ENT}/{PROJ}")
            if (r.config or {}).get("round_id", "").startswith("qe2efl-")]
    # (opt, xitag) -> list of (metric, clamp_frac)
    agg = defaultdict(list)
    for r in runs:
        c = r.config or {}
        opt = (r.name or "").split("/")[2] if len((r.name or "").split("/")) > 2 else "?"
        xt = c.get("round_id", "").split("-x")[-1]
        m = (r.summary or {}).get("eval/metric")
        cf = (r.summary or {}).get("diag/clamp_frac")
        if r.state == "finished" and isinstance(m, (int, float)):
            agg[(opt, xt)].append((m, cf))

    os.makedirs("docs/figures", exist_ok=True)
    os.makedirs("data", exist_ok=True)
    with open("data/qe2efl_summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["opt", "xi", "n", "bleu_mean", "bleu_std", "clamp_frac"])
        for (opt, xt), v in sorted(agg.items()):
            mets = [m for m, _ in v]
            cfs = [c for _, c in v if isinstance(c, (int, float))]
            w.writerow([opt, XI.get(xt, xt), len(mets),
                        round(st.mean(mets), 3),
                        round(st.pstdev(mets), 3) if len(mets) > 1 else 0.0,
                        round(st.mean(cfs), 3) if cfs else ""])

    # dp-adambc curve over xi
    bc_x, bc_y, bc_e = [], [], []
    for xt in ("8", "7", "6"):
        v = agg.get(("dp-adambc", xt))
        if v:
            mets = [m for m, _ in v]
            bc_x.append(XI[xt]); bc_y.append(st.mean(mets))
            bc_e.append(st.pstdev(mets) if len(mets) > 1 else 0.0)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    if bc_x:
        ax.errorbar(bc_x, bc_y, yerr=bc_e, marker="o", color="C3", capsize=3,
                    label="DP-AdamBC (subtract $\\Phi$, floor $\\xi$)")
    # dp-adam baseline (horizontal band)
    adam = agg.get(("dp-adam", "8"))
    if adam:
        mets = [m for m, _ in adam]; mu = st.mean(mets)
        sd = st.pstdev(mets) if len(mets) > 1 else 0.0
        ax.axhline(mu, color="C0", ls="--", label=f"DP-Adam baseline ({mu:.1f})")
        ax.axhspan(mu - sd, mu + sd, color="C0", alpha=0.12)
    # dp-adam-xi floor-only control points
    fo = agg.get(("dp-adam-xi", "6"))
    if fo:
        mets = [m for m, _ in fo]
        ax.scatter([1e-6], [st.mean(mets)], color="C2", marker="s", zorder=5,
                   label="floor-only control ($\\xi$=1e-6, no $\\Phi$-sub)")
    ax.set_xscale("log")
    ax.set_xlabel("xi floor on $(\\hat v-\\Phi)$")
    ax.set_ylabel("proxy BLEU (E2E, $\\varepsilon$=3)")
    ax.set_title("Floor dichotomy at $\\Phi/\\hat v\\approx1$ (Qwen-1.5B/E2E):\n"
                 "tiny floor $\\to$ BC$\\approx$Adam; large floor $\\to$ adaptivity off ($\\to$ scaled SGD)")
    ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig("docs/figures/floor_dichotomy.png", dpi=130)
    print("wrote docs/figures/floor_dichotomy.png and data/qe2efl_summary.csv")
    for (opt, xt), v in sorted(agg.items()):
        mets = [m for m, _ in v]
        print(f"  {opt:12s} xi={XI.get(xt,xt):<6} n={len(mets)} bleu={st.mean(mets):.3f}")


if __name__ == "__main__":
    main()
