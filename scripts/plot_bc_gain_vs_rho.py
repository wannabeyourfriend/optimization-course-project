#!/usr/bin/env python3
"""Money plot: DP-AdamBC utility gain over DP-Adam vs the noise share rho (and batch size).

The headline of the project: bias correction is inert/harmful at rho~=1 (standard recipe) and
helps once rho is pushed below ~1 (large batch on a learning setting). This plots
(dp-adambc - dp-adam) BLEU at a fixed eval step against the measured rho = phi/v_hat, pooling
whatever rho<1 (bcwin*, ratebatch*) and rho~1 (qe2efl) runs exist in W&B.

Usage: WANDB_API_KEY=... python scripts/plot_bc_gain_vs_rho.py [eval_step]
Writes docs/figures/bc_gain_vs_rho.png and data/bc_gain_summary.csv.
"""
import csv
import os
import sys
from collections import defaultdict
import statistics as st

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import wandb

ENT, PROJ = "ziw178-uc-san-diego", "dp-optimizer-finetuning"
# round-id prefixes that carry a dp-adam-vs-dp-adambc batch/eps comparison
PREFIXES = ("bcwin-", "ratebatch-", "qe2efl-")


def main():
    step = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    api = wandb.Api(timeout=30)
    runs = [r for r in api.runs(f"{ENT}/{PROJ}")
            if any((r.config or {}).get("round_id", "").startswith(p) for p in PREFIXES)]

    # group by (batch, eps, opt-kind) -> list of (bleu_at_step, rho_at_step)
    cells = defaultdict(list)
    for r in runs:
        c = r.config or {}
        opt = (r.name or "").split("/")[2] if len((r.name or "").split("/")) > 2 else "?"
        kind = "adam" if opt == "dp-adam" else ("bc" if opt == "dp-adambc" else opt)
        B = c.get("batch_size"); eps = c.get("epsilon")
        try:
            h = r.history(keys=["eval/metric", "diag/phi_over_vhat"], pandas=True)
        except Exception:
            continue
        ev = h.dropna(subset=["eval/metric"]) if "eval/metric" in h else None
        if ev is None or not len(ev):
            continue
        row = ev[ev["_step"] == step]
        if not len(row):
            row = ev.iloc[[-1]]  # fall back to last eval
        bleu = float(row["eval/metric"].iloc[0])
        rho = r.summary.get("diag/phi_over_vhat")
        cells[(B, eps, kind)].append((bleu, rho))

    # build per-(batch,eps) gain = mean(bc) - mean(adam)
    rows = []
    keys = sorted({(B, e) for (B, e, k) in cells}, key=lambda x: (str(x[1]), x[0] or 0))
    for (B, e) in keys:
        adam = [b for b, _ in cells.get((B, e, "adam"), [])]
        bc = [b for b, _ in cells.get((B, e, "bc"), [])]
        rhos = [rho for _, rho in cells.get((B, e, "adam"), []) if isinstance(rho, (int, float))]
        if not adam or not bc:
            continue
        gain = st.mean(bc) - st.mean(adam)
        rho = st.mean(rhos) if rhos else None
        rows.append({"batch": B, "eps": e, "rho": rho,
                     "adam": st.mean(adam), "bc": st.mean(bc), "gain": gain,
                     "n_adam": len(adam), "n_bc": len(bc)})

    os.makedirs("docs/figures", exist_ok=True); os.makedirs("data", exist_ok=True)
    with open("data/bc_gain_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["batch", "eps", "rho", "adam", "bc", "gain", "n_adam", "n_bc"])
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()})

    if rows:
        xs = [r["rho"] for r in rows if r["rho"] is not None]
        ys = [r["gain"] for r in rows if r["rho"] is not None]
        labs = [f"B={r['batch']},$\\varepsilon$={r['eps']}" for r in rows if r["rho"] is not None]
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.axhline(0, color="gray", lw=1)
        ax.scatter(xs, ys, c=["C2" if y > 0 else "C3" for y in ys], s=60, zorder=5)
        for x, y, l in zip(xs, ys, labs):
            ax.annotate(l, (x, y), fontsize=7, xytext=(4, 4), textcoords="offset points")
        ax.set_xlabel(r"noise share $\rho=\Phi/\hat v$ (measured)")
        ax.set_ylabel(r"BLEU gain: DP-AdamBC $-$ DP-Adam")
        ax.set_title("Bias correction helps only once $\\rho$ is pushed below $\\approx 1$")
        fig.tight_layout(); fig.savefig("docs/figures/bc_gain_vs_rho.png", dpi=130)
        print("wrote docs/figures/bc_gain_vs_rho.png")
    for r in rows:
        print(f"  B={r['batch']} eps={r['eps']} rho={r['rho']} gain={r['gain']:+.2f} "
              f"(adam {r['adam']:.2f} n{r['n_adam']} / bc {r['bc']:.2f} n{r['n_bc']})")


if __name__ == "__main__":
    main()
