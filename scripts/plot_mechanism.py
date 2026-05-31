#!/usr/bin/env python3
"""Mechanism figure: DP-AdamBC restores the DP-noise-shrunk effective step at rho<1.

Two panels from the bcwin runs (Qwen-1.5B/E2E, B=4096, eps=8, rho=0.955):
 (L) effective step-size (median |m_hat|/(sqrt(v_hat)+eps)) over training: dp-adam is shrunk by
     the noise-inflated v_hat; DP-AdamBC subtracts Phi and restores a larger step, scaling with how
     small the floor xi is.
 (R) the gain is NON-MONOTONE in the floor: BLEU(xi) peaks at xi~=v_true (1e-11) and falls for the
     too-small floor (1e-12), whose update_norm explodes from floored noise coordinates -- a
     per-coordinate signature, not a scalar learning-rate effect.

Usage: WANDB_API_KEY=... python scripts/plot_mechanism.py
Writes docs/figures/mechanism_effstep.png.
"""
import os
import statistics as st

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import wandb

ENT, PROJ = "ziw178-uc-san-diego", "dp-optimizer-finetuning"
ARMS = [("bcwin-adam", "DP-Adam", "C0", None),
        ("bcwin-x10", "DP-AdamBC $\\xi$=1e-10", "C1", 1e-10),
        ("bcwin-x11", "DP-AdamBC $\\xi$=1e-11", "C2", 1e-11),
        ("bcwin-x12", "DP-AdamBC $\\xi$=1e-12", "C3", 1e-12)]


def main():
    api = wandb.Api(timeout=30)
    runs = list(api.runs(f"{ENT}/{PROJ}"))

    def get(rid, seed=0):
        return next((r for r in runs if (r.config or {}).get("round_id") == rid
                     and (r.config or {}).get("seed") == seed), None)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    # Panel L: effective step over training
    for rid, lab, col, xi in ARMS:
        r = get(rid)
        if not r:
            continue
        h = r.history(keys=["diag/eff_stepsize_p50"], pandas=True).dropna(subset=["diag/eff_stepsize_p50"])
        if len(h):
            ax[0].plot(h["_step"], h["diag/eff_stepsize_p50"], label=lab, color=col, lw=1.6)
    ax[0].set_xlabel("optimizer step"); ax[0].set_ylabel(r"median effective step $|\hat m|/(\sqrt{\hat v}+\epsilon)$")
    ax[0].set_title(r"BC restores the noise-shrunk step (B=4096, $\varepsilon$=8, $\rho$=0.955)")
    ax[0].legend(fontsize=8)

    # Panel R: BLEU and update_norm vs floor xi (per-coordinate signature)
    xis, bleus, unorms = [], [], []
    for rid, lab, col, xi in ARMS:
        if xi is None:
            continue
        r = get(rid)
        if not r:
            continue
        h = r.history(keys=["eval/metric", "diag/update_norm"], pandas=True)
        ev = h.dropna(subset=["eval/metric"])
        un = h.dropna(subset=["diag/update_norm"])
        un = un[(un["_step"] >= 20) & (un["_step"] <= 60)]
        if len(ev) and len(un):
            xis.append(xi); bleus.append(float(ev["eval/metric"].iloc[-1]))
            unorms.append(float(st.median(un["diag/update_norm"])))
    if xis:
        ax2 = ax[1]
        ax2.plot(xis, bleus, "o-", color="C2", label="proxy BLEU (left axis)")
        ax2.set_xscale("log"); ax2.set_xlabel(r"floor $\xi$")
        ax2.set_ylabel("proxy BLEU", color="C2"); ax2.tick_params(axis="y", labelcolor="C2")
        # dp-adam baseline
        adam = get("bcwin-adam")
        if adam:
            ev = adam.history(keys=["eval/metric"], pandas=True).dropna(subset=["eval/metric"])
            if len(ev):
                ax2.axhline(float(ev["eval/metric"].iloc[-1]), color="C0", ls="--", lw=1, label="DP-Adam")
        ax3 = ax2.twinx()
        ax3.plot(xis, unorms, "s--", color="C3", alpha=0.7)
        ax3.set_ylabel(r"update norm (noise-coord blowup)", color="C3"); ax3.tick_params(axis="y", labelcolor="C3")
        ax2.set_title(r"Floor optimum at $\xi\approx v_{\rm true}$: too-small $\xi\to$ noise blowup")
        ax2.legend(fontsize=8, loc="lower center")
    os.makedirs("docs/figures", exist_ok=True)
    fig.tight_layout(); fig.savefig("docs/figures/mechanism_effstep.png", dpi=130)
    print("wrote docs/figures/mechanism_effstep.png")
    print("BLEU vs xi:", list(zip(xis, [round(b, 2) for b in bleus])), "update_norms:", [round(u, 2) for u in unorms])


if __name__ == "__main__":
    main()
