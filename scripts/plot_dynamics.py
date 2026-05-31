#!/usr/bin/env python3
"""Publication-quality DP-optimizer *dynamics* figures from the wandb history.

For a fixed epsilon, overlays the per-step trajectories of the 5 optimizers so the
mechanism differences are visible: how DP noise inflates Adam's 2nd moment, how
DP-AdamBC restores the effective step, and how that maps to loss/utility.

Usage:
  python scripts/plot_dynamics.py --entity ziw178-uc-san-diego \
      --project dp-optimizer-finetuning --round mnli-large-lora-8gpu \
      --epsilon 3 --out docs/figures/dynamics_mnli_eps3.png
"""
import argparse
import os
import sys

# Fixed optimizer order + colorblind-friendly palette; BC pairs share a hue family
# so the bias-correction effect reads at a glance.
OPT_ORDER = ["dp-sgd", "dp-adam", "dp-adambc", "dp-adamw", "dp-adamw-bc"]
OPT_COLOR = {
    "dp-sgd":      "#6b7280",  # gray  — non-adaptive baseline
    "dp-adam":     "#2563eb",  # blue
    "dp-adambc":   "#16a34a",  # green — Adam + bias correction
    "dp-adamw":    "#ea580c",  # orange
    "dp-adamw-bc": "#dc2626",  # red   — AdamW + bias correction
}
# (wandb key, panel title, y-label, log-y)
PANELS = [
    ("train/loss",            "Training loss",                 "loss",            False),
    ("eval/metric",           "Validation utility",            "accuracy / BLEU", False),
    ("diag/eff_stepsize_p50", "Effective step size (median)",  "|m̂|/(√v̂+ε)",     True),
    ("diag/phi_over_vhat",    "DP-noise share of 2nd moment",  "Φ / median(v̂)",   True),
    ("diag/grad_norm",        "Clipped+noised gradient norm",  "‖g‖",             True),
    ("diag/update_norm",      "Update magnitude",              "‖Δθ‖",            True),
]


def fetch(entity, project, round_id, epsilon):
    import wandb
    api = wandb.Api(timeout=60)
    path = f"{entity}/{project}" if entity else project
    want_eps = str(epsilon)
    series = {}  # optimizer -> dict(key -> (steps, values))
    keys = [p[0] for p in PANELS] + ["_step"]
    for r in api.runs(path):
        cfg = dict(r.config or {})
        if round_id and not str(cfg.get("round_id", "")).startswith(round_id):
            continue
        if str(cfg.get("epsilon")) != want_eps:
            continue
        opt = cfg.get("optimizer")
        if opt not in OPT_COLOR:
            continue
        hist = r.history(keys=keys, pandas=True)
        if hist is None or len(hist) == 0:
            continue
        per_key = {}
        xcol = "_step" if "_step" in hist.columns else hist.columns[0]
        for k, *_ in PANELS:
            if k in hist.columns:
                sub = hist[[xcol, k]].dropna()
                if len(sub):
                    per_key[k] = (sub[xcol].to_numpy(), sub[k].to_numpy())
        series[opt] = per_key
        print(f"  loaded {opt}: {len(hist)} rows ({r.name})", file=sys.stderr)
    return series


def render(series, epsilon, title, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.25, "figure.dpi": 130,
        "legend.frameon": False, "font.family": "DejaVu Sans",
    })
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))
    axes = axes.ravel()
    for ax, (key, ptitle, ylab, logy) in zip(axes, PANELS):
        any_data = False
        for opt in OPT_ORDER:
            pk = series.get(opt, {})
            if key not in pk:
                continue
            xs, ys = pk[key]
            marker = "o" if key == "eval/metric" else None
            ax.plot(xs, ys, color=OPT_COLOR[opt], lw=1.8, marker=marker,
                    ms=4, label=opt, alpha=0.9)
            any_data = True
        ax.set_title(ptitle)
        ax.set_xlabel("optimizer step")
        ax.set_ylabel(ylab)
        if logy:
            ax.set_yscale("log")
        if not any_data:
            ax.text(0.5, 0.5, "no data", ha="center", va="center",
                    transform=ax.transAxes, color="#999")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5,
               bbox_to_anchor=(0.5, 0.985))
    fig.suptitle(title, y=1.0, fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    print(f"[plot_dynamics] wrote {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--entity", default=os.environ.get("WANDB_ENTITY", "ziw178-uc-san-diego"))
    ap.add_argument("--project", default=os.environ.get("WANDB_PROJECT", "dp-optimizer-finetuning"))
    ap.add_argument("--round", default="", help="round_id prefix filter")
    ap.add_argument("--epsilon", default="3")
    ap.add_argument("--title", default=None)
    ap.add_argument("--out", default="docs/figures/dynamics.png")
    args = ap.parse_args()

    series = fetch(args.entity, args.project, args.round, args.epsilon)
    if not series:
        print("[plot_dynamics] no matching runs found", file=sys.stderr)
        return 1
    title = args.title or f"DP-optimizer dynamics — {args.round or args.project} (ε={args.epsilon})"
    render(series, args.epsilon, title, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
