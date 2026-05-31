#!/usr/bin/env python3
"""Build a single self-contained docs/dashboard.html from the wandb project (never crashes)."""

import argparse
import datetime as _dt
import html
import json
import os
import sys

# Exact wandb config/metric keys written by src/train.py (see SPEC WANDB SCHEMA).
CONFIG_KEYS = [
    "round_id", "model", "task", "optimizer", "epsilon", "delta", "seed",
    "max_grad_norm", "steps", "batch_size", "lr", "sample_rate", "sigma",
    "phi", "cluster", "method", "lora_r",
]
EVAL_METRIC_KEY = "eval/metric"
EPS_SPENT_KEY = "privacy/epsilon_spent"

PALETTE = ["#2563eb", "#dc2626", "#16a34a", "#9333ea", "#ea580c", "#0891b2", "#ca8a04"]


def fetch_runs(project, entity):
    """Return (runs, error_msg). runs is a list of flat dicts; error_msg is None on success."""
    try:
        import wandb  # noqa: F401
        from wandb import Api
    except Exception as e:  # wandb not installed
        return None, f"wandb is not importable ({e})."

    try:
        api = Api(timeout=30)
        path = f"{entity}/{project}" if entity else project
        wruns = api.runs(path)
        runs = []
        for r in wruns:
            cfg = dict(r.config or {})
            summ = dict(r.summary or {})
            row = {k: cfg.get(k) for k in CONFIG_KEYS}
            row["state"] = getattr(r, "state", None)
            row["url"] = getattr(r, "url", None)
            row["name"] = getattr(r, "name", None)
            row["eval_metric"] = _num(summ.get(EVAL_METRIC_KEY))
            row["metric_name"] = summ.get("eval/metric_name")
            row["epsilon_spent"] = _num(summ.get(EPS_SPENT_KEY))
            runs.append(row)
        return runs, None
    except Exception as e:  # unauthed / network / project missing
        return None, f"could not query wandb ({e})."


def _num(v):
    try:
        if v is None:
            return None
        return float(v)
    except (TypeError, ValueError):
        return None


def _svg_charts(runs):
    """One inline-SVG line chart per task: utility vs epsilon, one line per optimizer."""
    if not runs:
        return ""
    # group by task -> series(optimizer[+method]) -> {epsilon: [metrics]}
    tasks = {}
    for r in runs:
        task = r.get("task")
        opt = r.get("optimizer")
        eps = _num(r.get("epsilon"))
        m = r.get("eval_metric")
        if task is None or opt is None or eps is None or m is None:
            continue
        method = r.get("method")
        series = f"{opt} ({method})" if method else opt
        tasks.setdefault(task, {}).setdefault(series, {}).setdefault(eps, []).append(m)

    blocks = []
    for task in sorted(tasks):
        opts = tasks[task]
        # finite-epsilon points only for the x-axis; "inf" runs map to the max+pad tick.
        all_eps = sorted({e for o in opts.values() for e in o if e != float("inf")})
        if not all_eps:
            continue
        xmin, xmax = min(all_eps), max(all_eps)
        if xmax == xmin:
            xmax = xmin + 1.0
        all_m = [sum(v) / len(v) for o in opts.values() for v in o.values()]
        ymin, ymax = min(all_m), max(all_m)
        if ymax == ymin:
            ymax = ymin + 1.0
        W, H, PAD = 520, 280, 48

        def sx(e):
            return PAD + (e - xmin) / (xmax - xmin) * (W - 2 * PAD)

        def sy(m):
            return H - PAD - (m - ymin) / (ymax - ymin) * (H - 2 * PAD)

        parts = [f'<svg viewBox="0 0 {W} {H}" class="chart" role="img">']
        # axes
        parts.append(f'<line x1="{PAD}" y1="{H-PAD}" x2="{W-PAD}" y2="{H-PAD}" class="axis"/>')
        parts.append(f'<line x1="{PAD}" y1="{PAD}" x2="{PAD}" y2="{H-PAD}" class="axis"/>')
        # x ticks
        for e in all_eps:
            x = sx(e)
            parts.append(f'<line x1="{x:.1f}" y1="{H-PAD}" x2="{x:.1f}" y2="{H-PAD+5}" class="axis"/>')
            parts.append(f'<text x="{x:.1f}" y="{H-PAD+18}" class="tick" text-anchor="middle">{_fmt(e)}</text>')
        # y ticks (min/max)
        for m in (ymin, ymax):
            y = sy(m)
            parts.append(f'<text x="{PAD-6}" y="{y+3:.1f}" class="tick" text-anchor="end">{m:.3f}</text>')
        parts.append(f'<text x="{W/2:.0f}" y="{H-8}" class="axislbl" text-anchor="middle">epsilon</text>')
        parts.append(f'<text x="14" y="{H/2:.0f}" class="axislbl" text-anchor="middle" transform="rotate(-90 14 {H/2:.0f})">utility</text>')
        # lines
        legend = []
        for i, opt in enumerate(sorted(opts)):
            color = PALETTE[i % len(PALETTE)]
            pts = []
            for e in sorted(k for k in opts[opt] if k != float("inf")):
                vals = opts[opt][e]
                pts.append((sx(e), sy(sum(vals) / len(vals))))
            if not pts:
                continue
            d = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
            parts.append(f'<polyline points="{d}" fill="none" stroke="{color}" stroke-width="2"/>')
            for x, y in pts:
                parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3" fill="{color}"/>')
            legend.append(f'<span class="lg"><i style="background:{color}"></i>{html.escape(opt)}</span>')
        parts.append("</svg>")
        blocks.append(
            f'<div class="card"><h3>{html.escape(str(task))}: utility vs epsilon</h3>'
            + "".join(parts)
            + f'<div class="legend">{"".join(legend)}</div></div>'
        )
    return "".join(blocks)


def _fmt(e):
    return str(int(e)) if float(e).is_integer() else f"{e:g}"


def _runs_table(runs):
    cols = ["round_id", "model", "task", "method", "optimizer", "epsilon", "seed", "state",
            "metric", "epsilon_spent", "wandb"]
    head = "".join(f'<th data-c="{i}">{c}</th>' for i, c in enumerate(cols))
    body = []
    for r in runs:
        link = (f'<a href="{html.escape(str(r["url"]))}" target="_blank" rel="noopener">open</a>'
                if r.get("url") else "")
        metric = f'{r["eval_metric"]:.4f}' if r.get("eval_metric") is not None else ""
        esp = f'{r["epsilon_spent"]:.3f}' if r.get("epsilon_spent") is not None else ""
        cells = [
            r.get("round_id"), r.get("model"), r.get("task"), r.get("method"),
            r.get("optimizer"), r.get("epsilon"), r.get("seed"), r.get("state"),
            metric, esp, None,
        ]
        tds = []
        for i, c in enumerate(cells):
            if i == len(cells) - 1:
                tds.append(f"<td>{link}</td>")
            else:
                tds.append(f"<td>{html.escape('' if c is None else str(c))}</td>")
        body.append("<tr>" + "".join(tds) + "</tr>")
    return head, "".join(body)


def render_html(runs, project, entity, error):
    gen = _dt.datetime.now().isoformat(timespec="seconds")
    title = f"DP-Optimizer Finetuning — {html.escape(project)}"
    blob = json.dumps(runs or [], default=str)

    if runs:
        head, body = _runs_table(runs)
        table = f'<table id="runs"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>'
        charts = _svg_charts(runs) or '<p class="muted">No finite-epsilon utility points yet.</p>'
        note = f'<p class="muted">{len(runs)} runs from <code>{html.escape((entity + "/" if entity else "") + project)}</code>.</p>'
    else:
        table = ""
        charts = ""
        note = (f'<div class="placeholder"><strong>No wandb data.</strong> '
                f'{html.escape(error or "wandb unavailable.")}<br>'
                'Run <code>wandb login</code> (or set <code>WANDB_API_KEY</code>) and rebuild '
                'with <code>python scripts/build_dashboard.py</code>.</div>')

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title}</title>
<style>
:root{{color-scheme:light dark}}
body{{font:14px/1.5 -apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;margin:0;padding:24px;max-width:1100px;margin:auto;color:#111;background:#fafafa}}
h1{{margin:0 0 4px}} h3{{margin:0 0 8px;font-size:15px}}
.muted{{color:#666;font-size:13px}}
.placeholder{{padding:16px;border:1px solid #e0c000;background:#fffbe6;border-radius:8px}}
table{{border-collapse:collapse;width:100%;background:#fff;border-radius:8px;overflow:hidden;box-shadow:0 1px 2px rgba(0,0,0,.06)}}
th,td{{padding:7px 10px;text-align:left;border-bottom:1px solid #eee;font-size:13px;white-space:nowrap}}
th{{background:#f3f4f6;cursor:pointer;user-select:none}}
th:hover{{background:#e5e7eb}}
tbody tr:hover{{background:#f9fafb}}
code{{background:#eef;padding:1px 4px;border-radius:3px}}
.charts{{display:flex;flex-wrap:wrap;gap:16px;margin:18px 0}}
.card{{background:#fff;padding:14px;border-radius:8px;box-shadow:0 1px 2px rgba(0,0,0,.06)}}
.chart{{width:520px;max-width:100%;height:auto}}
.axis{{stroke:#999;stroke-width:1}} .tick{{font-size:10px;fill:#666}} .axislbl{{font-size:11px;fill:#444}}
.legend{{margin-top:6px}} .lg{{margin-right:12px;font-size:12px;color:#444}}
.lg i{{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:4px;vertical-align:middle}}
a{{color:#2563eb}}
</style></head>
<body>
<h1>{title}</h1>
<p class="muted">Generated {gen}</p>
{note}
<div class="charts">{charts}</div>
{table}
<script id="run-data" type="application/json">{blob}</script>
<script>
// Vanilla click-to-sort on the runs table (numeric-aware), no external deps.
(function(){{
  var t=document.getElementById('runs'); if(!t) return;
  var dir={{}};
  t.tHead.querySelectorAll('th').forEach(function(th,ci){{
    th.addEventListener('click',function(){{
      var rows=Array.prototype.slice.call(t.tBodies[0].rows);
      var d=dir[ci]=!dir[ci];
      rows.sort(function(a,b){{
        var x=a.cells[ci].innerText.trim(), y=b.cells[ci].innerText.trim();
        var nx=parseFloat(x), ny=parseFloat(y);
        var both=!isNaN(nx)&&!isNaN(ny);
        var c=both?(nx-ny):x.localeCompare(y);
        return d?c:-c;
      }});
      rows.forEach(function(r){{t.tBodies[0].appendChild(r);}});
    }});
  }});
}})();
</script>
</body></html>
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--project", default=os.environ.get("WANDB_PROJECT", "dp-optimizer-finetuning"))
    ap.add_argument("--entity", default=os.environ.get("WANDB_ENTITY"))
    ap.add_argument("--out", default="docs/dashboard.html")
    args = ap.parse_args()

    runs, error = fetch_runs(args.project, args.entity)
    out_html = render_html(runs, args.project, args.entity, error)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(out_html)

    if runs is None:
        print(f"[build_dashboard] wandb unavailable: {error} -> wrote placeholder {args.out}", file=sys.stderr)
    else:
        print(f"[build_dashboard] wrote {args.out} ({len(runs)} runs)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
