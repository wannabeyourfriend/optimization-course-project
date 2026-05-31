"""Optimizer diagnostics for wandb: PHI (DP-noise variance) and effective step sizes."""

from __future__ import annotations


def compute_phi(optimizer):
    """Per-coordinate DP-noise variance injected into the gradient (and Adam v_t).

    PHI = (noise_multiplier * max_grad_norm / B_eff)^2,  B_eff =
        expected_batch_size * accumulated_iterations * world_size

    The world_size factor accounts for DDP: DistributedDPOptimizer mean-reduces
    the summed gradient across ranks, scaling the noise by an extra 1/world_size.
    world_size is 1 for single-process optimizers, so this matches the non-DDP
    formula. Mirrors DPAdaptive.phi / DistributedDPAdaptive.phi exactly.

    Returns 0.0 when sigma==0 (non-private) or when the optimizer lacks DP fields.
    """
    nm = getattr(optimizer, "noise_multiplier", 0.0)
    if not nm:
        return 0.0
    mgn = getattr(optimizer, "max_grad_norm", 0.0)
    eb = getattr(optimizer, "expected_batch_size", None)
    try:
        acc = optimizer.accumulated_iterations or 1
    except (ValueError, AttributeError):
        # grad_sample cleared (pre-train / post-zero_grad): report nominal phi (acc=1).
        acc = 1
    if not eb:
        return 0.0
    ws = getattr(optimizer, "world_size", 1) or 1  # DistributedDP* sets this; else 1
    b_eff = eb * acc * ws
    return float((nm * mgn / b_eff) ** 2)


def _percentiles(values):
    """Return (p50, p90) of a sorted-on-the-fly 1-D list of floats."""
    if not values:
        return 0.0, 0.0
    vals = sorted(values)
    n = len(vals)

    def pct(p):
        idx = min(n - 1, max(0, int(round(p * (n - 1)))))
        return float(vals[idx])

    return pct(0.50), pct(0.90)


def effective_stepsize_percentiles(optimizer):
    """p50/p90 of the per-coordinate effective Adam step size |Δp| / lr.

    Reads m (exp_avg), v (exp_avg_sq), step t and PHI from DPAdaptive state and
    reconstructs the bias-corrected update magnitude:
        eff = |mhat| / (sqrt(vhat) + eps),   vhat optionally de-biased by PHI.
    Returns {"p50": float, "p90": float}; zeros if no Adam state is present
    (e.g. dp-sgd) so logging never crashes.
    """
    import torch

    phi = compute_phi(optimizer)
    eps = getattr(optimizer, "eps", 1e-8)
    xi = getattr(optimizer, "xi", 1e-8)
    bias_correction = getattr(optimizer, "bias_correction", False)
    betas = getattr(optimizer, "betas", (0.9, 0.999))
    b1, b2 = betas[0], betas[1]

    eff_all = []
    state = getattr(optimizer, "state", {})
    for st in state.values():
        if not isinstance(st, dict) or "exp_avg" not in st or "exp_avg_sq" not in st:
            continue
        m = st["exp_avg"]
        v = st["exp_avg_sq"]
        t = int(st.get("step", 0))
        if t <= 0:
            mhat, vhat = m, v
        else:
            mhat = m / (1 - b1 ** t)
            vhat = v / (1 - b2 ** t)
        if bias_correction:
            vhat = (vhat - phi).clamp_min(xi)
        eff = (mhat.abs() / (vhat.sqrt() + eps)).flatten()
        eff_all.append(eff)

    if not eff_all:
        return {"p50": 0.0, "p90": 0.0}

    flat = torch.cat(eff_all).detach().float().cpu().tolist()
    p50, p90 = _percentiles(flat)
    return {"p50": p50, "p90": p90}


def diagnostics_dict(optimizer):
    """Bundle diagnostics for wandb keys diag/phi, diag/eff_stepsize_p50/p90."""
    phi = compute_phi(optimizer)
    pct = effective_stepsize_percentiles(optimizer)
    return {
        "diag/phi": phi,
        "diag/eff_stepsize_p50": pct["p50"],
        "diag/eff_stepsize_p90": pct["p90"],
    }


def dynamics_dict(optimizer):
    """Rich per-step optimizer dynamics, for interpreting DP-optimizer behaviour.

    Returns (all best-effort; never raises):
      grad_norm     ‖p.grad‖ over trainable params (the CLIPPED+NOISED gradient).
      vhat_p50/p90  percentiles of the bias-corrected 2nd moment v̂ (Adam-family).
      phi_over_vhat Φ / median(v̂): the DP-noise SHARE of the 2nd moment. When this
                    is O(1), the noise dominates Adam's denominator -> DP-Adam
                    collapses toward DP-SGD, and AdamBC's correction matters most.
      clamp_frac    fraction of coords where (v̂ − Φ) hits the ξ floor (AdamBC only).
      update_norm   ‖Δθ/lr·lr‖ = ‖lr · m̂/(√v̂_c+eps)‖: the actual step magnitude.
    Adam-only fields are None for dp-sgd (no 2nd-moment state).
    """
    import torch

    out = {"grad_norm": None, "vhat_p50": None, "vhat_p90": None,
           "phi_over_vhat": None, "clamp_frac": None, "update_norm": None}
    try:
        phi = compute_phi(optimizer)
    except Exception:
        phi = 0.0

    # ‖clipped+noised gradient‖ over trainable params (p.grad is set by pre_step()).
    try:
        g_sq = 0.0
        for group in optimizer.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    g_sq += float(p.grad.detach().pow(2).sum())
        out["grad_norm"] = g_sq ** 0.5
    except Exception:
        pass

    eps = getattr(optimizer, "eps", 1e-8)
    xi = getattr(optimizer, "xi", 1e-8)
    bias_correction = getattr(optimizer, "bias_correction", False)
    b1, b2 = getattr(optimizer, "betas", (0.9, 0.999))
    state = getattr(optimizer, "state", {})
    vhat_chunks, upd_sq, clamped, total = [], 0.0, 0, 0
    try:
        for group in optimizer.param_groups:
            lr = group.get("lr", getattr(optimizer, "lr", 1e-3))
            for p in group["params"]:
                st = state.get(p)
                if not isinstance(st, dict) or "exp_avg_sq" not in st:
                    continue
                t = int(st.get("step", 0)) or 1
                mhat = st["exp_avg"] / (1 - b1 ** t)
                vhat = st["exp_avg_sq"] / (1 - b2 ** t)
                vhat_chunks.append(vhat.flatten())
                vc = vhat
                if bias_correction:
                    vc = (vhat - phi).clamp_min(xi)
                    clamped += int(((vhat - phi) < xi).sum())
                    total += vhat.numel()
                upd = lr * mhat / (vc.sqrt() + eps)
                upd_sq += float(upd.detach().pow(2).sum())
        if vhat_chunks:
            flat = torch.cat(vhat_chunks).detach().float()
            q = torch.quantile(flat, torch.tensor([0.5, 0.9], device=flat.device)) \
                if flat.numel() <= 4_000_000 else None
            if q is not None:
                out["vhat_p50"], out["vhat_p90"] = float(q[0]), float(q[1])
            else:  # subsample for very large param vectors
                idx = torch.randint(0, flat.numel(), (1_000_000,), device=flat.device)
                s = flat[idx]
                out["vhat_p50"] = float(s.median())
                out["vhat_p90"] = float(torch.quantile(s, 0.9))
            med = out["vhat_p50"] or 0.0
            out["phi_over_vhat"] = float(phi / med) if med > 0 else None
            out["clamp_frac"] = (clamped / total) if total else 0.0
            out["update_norm"] = upd_sq ** 0.5
        elif out["grad_norm"] is not None:  # dp-sgd: step ≈ lr·grad
            lr = optimizer.param_groups[0].get("lr", 1e-3)
            out["update_norm"] = lr * out["grad_norm"]
    except Exception:
        pass
    return out
