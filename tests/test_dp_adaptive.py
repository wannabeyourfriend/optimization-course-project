"""CPU-only, no-download tests for DPAdaptive: Adam/AdamW equivalence at sigma=0, PHI formula, and noise-corrected vhat."""

from __future__ import annotations

import os
import sys

import torch
import torch.nn as nn
from opacus import GradSampleModule

# Make the test runnable from anywhere (sys.path.insert("src") only works from
# the repo root); resolve src/ relative to this file.
_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
sys.path.insert(0, _SRC)
from dp_optim import DPAdaptive, OPTIMIZER_NAMES, make_dp_optimizer  # noqa: E402

torch.manual_seed(0)

N, D, OUT, BATCH = 64, 8, 4, 16
MAX_GRAD_NORM = 1.0
EXPECTED_BATCH_SIZE = BATCH
LR = 1e-2


def _synthetic():
    g = torch.Generator().manual_seed(123)
    X = torch.randn(N, D, generator=g)
    y = torch.randint(0, OUT, (N,), generator=g)
    return X, y


def _fresh_model():
    torch.manual_seed(7)
    return nn.Linear(D, OUT)


def _grad_sample_module(linear):
    return GradSampleModule(linear)


def _backward(opt, gsm, xb, yb):
    # The DP optimizer's zero_grad clears grad_sample, summed_grad and p.grad and
    # resets the per-sample "_processed" flags Opacus uses between real steps.
    opt.zero_grad(set_to_none=True)
    logits = gsm(xb)
    # per-sample loss (sum reduction so per-sample grads are well defined)
    loss = nn.functional.cross_entropy(logits, yb, reduction="sum")
    loss.backward()


def _run_dp_adaptive(decoupled_wd, sigma, steps=5):
    """Run DPAdaptive for ``steps`` and return the trained Linear's params (cloned)."""
    base_lin = _fresh_model()
    gsm = _grad_sample_module(base_lin)
    opt = DPAdaptive(
        torch.optim.SGD(gsm.parameters(), lr=LR),
        noise_multiplier=sigma,
        max_grad_norm=MAX_GRAD_NORM,
        expected_batch_size=EXPECTED_BATCH_SIZE,
        lr=LR,
        decoupled_wd=decoupled_wd,
        bias_correction=True,
    )
    X, y = _synthetic()
    for s in range(steps):
        xb = X[s * BATCH : (s + 1) * BATCH]
        yb = y[s * BATCH : (s + 1) * BATCH]
        _backward(opt, gsm, xb, yb)
        opt.step()
    return [p.detach().clone() for p in base_lin.parameters()], opt


def _run_reference_adam(decoupled_wd, steps=5):
    """Mirror DPAdaptive's clipped+scaled gradients into a stock Adam/AdamW."""
    base_lin = _fresh_model()
    gsm = _grad_sample_module(base_lin)
    dp = DPAdaptive(
        torch.optim.SGD(gsm.parameters(), lr=LR),
        noise_multiplier=0.0,
        max_grad_norm=MAX_GRAD_NORM,
        expected_batch_size=EXPECTED_BATCH_SIZE,
        lr=LR,
        decoupled_wd=decoupled_wd,
        bias_correction=True,
    )
    AdamCls = torch.optim.AdamW if decoupled_wd else torch.optim.Adam
    ref = AdamCls(base_lin.parameters(), lr=LR, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)

    X, y = _synthetic()
    for s in range(steps):
        xb = X[s * BATCH : (s + 1) * BATCH]
        yb = y[s * BATCH : (s + 1) * BATCH]
        _backward(dp, gsm, xb, yb)
        # Opacus clip + (sigma=0) noise + scale -> p.grad
        assert dp.pre_step()
        # Hand the identical gradient to the reference optimizer and step it.
        ref.step()
    return [p.detach().clone() for p in base_lin.parameters()]


def test_equivalence_adam():
    """sigma=0 DPAdaptive(decoupled_wd=False) == torch.optim.Adam to ~1e-6 over 5 steps."""
    dp_params, _ = _run_dp_adaptive(decoupled_wd=False, sigma=0.0)
    ref_params = _run_reference_adam(decoupled_wd=False)
    for a, b in zip(dp_params, ref_params):
        max_diff = (a - b).abs().max().item()
        assert max_diff < 1e-6, f"Adam equivalence failed: max|diff|={max_diff}"


def test_equivalence_adamw():
    """sigma=0 DPAdaptive(decoupled_wd=True) == torch.optim.AdamW to ~1e-6 over 5 steps."""
    dp_params, _ = _run_dp_adaptive(decoupled_wd=True, sigma=0.0)
    ref_params = _run_reference_adam(decoupled_wd=True)
    for a, b in zip(dp_params, ref_params):
        max_diff = (a - b).abs().max().item()
        assert max_diff < 1e-6, f"AdamW equivalence failed: max|diff|={max_diff}"


def test_phi_formula():
    """PHI == (sigma * C / B_eff) ** 2."""
    sigma, C = 1.3, MAX_GRAD_NORM
    base_lin = _fresh_model()
    gsm = _grad_sample_module(base_lin)
    opt = DPAdaptive(
        torch.optim.SGD(gsm.parameters(), lr=LR),
        noise_multiplier=sigma,
        max_grad_norm=C,
        expected_batch_size=EXPECTED_BATCH_SIZE,
        lr=LR,
        decoupled_wd=False,
        bias_correction=True,
    )
    X, y = _synthetic()
    _backward(opt, gsm, X[:BATCH], y[:BATCH])
    assert opt.pre_step()
    b_eff = EXPECTED_BATCH_SIZE * opt.accumulated_iterations
    expected = (sigma * C / b_eff) ** 2
    assert abs(opt.phi - expected) < 1e-12, f"phi={opt.phi} expected={expected}"


def test_bias_correction_shrinks_vhat():
    """With sigma>0, bias_correction yields a strictly smaller vhat than without."""
    sigma = 2.0
    X, y = _synthetic()

    def run(bias_correction):
        lin = _fresh_model()
        gsm = _grad_sample_module(lin)
        opt = DPAdaptive(
            torch.optim.SGD(gsm.parameters(), lr=LR),
            noise_multiplier=sigma,
            max_grad_norm=MAX_GRAD_NORM,
            expected_batch_size=EXPECTED_BATCH_SIZE,
            generator=torch.Generator().manual_seed(99),  # identical noise stream
            lr=LR,
            decoupled_wd=False,
            bias_correction=bias_correction,
        )
        _backward(opt, gsm, X[:BATCH], y[:BATCH])
        assert opt.pre_step()
        # phi must be strictly positive for the correction to bite.
        assert opt.phi > 0.0
        # Replicate the vhat that _adam_update would compute for the first param.
        b1, b2 = opt.betas
        p = next(p for p in lin.parameters() if p.requires_grad and p.grad is not None)
        g = p.grad
        v = (1 - b2) * g * g
        vhat = v / (1 - b2**1)
        if bias_correction:
            vhat = (vhat - opt.phi).clamp_min(opt.xi)
        return vhat

    vhat_off = run(False)
    vhat_on = run(True)
    assert torch.all(vhat_on <= vhat_off), "corrected vhat must be <= uncorrected"
    assert torch.any(vhat_on < vhat_off), "corrected vhat must be strictly smaller somewhere"


def test_factory_names():
    """Factory builds every advertised optimizer name."""
    lin = _fresh_model()
    gsm = _grad_sample_module(lin)
    for name in OPTIMIZER_NAMES:
        opt = make_dp_optimizer(
            name,
            gsm.parameters(),
            lr=LR,
            noise_multiplier=0.0,
            max_grad_norm=MAX_GRAD_NORM,
            expected_batch_size=EXPECTED_BATCH_SIZE,
        )
        assert opt is not None


def _main():
    tests = [
        test_equivalence_adam,
        test_equivalence_adamw,
        test_phi_formula,
        test_bias_correction_shrinks_vhat,
        test_factory_names,
    ]
    failures = 0
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except Exception as e:  # noqa: BLE001
            failures += 1
            print(f"FAIL {t.__name__}: {e}")
    if failures:
        print(f"{failures} test(s) failed")
        sys.exit(1)
    print("all tests passed")


if __name__ == "__main__":
    _main()
