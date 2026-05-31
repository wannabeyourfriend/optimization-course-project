"""CPU-only tests for DP-CorrMom: the noise-correlation structure and the privacy-critical
matrix-mechanism sensitivity. These guard the two things that, if wrong, silently break privacy."""

from __future__ import annotations

import math
import os
import sys

import torch
import torch.nn as nn
from opacus.optimizers.optimizer import DPOptimizer

_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
sys.path.insert(0, _SRC)
from dp_optim import DPCorrelatedAdaptive, OPTIMIZER_NAMES, corr_sensitivity, make_dp_optimizer  # noqa: E402
from dp_optim.dp_adaptive import _generate_noise  # noqa: E402

D, OUT = 6, 3
C = 1.0       # max_grad_norm
SIGMA = 1.3   # noise_multiplier (base)


def _lin():
    torch.manual_seed(7)
    return nn.Linear(D, OUT)


def _corrmom(lam, seed):
    lin = _lin()
    return DPCorrelatedAdaptive(
        torch.optim.SGD(lin.parameters(), lr=1e-2),
        noise_multiplier=SIGMA,
        max_grad_norm=C,
        expected_batch_size=16,
        generator=torch.Generator().manual_seed(seed),
        lr=1e-2,
        bias_correction=False,
        lambda_corr=lam,
    ), lin


def _noise_sequence(opt, lin, steps):
    """Drive add_noise() in isolation with summed_grad=0 so p.grad == injected noise."""
    seq = []
    for _ in range(steps):
        for p in lin.parameters():
            p.summed_grad = torch.zeros_like(p)   # fresh tensor -> unprocessed flag
        opt.add_noise()
        seq.append([p.grad.detach().clone() for p in lin.parameters()])
    return seq


def test_corr_sensitivity_matches_materialized_matrix():
    """kappa = maxcol(L^{-1}) for L = lower-bidiagonal[1, -lam], and is NOT sqrt(1+lam^2)."""
    for lam, T in [(0.5, 6), (0.7, 8), (0.9, 12), (0.95, 20)]:
        L = torch.eye(T, dtype=torch.float64)
        for i in range(1, T):
            L[i, i - 1] = -lam
        Linv = torch.linalg.inv(L)
        maxcol = Linv.norm(dim=0).max().item()        # max column L2 norm
        kappa = corr_sensitivity(lam, T)
        assert abs(maxcol - kappa) < 1e-9, f"lam={lam} T={T}: kappa={kappa} vs materialized {maxcol}"
        # guard against the wrong sqrt(1+lam^2) formula (the synthesis's error)
        wrong = math.sqrt(1 + lam * lam)
        assert abs(kappa - wrong) > 1e-3, f"kappa accidentally equals the WRONG sqrt(1+lam^2)={wrong}"
    assert corr_sensitivity(0.0, 10) == 1.0          # lam=0 -> DP-SGD (no inflation)


def test_lambda0_reproduces_iid_noise_bit_for_bit():
    """lambda_corr=0 add_noise == stock Opacus DPOptimizer.add_noise with the same generator."""
    lam0, lin0 = _corrmom(0.0, seed=2024)
    # stock DPOptimizer on a *separate* identical model, same generator seed
    lin_ref = _lin()
    ref = DPOptimizer(
        torch.optim.SGD(lin_ref.parameters(), lr=1e-2),
        noise_multiplier=SIGMA, max_grad_norm=C, expected_batch_size=16,
        generator=torch.Generator().manual_seed(2024),
    )
    a = _noise_sequence(lam0, lin0, steps=4)
    # drive the reference identically
    b = []
    for _ in range(4):
        for p in lin_ref.parameters():
            p.summed_grad = torch.zeros_like(p)
        ref.add_noise()
        b.append([p.grad.detach().clone() for p in lin_ref.parameters()])
    for sa, sb in zip(a, b):
        for ta, tb in zip(sa, sb):
            assert torch.equal(ta, tb), "lambda=0 must reproduce stock i.i.d. noise bit-for-bit"


def test_correlation_structure():
    """lambda>0 injects w_t = z_t - lambda*z_{t-1} (w_1 = z_1)."""
    lam = 0.6
    # z_t = the i.i.d. sequence (lam=0 run, seed S); w_t = the lam=0.6 run with the SAME seed S.
    z_run, z_lin = _corrmom(0.0, seed=99)
    w_run, w_lin = _corrmom(lam, seed=99)
    z = _noise_sequence(z_run, z_lin, steps=4)
    w = _noise_sequence(w_run, w_lin, steps=4)
    for ti in range(4):
        for pi in range(len(z[ti])):
            zt = z[ti][pi]
            zt_prev = z[ti - 1][pi] if ti >= 1 else torch.zeros_like(zt)
            expected = zt - lam * zt_prev
            assert torch.allclose(w[ti][pi], expected, atol=1e-6), \
                f"step {ti}: w_t != z_t - {lam}*z_(t-1)"


def test_corrmom_in_factory():
    lin = _lin()
    opt = make_dp_optimizer(
        "dp-corrmom", lin.parameters(), lr=1e-2, noise_multiplier=SIGMA,
        max_grad_norm=C, expected_batch_size=16, lambda_corr=0.8,
    )
    assert isinstance(opt, DPCorrelatedAdaptive) and opt.lambda_corr == 0.8
    assert "dp-corrmom" in OPTIMIZER_NAMES


def _main():
    tests = [
        test_corr_sensitivity_matches_materialized_matrix,
        test_lambda0_reproduces_iid_noise_bit_for_bit,
        test_correlation_structure,
        test_corrmom_in_factory,
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
