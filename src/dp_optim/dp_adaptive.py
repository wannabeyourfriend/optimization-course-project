"""DP adaptive (Adam-family) optimizer that fuses Opacus clip+noise with a custom bias-corrected Adam update."""

from __future__ import annotations

from typing import Optional

import torch
from opacus.optimizers.optimizer import (
    DPOptimizer,
    _check_processed_flag,
    _generate_noise,
    _mark_as_processed,
)
from opacus.optimizers.ddpoptimizer import DistributedDPOptimizer
from torch.optim import Optimizer


def corr_sensitivity(lam: float, steps: int) -> float:
    """Matrix-mechanism column sensitivity for injecting ``n_t = z_t - lam * z_{t-1}``.

    Injecting ``n = L z`` (L lower-bidiagonal with 1 on the diagonal and ``-lam`` on the
    sub-diagonal) realises the prefix-sum workload ``A`` with strategy ``C = L^{-1}`` (the
    lower-triangular Toeplitz matrix ``[1, lam, lam^2, ...]``). The privacy cost is the max
    column L2 norm of ``C``::

        kappa = sqrt(sum_{k=0}^{steps-1} lam^{2k}) = sqrt((1 - lam^{2*steps}) / (1 - lam^2))

    So to match a target noise multiplier ``sigma`` (same RDP as i.i.d. DP-SGD), the per-step
    base noise std must be inflated to `` sigma * max_grad_norm * kappa``. NOTE: this is the
    CORRECT sensitivity; the naive ``sqrt(1 + lam^2)`` (column norm of L, not L^{-1}) UNDER-noises
    and breaks privacy (at lam=0.9: 1.35 vs the correct 2.29). ``lam=0`` -> kappa=1 (DP-SGD).
    """
    if lam <= 0.0:
        return 1.0
    if lam >= 1.0:
        raise ValueError("lambda_corr must be in [0, 1)")
    return float(((1.0 - lam ** (2 * steps)) / (1.0 - lam * lam)) ** 0.5)


class DPAdaptive(DPOptimizer):
    """Opacus ``DPOptimizer`` with an in-house Adam/AdamW update.

    After ``pre_step()`` has populated ``p.grad`` with the clipped, noised and
    batch-averaged gradient, this optimizer applies its OWN Adam-family update
    instead of delegating to a wrapped Adam. This lets us subtract the known
    DP-noise variance bias (DP-AdamBC) from the second-moment estimate.

    The final per-coordinate noise std baked into ``p.grad`` by Opacus is
    ``noise_multiplier * max_grad_norm / B_eff`` where
    ``B_eff = expected_batch_size * accumulated_iterations``. The variance that
    inflates the Adam second moment is therefore::

        PHI = (noise_multiplier * max_grad_norm / B_eff) ** 2

    When ``noise_multiplier == 0`` we have ``PHI == 0``; subtracting it is a no-op
    and the update reduces to ``torch.optim.Adam`` (``decoupled_wd=False``) /
    ``torch.optim.AdamW`` (``decoupled_wd=True``) applied to the same gradients.

    Privacy note: clipping and noising happen entirely inside ``pre_step()``
    (``clip_and_accumulate`` -> ``add_noise`` -> ``scale_grad`` -> accountant
    ``step_hook``). The bias correction below touches only the optimizer's
    private second-moment estimate AFTER ``pre_step()`` has returned, so it can
    never change what the privacy accountant sees.

    TODO(ghost-clipping): large models need a fast-gradient/ghost-clipping variant
    built on ``DPOptimizerFastGradientClipping``; not implemented here on purpose.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        *,
        noise_multiplier: float,
        max_grad_norm: float,
        expected_batch_size: Optional[int],
        loss_reduction: str = "mean",
        generator=None,
        secure_mode: bool = False,
        lr: float = 1e-3,
        betas=(0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        decoupled_wd: bool = False,
        bias_correction: bool = True,
        floor_only: bool = False,
        xi: float = 1e-8,
        probe: bool = False,
        lowpass_beta: float = 0.0,
        **kwargs,
    ):
        super().__init__(
            optimizer=optimizer,
            noise_multiplier=noise_multiplier,
            max_grad_norm=max_grad_norm,
            expected_batch_size=expected_batch_size,
            loss_reduction=loss_reduction,
            generator=generator,
            secure_mode=secure_mode,
            **kwargs,
        )
        self.lr = lr
        self.betas = tuple(betas)
        self.eps = eps
        self.weight_decay = weight_decay
        self.decoupled_wd = decoupled_wd
        self.bias_correction = bias_correction
        self.floor_only = floor_only  # xi-floor WITHOUT phi-subtraction (control)
        self.xi = xi
        self.probe = probe            # direction-fidelity probe (Muon viability gate)
        self.probe_stats = {}
        # STEP-0 lever validator: an extra causal low-pass (EMA) on the bias-corrected
        # first moment m_hat. Pure post-processing of the already-private gradient, so it
        # costs ZERO extra privacy budget and touches neither add_noise nor the accountant.
        # lowpass_beta=0 disables it (exact stock behaviour). Tests whether denoising the
        # first-moment path moves utility at rho~=1, before building correlated-noise (DP-CorrMom).
        self.lowpass_beta = lowpass_beta

    @torch.no_grad()
    def _run_probe(self) -> None:
        """Direction-fidelity probe (diagnostic only; does NOT touch the update or accountant).

        For each 2D (matrix) parameter, compares the noisy momentum ``M`` against the
        ORTHOGONALIZED momentum ``msign(M)`` (singular values flattened to 1, the Muon /
        modular-manifold update direction) in terms of cosine similarity to the *clean* batch
        gradient ``g_clean`` (the un-clipped, un-noised per-sample mean from ``grad_sample``,
        available before ``pre_step`` clips). If ``cos_after > cos_before``, orthogonalization
        recovers a low-rank signal direction (DP-Muon worth building); if ``cos_after <
        cos_before``, msign amplifies the DP-noise tail (abandon Muon).
        """
        import torch.nn.functional as F

        b_eff = self.expected_batch_size * self.accumulated_iterations
        cb = ca = 0.0
        n = 0
        for group in self.param_groups:
            for p in group["params"]:
                gs = getattr(p, "grad_sample", None)
                st = self.state.get(p)
                if gs is None or not isinstance(st, dict) or "exp_avg" not in st:
                    continue
                if int(st.get("step", 0)) < 3 or p.dim() != 2:
                    continue
                g_clean = gs.sum(dim=0).div(b_eff)          # un-clipped, un-noised batch grad
                M = st["exp_avg"]
                try:                                         # msign = polar factor U Vh
                    U, _, Vh = torch.linalg.svd(M.float(), full_matrices=False)
                    A = U @ Vh
                except Exception:
                    continue
                cb += float(F.cosine_similarity(M.flatten().float(), g_clean.flatten().float(), dim=0))
                ca += float(F.cosine_similarity(A.flatten(), g_clean.flatten().float(), dim=0))
                n += 1
        if n:
            self.probe_stats = {"cos_before": cb / n, "cos_after": ca / n,
                                "cos_gain": (ca - cb) / n, "probe_n": n}

    @property
    def phi(self) -> float:
        """Per-coordinate DP-noise variance present in ``p.grad`` after ``pre_step``."""
        b_eff = self.expected_batch_size * self.accumulated_iterations
        return (self.noise_multiplier * self.max_grad_norm / b_eff) ** 2

    @torch.no_grad()
    def _adam_update(self) -> None:
        """Apply the Adam/AdamW update in place using the gradients in ``p.grad``."""
        phi = self.phi
        b1, b2 = self.betas

        for group in self.param_groups:
            # lr can be varied by a scheduler via param_groups; fall back to ctor lr.
            lr = group.get("lr", self.lr)
            for p in group["params"]:
                if not p.requires_grad or p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]

                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)

                if self.weight_decay > 0:
                    if self.decoupled_wd:
                        # AdamW: decoupled decay applied directly to the params,
                        # matching torch.optim.AdamW (param *= 1 - lr*wd).
                        p.mul_(1 - lr * self.weight_decay)
                    else:
                        # Adam: L2 penalty folded into the gradient (out-of-place
                        # so we do not mutate the shared p.grad tensor).
                        g = g.add(p, alpha=self.weight_decay)

                state["step"] += 1
                t = state["step"]
                m, v = state["exp_avg"], state["exp_avg_sq"]

                m.mul_(b1).add_(g, alpha=1 - b1)
                v.mul_(b2).addcmul_(g, g, value=1 - b2)

                mhat = m / (1 - b1**t)
                vhat = v / (1 - b2**t)

                # STEP-0 low-pass: an extra causal EMA on the (already-private) first
                # moment. Privacy-free post-processing (lowpass_beta=0 -> mhat unchanged).
                if self.lowpass_beta > 0:
                    lp = self.lowpass_beta
                    mlp = state.get("m_lp")
                    if mlp is None:
                        mlp = state["m_lp"] = torch.zeros_like(p)
                    mlp.mul_(lp).add_(mhat, alpha=1 - lp)
                    mhat = mlp / (1 - lp**t)

                # DP-AdamBC: subtract the known DP-noise variance from the second
                # moment, then floor at xi for numerical stability. When sigma==0,
                # phi==0 so this only floors (xi << any real vhat in practice),
                # leaving the stock-Adam invariant intact.
                if self.bias_correction:
                    vhat = (vhat - phi).clamp_min(self.xi)
                elif self.floor_only:
                    # Control: apply the SAME xi-floor but do NOT subtract phi, so
                    # the BC win can be attributed to phi-subtraction vs. the floor.
                    vhat = vhat.clamp_min(self.xi)

                # torch.optim.Adam denominator is sqrt(vhat) + eps; match exactly.
                p.addcdiv_(mhat, vhat.sqrt().add_(self.eps), value=-lr)

    def step(self, closure=None) -> Optional[float]:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        # pre_step() clips, noises, scales and runs the accountant hook ->
        # populates p.grad. It returns False on a skipped (accumulation) step;
        # only run our update on a real step.
        if self.probe:
            self._run_probe()  # BEFORE pre_step: grad_sample still holds the clean per-sample grads
        if self.pre_step():
            self._adam_update()
        return loss


def _inject_correlated_noise(opt) -> None:
    """Inject ``w_t = z_t - opt.lambda_corr * z_{t-1}`` into ``p.grad`` (shared by the Adam- and
    SGD-based correlated optimizers). ``lambda_corr == 0`` reproduces stock i.i.d. noise bit-for-bit.
    The caller must pre-inflate ``noise_multiplier`` by ``corr_sensitivity(lambda, steps)`` and use
    an unamplified accountant (correlated noise voids Poisson amplification)."""
    for p in opt.params:
        _check_processed_flag(p.summed_grad)
        z = _generate_noise(
            std=opt.noise_multiplier * opt.max_grad_norm,
            reference=p.summed_grad,
            generator=opt.generator,
            secure_mode=opt.secure_mode,
        )
        if opt.lambda_corr > 0.0:
            prev = opt._prev_noise.get(p)
            w = z if (prev is None or prev.shape != z.shape) else z - opt.lambda_corr * prev
            opt._prev_noise[p] = z
        else:
            w = z
        p.grad = (p.summed_grad + w).view_as(p)
        _mark_as_processed(p.summed_grad)


class DPCorrelatedOptimizer(DPOptimizer):
    """DP-CorrSGD: anti-correlated noise on a PLAIN SGD update (no Adam second-moment denominator).

    This is the textbook DP-FTRL / matrix-factorization setting: with plain SGD (momentum 0) the
    parameter trajectory IS the gradient prefix-sum ``theta_T = theta_0 - eta * sum_t (g_t + w_t)``,
    so the bidiagonal anti-correlation cancels exactly the noise that limits learning, with NO
    momentum-window mismatch and NO ``sqrt(v_hat)`` denominator to be poisoned by the kappa-inflated
    per-step noise (the two confounds seen with the Adam-based dp-corrmom). The wrapped optimizer's
    ``.step()`` (SGD) is delegated to by ``DPOptimizer`` after ``add_noise``; only the noise changes.
    """

    def __init__(self, *args, lambda_corr: float = 0.0, **kwargs):
        super().__init__(*args, **kwargs)
        if not (0.0 <= lambda_corr < 1.0):
            raise ValueError("lambda_corr must be in [0, 1)")
        self.lambda_corr = lambda_corr
        self._prev_noise: dict = {}

    def add_noise(self) -> None:
        _inject_correlated_noise(self)


class DPCorrelatedAdaptive(DPAdaptive):
    """DP-CorrMom: anti-correlated DP noise on the first-moment / prefix-sum path.

    Instead of Opacus' i.i.d. per-step noise ``z_t``, inject ``w_t = z_t - lambda_corr * z_{t-1}``
    (a bidiagonal Toeplitz matrix-factorization mechanism, the single-scalar DP-lambda-CGD
    strategy). Because momentum integrates the gradient into a running PREFIX SUM, the
    anti-correlated injections partially cancel along that sum: the integrated noise variance
    drops by roughly ``(1 - lambda)/(1 + lambda)`` vs i.i.d., at the SAME (eps, delta).

    Privacy: the i.i.d. ``z_t`` carry the entire DP guarantee; the linear mixing is a fixed
    post-processing matrix. The accountant must inflate the per-step base std by
    ``corr_sensitivity(lambda_corr, steps)`` (= ``kappa``, the column norm of ``L^{-1}``).
    The CALLER (train.py) is responsible for passing a ``noise_multiplier`` already inflated by
    kappa AND for using an UNAMPLIFIED accountant (correlated noise voids Poisson amplification).
    ``lambda_corr == 0`` reproduces stock i.i.d. ``add_noise`` bit-for-bit (the hard control).

    Only ``add_noise`` changes; ``phi`` and ``_adam_update`` are inherited unchanged. Typically
    used with ``bias_correction=False`` (BC is inert at rho~=1; this method fixes the first
    moment, not the second).
    """

    def __init__(self, *args, lambda_corr: float = 0.0, **kwargs):
        super().__init__(*args, **kwargs)
        if not (0.0 <= lambda_corr < 1.0):
            raise ValueError("lambda_corr must be in [0, 1)")
        self.lambda_corr = lambda_corr
        self._prev_noise: dict = {}

    def add_noise(self) -> None:
        """Override: inject ``w_t = z_t - lambda_corr * z_{t-1}`` instead of i.i.d. ``z_t``."""
        _inject_correlated_noise(self)


class DistributedDPAdaptive(DPAdaptive, DistributedDPOptimizer):
    """DDP-correct :class:`DPAdaptive`.

    MRO is ``DistributedDPAdaptive -> DPAdaptive -> DistributedDPOptimizer ->
    DPOptimizer``. That ordering is deliberate:

    - ``DPAdaptive.__init__`` runs first (stores lr/betas/bias_correction/...),
      and its ``super().__init__`` lands on ``DistributedDPOptimizer.__init__``,
      which records ``self.rank``/``self.world_size`` before delegating to
      ``DPOptimizer``.
    - ``add_noise`` resolves to ``DistributedDPOptimizer`` (noise added ONLY on
      rank 0, so the total injected noise matches a single global batch), while
      ``phi``/``_adam_update`` come from ``DPAdaptive``.

    We override two things:

    - ``phi``: ``reduce_gradients`` divides the summed gradient by ``world_size``
      (mean reduction), so the DP noise in the FINAL gradient is scaled by an
      extra ``1/world_size`` beyond the per-rank ``expected_batch_size``. The
      effective denominator is therefore ``expected_batch_size *
      accumulated_iterations * world_size`` (= the global batch). Without this the
      DP-AdamBC bias correction would subtract ``world_size**2`` too much. Verified
      by single-vs-DDP equivalence (matching ``phi`` and ``epsilon_spent``).
    - ``step``: insert the cross-rank gradient all-reduce (summed clipped grads +
      rank-0 noise) BETWEEN ``pre_step`` and our Adam update, mirroring
      ``DistributedDPOptimizer.step``.
    """

    @property
    def phi(self) -> float:
        b_eff = self.expected_batch_size * self.accumulated_iterations * self.world_size
        return (self.noise_multiplier * self.max_grad_norm / b_eff) ** 2

    def step(self, closure=None) -> Optional[float]:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        if self.pre_step():
            self.reduce_gradients()  # all_reduce SUM across ranks, /world_size if mean
            self._adam_update()
        return loss
