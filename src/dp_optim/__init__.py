"""Factory for DP optimizers (DP-SGD and the DPAdaptive Adam-family variants)."""

from __future__ import annotations

from typing import Iterable

import torch
from opacus.optimizers.optimizer import DPOptimizer
from opacus.optimizers.ddpoptimizer import DistributedDPOptimizer

from .dp_adaptive import (
    DPAdaptive,
    DPCorrelatedAdaptive,
    DPCorrelatedOptimizer,
    DistributedDPAdaptive,
    corr_sensitivity,
)

# Optimizer names understood by make_dp_optimizer.
OPTIMIZER_NAMES = [
    "dp-sgd",
    "dp-adam",
    "dp-adam-xi",  # control: xi-floor on v_hat WITHOUT phi-subtraction
    "dp-adam-lp",  # STEP-0 lever validator: extra low-pass on m_hat (set --lowpass-beta)
    "dp-corrmom",  # DP-CorrMom: anti-correlated noise on Adam update (set --lambda-corr)
    "dp-corrsgd",  # DP-CorrSGD: anti-correlated noise on PLAIN SGD (matched prefix-sum workload)
    "dp-adambc",
    "dp-adamw",
    "dp-adamw-bc",
]

# (decoupled_wd, bias_correction, floor_only) per DPAdaptive variant.
_ADAPTIVE_FLAGS = {
    "dp-adam": (False, False, False),
    "dp-adam-xi": (False, False, True),   # xi-floor only (isolates the floor effect)
    "dp-adam-lp": (False, False, False),  # plain Adam + first-moment low-pass (lowpass_beta via hp)
    "dp-corrmom": (False, False, False),  # plain Adam + anti-correlated noise (lambda_corr via hp)
    "dp-adambc": (False, True, False),
    "dp-adamw": (True, False, False),
    "dp-adamw-bc": (True, True, False),
}


def make_dp_optimizer(
    name: str,
    params: Iterable[torch.nn.Parameter],
    lr: float,
    noise_multiplier: float,
    max_grad_norm: float,
    expected_batch_size: int,
    distributed: bool = False,
    **hp,
) -> DPOptimizer:
    """Build a DP optimizer by name.

    "dp-sgd" wraps ``torch.optim.SGD(momentum=0.9)`` in a plain Opacus
    ``DPOptimizer``. The four adaptive names build :class:`DPAdaptive` with the
    appropriate ``decoupled_wd`` / ``bias_correction`` flags.

    When ``distributed=True`` (one process per GPU, under torchrun), the DDP-aware
    variants are returned: noise is injected on rank 0 only and clipped gradients
    are summed across ranks, so the privacy guarantee matches a single global
    batch. Requires an initialized ``torch.distributed`` process group.
    """
    params = list(params)
    dp_kwargs = dict(
        noise_multiplier=noise_multiplier,
        max_grad_norm=max_grad_norm,
        expected_batch_size=expected_batch_size,
    )

    if name == "dp-sgd":
        base = torch.optim.SGD(params, lr=lr, momentum=hp.get("momentum", 0.9))
        cls = DistributedDPOptimizer if distributed else DPOptimizer
        return cls(base, **dp_kwargs)

    if name == "dp-corrsgd":
        # plain SGD (momentum 0 by default) + anti-correlated noise: the matched prefix-sum
        # workload. noise_multiplier must already be kappa-inflated + unamplified (see train.py).
        if distributed:
            raise NotImplementedError("dp-corrsgd DDP variant not implemented yet")
        base = torch.optim.SGD(params, lr=lr, momentum=hp.get("momentum", 0.0))
        return DPCorrelatedOptimizer(base, lambda_corr=hp.get("lambda_corr", 0.0), **dp_kwargs)

    if name in _ADAPTIVE_FLAGS:
        decoupled_wd, bias_correction, floor_only = _ADAPTIVE_FLAGS[name]
        betas = hp.get("betas", (0.9, 0.999))
        # The base optimizer only stores param_groups (lr lives here); DPAdaptive
        # never calls its .step(), so its type is irrelevant for the update.
        base = torch.optim.SGD(params, lr=lr)
        kw = dict(
            lr=lr,
            betas=betas,
            eps=hp.get("eps", 1e-8),
            weight_decay=hp.get("weight_decay", 0.0),
            decoupled_wd=decoupled_wd,
            bias_correction=bias_correction,
            floor_only=floor_only,
            xi=hp.get("xi", 1e-8),
            probe=hp.get("probe", False),
            lowpass_beta=hp.get("lowpass_beta", 0.0),
            **dp_kwargs,
        )
        if name == "dp-corrmom":
            # DP-CorrMom: correlated-noise injection. NOTE: noise_multiplier must already be
            # inflated by corr_sensitivity(lambda, steps) and accounted UNAMPLIFIED upstream.
            if distributed:
                raise NotImplementedError("dp-corrmom DDP variant not implemented yet")
            return DPCorrelatedAdaptive(base, lambda_corr=hp.get("lambda_corr", 0.0), **kw)
        cls = DistributedDPAdaptive if distributed else DPAdaptive
        return cls(base, **kw)

    raise ValueError(f"Unknown optimizer name: {name!r}. Choose from {OPTIMIZER_NAMES}.")


__all__ = [
    "DPAdaptive",
    "DPCorrelatedAdaptive",
    "DPCorrelatedOptimizer",
    "DistributedDPAdaptive",
    "corr_sensitivity",
    "make_dp_optimizer",
    "OPTIMIZER_NAMES",
]
