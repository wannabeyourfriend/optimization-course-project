"""DP-SGD/Adam finetuning entrypoint: clip+noise via Opacus, custom adaptive update, wandb logging."""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch

# dp_optim lives under src/ (run with PYTHONPATH=src).
from dp_optim import OPTIMIZER_NAMES, make_dp_optimizer

MODEL_ALIASES = {
    "roberta-base": "roberta-base",
    "roberta-large": "roberta-large",
    "gpt2": "gpt2",
    "gpt2-large": "gpt2-large",
    "qwen2.5-1.5b": "Qwen/Qwen2.5-1.5B",
    "qwen2.5-3b": "Qwen/Qwen2.5-3B",
}

# Causal-LM tasks (vs. sequence-classification GLUE tasks).
CAUSAL_TASKS = {"e2e"}


# --------------------------------------------------------------------------- #
# Config / CLI
# --------------------------------------------------------------------------- #
def parse_args(argv=None):
    p = argparse.ArgumentParser(description="DP finetuning with adaptive DP optimizers.")
    p.add_argument("--model", default="roberta-base")
    p.add_argument("--task", choices=["sst2", "qnli", "mnli", "e2e"], default="sst2")
    p.add_argument("--optimizer", choices=OPTIMIZER_NAMES, default="dp-adam")
    p.add_argument("--epsilon", default="8", help="float budget or 'inf' for non-private (sigma=0).")
    p.add_argument("--delta", type=float, default=1e-5)
    p.add_argument("--max-grad-norm", type=float, default=0.1)
    p.add_argument("--probe", action="store_true",
                   help="direction-fidelity probe: log cos(momentum,clean) vs cos(msign(momentum),clean).")
    p.add_argument("--xi", type=float, default=1e-8,
                   help="v_hat floor for DP-AdamBC / dp-adam-xi (set ~1e-10 so the "
                        "floor doesn't manufacture a fake BC separation).")
    p.add_argument("--lowpass-beta", type=float, default=0.0,
                   help="STEP-0 lever validator (dp-adam-lp): extra causal EMA on m_hat; "
                        "privacy-free post-processing. 0 disables (= plain dp-adam).")
    p.add_argument("--lambda-corr", type=float, default=0.0,
                   help="dp-corrmom: anti-correlated noise w_t=z_t-lambda*z_{t-1} in [0,1). "
                        "Uses SINGLE-PARTICIPATION unamplified accounting (deterministic loader, "
                        "no Poisson); sigma inflated by corr_sensitivity(lambda,steps). 0 = "
                        "unamplified DP-SGD-momentum (the hard control).")
    p.add_argument("--beta1", type=float, default=0.9,
                   help="Adam first-moment decay. 0 disables momentum (the parameter trajectory "
                        "then IS the gradient prefix-sum the dp-corrmom strategy is matched to).")
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument("--micro-batch", type=int, default=8)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--lora", action="store_true")
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--ddp", action="store_true")
    p.add_argument("--wandb-project", default="dp-optimizer-finetuning")
    p.add_argument("--round-id", default="adhoc")
    p.add_argument("--config", default=None, help="YAML; CLI flags override its values.")
    p.add_argument("--out", default="results/")
    p.add_argument("--eval-every", type=int, default=200)
    p.add_argument("--max-len", type=int, default=128)
    p.add_argument("--cluster", default=os.environ.get("CLUSTER", "local"))
    p.add_argument("--dry-run", action="store_true",
                   help="Build everything and run 2 steps on tiny synthetic data (CPU, no network).")
    return p.parse_args(argv)


def apply_yaml_config(args, argv):
    """Merge a YAML config under the CLI flags. CLI values explicitly passed win."""
    if not args.config:
        return args
    import yaml

    with open(args.config) as f:
        cfg = yaml.safe_load(f) or {}
    # lr_by_optimizer map -> per-optimizer default lr.
    if args.lr is None and isinstance(cfg.get("lr_by_optimizer"), dict):
        args.lr = cfg["lr_by_optimizer"].get(args.optimizer)
    passed = {a.lstrip("-").split("=")[0].replace("-", "_") for a in (argv or []) if a.startswith("--")}
    for key, val in cfg.items():
        attr = key.replace("-", "_")
        if hasattr(args, attr) and attr not in passed:
            setattr(args, attr, val)
    return args


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# --------------------------------------------------------------------------- #
# Model / data
# --------------------------------------------------------------------------- #
def _prep_causal_for_opacus(model):
    """Make a causal LM safe for Opacus per-sample grads.

    Two GPT-2-style hazards: (1) tied lm_head/wte (output weight aliases the input
    embedding) yields an inconsistent grad_sample batch dim -> clone to untie. (2) learned
    absolute position embeddings (wpe) are indexed by batch-shared position_ids, so Opacus
    collapses their grad_sample to batch dim 1 -> freeze them. RoBERTa (per-sample
    position_ids) and RoPE models (Qwen, no wpe) are unaffected.
    """
    import torch.nn as nn

    if getattr(model.config, "tie_word_embeddings", False):
        out, inp = model.get_output_embeddings(), model.get_input_embeddings()
        if out is not None and inp is not None and out.weight is inp.weight:
            out.weight = nn.Parameter(out.weight.detach().clone())
        model.config.tie_word_embeddings = False
    for name, p in model.named_parameters():
        if name.endswith("wpe.weight"):  # GPT-2 absolute position embedding
            p.requires_grad_(False)


def load_model_and_tokenizer(args, num_labels):
    from transformers import AutoTokenizer

    model_id = MODEL_ALIASES.get(args.model, args.model)
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    if args.task in CAUSAL_TASKS:
        from transformers import AutoModelForCausalLM

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(model_id)
        _prep_causal_for_opacus(model)  # untie + freeze wpe for Opacus per-sample grads
    else:
        from transformers import AutoModelForSequenceClassification

        model = AutoModelForSequenceClassification.from_pretrained(
            model_id, num_labels=num_labels
        )
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id

    if args.lora:
        from peft import LoraConfig, TaskType, get_peft_model

        task_type = TaskType.CAUSAL_LM if args.task in CAUSAL_TASKS else TaskType.SEQ_CLS
        model = get_peft_model(
            model,
            LoraConfig(task_type=task_type, r=args.lora_r, lora_alpha=2 * args.lora_r,
                       lora_dropout=0.0),
        )
    return model, tokenizer


def make_loader(dataset, micro_batch, sample_rate, generator):
    """Poisson-sampled DPDataLoader (the DP-correct sampler Opacus accounting assumes)."""
    from opacus.data_loader import DPDataLoader

    return DPDataLoader(
        dataset,
        sample_rate=sample_rate,
        collate_fn=_collate,
        generator=generator,
    )


def make_corr_loader(dataset, batch_size, steps, seed):
    """Deterministic SINGLE-PARTICIPATION loader for dp-corrmom.

    Correlated noise voids Poisson-subsampling amplification, and the matrix-mechanism
    sensitivity ``corr_sensitivity`` is only valid when each example participates in at most
    ONE step. So we shuffle once and take ``steps * batch_size`` DISTINCT examples (each used
    exactly once), yielding fixed logical batches. BatchMemoryManager then splits each into
    micro-batches exactly as for the Poisson path.
    """
    n_need = steps * batch_size
    if n_need > len(dataset):
        raise ValueError(
            f"dp-corrmom single-participation needs steps*batch_size={n_need} <= "
            f"n_train={len(dataset)}; reduce steps/batch_size or use more data."
        )
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(dataset), generator=g)[:n_need].tolist()
    subset = torch.utils.data.Subset(dataset, perm)
    return torch.utils.data.DataLoader(
        subset, batch_size=batch_size, shuffle=False, drop_last=True, collate_fn=_collate
    )


def _collate(batch):
    """Stack a list of {input_ids, attention_mask, labels} dicts into batched tensors."""
    out = {}
    for key in batch[0]:
        out[key] = torch.stack([torch.as_tensor(b[key]) for b in batch])
    return out


# --------------------------------------------------------------------------- #
# Diagnostics (graceful: optional sibling module)
# --------------------------------------------------------------------------- #
def diag_dict(optimizer):
    try:
        from diagnostics import compute_phi, effective_stepsize_percentiles, dynamics_dict

        phi = compute_phi(optimizer)
        pct = effective_stepsize_percentiles(optimizer)
        d = {"phi": phi, "p50": pct.get("p50"), "p90": pct.get("p90")}
        d.update(dynamics_dict(optimizer))  # grad_norm, vhat_*, phi_over_vhat, clamp_frac, update_norm
        d.update(getattr(optimizer, "probe_stats", {}) or {})  # cos_before/after/gain (--probe)
        return d
    except Exception:
        nm, c = optimizer.noise_multiplier, optimizer.max_grad_norm
        beff = optimizer.expected_batch_size * max(optimizer.accumulated_iterations, 1)
        return {"phi": (nm * c / beff) ** 2, "p50": None, "p90": None}


# --------------------------------------------------------------------------- #
# Eval
# --------------------------------------------------------------------------- #
@torch.no_grad()
def evaluate(model, loader, task, metric_name, tokenizer, device, max_batches=None):
    from eval_metrics import compute_metric

    model.eval()
    preds, refs = [], []
    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        batch = {k: v.to(device) for k, v in batch.items()}
        if task in CAUSAL_TASKS:
            # Cheap proxy: decode greedy argmax of the LM head over the labels span.
            logits = model(input_ids=batch["input_ids"],
                           attention_mask=batch["attention_mask"]).logits
            pred_ids = logits.argmax(-1)
            if tokenizer is not None:
                preds += tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
                refs += tokenizer.batch_decode(batch["labels"], skip_special_tokens=True)
            else:  # offline dry-run: no tokenizer, compare token-id strings
                preds += [" ".join(map(str, r.tolist())) for r in pred_ids]
                refs += [" ".join(map(str, r.tolist())) for r in batch["labels"]]
        else:
            logits = model(input_ids=batch["input_ids"],
                           attention_mask=batch["attention_mask"]).logits
            preds += logits.argmax(-1).cpu().tolist()
            refs += batch["labels"].cpu().tolist()
    model.train()
    if not preds:
        return {metric_name: 0.0}
    return compute_metric(metric_name, preds, refs)


# --------------------------------------------------------------------------- #
# Synthetic dry-run data + tiny local model (no network, CPU-safe)
# --------------------------------------------------------------------------- #
def _tiny_model(args, num_labels):
    """Tiny randomly-initialized model from a LOCAL config (no HF download) for --dry-run."""
    import transformers

    vocab_size = 512
    if args.task in CAUSAL_TASKS:
        # tie_word_embeddings=False: Opacus per-sample grads break on tied embeddings
        # (lm_head.weight aliasing wte.weight -> inconsistent grad_sample batch dim).
        # Real causal runs (gpt2/qwen) need the same untie; see load_model_and_tokenizer.
        cfg = transformers.GPT2Config(vocab_size=vocab_size, n_positions=max(args.max_len, 8),
                                      n_embd=64, n_layer=2, n_head=2, n_inner=128,
                                      tie_word_embeddings=False)
        model = transformers.GPT2LMHeadModel(cfg)
        model.config.pad_token_id = 0
        _prep_causal_for_opacus(model)
    else:
        cfg = transformers.RobertaConfig(vocab_size=vocab_size, hidden_size=64,
                                         num_hidden_layers=2, num_attention_heads=2,
                                         intermediate_size=128,
                                         max_position_embeddings=max(args.max_len + 2, 16),
                                         num_labels=num_labels or 2, pad_token_id=0)
        model = transformers.RobertaForSequenceClassification(cfg)
    if args.lora:
        from peft import LoraConfig, TaskType, get_peft_model

        task_type = TaskType.CAUSAL_LM if args.task in CAUSAL_TASKS else TaskType.SEQ_CLS
        model = get_peft_model(model, LoraConfig(task_type=task_type, r=args.lora_r,
                                                 lora_alpha=2 * args.lora_r, lora_dropout=0.0))
    return model, vocab_size


def _synthetic_dataset(task, vocab_size, n=16, max_len=16, num_labels=2):
    vocab = max(vocab_size - 1, 2)
    rng = torch.Generator().manual_seed(0)
    rows = []
    for _ in range(n):
        ids = torch.randint(1, vocab, (max_len,), generator=rng)
        mask = torch.ones(max_len, dtype=torch.long)
        if task in CAUSAL_TASKS:
            rows.append({"input_ids": ids, "attention_mask": mask, "labels": ids.clone()})
        else:
            label = torch.randint(0, num_labels, (1,), generator=rng).item()
            rows.append({"input_ids": ids, "attention_mask": mask, "labels": label})
    return rows


# --------------------------------------------------------------------------- #
# Train
# --------------------------------------------------------------------------- #
def compute_sigma(epsilon, delta, sample_rate, steps):
    """sigma=0 for non-private (epsilon=inf); else solve via Opacus get_noise_multiplier."""
    if epsilon == float("inf"):
        return 0.0
    from opacus.accountants.utils import get_noise_multiplier

    return get_noise_multiplier(
        target_epsilon=epsilon, target_delta=delta,
        sample_rate=sample_rate, steps=steps,
    )


def model_loss(model, batch, task):
    """Forward pass returning the HF loss (labels passed through)."""
    if task in CAUSAL_TASKS:
        return model(input_ids=batch["input_ids"],
                     attention_mask=batch["attention_mask"],
                     labels=batch["labels"]).loss
    return model(input_ids=batch["input_ids"],
                 attention_mask=batch["attention_mask"],
                 labels=batch["labels"]).loss


def _ddp_setup(args):
    """Init the process group for a torchrun-launched DDP run. Returns
    (ddp, rank, world_size, device, is_main). No-op (single process) otherwise."""
    use_ddp = bool(getattr(args, "ddp", False)) and "RANK" in os.environ
    if not use_ddp:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return False, 0, 1, device, True
    import torch.distributed as dist

    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank % max(1, torch.cuda.device_count())))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    return True, rank, world_size, device, (rank == 0)


def train(args, run=None):
    """Train one config. If `run` is given (a wandb run created by a sweep agent),
    log to it instead of creating a new one; otherwise CLI/bash-sweep behaviour."""
    set_seed(args.seed)
    ddp, rank, world_size, device, is_main = _ddp_setup(args)
    epsilon = float("inf") if str(args.epsilon).lower() == "inf" else float(args.epsilon)

    from opacus import GradSampleModule
    from opacus.accountants import PRVAccountant, RDPAccountant

    # ---- data + model ----
    if args.dry_run:
        # Fully offline: tiny local model + random-token synthetic data (no HF download).
        args.steps, args.batch_size, args.micro_batch, args.max_len = 2, 8, 4, 16
        num_labels = None if args.task in CAUSAL_TASKS else 3
        model, vocab_size = _tiny_model(args, num_labels)
        tokenizer = None
        n_train = 16
        train_ds = _synthetic_dataset(args.task, vocab_size, n=n_train,
                                      max_len=args.max_len, num_labels=num_labels or 2)
        val_ds = train_ds
        metric_name = "bleu" if args.task in CAUSAL_TASKS else "accuracy"
    else:
        from data import load_task
        from transformers import AutoTokenizer

        model_id = MODEL_ALIASES.get(args.model, args.model)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        train_ds, val_ds, num_labels, metric_name = load_task(args.task, tokenizer, args.max_len)
        model, tokenizer = load_model_and_tokenizer(args, num_labels)
        n_train = len(train_ds)

    model = model.to(device)
    model.train()

    # ---- privacy params ----
    # The privacy mechanism is defined on the GLOBAL batch: sigma and the
    # accountant both use the global sample rate (independent of how many GPUs
    # parallelize one step). Under DDP each rank owns 1/world_size of the logical
    # batch, so its loader samples at a reduced rate and the optimizer's
    # expected_batch_size is the per-rank share; DistributedDPOptimizer sums the
    # clipped grads across ranks and adds noise once (rank 0), reproducing the
    # single-GPU mechanism. See DistributedDPAdaptive.
    CORR = args.optimizer in ("dp-corrmom", "dp-corrsgd")
    sigma_eff = None
    if CORR:
        # dp-corrmom: SINGLE-PARTICIPATION, UNAMPLIFIED matrix-mechanism accounting.
        # The whole correlated run is ONE Gaussian release of C_strat*g with column
        # sensitivity kappa=corr_sensitivity(lambda,steps); so solve the single unamplified
        # Gaussian (sample_rate=1, 1 step) for (eps,delta) and inject base noise sigma_eff*kappa.
        # lambda=0 -> kappa=1 -> unamplified DP-SGD-momentum (the hard control).
        if ddp:
            raise NotImplementedError("dp-corrmom DDP not implemented")
        from dp_optim import corr_sensitivity
        kappa = corr_sensitivity(args.lambda_corr, args.steps)
        sigma_eff = compute_sigma(epsilon, args.delta, sample_rate=1.0, steps=1)
        sigma = sigma_eff * kappa
        sample_rate = 1.0
        per_rank_sample_rate = 1.0
        per_rank_batch = args.batch_size
    else:
        sample_rate = min(args.batch_size / n_train, 1.0)          # global
        sigma = compute_sigma(epsilon, args.delta, sample_rate, args.steps)
        per_rank_batch = max(1, args.batch_size // world_size)
        per_rank_sample_rate = min(per_rank_batch / n_train, 1.0)

    # ---- wrap model: DPDDP (broadcasts params, no grad-averaging hooks) then
    # GradSampleModule (per-sample grad hooks) ----
    if ddp:
        from opacus.distributed import (
            DifferentiallyPrivateDistributedDataParallel as DPDDP,
        )
        model = DPDDP(model)
    model = GradSampleModule(model)

    # ---- Poisson-sampled loader (per-rank rate; distinct seed per rank so ranks
    # sample independent subsets whose union is the logical batch) ----
    if CORR:
        # deterministic single-participation loader (no Poisson); BatchMemoryManager
        # below splits each logical batch into micro-batches exactly as for the Poisson path.
        train_loader = make_corr_loader(train_ds, args.batch_size, args.steps, args.seed)
    else:
        gen = torch.Generator(device="cpu").manual_seed(args.seed + rank)
        train_loader = make_loader(train_ds, args.micro_batch, per_rank_sample_rate, gen)
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=args.micro_batch, collate_fn=_collate
    )

    # ---- optimizer (factory) ----
    lr = args.lr if args.lr is not None else 1e-3
    optimizer = make_dp_optimizer(
        args.optimizer,
        params=[p for p in model.parameters() if p.requires_grad],
        lr=lr,
        noise_multiplier=sigma,
        max_grad_norm=args.max_grad_norm,
        expected_batch_size=per_rank_batch,
        distributed=ddp,
        xi=args.xi,
        probe=args.probe,
        lowpass_beta=args.lowpass_beta,
        lambda_corr=args.lambda_corr,
        betas=(args.beta1, 0.999),
    )

    # ---- accountants: RDP gives running eps; PRV certifies eps at the end.
    # attach_step_hook REPLACES the hook, so both must share one combined hook. ----
    rdp = RDPAccountant()
    prv = PRVAccountant()
    if not CORR:
        # dp-corrmom has correlated cross-step noise, so the per-step subsampled composition
        # is INVALID; it is certified as one unamplified Gaussian at the end instead.
        optimizer.attach_step_hook(_dual_hook(rdp, prv, sample_rate))

    phi = diag_dict(optimizer)["phi"]
    if run is not None:
        # Running under a wandb-sweep agent: enrich the agent-created run's config.
        try:
            run.config.update(
                {"model": args.model, "task": args.task, "optimizer": args.optimizer,
                 "epsilon": epsilon, "delta": args.delta, "seed": args.seed,
                 "max_grad_norm": args.max_grad_norm, "steps": args.steps,
                 "batch_size": args.batch_size, "lr": args.lr, "sample_rate": sample_rate,
                 "sigma": sigma, "phi": phi, "round_id": args.round_id,
                 "cluster": args.cluster,
                 "method": "lora" if args.lora else "full", "lora": bool(args.lora),
                 "lora_r": args.lora_r},
                allow_val_change=True)
        except Exception:
            pass
    elif is_main:
        run = _init_wandb(args, epsilon, sample_rate, sigma, phi)

    # ---- micro-batch accumulation via BatchMemoryManager ----
    from opacus.utils.batch_memory_manager import BatchMemoryManager

    step = 0
    t0 = time.time()
    with BatchMemoryManager(
        data_loader=train_loader,
        max_physical_batch_size=args.micro_batch,
        optimizer=optimizer,
    ) as mem_loader:
        while step < args.steps:
            for batch in mem_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                if batch["input_ids"].shape[0] == 0:
                    continue  # Poisson sampling can yield empty physical batches.
                loss = model_loss(model, batch, args.task)
                loss.backward()
                optimizer.step()
                if optimizer._is_last_step_skipped:
                    optimizer.zero_grad()
                    continue
                d = diag_dict(optimizer)  # before zero_grad: accumulated_iterations still valid
                optimizer.zero_grad()
                step += 1

                # CORR has no per-step hook (single mechanism certified at the end); log the
                # target eps so the dashboard shows the calibrated budget, not 0.
                eps_spent = epsilon if CORR else _safe_eps(rdp, args.delta)
                _log(run, {
                    "train/loss": float(loss.detach()),
                    "privacy/epsilon_spent": eps_spent,
                    "diag/phi": d["phi"],
                    "diag/eff_stepsize_p50": d["p50"],
                    "diag/eff_stepsize_p90": d["p90"],
                    # rich optimization dynamics (interpret DP-optimizer behaviour)
                    "diag/grad_norm": d.get("grad_norm"),
                    "diag/vhat_p50": d.get("vhat_p50"),
                    "diag/vhat_p90": d.get("vhat_p90"),
                    "diag/phi_over_vhat": d.get("phi_over_vhat"),
                    "diag/clamp_frac": d.get("clamp_frac"),
                    "diag/update_norm": d.get("update_norm"),
                    "diag/cos_before": d.get("cos_before"),
                    "diag/cos_after": d.get("cos_after"),
                    "diag/cos_gain": d.get("cos_gain"),
                }, step)

                if step % args.eval_every == 0 or step >= args.steps:
                    m = evaluate(model, val_loader, args.task, metric_name,
                                 tokenizer, device,
                                 max_batches=2 if args.dry_run else None)
                    _log(run, {"eval/metric": float(m.get(metric_name, 0.0)),
                               "eval/metric_name": metric_name}, step)
                if step >= args.steps:
                    break

    # ---- final eval + PRV-certified epsilon ----
    final_metric = evaluate(model, val_loader, args.task, metric_name, tokenizer,
                            device, max_batches=2 if args.dry_run else None)
    if CORR:
        # certify the single unamplified Gaussian (sigma_eff vs the kappa-sensitivity).
        if sigma > 0:
            cert = PRVAccountant()
            cert.step(noise_multiplier=sigma_eff, sample_rate=1.0)
            eps_final = _safe_eps(cert, args.delta)
        else:
            eps_final = 0.0
    else:
        eps_final = _safe_eps(prv, args.delta) if sigma > 0 else 0.0
    elapsed = time.time() - t0

    result = {
        "round_id": args.round_id, "model": args.model, "task": args.task,
        "optimizer": args.optimizer, "epsilon": args.epsilon, "delta": args.delta,
        "seed": args.seed, "max_grad_norm": args.max_grad_norm, "steps": args.steps,
        "batch_size": args.batch_size, "lr": lr, "sample_rate": sample_rate,
        "sigma": sigma, "phi": phi, "cluster": args.cluster,
        "method": "lora" if args.lora else "full", "lora_r": args.lora_r,
        "metric_name": metric_name,
        "eval_metric": float(final_metric.get(metric_name, 0.0)),
        "epsilon_spent": eps_final, "elapsed_sec": elapsed,
    }
    if is_main:
        _write_result(args, result)
    if run is not None:
        run.summary.update({"eval/metric": result["eval_metric"],
                            "privacy/epsilon_spent": eps_final})
        run.finish()
    if ddp:
        import torch.distributed as dist
        dist.destroy_process_group()
    return result


# --------------------------------------------------------------------------- #
# wandb / accounting helpers
# --------------------------------------------------------------------------- #
def _dual_hook(rdp, prv, sample_rate):
    """Step BOTH accountants per optimizer step (RDP for running eps, PRV to certify)."""
    def hook(optim):
        eff_rate = sample_rate * optim.accumulated_iterations
        rdp.step(noise_multiplier=optim.noise_multiplier, sample_rate=eff_rate)
        prv.step(noise_multiplier=optim.noise_multiplier, sample_rate=eff_rate)
    return hook


def _safe_eps(accountant, delta):
    try:
        if len(accountant) == 0:
            return 0.0
        return float(accountant.get_epsilon(delta=delta))
    except Exception:
        return None


def _init_wandb(args, epsilon, sample_rate, sigma, phi):
    try:
        import wandb
    except Exception:
        return None
    if args.dry_run or os.environ.get("WANDB_MODE") == "disabled":
        return None
    method = "lora" if args.lora else "full"
    name = f"{args.round_id}/{method}/{args.optimizer}/eps{args.epsilon}/s{args.seed}"
    return wandb.init(
        project=args.wandb_project,
        name=name,
        config={
            "model": args.model, "task": args.task, "optimizer": args.optimizer,
            "epsilon": epsilon, "delta": args.delta, "seed": args.seed,
            "max_grad_norm": args.max_grad_norm, "steps": args.steps,
            "batch_size": args.batch_size, "lr": args.lr, "sample_rate": sample_rate,
            "sigma": sigma, "phi": phi, "round_id": args.round_id,
            "cluster": args.cluster,
            "method": method, "lora": bool(args.lora), "lora_r": args.lora_r,
        },
    )


def _log(run, metrics, step):
    metrics = {k: v for k, v in metrics.items() if v is not None}
    if run is not None:
        run.log(metrics, step=step)


def _write_result(args, result):
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = (f"{args.round_id}__{args.model}__{args.task}__{args.optimizer}"
             f"__eps{args.epsilon}__s{args.seed}.json")
    (out_dir / fname).write_text(json.dumps(result, indent=2))


def main(argv=None):
    import sys

    raw = sys.argv[1:] if argv is None else argv
    args = parse_args(raw)
    args = apply_yaml_config(args, raw)
    result = train(args)
    # Under DDP every rank returns the same result; only rank 0 prints it.
    if os.environ.get("RANK", "0") == "0":
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
