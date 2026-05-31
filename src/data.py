"""Task/dataset loading for GLUE classification and E2E NLG causal-LM finetuning."""

from __future__ import annotations

GLUE_TASKS = {
    "sst2": ("sentence", None),
    "qnli": ("question", "sentence"),
    "mnli": ("premise", "hypothesis"),
}

GLUE_NUM_LABELS = {"sst2": 2, "qnli": 2, "mnli": 3}


def _require_datasets():
    try:
        from datasets import load_dataset  # noqa: F401
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "The 'datasets' package is required to load tasks. "
            "Install it with `uv pip install datasets`."
        ) from e
    from datasets import load_dataset

    return load_dataset


def _load_glue(task, tokenizer, max_len):
    load_dataset = _require_datasets()
    key_a, key_b = GLUE_TASKS[task]
    # Use the namespaced GLUE repo: recent huggingface_hub rejects the legacy
    # bare-name "glue" ("Repository id must be 'namespace/name'").
    raw = load_dataset("nyu-mll/glue", task)
    val_split = "validation_matched" if task == "mnli" else "validation"

    def tok(batch):
        args = (batch[key_a],) if key_b is None else (batch[key_a], batch[key_b])
        out = tokenizer(
            *args, truncation=True, max_length=max_len, padding="max_length"
        )
        out["labels"] = batch["label"]
        return out

    cols_to_remove = [c for c in raw["train"].column_names if c != "label"]
    train_ds = raw["train"].map(tok, batched=True, remove_columns=cols_to_remove)
    val_ds = raw[val_split].map(tok, batched=True, remove_columns=cols_to_remove)
    train_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    val_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    return train_ds, val_ds, GLUE_NUM_LABELS[task], "accuracy"


def _load_e2e(tokenizer, max_len):
    """Causal-LM: 'meaning_representation -> human_reference' as a single text."""
    load_dataset = _require_datasets()
    # The default e2e_nlg loader script pulls CSVs from raw.githubusercontent.com,
    # which is unreachable on the GPU box. Load HF's auto-converted parquet revision
    # (served via HF_ENDPOINT mirror) instead.
    raw = load_dataset("tuetschek/e2e_nlg", revision="refs/convert/parquet")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    eos = tokenizer.eos_token or ""

    def tok(batch):
        texts = [
            f"{mr} = {ref}{eos}"
            for mr, ref in zip(
                batch["meaning_representation"], batch["human_reference"]
            )
        ]
        out = tokenizer(
            texts, truncation=True, max_length=max_len, padding="max_length"
        )
        out["labels"] = [list(ids) for ids in out["input_ids"]]
        return out

    cols = raw["train"].column_names
    train_ds = raw["train"].map(tok, batched=True, remove_columns=cols)
    val_ds = raw["validation"].map(tok, batched=True, remove_columns=cols)
    train_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    val_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    return train_ds, val_ds, None, "bleu"


def load_task(task, tokenizer, max_len=128):
    """Return (train_ds, val_ds, num_labels_or_None, metric_name) for a task.

    GLUE (sst2/qnli/mnli) -> sequence classification, metric 'accuracy'.
    e2e -> causal-LM text generation, metric 'bleu' (BLEU + ROUGE-L at eval).
    """
    if task in GLUE_TASKS:
        return _load_glue(task, tokenizer, max_len)
    if task == "e2e":
        return _load_e2e(tokenizer, max_len)
    raise ValueError(f"Unknown task '{task}'. Expected one of sst2/qnli/mnli/e2e.")
