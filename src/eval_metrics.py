"""Evaluation metrics: accuracy for GLUE, BLEU + ROUGE-L for E2E NLG."""

from __future__ import annotations


def accuracy(predictions, references):
    """Classification accuracy via sklearn. predictions/references: 1-D label arrays."""
    try:
        from sklearn.metrics import accuracy_score
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "scikit-learn is required for accuracy. "
            "Install it with `uv pip install scikit-learn`."
        ) from e
    return {"accuracy": float(accuracy_score(references, predictions))}


def bleu(predictions, references):
    """Corpus BLEU via sacrebleu. references: list[str] (single ref per prediction)."""
    try:
        import sacrebleu
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "sacrebleu is required for BLEU. "
            "Install it with `uv pip install sacrebleu`."
        ) from e
    score = sacrebleu.corpus_bleu(list(predictions), [list(references)])
    return {"bleu": float(score.score)}


def rouge_l(predictions, references):
    """Mean ROUGE-L F-measure via rouge_score."""
    try:
        from rouge_score import rouge_scorer
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "rouge_score is required for ROUGE-L. "
            "Install it with `uv pip install rouge-score`."
        ) from e
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    fs = [
        scorer.score(ref, pred)["rougeL"].fmeasure
        for pred, ref in zip(predictions, references)
    ]
    mean = float(sum(fs) / len(fs)) if fs else 0.0
    return {"rougeL": mean}


def compute_metric(metric_name, predictions, references):
    """Dispatch on metric_name. For 'bleu' (e2e) also returns ROUGE-L.

    Returns a dict; the primary score is keyed by ``metric_name``.
    """
    if metric_name == "accuracy":
        return accuracy(predictions, references)
    if metric_name == "bleu":
        out = bleu(predictions, references)
        out.update(rouge_l(predictions, references))
        return out
    raise ValueError(f"Unknown metric '{metric_name}'. Expected 'accuracy' or 'bleu'.")
