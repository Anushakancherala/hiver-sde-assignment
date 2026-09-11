"""Metric helpers for Golden Set evaluation."""

from __future__ import annotations

from typing import Any

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
)


def classification_metrics(
    expected: list[str],
    predicted: list[str],
    labels: list[str],
) -> dict[str, Any]:
    """Return aggregate, per-label, and confusion-matrix metrics."""
    precision, recall, f1, support = precision_recall_fscore_support(
        expected, predicted, labels=labels, zero_division=0
    )
    per_label = {
        label: {
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "support": int(support[index]),
        }
        for index, label in enumerate(labels)
    }
    return {
        "accuracy": float(accuracy_score(expected, predicted)),
        "macro_precision": float(precision_score(expected, predicted, labels=labels, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(expected, predicted, labels=labels, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(expected, predicted, labels=labels, average="macro", zero_division=0)),
        "per_label": per_label,
        "confusion_matrix": {
            "labels": labels,
            "values": confusion_matrix(expected, predicted, labels=labels).tolist(),
        },
    }


def intent_metrics(expected: list[str], predicted: list[str], labels: list[str]) -> dict[str, Any]:
    """Calculate intent classification metrics."""
    return classification_metrics(expected, predicted, labels)


def action_metrics(expected: list[str], predicted: list[str]) -> dict[str, Any]:
    """Calculate binary AUTO_HANDLE/ESCALATE metrics."""
    metrics = classification_metrics(expected, predicted, ["AUTO_HANDLE", "ESCALATE"])
    metrics["precision"] = metrics["macro_precision"]
    metrics["recall"] = metrics["macro_recall"]
    metrics["f1"] = metrics["macro_f1"]
    return metrics