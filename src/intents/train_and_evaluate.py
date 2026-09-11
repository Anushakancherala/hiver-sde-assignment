"""Train and evaluate the baseline intent classifier.

This script uses only the rule-assisted historical training CSV. The Golden Set
is intentionally not read or used for either the train or test split.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import joblib
from sklearn.metrics import classification_report, precision_score, recall_score
from sklearn.model_selection import train_test_split

from .classifier import INTENTS, load_training_data, train_classifier


DEFAULT_DATA_PATH = Path("data/amazonhelp_training_labeled.csv")
DEFAULT_MODEL_PATH = Path("models/intent_classifier.joblib")
RANDOM_SEED = 42


def train_and_evaluate(
    data_path: str | Path = DEFAULT_DATA_PATH,
    model_path: str | Path = DEFAULT_MODEL_PATH,
    test_size: float = 0.2,
    random_state: int = RANDOM_SEED,
) -> dict[str, Any]:
    """Train the classifier, print metrics, and save the fitted pipeline."""
    texts, labels = load_training_data(
        data_path,
        text_column="customer_message",
        label_column="intent",
    )

    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts,
        labels,
        test_size=test_size,
        random_state=random_state,
        stratify=labels,
    )

    classifier = train_classifier(train_texts, train_labels)
    predictions = classifier.predict(test_texts)

    metrics = {
        "accuracy": float((predictions == test_labels).mean()),
        "macro_precision": float(
            precision_score(test_labels, predictions, labels=list(INTENTS), average="macro", zero_division=0)
        ),
        "macro_recall": float(
            recall_score(test_labels, predictions, labels=list(INTENTS), average="macro", zero_division=0)
        ),
        "macro_f1": float(
            classification_report(
                test_labels,
                predictions,
                labels=list(INTENTS),
                output_dict=True,
                zero_division=0,
            )["macro avg"]["f1-score"]
        ),
    }

    output_path = Path(model_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(classifier, output_path)

    print(f"Training examples: {len(train_texts)}")
    print(f"Test examples: {len(test_texts)}")
    print(f"Random seed: {random_state}")
    print("\nOverall metrics:")
    for name, value in metrics.items():
        print(f"  {name}: {value:.4f}")
    print("\nPer-intent classification report:")
    print(
        classification_report(
            test_labels,
            predictions,
            labels=list(INTENTS),
            zero_division=0,
        )
    )
    print(f"Saved model pipeline to {output_path}")

    return {
        "classifier": classifier,
        "metrics": metrics,
        "test_labels": test_labels,
        "predictions": list(predictions),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    args = parser.parse_args()
    train_and_evaluate(data_path=args.data, model_path=args.model)


if __name__ == "__main__":
    main()