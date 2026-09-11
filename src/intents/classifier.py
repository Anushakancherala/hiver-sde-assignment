"""A small, explainable intent classifier for AmazonHelp support messages."""

from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline


INTENTS = (
    "DELIVERY_ISSUE",
    "ORDER_ISSUE",
    "RETURN_REFUND",
    "PAYMENT_BILLING",
    "ACCOUNT_MEMBERSHIP",
    "PRODUCT_TECHNICAL",
    "PRODUCT_QUALITY",
    "SAFETY_ISSUE",
    "GENERAL_SUPPORT",
)


def preprocess_text(text: Any) -> str:
    """Normalize a support message before it reaches the vectorizer."""
    if text is None or pd.isna(text):
        return ""

    cleaned_text = " ".join(str(text).lower().split())
    return cleaned_text


def _read_labeled_file(file_path: str | Path) -> pd.DataFrame:
    """Read a supported labeled-data file without assigning any labels."""
    path = Path(file_path)
    suffix = path.suffix.lower()

    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".json", ".jsonl"}:
        return pd.read_json(path, lines=suffix == ".jsonl")
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)

    raise ValueError("Supported training-data formats are CSV, JSON, JSONL, and XLSX")


def load_training_data(
    file_path: str | Path,
    text_column: str = "text",
    label_column: str = "intent",
) -> tuple[list[str], list[str]]:
    """Load existing labeled training data from disk.

    The function only reads labels already present in the file. It never creates
    labels from message text or from the Golden Set.
    """
    data = _read_labeled_file(file_path)
    missing_columns = {text_column, label_column} - set(data.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Missing required column(s): {missing}")

    texts = [preprocess_text(value) for value in data[text_column]]
    labels = [str(value).strip() for value in data[label_column]]

    if not texts or any(not text for text in texts):
        raise ValueError("Training data must contain non-empty text values")
    unknown_labels = sorted(set(labels) - set(INTENTS))
    if unknown_labels:
        raise ValueError(f"Unknown intent label(s): {', '.join(unknown_labels)}")

    return texts, labels


def train_classifier(texts: list[str], labels: list[str]) -> Pipeline:
    """Train and return a TF-IDF plus Logistic Regression classifier."""
    if len(texts) != len(labels) or not texts:
        raise ValueError("texts and labels must be non-empty lists of equal length")

    unknown_labels = sorted(set(labels) - set(INTENTS))
    if unknown_labels:
        raise ValueError(f"Unknown intent label(s): {', '.join(unknown_labels)}")

    classifier = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(preprocessor=preprocess_text, ngram_range=(1, 2)),
            ),
            (
                "logistic_regression",
                LogisticRegression(max_iter=1000, random_state=42),
            ),
        ]
    )
    classifier.fit(texts, labels)
    return classifier


def predict_intent(classifier: Pipeline, text: str) -> str:
    """Predict one intent for a single support message."""
    if not preprocess_text(text):
        raise ValueError("text must be non-empty")
    return str(classifier.predict([text])[0])


def evaluate_classifier(
    classifier: Pipeline,
    texts: list[str],
    labels: list[str],
) -> dict[str, float]:
    """Return accuracy and macro F1 for a labeled evaluation set."""
    if len(texts) != len(labels) or not texts:
        raise ValueError("texts and labels must be non-empty lists of equal length")

    predictions = classifier.predict(texts)
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(
            f1_score(labels, predictions, labels=list(INTENTS), average="macro", zero_division=0)
        ),
    }