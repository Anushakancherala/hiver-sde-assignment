"""Focused tests for the intent-classification baseline."""

import pytest

from src.intents import (
    evaluate_classifier,
    predict_intent,
    preprocess_text,
    train_classifier,
)


def test_preprocess_text_normalizes_whitespace_and_case():
    assert preprocess_text("  Package   NOT here ") == "package not here"


def test_classifier_predicts_known_intent_and_evaluates():
    texts = [
        "Where is my package?",
        "My delivery has not arrived",
        "I want a refund",
        "Please return this order",
    ]
    labels = ["DELIVERY_ISSUE", "DELIVERY_ISSUE", "RETURN_REFUND", "RETURN_REFUND"]

    classifier = train_classifier(texts, labels)

    assert predict_intent(classifier, "My package is late") == "DELIVERY_ISSUE"
    metrics = evaluate_classifier(classifier, texts, labels)
    assert metrics["accuracy"] == pytest.approx(1.0)
    assert metrics["macro_f1"] == pytest.approx(2 / 9)


def test_training_rejects_unknown_intent():
    with pytest.raises(ValueError, match="Unknown intent"):
        train_classifier(["A message"], ["MADE_UP_INTENT"])