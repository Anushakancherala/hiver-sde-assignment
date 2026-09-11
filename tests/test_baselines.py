"""Tests for non-Golden baseline comparisons."""

from pathlib import Path

from src.evaluation.baselines import (
    majority_intent,
    predict_trivial,
    simple_action,
    train_simple_intent_classifier,
)
from src.evaluation.run_baselines import run_baseline_evaluation
from src.evaluation.metrics import action_metrics, intent_metrics


TRAINING_PATH = Path("data/amazonhelp_training_labeled.csv")


def test_trivial_majority_intent_and_constant_action():
    intents, actions = predict_trivial(["one", "two", "three"], TRAINING_PATH)

    assert majority_intent(TRAINING_PATH) == "GENERAL_SUPPORT"
    assert intents == ["GENERAL_SUPPORT"] * 3
    assert actions == ["AUTO_HANDLE"] * 3


def test_simple_classifier_trains_from_training_data():
    classifier = train_simple_intent_classifier(TRAINING_PATH)

    assert hasattr(classifier, "predict")
    assert classifier.predict(["Where is my package?"])[0]


def test_simple_action_rules_escalate_high_risk_only():
    assert simple_action("There was a fire and someone was injured") == "ESCALATE"
    assert simple_action("I see an unauthorized charge") == "ESCALATE"
    assert simple_action("My package is arriving tomorrow") == "AUTO_HANDLE"


def test_baseline_metrics_include_required_fields():
    intent = intent_metrics(["A", "B"], ["A", "A"], ["A", "B"])
    action = action_metrics(["AUTO_HANDLE", "ESCALATE"], ["AUTO_HANDLE", "AUTO_HANDLE"])

    assert {"accuracy", "macro_precision", "macro_recall", "macro_f1", "per_label", "confusion_matrix"}.issubset(intent)
    assert {"accuracy", "precision", "recall", "f1", "confusion_matrix"}.issubset(action)


def test_baseline_source_does_not_reference_golden_derived_training_files():
    source = Path("src/evaluation/baselines.py").read_text(encoding="utf-8")
    assert "amazon_golden_set_final" not in source
    assert "amazon_golden_set_labeled" not in source
    assert "golden_set_review" not in source
    assert "expected_action" not in source


def test_baseline_prediction_output_schema():
    predictions, results = run_baseline_evaluation()

    assert len(predictions) == 200
    assert {
        "row_number",
        "customer_message",
        "expected_intent",
        "trivial_predicted_intent",
        "simple_predicted_intent",
        "expected_action",
        "trivial_predicted_action",
        "simple_predicted_action",
    }.issubset(predictions.columns)
    assert set(results["systems"]) == {"trivial_baseline", "simple_baseline"}