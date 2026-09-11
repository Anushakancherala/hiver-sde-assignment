"""Offline tests for the complete AmazonHelp evaluation harness."""

import json
from pathlib import Path

import pandas as pd

from src.evaluation.evaluator import MockProvider, evaluate_pipeline, load_golden_set
from src.evaluation.metrics import action_metrics, intent_metrics


GOLDEN_PATH = Path("evaluation/amazon_golden_set_final.csv")


def test_final_golden_set_loads_with_required_columns_and_200_rows():
    data = load_golden_set(GOLDEN_PATH)
    assert len(data) == 200
    assert data.columns.tolist() == ["customer_message", "amazon_reply", "intent", "expected_action", "reason"]


def test_metric_calculation_includes_expected_scores_and_confusion_matrix():
    intent = intent_metrics(["A", "B", "A"], ["A", "A", "A"], ["A", "B"])
    action = action_metrics(["AUTO_HANDLE", "ESCALATE"], ["AUTO_HANDLE", "AUTO_HANDLE"])
    assert intent["accuracy"] == 2 / 3
    assert len(intent["confusion_matrix"]["values"]) == 2
    assert action["accuracy"] == 0.5
    assert "macro_f1" in action


def test_mock_reply_generation_is_deterministic():
    assert MockProvider().generate("anything") == MockProvider().generate("different")


def test_end_to_end_evaluation_small_subset_and_output_schema(tmp_path: Path):
    predictions, results = evaluate_pipeline(limit=3, provider=MockProvider())
    required = {
        "customer_message", "amazon_reply", "expected_intent", "predicted_intent",
        "expected_action", "predicted_action", "generated_reply", "escalation_reason",
        "retrieval_similarity", "retrieved_examples", "intent_confidence",
    }
    assert len(predictions) == 3
    assert required.issubset(predictions.columns)
    assert results["dataset"]["golden_set_used_for_training"] is False
    assert all(predictions["generated_reply"].str.len() > 0)

    output_path = tmp_path / "predictions.csv"
    predictions.to_csv(output_path, index=False)
    assert pd.read_csv(output_path).shape[0] == 3


def test_evaluation_source_is_final_golden_set_only():
    source = Path("src/evaluation/evaluator.py").read_text(encoding="utf-8")
    assert "amazon_golden_set_final.csv" in source
    assert "amazon_golden_set_labeled.csv" not in source
    assert "amazon_golden_set (1).csv" not in source
    assert "train_classifier" not in source