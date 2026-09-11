"""Tests for the baseline training and evaluation script."""

import csv
from pathlib import Path

from src.intents.train_and_evaluate import train_and_evaluate


def test_train_and_evaluate_uses_stratified_reproducible_split(tmp_path: Path):
    data_path = tmp_path / "training.csv"
    model_path = tmp_path / "model.joblib"
    rows = []
    for index in range(9):
        intent = (
            "DELIVERY_ISSUE",
            "ORDER_ISSUE",
            "RETURN_REFUND",
            "PAYMENT_BILLING",
            "ACCOUNT_MEMBERSHIP",
            "PRODUCT_TECHNICAL",
            "PRODUCT_QUALITY",
            "SAFETY_ISSUE",
            "GENERAL_SUPPORT",
        )[index]
        for example in range(3):
            rows.append(f"message {index} {example}, {intent}, rule_assisted, high")
    with data_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(["customer_message", "intent", "label_source", "confidence"])
        writer.writerows(row.split(", ") for row in rows)

    first = train_and_evaluate(data_path, model_path, test_size=1 / 3)
    second = train_and_evaluate(data_path, tmp_path / "model-2.joblib", test_size=1 / 3)

    assert model_path.exists()
    assert first["metrics"] == second["metrics"]
    assert first["test_labels"] == second["test_labels"]
    assert set(first["metrics"]) == {
        "accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
    }