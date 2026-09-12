"""Evaluate trivial and simple baselines against the final Golden Set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.evaluation.evaluator import GOLDEN_SET_PATH, load_golden_set
from src.evaluation.metrics import action_metrics, intent_metrics
from src.intents import INTENTS

from .baselines import TRAINING_DATA_PATH, predict_simple, predict_trivial


OUTPUT_RESULTS = Path("evaluation/baseline_results.json")
OUTPUT_PREDICTIONS = Path("evaluation/baseline_predictions.csv")


def run_baseline_evaluation(
    golden_set_path: str | Path = GOLDEN_SET_PATH,
    training_path: str | Path = TRAINING_DATA_PATH,
) -> tuple[pd.DataFrame, dict]:
    """Evaluate both baselines using the Golden Set only for final scoring."""
    golden = load_golden_set(golden_set_path)
    messages = golden["customer_message"].tolist()

    trivial_intents, trivial_actions = predict_trivial(
        messages, training_path
    )
    simple_intents, simple_actions = predict_simple(
        messages, training_path
    )

    systems = {
        "trivial_baseline": (trivial_intents, trivial_actions),
        "simple_baseline": (simple_intents, simple_actions),
    }

    prediction_rows = []

    results = {
        "dataset": {
            "name": "AmazonHelp Golden Set (final reviewed)",
            "path": str(golden_set_path),
            "rows": len(golden),
            "training_data": str(training_path),
            "golden_set_used_for_training": False,
        },
        "systems": {},
    }

    for system_name, (predicted_intents, predicted_actions) in systems.items():
        results["systems"][system_name] = {
            "intent_metrics": intent_metrics(
                golden["intent"].tolist(),
                predicted_intents,
                list(INTENTS),
            ),
            "action_metrics": action_metrics(
                golden["expected_action"].tolist(),
                predicted_actions,
            ),
        }

    for index, row in golden.iterrows():
        prediction_rows.append(
            {
                "row_number": index + 1,
                "customer_message": row["customer_message"],
                "expected_intent": row["intent"],
                "trivial_predicted_intent": trivial_intents[index],
                "simple_predicted_intent": simple_intents[index],
                "expected_action": row["expected_action"],
                "trivial_predicted_action": trivial_actions[index],
                "simple_predicted_action": simple_actions[index],
            }
        )

    predictions = pd.DataFrame(prediction_rows)

    return predictions, results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--golden-set",
        type=Path,
        default=GOLDEN_SET_PATH,
    )

    parser.add_argument(
        "--training-data",
        type=Path,
        default=TRAINING_DATA_PATH,
    )

    args = parser.parse_args()

    predictions, results = run_baseline_evaluation(
        args.golden_set,
        args.training_data,
    )

    # Require existing-agent evaluation results so the
    # baseline comparison is complete on a clean checkout.
    current_results = Path("evaluation/results.json")

    if not current_results.exists():
        raise FileNotFoundError(
            "evaluation/results.json not found. "
            "Run `python -m src.evaluation.run_evaluation --mock` first."
        )

    results["existing_agent"] = json.loads(
        current_results.read_text(encoding="utf-8")
    )

    predictions.to_csv(
        OUTPUT_PREDICTIONS,
        index=False,
    )

    OUTPUT_RESULTS.write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )

    for system_name, system_results in results["systems"].items():
        print(
            f"{system_name}: "
            f"intent accuracy="
            f"{system_results['intent_metrics']['accuracy']:.4f}, "
            f"intent macro F1="
            f"{system_results['intent_metrics']['macro_f1']:.4f}, "
            f"action accuracy="
            f"{system_results['action_metrics']['accuracy']:.4f}, "
            f"action F1="
            f"{system_results['action_metrics']['f1']:.4f}"
        )

    print(
        "Created evaluation/baseline_predictions.csv "
        "and evaluation/baseline_results.json"
    )


if __name__ == "__main__":
    main()