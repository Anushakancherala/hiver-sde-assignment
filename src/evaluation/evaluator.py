"""End-to-end evaluation of the AmazonHelp support-agent pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from src.agent import (
    ESCALATION_RECOMMENDATION,
    decide_action,
    generate_reply,
)
from src.intents import INTENTS, predict_intent
from src.retrieval import build_retriever, retrieve_similar_conversations

from .metrics import action_metrics, intent_metrics


GOLDEN_SET_PATH = Path("evaluation/amazon_golden_set_final.csv")
TRAINING_DATA_PATH = Path("data/amazonhelp_training_labeled.csv")
HISTORICAL_PAIRS_PATH = Path("data/amazonhelp_conversation_pairs.csv")
MODEL_PATH = Path("models/intent_classifier.joblib")

REQUIRED_GOLDEN_COLUMNS = [
    "customer_message",
    "amazon_reply",
    "intent",
    "expected_action",
    "reason",
]


class MockProvider:
    """Deterministic provider used only for offline testing."""

    def generate(self, prompt: str) -> str:
        return (
            "Based on similar AmazonHelp cases, please contact "
            "Amazon customer support for assistance."
        )


def load_golden_set(
    path: str | Path = GOLDEN_SET_PATH,
) -> pd.DataFrame:
    """Load and validate the final reviewed Golden Set only."""

    data = pd.read_csv(path, keep_default_na=False)

    if list(data.columns) != REQUIRED_GOLDEN_COLUMNS:
        raise ValueError(
            f"Expected Golden Set columns: {REQUIRED_GOLDEN_COLUMNS}"
        )

    if len(data) != 200:
        raise ValueError(
            f"Expected 200 Golden Set rows, found {len(data)}"
        )

    if (
        data[REQUIRED_GOLDEN_COLUMNS]
        .apply(lambda column: column.astype(str).str.strip().eq(""))
        .any()
        .any()
    ):
        raise ValueError("Golden Set contains empty required values")

    return data


def _serialize_examples(
    examples: list[dict[str, Any]],
) -> str:
    """Serialize retrieved examples for storage in predictions.csv."""

    return json.dumps(examples, ensure_ascii=False)


def evaluate_pipeline(
    golden_set_path: str | Path = GOLDEN_SET_PATH,
    model_path: str | Path = MODEL_PATH,
    historical_pairs_path: str | Path = HISTORICAL_PAIRS_PATH,
    provider: Any | None = None,
    limit: int | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Run classification, retrieval, generation, and escalation decisions."""

    if provider is None:
        raise ValueError(
            "A reply-generation provider is required. "
            "Use MockProvider() for offline testing or OpenAIProvider() "
            "for real LLM evaluation."
        )

    golden_set = load_golden_set(golden_set_path)

    if limit is not None:
        golden_set = golden_set.head(limit).copy()

    classifier = joblib.load(model_path)
    retriever = build_retriever(historical_pairs_path)

    prediction_rows = []

    for _, row in golden_set.iterrows():
        customer_message = str(row["customer_message"])

        # 1. Predict intent
        predicted_intent = predict_intent(
            classifier,
            customer_message,
        )

        # 2. Retrieve similar historical conversations
        retrieved_examples = retrieve_similar_conversations(
            retriever,
            customer_message,
        )

        # 3. Generate grounded customer-facing reply
        generated_reply = generate_reply(
            customer_message,
            predicted_intent,
            retrieved_examples,
            provider,
        )

        # 4. Decide AUTO_HANDLE vs ESCALATE
        decision = decide_action(
            customer_message,
            predicted_intent,
            retrieved_examples,
            generated_reply,
        )

        similarities = [
            float(example["similarity"])
            for example in retrieved_examples
        ]

        confidence = None

        if hasattr(classifier, "predict_proba"):
            confidence = float(
                max(
                    classifier.predict_proba(
                        [customer_message]
                    )[0]
                )
            )

        prediction_rows.append(
            {
                "customer_message": customer_message,
                "amazon_reply": row["amazon_reply"],
                "expected_intent": row["intent"],
                "predicted_intent": predicted_intent,
                "expected_action": row["expected_action"],
                "predicted_action": decision["action"],
                "generated_reply": generated_reply,
                "escalation_reason": decision["reason"],
                "retrieval_similarity": max(
                    similarities,
                    default=0.0,
                ),
                "retrieved_examples": _serialize_examples(
                    retrieved_examples
                ),
                "intent_confidence": confidence,
                "reply_relevance": None,
                "reply_groundedness": None,
                "reply_correctness": None,
                "reply_helpfulness": None,
                "reply_brand_consistency": None,
                "reply_unsupported_claims": None,
            }
        )

    predictions = pd.DataFrame(prediction_rows)

    results = {
        "dataset": {
            "name": "AmazonHelp Golden Set (final reviewed)",
            "path": str(golden_set_path),
            "rows": len(predictions),
            "training_data": str(TRAINING_DATA_PATH),
            "historical_retrieval_data": str(
                historical_pairs_path
            ),
            "golden_set_used_for_training": False,
        },
        "intent_metrics": intent_metrics(
            predictions["expected_intent"].tolist(),
            predictions["predicted_intent"].tolist(),
            list(INTENTS),
        ),
        "action_metrics": action_metrics(
            predictions["expected_action"].tolist(),
            predictions["predicted_action"].tolist(),
        ),
        "reply_quality": {
            "status": "pending_llm_judge",
            "message": (
                "Reply quality will be evaluated separately "
                "using the LLM-as-judge pipeline."
            ),
        },
    }

    return predictions, results


def write_evaluation_outputs(
    predictions: pd.DataFrame,
    results: dict[str, Any],
    predictions_path: str | Path = "evaluation/predictions.csv",
    results_path: str | Path = "evaluation/results.json",
) -> None:
    """Write per-example predictions and aggregate results."""

    predictions.to_csv(
        predictions_path,
        index=False,
    )

    Path(results_path).write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )