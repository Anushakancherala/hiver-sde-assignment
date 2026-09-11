"""LLM-as-judge utilities for reply-quality evaluation."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

from src.agent import LLMProvider


JUDGE_INPUT_COLUMNS = (
    "customer_message",
    "generated_reply",
    "predicted_intent",
    "predicted_action",
    "escalation_reason",
    "retrieval_similarity",
    "retrieved_examples",
)
JUDGE_OUTPUT_COLUMNS = (
    "relevance",
    "groundedness",
    "correctness",
    "helpfulness",
    "brand_consistency",
    "unsupported_claims",
    "overall_score",
    "brief_reason",
)
QUALITY_DIMENSIONS = (
    "relevance",
    "groundedness",
    "correctness",
    "helpfulness",
    "brand_consistency",
)


def build_judge_prompt(row: dict[str, Any]) -> str:
    """Build a judge prompt without exposing evaluation ground truth."""
    missing = set(JUDGE_INPUT_COLUMNS) - set(row)
    if missing:
        raise ValueError(f"Missing judge input field(s): {', '.join(sorted(missing))}")
    return f"""You are evaluating the quality of an AmazonHelp customer-support reply.

Score each quality dimension from 1 (poor) to 5 (excellent): relevance,
groundedness, correctness, helpfulness, and brand_consistency. Set
unsupported_claims to true if the reply makes a claim not supported by the
customer message or historical evidence. Calculate overall_score as your
overall 1-5 quality score and give a brief reason.

Customer message:
{row['customer_message']}

Generated reply:
{row['generated_reply']}

Predicted intent:
{row['predicted_intent']}

Predicted action:
{row['predicted_action']}

Escalation reason:
{row['escalation_reason']}

Top retrieval similarity:
{row['retrieval_similarity']}

Retrieved historical evidence (customer messages and AmazonHelp replies):
{row['retrieved_examples']}

Return only valid JSON with exactly these keys:
relevance, groundedness, correctness, helpfulness, brand_consistency,
unsupported_claims, overall_score, brief_reason."""


def parse_judge_response(response: str) -> dict[str, Any]:
    """Parse and strictly validate one structured judge response."""
    text = response.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1]).strip()
    try:
        result = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError("Judge response must be valid JSON") from exc
    if set(result) != set(JUDGE_OUTPUT_COLUMNS):
        raise ValueError("Judge JSON has unexpected or missing keys")
    for dimension in (*QUALITY_DIMENSIONS, "overall_score"):
        value = result[dimension]
        if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 5:
            raise ValueError(f"{dimension} must be an integer from 1 to 5")
    if not isinstance(result["unsupported_claims"], bool):
        raise ValueError("unsupported_claims must be boolean")
    if not isinstance(result["brief_reason"], str) or not result["brief_reason"].strip():
        raise ValueError("brief_reason must be a non-empty string")
    return result


class MockJudge:
    """Deterministic offline judge for tests only."""

    def generate(self, prompt: str) -> str:
        return json.dumps(
            {
                "relevance": 3,
                "groundedness": 3,
                "correctness": 3,
                "helpfulness": 3,
                "brand_consistency": 4,
                "unsupported_claims": False,
                "overall_score": 3,
                "brief_reason": "Deterministic mock score; not a final quality assessment.",
            }
        )


def load_prediction_input(path: str | Path = "evaluation/predictions.csv") -> pd.DataFrame:
    """Load the existing predictions artifact and validate judge inputs."""
    predictions = pd.read_csv(path, keep_default_na=False)
    missing = set(JUDGE_INPUT_COLUMNS) - set(predictions.columns)
    if missing:
        raise ValueError(f"Predictions are missing judge input field(s): {', '.join(sorted(missing))}")
    return predictions


def judge_predictions(
    predictions: pd.DataFrame,
    provider: LLMProvider,
    sample_size: int = 40,
    random_state: int = 42,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Judge a deterministic sample of predictions."""
    if sample_size < 1 or sample_size > len(predictions):
        raise ValueError("sample_size must be between 1 and the number of predictions")
    sample = predictions.sample(n=sample_size, random_state=random_state).sort_index()
    scored_rows = []
    for index, row in sample.iterrows():
        judge_result = parse_judge_response(provider.generate(build_judge_prompt(row.to_dict())))
        scored_rows.append(
            {
                "row_number": int(row.get("row_number", index + 1)),
                "customer_message": row["customer_message"],
                "generated_reply": row["generated_reply"],
                **judge_result,
            }
        )
    scores = pd.DataFrame(scored_rows)
    results = summarize_judge_scores(scores)
    return scores, results


def summarize_judge_scores(scores: pd.DataFrame) -> dict[str, Any]:
    """Summarize validated judge scores for JSON output."""
    means = {dimension: float(scores[dimension].mean()) for dimension in QUALITY_DIMENSIONS}
    means["overall_score"] = float(scores["overall_score"].mean())
    distribution = {
        dimension: {str(score): int((scores[dimension] == score).sum()) for score in range(1, 6)}
        for dimension in (*QUALITY_DIMENSIONS, "overall_score")
    }
    return {
        "judged_examples": len(scores),
        "mean_scores": means,
        "overall_mean_score": means["overall_score"],
        "unsupported_claims_percentage": float(scores["unsupported_claims"].mean() * 100),
        "score_distribution": distribution,
    }