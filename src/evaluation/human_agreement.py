"""Calculate agreement between human and LLM reply-quality ratings."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.metrics import cohen_kappa_score


HUMAN_RATINGS_PATH = Path("evaluation/human_judge_ratings.csv")
LLM_SCORES_PATH = Path("evaluation/judge_scores.csv")
RESULTS_PATH = Path("evaluation/human_agreement_results.json")
DIMENSIONS = (
    "relevance",
    "groundedness",
    "correctness",
    "helpfulness",
    "brand_consistency",
    "overall_score",
)
REQUIRED_HUMAN_COLUMNS = ("row_number", "customer_message", "generated_reply", *DIMENSIONS, "human_notes")
LLM_SCORE_COLUMNS = ("row_number", *DIMENSIONS)


def _validated_ratings(path: str | Path) -> pd.DataFrame:
    ratings = pd.read_csv(path, keep_default_na=False)
    missing = set(REQUIRED_HUMAN_COLUMNS) - set(ratings.columns)
    if missing:
        raise ValueError(f"Human ratings are missing columns: {', '.join(sorted(missing))}")
    for dimension in DIMENSIONS:
        values = ratings[dimension].astype(str).str.strip()
        if values.eq("").any():
            raise ValueError(
                f"Human ratings are incomplete: fill every {dimension} value before calculating agreement"
            )
        numeric = pd.to_numeric(values, errors="coerce")
        if numeric.isna().any() or ((numeric < 1) | (numeric > 5) | (numeric % 1 != 0)).any():
            raise ValueError(f"Human {dimension} ratings must be integer values from 1 to 5")
        ratings[dimension] = numeric.astype(int)
    return ratings


def _correlation(human: pd.Series, llm: pd.Series, method: str) -> float | None:
    if human.nunique() < 2 or llm.nunique() < 2:
        return None
    return float(human.corr(llm, method=method))


def calculate_agreement(
    human_path: str | Path = HUMAN_RATINGS_PATH,
    llm_path: str | Path = LLM_SCORES_PATH,
) -> dict[str, Any]:
    """Calculate agreement only after complete valid human ratings exist."""
    human = _validated_ratings(human_path)
    llm = pd.read_csv(llm_path, keep_default_na=False)
    missing = set(LLM_SCORE_COLUMNS) - set(llm.columns)
    if missing:
        raise ValueError(f"LLM scores are missing columns: {', '.join(sorted(missing))}")
    if len(human) != len(llm) or not human["row_number"].equals(llm["row_number"]):
        raise ValueError("Human ratings must contain the same judged examples in the same order as LLM scores")

    exact = {}
    within_one = {}
    correlations = {}
    for dimension in DIMENSIONS:
        llm_values = pd.to_numeric(llm[dimension], errors="coerce")
        if llm_values.isna().any() or ((llm_values < 1) | (llm_values > 5)).any():
            raise ValueError(f"LLM {dimension} ratings must be between 1 and 5")
        human_values = human[dimension]
        exact[dimension] = float((human_values == llm_values).mean() * 100)
        within_one[dimension] = float(((human_values - llm_values).abs() <= 1).mean() * 100)
        correlations[dimension] = {
            "pearson": _correlation(human_values, llm_values, "pearson"),
            "spearman": _correlation(human_values, llm_values, "spearman"),
        }

    return {
        "examples": len(human),
        "exact_agreement_percentage": exact,
        "within_one_agreement_percentage": within_one,
        "correlation": correlations,
        "overall_score_agreement": {
            "weighted_kappa": float(
                cohen_kappa_score(human["overall_score"], llm["overall_score"], weights="quadratic")
            ),
            "note": "Quadratic weighted Cohen's kappa; higher values indicate stronger agreement.",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--human-ratings", type=Path, default=HUMAN_RATINGS_PATH)
    parser.add_argument("--llm-scores", type=Path, default=LLM_SCORES_PATH)
    parser.add_argument("--output", type=Path, default=RESULTS_PATH)
    args = parser.parse_args()
    results = calculate_agreement(args.human_ratings, args.llm_scores)
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Calculated agreement for {results['examples']} examples.")


if __name__ == "__main__":
    main()