"""Prepare a blank human-rating sheet from existing LLM judge examples."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


JUDGE_PATH = Path("evaluation/judge_scores.csv")
OUTPUT_PATH = Path("evaluation/human_judge_ratings.csv")
HUMAN_DIMENSIONS = (
    "relevance",
    "groundedness",
    "correctness",
    "helpfulness",
    "brand_consistency",
    "overall_score",
)
OUTPUT_COLUMNS = ("row_number", "customer_message", "generated_reply", *HUMAN_DIMENSIONS, "human_notes")
SOURCE_COLUMNS = ("row_number", "customer_message", "generated_reply")


def prepare_human_ratings(
    judge_path: str | Path = JUDGE_PATH,
    output_path: str | Path = OUTPUT_PATH,
) -> pd.DataFrame:
    """Copy judged examples while leaving every human field empty."""
    judge_scores = pd.read_csv(judge_path, keep_default_na=False)
    missing = set(SOURCE_COLUMNS) - set(judge_scores.columns)
    if missing:
        raise ValueError(f"Judge scores are missing required columns: {', '.join(sorted(missing))}")
    human_ratings = judge_scores.loc[:, list(SOURCE_COLUMNS)].copy()
    for dimension in (*HUMAN_DIMENSIONS, "human_notes"):
        human_ratings[dimension] = ""
    human_ratings = human_ratings.loc[:, list(OUTPUT_COLUMNS)]
    human_ratings.to_csv(output_path, index=False)
    return human_ratings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--judge-scores", type=Path, default=JUDGE_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args()
    ratings = prepare_human_ratings(args.judge_scores, args.output)
    print(f"Prepared {len(ratings)} examples for independent human review.")
    print("Human ratings remain empty and must be filled manually.")


if __name__ == "__main__":
    main()