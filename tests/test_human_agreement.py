"""Tests for human-review preparation and agreement calculations."""

import json
from pathlib import Path

import pandas as pd
import pytest

from src.evaluation.human_agreement import calculate_agreement
from src.evaluation.prepare_human_judge import OUTPUT_COLUMNS, prepare_human_ratings


def make_judge_scores(path: Path) -> None:
    pd.DataFrame(
        [
            {"row_number": 2, "customer_message": "One", "generated_reply": "Reply one", "relevance": 4, "groundedness": 4, "correctness": 3, "helpfulness": 4, "brand_consistency": 5, "unsupported_claims": False, "overall_score": 4, "brief_reason": "ok"},
            {"row_number": 7, "customer_message": "Two", "generated_reply": "Reply two", "relevance": 2, "groundedness": 3, "correctness": 2, "helpfulness": 2, "brand_consistency": 4, "unsupported_claims": True, "overall_score": 2, "brief_reason": "weak"},
        ]
    ).to_csv(path, index=False)


def test_prepare_human_rating_schema_excludes_llm_scores(tmp_path: Path):
    judge_path = tmp_path / "judge.csv"
    output_path = tmp_path / "human.csv"
    make_judge_scores(judge_path)

    ratings = prepare_human_ratings(judge_path, output_path)

    assert ratings.columns.tolist() == list(OUTPUT_COLUMNS)
    assert len(ratings) == 2
    assert ratings["relevance"].eq("").all()
    assert "unsupported_claims" not in ratings.columns
    assert "brief_reason" not in ratings.columns


def test_empty_human_ratings_are_rejected_without_results(tmp_path: Path):
    judge_path = tmp_path / "judge.csv"
    human_path = tmp_path / "human.csv"
    make_judge_scores(judge_path)
    prepare_human_ratings(judge_path, human_path)

    with pytest.raises(ValueError, match="incomplete"):
        calculate_agreement(human_path, judge_path)
    assert not (tmp_path / "results.json").exists()


def test_agreement_calculation_and_correlations(tmp_path: Path):
    judge_path = tmp_path / "judge.csv"
    human_path = tmp_path / "human.csv"
    make_judge_scores(judge_path)
    human = pd.DataFrame(
        {
            "row_number": [2, 7],
            "customer_message": ["One", "Two"],
            "generated_reply": ["Reply one", "Reply two"],
            "relevance": [4, 2],
            "groundedness": [4, 2],
            "correctness": [3, 2],
            "helpfulness": [5, 2],
            "brand_consistency": [5, 4],
            "overall_score": [4, 2],
            "human_notes": ["good", "needs work"],
        }
    )
    human.to_csv(human_path, index=False)

    results = calculate_agreement(human_path, judge_path)

    assert results["examples"] == 2
    assert results["exact_agreement_percentage"]["relevance"] == 100.0
    assert results["within_one_agreement_percentage"]["helpfulness"] == 100.0
    assert "pearson" in results["correlation"]["overall_score"]
    assert "weighted_kappa" in results["overall_score_agreement"]


def test_invalid_human_rating_is_rejected(tmp_path: Path):
    judge_path = tmp_path / "judge.csv"
    human_path = tmp_path / "human.csv"
    make_judge_scores(judge_path)
    prepare_human_ratings(judge_path, human_path)
    human = pd.read_csv(human_path, keep_default_na=False)
    for dimension in ("relevance", "groundedness", "correctness", "helpfulness", "brand_consistency", "overall_score"):
        human[dimension] = 3
    human.loc[0, "relevance"] = 6
    human.to_csv(human_path, index=False)

    with pytest.raises(ValueError, match="1 to 5"):
        calculate_agreement(human_path, judge_path)