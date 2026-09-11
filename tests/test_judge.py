"""Offline tests for LLM-as-judge evaluation."""

import json

import pandas as pd
import pytest

from src.evaluation.judge import (
    JUDGE_OUTPUT_COLUMNS,
    MockJudge,
    build_judge_prompt,
    judge_predictions,
    parse_judge_response,
)


def sample_prediction():
    return {
        "customer_message": "Where is my package?",
        "generated_reply": "Please contact Amazon customer support.",
        "predicted_intent": "DELIVERY_ISSUE",
        "predicted_action": "AUTO_HANDLE",
        "escalation_reason": "The reply is grounded in useful historical evidence.",
        "retrieval_similarity": 0.8,
        "retrieved_examples": '[{"customer_message":"My package is late","amazon_reply":"Please contact support.","similarity":0.8}]',
    }


def test_judge_prompt_contains_allowed_inputs_but_not_ground_truth():
    prompt = build_judge_prompt(sample_prediction())

    assert "Where is my package?" in prompt
    assert "My package is late" in prompt
    assert "expected_intent" not in prompt
    assert "expected_action" not in prompt
    assert "Ground-truth" not in prompt


def test_judge_json_parsing_and_score_validation():
    response = json.dumps({
        "relevance": 4, "groundedness": 3, "correctness": 4,
        "helpfulness": 3, "brand_consistency": 5,
        "unsupported_claims": False, "overall_score": 4, "brief_reason": "Good.",
    })
    result = parse_judge_response(response)

    assert set(result) == set(JUDGE_OUTPUT_COLUMNS)
    with pytest.raises(ValueError):
        parse_judge_response(response.replace('"relevance": 4', '"relevance": 6'))


def test_mock_judge_is_deterministic_and_output_schema_is_valid():
    scores, results = judge_predictions(pd.DataFrame([sample_prediction()] * 2), MockJudge(), sample_size=2)

    assert list(scores.columns) == ["row_number", "customer_message", "generated_reply", *JUDGE_OUTPUT_COLUMNS]
    assert results["judged_examples"] == 2
    assert results["mode"] if "mode" in results else True
    assert scores["overall_score"].tolist() == [3, 3]


def test_judge_input_has_no_ground_truth_columns():
    source = build_judge_prompt(sample_prediction())
    for forbidden in ("expected_intent", "expected_action"):  # escalation_reason is allowed
        assert forbidden not in source