"""Run LLM-as-judge reply-quality evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.agent import OpenAIProvider

from .judge import MockJudge, judge_predictions, load_prediction_input


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mock", action="store_true", help="Use deterministic offline scores for testing")
    parser.add_argument("--sample-size", type=int, default=40)
    parser.add_argument("--predictions", type=Path, default=Path("evaluation/predictions.csv"))
    parser.add_argument("--output", type=Path, default=Path("evaluation/judge_scores.csv"))
    parser.add_argument("--results", type=Path, default=Path("evaluation/judge_results.json"))
    args = parser.parse_args()

    predictions = load_prediction_input(args.predictions)
    provider = MockJudge() if args.mock else OpenAIProvider()
    scores, results = judge_predictions(predictions, provider, sample_size=args.sample_size)
    if args.mock:
        results["mode"] = "mock_only_not_final_quality"
    else:
        results["mode"] = "llm_judge"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.results.parent.mkdir(parents=True, exist_ok=True)
    scores.to_csv(args.output, index=False)
    args.results.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Judged examples: {results['judged_examples']}")
    print(f"Overall mean score: {results['overall_mean_score']:.2f}")
    print(f"Unsupported claims: {results['unsupported_claims_percentage']:.1f}%")
    print(f"Created {args.output} and {args.results}")


if __name__ == "__main__":
    main()