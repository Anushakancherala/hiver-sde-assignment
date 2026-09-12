"""Command-line runner for the complete AmazonHelp evaluation pipeline."""

from __future__ import annotations

import argparse

from src.agent import OpenAIProvider

from .evaluator import MockProvider, evaluate_pipeline, write_evaluation_outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mock", action="store_true", help="Use deterministic offline reply generation")
    parser.add_argument("--limit", type=int, default=None, help="Evaluate only the first N Golden Set rows")
    args = parser.parse_args()

    provider = MockProvider() if args.mock else OpenAIProvider()
    predictions, results = evaluate_pipeline(provider=provider, limit=args.limit)
    write_evaluation_outputs(predictions, results)
    print(f"Evaluated Golden Set rows: {len(predictions)}")
    print(f"Intent accuracy: {results['intent_metrics']['accuracy']:.4f}")
    print(f"Intent macro F1: {results['intent_metrics']['macro_f1']:.4f}")
    print(f"Action accuracy: {results['action_metrics']['accuracy']:.4f}")
    print(f"Action F1: {results['action_metrics']['macro_f1']:.4f}")
    print("Created evaluation/predictions.csv and evaluation/results.json")


if __name__ == "__main__":
    main()