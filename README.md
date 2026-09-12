# Hiver SDE Assignment

This repository contains a lightweight support-agent pipeline for the AmazonHelp-style customer message task. The project is intentionally simple and auditable: it uses a TF-IDF + logistic regression intent model, historical retrieval, a grounded reply generator, and a rule-based escalation policy.

## Setup

1. Create a Python environment and install dependencies:

   ```bash
   python -m pip install -r requirements.txt
   ```

2. If you want to use the real OpenAI-compatible provider, copy the example environment file and fill in your key:

   ```bash
   cp .env.example .env
   ```

   Then edit `.env` and set `OPENAI_API_KEY` if you plan to run non-mock LLM calls.

## Dataset preparation

The training data and evaluation data are already in the repository:

- `data/amazonhelp_training_labeled.csv` — historical training data used for intent training
- `evaluation/amazon_golden_set_final.csv` — final reviewed Golden Set used for evaluation only
- `data/amazonhelp_conversation_pairs.csv` — historical retrieval pairs

Important: the project explicitly does not use the Golden Set for training. The evaluation files record `golden_set_used_for_training: false`.

## Train the intent classifier

From the repo root:

```bash
python -m src.intents.train_and_evaluate
```

This trains the model and saves it to:

```text
models/intent_classifier.joblib
```

## Run the agent in mock mode

The repo includes a deterministic mock provider for offline runs:

```bash
python -m src.evaluation.run_evaluation --mock
```

This writes:

- `evaluation/predictions.csv`
- `evaluation/results.json`

The mock mode is a safe, deterministic evaluation path for local testing and demos.

## Run the agent with a real LLM provider

If `OPENAI_API_KEY` is set in `.env` or the environment, you can run the real provider path without `--mock`:

```bash
python -m src.evaluation.run_evaluation
```

This will attempt to call the configured OpenAI-compatible API.

## Baselines

To reproduce the baseline comparison against the final Golden Set:

```bash
python -m src.evaluation.run_baselines
```

This creates:

- `evaluation/baseline_predictions.csv`
- `evaluation/baseline_results.json`

## LLM-as-judge evaluation

A judge pass is available for reply-quality scoring. The mock mode is the safe default:

```bash
python -m src.evaluation.run_judge --mock --sample-size 40
```

This creates:

- `evaluation/judge_scores.csv`
- `evaluation/judge_results.json`

The project explicitly labels mock judge output as `mock_only_not_final_quality`.

## Human agreement workflow

The repository prepares a blank human review sheet:

```bash
python -m src.evaluation.prepare_human_judge
```

This creates `evaluation/human_judge_ratings.csv` for manual review. After the ratings are filled in, agreement can be computed with:

```bash
python -m src.evaluation.human_agreement
```

The script validates that all human ratings are complete before producing agreement statistics.

## Reproduce the key headline results quickly

Under about 15 minutes on a standard local setup:

```bash
python -m pip install -r requirements.txt
python -m src.intents.train_and_evaluate
python -m src.evaluation.run_baselines
python -m src.evaluation.run_evaluation --mock
python -m src.evaluation.run_judge --mock --sample-size 40
```

This reproduces the repository’s existing evaluation outputs without claiming that the judge score is a real LLM-grade benchmark.

## Honesty about the data and judge output

- The final reviewed Golden Set is used for evaluation only; it is not treated as a manually labeled source for training.
- The project contains AI-assisted Golden Set preparation steps, but the final evaluation uses the final reviewed file in `evaluation/amazon_golden_set_final.csv`.
- The judge output in `evaluation/judge_results.json` is a mock run, not a final LLM-quality certificate.

## Notes

- The repository is intentionally simple and deterministic.
- Most offline work is designed to run without a live API key.
- If you want a real LLM evaluation, set `OPENAI_API_KEY` and run the non-mock path.
