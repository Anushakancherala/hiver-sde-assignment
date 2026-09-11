# Decision log

This log records engineering and evaluation decisions that are already present in the project. Each item reflects a real trade-off in the repository rather than a hypothetical design.

## 1) Keep the Golden Set out of training

- Decision: The final reviewed Golden Set is used only for final scoring and never for model training.
- Why: The evaluation files explicitly record `golden_set_used_for_training: false`, and the training script reads only `data/amazonhelp_training_labeled.csv`.
- Alternative considered: Use the Golden Set in the train/test split or for iterative tuning.
- Trade-off: This avoids leakage, but it reduces the amount of labeled data available for optimization and makes the task harder.

## 2) Validate the Golden Set schema and row count before scoring

- Decision: The evaluator enforces exact column names and a 200-row expectation.
- Why: `load_golden_set` checks `customer_message`, `amazon_reply`, `intent`, `expected_action`, and `reason`, and raises an error if the file is malformed or has the wrong size.
- Alternative considered: Trust the file and proceed.
- Trade-off: Early failure is safer than silently scoring the wrong dataset, but strict validation can block experiments if the data is slightly inconsistent.

## 3) Save the trained intent model to a fixed path

- Decision: The classifier is saved to `models/intent_classifier.joblib`.
- Why: The training script and evaluation pipeline both load the same model artifact from a stable path.
- Alternative considered: Keep the model only in memory or save under a timestamped name.
- Trade-off: Reproducibility improves, while the repo becomes slightly more dependent on the model artifact existing at a fixed location.

## 4) Use stratified train/test splits with a fixed random seed

- Decision: The training script uses `train_test_split(..., stratify=labels, random_state=42)`.
- Why: This preserves label proportions in the train/test split and makes the split deterministic.
- Alternative considered: A random split without stratification or a different random seed.
- Trade-off: Better label coverage in evaluation, but the measured result is tied to one specific split rather than a broader cross-validation estimate.

## 5) Use zero-division-safe metric calculation in sparse classes

- Decision: The reporting code calls precision/recall/F1 with `zero_division=0`.
- Why: This prevents divide-by-zero failures when a class is not predicted in a fold or evaluation run.
- Alternative considered: Let the metric library raise or drop missing classes.
- Trade-off: Zero scores are explicit and stable, but they can look harsher than a model would if the class were absent from the evaluation set.

## 6) Keep baseline comparison separate from the agent evaluation

- Decision: Baselines are run through `src/evaluation/run_baselines.py`, and the main evaluation path is separate.
- Why: The repo computes baseline results and existing-agent results in distinct outputs (`baseline_results.json` vs `results.json`).
- Alternative considered: Fold baselines into the main evaluation script only.
- Trade-off: Clearer reporting, but more code paths to maintain.

## 7) Keep the mock provider for offline and test-safe execution

- Decision: `MockProvider` returns a deterministic canned reply string and is used in offline evaluation and tests.
- Why: This is the default used by `run_evaluation.py --mock` and by the evaluation harness when no real provider is desired.
- Alternative considered: Force all runs to call an external LLM.
- Trade-off: Deterministic behavior makes testing and demos easy, but it limits realism and prevents a true LLM-quality evaluation without setup.

## 8) Use an environment-based OpenAI client only when a real provider is needed

- Decision: `OpenAIProvider` reads `OPENAI_API_KEY`, `OPENAI_MODEL`, and `OPENAI_BASE_URL` from the environment.
- Why: The project includes `.env.example` and a provider that raises if the API key is missing.
- Alternative considered: Hard-code the API key or a fixed model into the code.
- Trade-off: This is portable and easy to configure, but it requires the user to manage environment variables correctly.

## 9) Keep retrieval evidence separate from Golden Set data

- Decision: The retrieval layer uses `data/amazonhelp_conversation_pairs.csv`, not the Golden Set.
- Why: The evaluation metadata mentions `historical_retrieval_data` and the code loads historical conversation pairs separately from the Golden Set.
- Alternative considered: Include Golden Set rows in retrieval memory.
- Trade-off: This keeps the evaluation fairer and reduces leakage risk, but it limits the system to historical support pairs rather than using final labels as retrieval evidence.

## 10) Use a deterministic mock LLM judge, clearly labeled as non-final

- Decision: The judge script supports `--mock` and sets `results["mode"] = "mock_only_not_final_quality"` when that mode is used.
- Why: This matches the repository’s honesty requirement: the mock judge is useful for debugging, but it is not a final quality benchmark.
- Alternative considered: Report the mock scores as final judge quality.
- Trade-off: This prevents over-claiming, but it means the repository cannot claim a real judge score unless a non-mock run is performed.

## 11) Prepare human review rows but refuse agreement until the file is complete

- Decision: `human_agreement.py` validates that every human rating is filled before computing agreement.
- Why: It raises an error if any human value is missing or invalid; this avoids fake agreement numbers.
- Alternative considered: Fill missing data with defaults or compute partial agreement.
- Trade-off: The system becomes safer and more honest, but human-review workflows must be completed manually before final agreement statistics can be computed.

## 12) Treat reply-quality metrics as placeholders until judge output is explicitly scored

- Decision: The end-to-end evaluator writes a `reply_quality` section with status `placeholder`.
- Why: The code explicitly says the reply-quality fields are prepared for later judge scoring, rather than claiming they have already been scored.
- Alternative considered: Fill in reply-quality metrics immediately during pipeline execution.
- Trade-off: This reduces false confidence, but it leaves the final quality signal incomplete until the judge step runs.

## 13) Keep the final scored outputs in CSV and JSON formats

- Decision: Predicted rows and aggregate summaries are written to both `evaluation/predictions.csv` and `evaluation/results.json` (and similarly for baselines and judge output).
- Why: This supports review, reproducibility, and downstream analysis without depending on a notebook or hidden state.
- Alternative considered: Keep only console logs or only one output format.
- Trade-off: More files to inspect, but the outputs are easier to compare and debug across runs.

## 14) Make the repository reproducible from a standard Python environment

- Decision: The project uses a simple requirements file (`scikit-learn`, `pandas`, `openpyxl`, `pytest`) and standard `python -m` commands.
- Why: This keeps setup lightweight and compatible with the code paths used for training and evaluation.
- Alternative considered: Use a larger framework or a containerized workflow.
- Trade-off: Simpler setup and easier inspection, but less operational complexity than a full stack deployment environment.
