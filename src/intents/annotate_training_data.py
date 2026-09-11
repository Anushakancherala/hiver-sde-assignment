"""Create a rule-assisted intent dataset from AmazonHelp conversations.

This module does not learn labels and does not read the Golden Set. It applies
the explicit keyword rules below to historical customer messages so that the
result can be inspected and revised before it is used for classifier training.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

from .classifier import INTENTS


SOURCE_COLUMNS = ("customer_message", "amazon_reply")
OUTPUT_COLUMNS = ("customer_message", "intent", "label_source", "confidence")
LABEL_SOURCE = "rule_assisted"


# Each expression is intentionally visible and editable. Rules are scored in
# their listed intent; the highest score wins, while ties are sent to review.
RULES: dict[str, tuple[tuple[str, int], ...]] = {
    "SAFETY_ISSUE": (
        (r"\b(unsafe|dangerous|hazard|fire|smoke|explod|electric shock|injur|hurt)\w*\b", 3),
    ),
    "DELIVERY_ISSUE": (
        (r"\b(not arrived|hasn't arrived|has not arrived|didn't arrive|wasn't delivered)\b", 3),
        (r"\b(deliver|delivery|delivered|shipping|shipment|tracking|courier|carrier|package|parcel|doorstep)\w*\b", 1),
        (r"\b(late|missing|where is my)\b", 1),
    ),
    "RETURN_REFUND": (
        (r"\b(return|refund|refunded|money back|reimburse|exchange)\w*\b", 2),
        (r"\b(send|sending)\s+(it|this|that)\s+back\b", 2),
    ),
    "PAYMENT_BILLING": (
        (r"\b(payment|billing|bill|charged|charge|credit card|debit card|payment method)\w*\b", 2),
        (r"\b(overcharged|double charged|wrong amount|invoice)\w*\b", 3),
    ),
    "ACCOUNT_MEMBERSHIP": (
        (r"\b(account|log in|login|sign in|password|membership|prime|subscription|household)\w*\b", 2),
        (r"\b(add|remove)\s+(an?\s+)?adult\b", 2),
    ),
    "PRODUCT_TECHNICAL": (
        (r"\b(echo|alexa|kindle|fire tv|device|app|wifi|bluetooth)\w*\b", 2),
        (r"\b(not working|won't work|wont work|doesn't work|does not work|not supported|troubleshoot|setup|connect)\b", 3),
    ),
    "PRODUCT_QUALITY": (
        (r"\b(broken|defective|damaged|faulty|poor quality|scratched|counterfeit|fake)\w*\b", 3),
        (r"\b(wrong item|missing parts|missing piece)\b", 2),
    ),
    "ORDER_ISSUE": (
        (r"\b(order|ordered|pre-?order|purchase|buy|bought|item)\w*\b", 1),
        (r"\b(cancel|cancellation|modify|change)\s+(my\s+)?order\b", 3),
        (r"\b(out of stock|availability|available)\b", 2),
    ),
    "GENERAL_SUPPORT": (
        (r"\b(help|support|question|information|info|contact|talk|speak)\w*\b", 1),
        (r"\b(thanks|thank you|hello|hi|okay|ok)\b", 1),
    ),
}


def _validate_rules() -> None:
    """Fail early if the editable rules drift from the classifier intents."""
    if set(RULES) != set(INTENTS):
        raise ValueError("Annotation rules must define exactly the classifier intents")


def _rule_scores(message: str) -> dict[str, int]:
    normalized = " ".join(str(message).lower().split())
    scores = {intent: 0 for intent in INTENTS}
    for intent, patterns in RULES.items():
        for pattern, weight in patterns:
            if re.search(pattern, normalized):
                scores[intent] += weight
    return scores


def annotate_message(message: str) -> tuple[str, str]:
    """Return a rule-based intent and confidence for one customer message."""
    scores = _rule_scores(message)
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    top_intent, top_score = ranked[0]
    second_score = ranked[1][1]

    if top_score == 0 or top_score == second_score:
        return "GENERAL_SUPPORT", "low"
    if top_score >= 3 and top_score - second_score >= 2:
        return top_intent, "high"
    return top_intent, "medium"


def create_training_dataset(input_path: str | Path, output_path: str | Path) -> pd.DataFrame:
    """Annotate historical messages and write the reproducible CSV output."""
    _validate_rules()
    source = pd.read_csv(input_path)
    missing_columns = set(SOURCE_COLUMNS) - set(source.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Missing required source column(s): {missing}")

    records = []
    for message in source["customer_message"]:
        if pd.isna(message) or not str(message).strip():
            raise ValueError("Source contains an empty customer_message")
        intent, confidence = annotate_message(str(message))
        records.append(
            {
                "customer_message": message,
                "intent": intent,
                "label_source": LABEL_SOURCE,
                "confidence": confidence,
            }
        )

    labeled = pd.DataFrame(records, columns=OUTPUT_COLUMNS)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    labeled.to_csv(output, index=False)
    return labeled


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/amazonhelp_conversation_pairs.csv"),
        help="Historical AmazonHelp conversation pairs CSV",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/amazonhelp_training_labeled.csv"),
        help="Output CSV for rule-assisted training examples",
    )
    args = parser.parse_args()

    labeled = create_training_dataset(args.input, args.output)
    print("Examples assigned to each intent:")
    for intent in INTENTS:
        print(f"  {intent}: {int((labeled['intent'] == intent).sum())}")
    low_confidence = int((labeled["confidence"] == "low").sum())
    print(f"Low-confidence examples to review: {low_confidence}")
    print(f"Saved {len(labeled)} examples to {args.output}")


if __name__ == "__main__":
    main()