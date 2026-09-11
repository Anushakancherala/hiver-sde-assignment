"""Create an AI-assisted draft label file for the existing Golden Set.

This is a deterministic annotation utility, not model training. It reads only
the existing Golden Set rows and applies the visible rules below to each
customer message. The generated labels are a draft and are not human labels.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = ("customer_message", "amazon_reply", "intent", "expected_action", "reason")
INTENTS = (
    "DELIVERY_ISSUE",
    "ORDER_ISSUE",
    "RETURN_REFUND",
    "PAYMENT_BILLING",
    "ACCOUNT_MEMBERSHIP",
    "PRODUCT_TECHNICAL",
    "PRODUCT_QUALITY",
    "SAFETY_ISSUE",
    "GENERAL_SUPPORT",
)
AUTO_HANDLE = "AUTO_HANDLE"
ESCALATE = "ESCALATE"

# These rules are checked in order. More specific/high-risk categories come
# first so that, for example, a fraud complaint is not treated as routine help.
INTENT_RULES = (
    (
        "SAFETY_ISSUE",
        re.compile(r"\b(fire|fired|smoke|burn|danger|dangerous|unsafe|hazard|injur|explod)\w*\b", re.I),
        "The message describes a potential safety incident.",
    ),
    (
        "PAYMENT_BILLING",
        re.compile(r"\b(fraud|scam|charge|charged|payment|bank|cashback|cash back|price|mrp|billing|invoice|amount|money)\w*\b", re.I),
        "The message concerns payment, billing, a charge, price, or money.",
    ),
    (
        "RETURN_REFUND",
        re.compile(r"\b(return|refund|refunded|reimburse|money back|pick.?up|send back|warranty)\w*\b", re.I),
        "The message concerns returning an item, a refund, or a post-purchase resolution.",
    ),
    (
        "ACCOUNT_MEMBERSHIP",
        re.compile(r"\b(password|log\s*in|login|account|membership|prime|sign\s*in|contactable|call|email|mail|customer care|support team)\w*\b", re.I),
        "The message concerns account access, membership, or contacting support.",
    ),
    (
        "PRODUCT_TECHNICAL",
        re.compile(r"\b(echo|alexa|kindle|tablet|router|app|website|browser|chrome|connection|connected|code|password|supported|work|working|error|setting|language)\w*\b", re.I),
        "The message asks about product, app, website, or device behavior.",
    ),
    (
        "PRODUCT_QUALITY",
        re.compile(r"\b(counterfeit|fake|wrong item|damaged|broken|defective|quality|size|missing|unwarranted|out of warranty)\w*\b", re.I),
        "The message concerns product condition, authenticity, fit, or quality.",
    ),
    (
        "DELIVERY_ISSUE",
        re.compile(r"\b(deliver|delivery|delivered|package|parcel|shipment|shipping|courier|carrier|arriv|received|receive|tracking|out for delivery|not here|doorstep|driver)\w*\b", re.I),
        "The message concerns delivery status, a carrier, or a missing/late package.",
    ),
    (
        "ORDER_ISSUE",
        re.compile(r"\b(order|ordered|ordering|cancel|cancellation|pre.?order|purchase|book|item|seller)\w*\b", re.I),
        "The message concerns an order, cancellation, purchase, or seller issue.",
    ),
)

HIGH_RISK_RULES = (
    (re.compile(r"\b(fire|fired|smoke|burn|danger|dangerous|unsafe|hazard|injur|explod)\w*\b", re.I), "Potential safety incident requires human review."),
    (re.compile(r"\b(threat|threaten|harm|hurt|attack|violence|weapon|kill)\w*\b", re.I), "Threat or harm language requires human review."),
    (re.compile(r"\b(fraud|scam|counterfeit|fake|unauthori[sz]ed|stolen|identity theft)\w*\b", re.I), "Possible fraud, counterfeit, or unauthorized activity requires human review."),
    (re.compile(r"\b(password|log\s*in|login|account)\w*\b.*\b(locked|incorrect|hacked|security|code|access)\b|\b(locked|hacked|security)\b.*\b(password|account|login)\b", re.I), "Account security or access concern requires human review."),
)

LOW_INFORMATION = re.compile(
    r"^\s*(?:@?amazonhelp\s*)?(?:ok(?:ay)?|thanks?|thank you|done|solved|all of them|\.com|https?://\S+)?\s*$",
    re.I,
)


def annotate_message(customer_message: str) -> tuple[str, str, str]:
    """Return deterministic draft intent, action, and explanation."""
    message = " ".join(str(customer_message).split())
    for high_risk_pattern, action_reason in HIGH_RISK_RULES:
        if high_risk_pattern.search(message):
            for intent, intent_pattern, intent_reason in INTENT_RULES:
                if intent_pattern.search(message):
                    return intent, ESCALATE, f"{intent_reason} {action_reason}"
            return "GENERAL_SUPPORT", ESCALATE, action_reason

    if LOW_INFORMATION.fullmatch(message):
        return "GENERAL_SUPPORT", ESCALATE, "The message contains too little issue detail for safe automatic handling."

    for intent, pattern, intent_reason in INTENT_RULES:
        if pattern.search(message):
            return intent, AUTO_HANDLE, f"{intent_reason} No explicit high-risk signal was found."

    return "GENERAL_SUPPORT", ESCALATE, "The issue is too ambiguous for safe automatic handling."


def create_labeled_golden_set(input_path: str | Path, output_path: str | Path) -> pd.DataFrame:
    """Label the existing rows without changing their source content."""
    source = pd.read_csv(input_path, keep_default_na=False)
    if list(source.columns) != list(REQUIRED_COLUMNS):
        raise ValueError(f"Expected columns exactly: {list(REQUIRED_COLUMNS)}")
    if len(source) != 200:
        raise ValueError(f"Expected exactly 200 Golden Set rows, found {len(source)}")

    labeled = source.copy()
    decisions = [annotate_message(message) for message in source["customer_message"]]
    labeled["intent"] = [decision[0] for decision in decisions]
    labeled["expected_action"] = [decision[1] for decision in decisions]
    labeled["reason"] = [decision[2] for decision in decisions]
    labeled.to_csv(output_path, index=False)
    return labeled


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("evaluation/amazon_golden_set (1).csv"))
    parser.add_argument("--output", type=Path, default=Path("evaluation/amazon_golden_set_labeled.csv"))
    args = parser.parse_args()

    labeled = create_labeled_golden_set(args.input, args.output)
    print(f"Rows written: {len(labeled)}")
    print("Intent counts:")
    print(labeled["intent"].value_counts().reindex(INTENTS, fill_value=0).to_string())
    print("Action counts:")
    print(labeled["expected_action"].value_counts().reindex([AUTO_HANDLE, ESCALATE], fill_value=0).to_string())
    print("Label provenance: AI-assisted deterministic draft; not manually labelled.")


if __name__ == "__main__":
    main()