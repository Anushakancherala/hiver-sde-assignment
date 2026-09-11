"""Deterministic policy for deciding between auto-handling and escalation."""

from __future__ import annotations

import re
from typing import Any

from .reply_generator import ESCALATION_RECOMMENDATION


AUTO_HANDLE = "AUTO_HANDLE"
ESCALATE = "ESCALATE"

# These values are deliberately centralized so the policy is easy to tune.
MINIMUM_RETRIEVAL_SIMILARITY = 0.20
HIGH_RETRIEVAL_SIMILARITY = 0.50

HIGH_RISK_INTENTS = {"SAFETY_ISSUE"}
HIGH_RISK_PATTERNS = (
    re.compile(r"\b(threat|threaten|kill|hurt|harm|attack|violence|weapon)\w*\b"),
    re.compile(r"\b(unauthori[sz]ed|fraud|scam|stolen|identity theft|account hacked)\b"),
    re.compile(r"\b(security breach|password stolen|someone accessed my account)\b"),
    re.compile(r"\b(unknown|unrecognized|unrecognised|fraudulent)\s+(charge|payment|purchase)\b"),
)


def _has_high_risk_message(message: str) -> bool:
    normalized = " ".join(message.lower().split())
    return any(pattern.search(normalized) for pattern in HIGH_RISK_PATTERNS)


def _top_similarity(retrieved_examples: list[dict[str, Any]]) -> float:
    scores = [
        float(example.get("similarity", 0))
        for example in retrieved_examples
        if example.get("customer_message") and example.get("amazon_reply")
    ]
    return max(scores, default=0.0)


def _decision(action: str, reason: str, confidence: str) -> dict[str, str]:
    return {"action": action, "reason": reason, "confidence": confidence}


def decide_action(
    customer_message: str,
    intent: str,
    retrieved_examples: list[dict[str, Any]],
    draft_reply: str,
) -> dict[str, str]:
    """Decide whether a generated reply is safe to auto-handle.

    This policy only examines the customer message, predicted intent, retrieved
    evidence, and draft reply. It does not call an LLM or invent business rules.
    """
    if not customer_message.strip():
        raise ValueError("customer_message must be non-empty")
    if not intent.strip():
        raise ValueError("intent must be non-empty")

    if intent in HIGH_RISK_INTENTS or _has_high_risk_message(customer_message):
        return _decision(ESCALATE, "The message contains a safety or high-risk concern.", "high")

    top_similarity = _top_similarity(retrieved_examples)
    if top_similarity < MINIMUM_RETRIEVAL_SIMILARITY:
        return _decision(
            ESCALATE,
            f"Historical retrieval evidence is insufficient (top similarity: {top_similarity:.2f}).",
            "high",
        )

    if not draft_reply.strip() or draft_reply.strip() == ESCALATION_RECOMMENDATION:
        return _decision(
            ESCALATE,
            "The reply generator indicated that the evidence is insufficient.",
            "high",
        )

    if top_similarity >= HIGH_RETRIEVAL_SIMILARITY:
        return _decision(AUTO_HANDLE, "The reply is grounded in strongly similar historical evidence.", "high")
    return _decision(AUTO_HANDLE, "The reply is grounded in useful historical evidence.", "medium")