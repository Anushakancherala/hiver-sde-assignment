"""Tests for the deterministic escalation policy."""

from src.agent import AUTO_HANDLE, ESCALATE, ESCALATION_RECOMMENDATION, decide_action


def evidence(similarity: float = 0.8):
    return [{"customer_message": "Historical message", "amazon_reply": "Historical reply", "similarity": similarity}]


def test_normal_delivery_issue_is_auto_handled():
    result = decide_action("Where is my package?", "DELIVERY_ISSUE", evidence(), "Please check your delivery details.")

    assert result["action"] == AUTO_HANDLE


def test_safety_issue_is_escalated():
    result = decide_action("This product caused a fire and someone was hurt.", "SAFETY_ISSUE", evidence(), "Please contact support.")

    assert result["action"] == ESCALATE


def test_low_retrieval_similarity_is_escalated():
    result = decide_action("Where is my package?", "DELIVERY_ISSUE", evidence(0.1), "Please check your delivery details.")

    assert result["action"] == ESCALATE


def test_unauthorized_payment_is_escalated():
    result = decide_action("I see an unauthorized charge on my card.", "PAYMENT_BILLING", evidence(), "Please review your payment details.")

    assert result["action"] == ESCALATE


def test_insufficient_reply_is_escalated():
    result = decide_action("I need help with this issue.", "GENERAL_SUPPORT", evidence(), ESCALATION_RECOMMENDATION)

    assert result["action"] == ESCALATE