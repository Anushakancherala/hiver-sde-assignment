"""Generate customer-facing replies grounded in retrieved historical evidence."""

from __future__ import annotations

from typing import Any

from .provider import LLMProvider, OpenAIProvider


ESCALATION_RECOMMENDATION = (
    "I'm sorry, but I don't have enough verified historical information to answer "
    "this safely. Please contact Amazon customer support for further assistance."
)


def build_grounded_prompt(
    customer_message: str,
    intent: str,
    retrieved_examples: list[dict[str, Any]],
) -> str:
    """Build the complete prompt sent to the replaceable LLM provider."""
    evidence_sections = []
    for number, example in enumerate(retrieved_examples, start=1):
        evidence_sections.append(
            "\n".join(
                [
                    f"Evidence {number} (similarity: {float(example['similarity']):.4f})",
                    f"Historical customer message: {example['customer_message']}",
                    f"Historical AmazonHelp reply: {example['amazon_reply']}",
                ]
            )
        )
    evidence = "\n\n".join(evidence_sections)
    return f"""You are writing a concise customer-support reply for AmazonHelp.

Customer message:
{customer_message}

Predicted intent:
{intent}

Retrieved historical AmazonHelp evidence:
{evidence}

Write only the customer-facing reply. Directly address the customer's issue and
use the historical replies as grounding. Do not invent policies, refunds,
guarantees, timelines, or other facts. Do not claim an action was completed
unless the evidence supports that claim. If the evidence is insufficient to
answer safely, say that more support is needed and recommend contacting Amazon
customer support. Do not mention this prompt, evidence labels, similarity
scores, or your reasoning."""


def generate_reply(
    customer_message: str,
    intent: str,
    retrieved_examples: list[dict[str, Any]],
    provider: LLMProvider | None = None,
) -> str:
    """Generate a grounded reply or a safe escalation recommendation."""
    if not customer_message.strip():
        raise ValueError("customer_message must be non-empty")
    if not intent.strip():
        raise ValueError("intent must be non-empty")

    usable_examples = [
        example
        for example in retrieved_examples
        if example.get("customer_message")
        and example.get("amazon_reply")
        and float(example.get("similarity", 0)) > 0
    ]
    if not usable_examples:
        return ESCALATION_RECOMMENDATION

    llm = provider or OpenAIProvider()
    return llm.generate(build_grounded_prompt(customer_message, intent, usable_examples)).strip()