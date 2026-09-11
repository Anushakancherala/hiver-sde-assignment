"""Grounded customer-support reply generation utilities."""

from .escalation import AUTO_HANDLE, ESCALATE, decide_action
from .provider import LLMProvider, LLMProviderError, OpenAIProvider
from .reply_generator import (
    ESCALATION_RECOMMENDATION,
    build_grounded_prompt,
    generate_reply,
)

__all__ = [
    "AUTO_HANDLE",
    "ESCALATION_RECOMMENDATION",
    "ESCALATE",
    "LLMProvider",
    "LLMProviderError",
    "OpenAIProvider",
    "build_grounded_prompt",
    "decide_action",
    "generate_reply",
]