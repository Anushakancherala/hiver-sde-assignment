"""Simple non-LLM baselines for AmazonHelp evaluation."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split

from src.agent import AUTO_HANDLE, ESCALATE
from src.intents import INTENTS, load_training_data, train_classifier
from src.intents.classifier import predict_intent


TRAINING_DATA_PATH = Path("data/amazonhelp_training_labeled.csv")
RANDOM_SEED = 42
SIMPLE_ESCALATION_RULES = (
    re.compile(r"\b(safety|fire|smoke|burn|danger|dangerous|unsafe|hazard|injur|explod)\w*\b", re.I),
    re.compile(r"\b(fraud|scam|unauthori[sz]ed|stolen|identity theft|counterfeit)\w*\b", re.I),
    re.compile(r"\b(hacked|hack|security breach|account compromise|compromised)\b", re.I),
    re.compile(r"\b(threat|threaten|harm|hurt|attack|violence|weapon|abuse)\w*\b", re.I),
)


def majority_intent(training_path: str | Path = TRAINING_DATA_PATH) -> str:
    """Return the majority intent from non-Golden training data."""
    _, labels = load_training_data(
        training_path,
        text_column="customer_message",
        label_column="intent",
    )
    return pd.Series(labels).value_counts().idxmax()


def train_simple_intent_classifier(
    training_path: str | Path = TRAINING_DATA_PATH,
    random_state: int = RANDOM_SEED,
) -> Any:
    """Train TF-IDF + Logistic Regression on a reproducible training split."""
    texts, labels = load_training_data(
        training_path,
        text_column="customer_message",
        label_column="intent",
    )
    train_texts, _, train_labels, _ = train_test_split(
        texts,
        labels,
        test_size=0.2,
        random_state=random_state,
        stratify=labels,
    )
    return train_classifier(train_texts, train_labels)


def simple_action(customer_message: str) -> str:
    """Escalate clear high-risk messages; auto-handle all other messages."""
    if any(rule.search(customer_message) for rule in SIMPLE_ESCALATION_RULES):
        return ESCALATE
    return AUTO_HANDLE


def predict_trivial(
    customer_messages: list[str],
    training_path: str | Path = TRAINING_DATA_PATH,
) -> tuple[list[str], list[str]]:
    """Predict majority intent and constant AUTO_HANDLE action."""
    intent = majority_intent(training_path)
    return [intent] * len(customer_messages), [AUTO_HANDLE] * len(customer_messages)


def predict_simple(
    customer_messages: list[str],
    training_path: str | Path = TRAINING_DATA_PATH,
) -> tuple[list[str], list[str]]:
    """Predict intent with the simple classifier and action with message rules."""
    classifier = train_simple_intent_classifier(training_path)
    intents = [predict_intent(classifier, message) for message in customer_messages]
    actions = [simple_action(message) for message in customer_messages]
    return intents, actions