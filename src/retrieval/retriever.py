"""TF-IDF retrieval of historical AmazonHelp replies."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.intents.classifier import preprocess_text


REQUIRED_COLUMNS = ("customer_message", "amazon_reply")


@dataclass
class TfidfRetriever:
    """Fitted TF-IDF representation and the rows it represents."""

    vectorizer: TfidfVectorizer
    matrix: Any
    conversations: pd.DataFrame


def load_conversation_pairs(file_path: str | Path) -> pd.DataFrame:
    """Load historical customer messages and their actual Amazon replies."""
    conversations = pd.read_csv(file_path)
    missing_columns = set(REQUIRED_COLUMNS) - set(conversations.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Missing required source column(s): {missing}")

    conversations = conversations.loc[:, list(REQUIRED_COLUMNS)].copy()
    if conversations.empty:
        raise ValueError("Conversation-pair data must not be empty")
    if conversations[list(REQUIRED_COLUMNS)].isna().any().any():
        raise ValueError("Conversation-pair data must not contain missing values")
    if (conversations["customer_message"].astype(str).str.strip() == "").any():
        raise ValueError("Conversation-pair data must contain non-empty customer messages")
    return conversations


def fit_tfidf_retriever(conversations: pd.DataFrame) -> TfidfRetriever:
    """Fit a TF-IDF vectorizer on historical customer messages."""
    missing_columns = set(REQUIRED_COLUMNS) - set(conversations.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Missing required source column(s): {missing}")
    if conversations.empty:
        raise ValueError("Conversation-pair data must not be empty")

    stored_conversations = conversations.loc[:, list(REQUIRED_COLUMNS)].reset_index(drop=True).copy()
    vectorizer = TfidfVectorizer(
        preprocessor=preprocess_text,
        ngram_range=(1, 2),
    )
    matrix = vectorizer.fit_transform(stored_conversations["customer_message"])
    return TfidfRetriever(vectorizer, matrix, stored_conversations)


def retrieve_similar_conversations(
    retriever: TfidfRetriever,
    customer_message: str,
    top_k: int = 3,
) -> list[dict[str, Any]]:
    """Return the top historical messages, replies, and cosine scores."""
    if not preprocess_text(customer_message):
        raise ValueError("customer_message must be non-empty")
    if top_k < 1:
        raise ValueError("top_k must be at least 1")

    query_vector = retriever.vectorizer.transform([customer_message])
    scores = cosine_similarity(query_vector, retriever.matrix).ravel()
    result_count = min(top_k, len(scores))
    ranked_indices = scores.argsort()[::-1][:result_count]

    results = []
    for index in ranked_indices:
        row = retriever.conversations.iloc[int(index)]
        results.append(
            {
                "customer_message": row["customer_message"],
                "amazon_reply": row["amazon_reply"],
                "similarity": float(scores[index]),
            }
        )
    return results


def build_retriever(file_path: str | Path) -> TfidfRetriever:
    """Load conversation pairs and fit a retriever in one call."""
    return fit_tfidf_retriever(load_conversation_pairs(file_path))