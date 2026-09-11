"""Tests for historical AmazonHelp reply retrieval."""

from pathlib import Path

from src.retrieval import (
    build_retriever,
    load_conversation_pairs,
    retrieve_similar_conversations,
)


DATA_PATH = Path("data/amazonhelp_conversation_pairs.csv")


def test_retriever_loads_conversation_pairs():
    conversations = load_conversation_pairs(DATA_PATH)

    assert len(conversations) == 869
    assert list(conversations.columns) == ["customer_message", "amazon_reply"]


def test_retrieval_returns_at_most_top_k_results():
    retriever = build_retriever(DATA_PATH)

    results = retrieve_similar_conversations(retriever, "Where is my package?", top_k=3)

    assert 0 < len(results) <= 3


def test_retrieval_results_contain_historical_evidence_and_score():
    retriever = build_retriever(DATA_PATH)

    results = retrieve_similar_conversations(retriever, "My delivery has not arrived", top_k=1)

    assert set(results[0]) == {"customer_message", "amazon_reply", "similarity"}
    assert results[0]["customer_message"]
    assert results[0]["amazon_reply"]
    assert 0.0 <= results[0]["similarity"] <= 1.0