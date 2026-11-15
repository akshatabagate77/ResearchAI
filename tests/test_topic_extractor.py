
import pytest

from src.topic_extractor import (
    extract_keywords,
    extract_document_keywords,
    extract_common_keywords
)


# Test 1: Extract keywords from text
def test_extract_keywords_returns_results():
    texts = [
        "Machine learning helps computers learn from data.",
        "Deep learning uses neural networks for machine learning."
    ]

    result = extract_keywords(texts)

    assert len(result) > 0
    assert "keyword" in result[0]
    assert "score" in result[0]


# Test 2: Return empty list for empty input
def test_extract_keywords_with_empty_input():
    result = extract_keywords([])

    assert result == []


# Test 3: Handle text containing only spaces
def test_extract_keywords_with_blank_text():
    result = extract_keywords(["   ", ""])

    assert result == []


# Test 4: Respect top_n limit
def test_extract_keywords_respects_top_n():
    texts = [
        "Machine learning and deep learning use neural networks."
    ]

    result = extract_keywords(texts, top_n=2)

    assert len(result) <= 2


# Test 5: Return keywords with numeric scores
def test_extract_keywords_scores_are_numeric():
    texts = [
        "Artificial intelligence uses machine learning."
    ]

    result = extract_keywords(texts)

    for item in result:
        assert isinstance(item["score"], float)
        assert item["score"] >= 0


# Test 6: Extract keywords separately for each document
def test_extract_document_keywords():
    chunks = [
        {
            "document_name": "Paper1.pdf",
            "text": "Machine learning studies computer algorithms."
        },
        {
            "document_name": "Paper2.pdf",
            "text": "Deep learning uses neural networks."
        }
    ]

    result = extract_document_keywords(chunks)

    assert "Paper1.pdf" in result
    assert "Paper2.pdf" in result
    assert len(result["Paper1.pdf"]) > 0
    assert len(result["Paper2.pdf"]) > 0


# Test 7: Handle empty document chunks
def test_extract_document_keywords_with_empty_chunks():
    result = extract_document_keywords([])

    assert result == {}


# Test 8: Use default name for unknown documents
def test_extract_document_keywords_with_missing_name():
    chunks = [
        {
            "text": "Machine learning helps computers learn."
        }
    ]

    result = extract_document_keywords(chunks)

    assert "Unknown document" in result


# Test 9: Find common keywords across documents
def test_extract_common_keywords():
    document_keywords = {
        "Paper1.pdf": [
            {"keyword": "machine learning", "score": 0.8},
            {"keyword": "neural network", "score": 0.6}
        ],
        "Paper2.pdf": [
            {"keyword": "machine learning", "score": 0.7},
            {"keyword": "deep learning", "score": 0.5}
        ]
    }

    result = extract_common_keywords(document_keywords)

    assert result[0]["keyword"] == "machine learning"
    assert result[0]["paper_count"] == 2


# Test 10: Return empty list when no common keywords exist
def test_extract_common_keywords_with_empty_input():
    result = extract_common_keywords({})

    assert result == []

# Test: Handle text containing only English stop words
def test_extract_keywords_with_only_stop_words():

    result = extract_keywords([
        "the is and are"
    ])

    assert result == []