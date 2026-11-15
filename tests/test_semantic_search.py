
import numpy as np
import pytest

from src.semantic_search import (
    search_similar_chunks,
    search_keyword_chunks,
    search_hybrid_chunks
)

from src.vector_store import create_vector_store


# Fake embedding model for testing
class FakeModel:

    def encode(self, texts, normalize_embeddings=True):

        return np.array([
            [1.0, 0.0]
            for text in texts
        ])


def create_sample_data():

    chunks = [
        {
            "chunk_id": 1,
            "page_number": 1,
            "text": "Machine learning helps computers learn from data."
        },
        {
            "chunk_id": 2,
            "page_number": 2,
            "text": "Deep learning uses artificial neural networks."
        },
        {
            "chunk_id": 3,
            "page_number": 3,
            "text": "Python is a popular programming language."
        }
    ]

    embeddings = np.array([
        [1.0, 0.0],
        [0.0, 1.0],
        [0.0, 0.5]
    ])

    index, stored_chunks = create_vector_store(
        chunks,
        embeddings
    )

    model = FakeModel()

    return model, index, stored_chunks


def test_semantic_search_returns_results():

    model, index, chunks = create_sample_data()

    results = search_similar_chunks(
        "Explain machine learning",
        model,
        index,
        chunks,
        top_k=2
    )

    assert len(results) == 2
    assert "score" in results[0]
    assert "chunk" in results[0]


def test_semantic_search_rejects_empty_query():

    model, index, chunks = create_sample_data()

    with pytest.raises(ValueError):
        search_similar_chunks(
            "",
            model,
            index,
            chunks
        )


def test_keyword_search_finds_matching_text():

    _, _, chunks = create_sample_data()

    results = search_keyword_chunks(
        "machine learning",
        chunks
    )

    assert len(results) > 0
    assert "Machine learning" in results[0]["chunk"]["text"]


def test_keyword_search_handles_empty_chunks():

    results = search_keyword_chunks(
        "machine learning",
        []
    )

    assert results == []


def test_hybrid_search_returns_combined_scores():

    model, index, chunks = create_sample_data()

    results = search_hybrid_chunks(
        "machine learning",
        model,
        index,
        chunks,
        top_k=2
    )

    assert len(results) > 0
    assert "semantic_score" in results[0]
    assert "keyword_score" in results[0]
    assert "score" in results[0]


def test_hybrid_search_rejects_invalid_weight():

    model, index, chunks = create_sample_data()

    with pytest.raises(ValueError):
        search_hybrid_chunks(
            "machine learning",
            model,
            index,
            chunks,
            semantic_weight=1.5
        )

# Test 7: Reject search when FAISS index is empty
def test_semantic_search_rejects_empty_index():
    model, index, chunks = create_sample_data()

    class EmptyIndex:
        ntotal = 0

    with pytest.raises(ValueError, match="FAISS index is empty"):
        search_similar_chunks(
            "machine learning",
            model,
            EmptyIndex(),
            chunks
        )


# Test 8: Skip invalid FAISS search indices
def test_semantic_search_skips_invalid_indices():
    class FakeIndex:
        ntotal = 2

        def search(self, query_vector, top_k):
            scores = np.array([[0.9, 0.0]], dtype=np.float32)
            indices = np.array([[0, -1]], dtype=np.int64)
            return scores, indices

    model, _, chunks = create_sample_data()

    results = search_similar_chunks(
        "machine learning",
        model,
        FakeIndex(),
        chunks,
        top_k=2
    )

    assert len(results) == 1
    assert results[0]["chunk"] == chunks[0]


# Test 9: Reject empty keyword search query
def test_keyword_search_rejects_empty_query():
    _, _, chunks = create_sample_data()

    with pytest.raises(ValueError, match="Please enter a question"):
        search_keyword_chunks(
            "",
            chunks
        )


# Test 10: Handle documents with no useful vocabulary
def test_keyword_search_handles_empty_vocabulary():
    chunks = [
        {"chunk_id": 1, "text": "the and or"},
        {"chunk_id": 2, "text": ""}
    ]

    results = search_keyword_chunks(
        "machine learning",
        chunks
    )

    assert results == []


# Test 11: Reject empty hybrid search query
def test_hybrid_search_rejects_empty_query():
    model, index, chunks = create_sample_data()

    with pytest.raises(ValueError, match="Please enter a question"):
        search_hybrid_chunks(
            "",
            model,
            index,
            chunks
        )


# Test 12: Reject zero or negative top_k
def test_hybrid_search_rejects_invalid_top_k():
    model, index, chunks = create_sample_data()

    with pytest.raises(ValueError, match="top_k must be greater than zero"):
        search_hybrid_chunks(
            "machine learning",
            model,
            index,
            chunks,
            top_k=0
        )


# Test 13: Reject empty FAISS index in hybrid search
def test_hybrid_search_rejects_empty_index():
    model, _, chunks = create_sample_data()

    class EmptyIndex:
        ntotal = 0

    with pytest.raises(ValueError, match="FAISS index is empty"):
        search_hybrid_chunks(
            "machine learning",
            model,
            EmptyIndex(),
            chunks
        )


# Test 14: Reject mismatched chunk and vector counts
def test_hybrid_search_rejects_mismatched_chunks():
    model, index, chunks = create_sample_data()

    with pytest.raises(
        ValueError,
        match="number of chunks does not match"
    ):
        search_hybrid_chunks(
            "machine learning",
            model,
            index,
            chunks[:2]
        )
