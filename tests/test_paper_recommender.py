
import numpy as np
import pytest

from src.paper_recommender import recommend_papers


# Fake embedding model
class FakeModel:

    def encode(self, texts, normalize_embeddings=True):
        return np.array([[1.0, 0.0]], dtype="float32")


# Fake FAISS index
class FakeIndex:

    def __init__(self, scores, indices):
        self.scores = np.array([scores], dtype="float32")
        self.indices = np.array([indices], dtype="int64")
        self.received_k = None

    def search(self, query_embedding, k):
        self.received_k = k
        return self.scores[:, :k], self.indices[:, :k]


# Sample research paper chunks
def sample_chunks():
    return [
        {
            "document_name": "Paper1.pdf",
            "page_number": 1,
            "text": "Machine learning research."
        },
        {
            "document_name": "Paper1.pdf",
            "page_number": 2,
            "text": "Deep learning methods."
        },
        {
            "document_name": "Paper2.pdf",
            "page_number": 3,
            "text": "Neural network research."
        }
    ]


# Test 1: Reject empty query
def test_rejects_empty_query():
    with pytest.raises(ValueError):
        recommend_papers(
            "",
            FakeModel(),
            FakeIndex([0.9], [0]),
            sample_chunks()
        )


# Test 2: Return empty list when index is missing
def test_returns_empty_when_index_is_none():
    result = recommend_papers(
        "Machine learning",
        FakeModel(),
        None,
        sample_chunks()
    )

    assert result == []


# Test 3: Return empty list when chunks are empty
def test_returns_empty_when_chunks_are_empty():
    result = recommend_papers(
        "Machine learning",
        FakeModel(),
        FakeIndex([0.9], [0]),
        []
    )

    assert result == []


# Test 4: Generate paper recommendations
def test_generates_recommendations():
    chunks = sample_chunks()

    index = FakeIndex(
        [0.95, 0.85, 0.75],
        [0, 1, 2]
    )

    result = recommend_papers(
        "Machine learning",
        FakeModel(),
        index,
        chunks
    )

    assert len(result) == 2
    assert result[0]["document_name"] == "Paper1.pdf"
    assert result[1]["document_name"] == "Paper2.pdf"


# Test 5: Calculate the best relevance score for each paper
def test_uses_best_score_for_each_paper():
    chunks = sample_chunks()

    index = FakeIndex(
        [0.95, 0.80, 0.70],
        [0, 1, 2]
    )

    result = recommend_papers(
        "Machine learning",
        FakeModel(),
        index,
        chunks
    )

    assert result[0]["relevance_score"] == 95.0


# Test 6: Ignore results below minimum score
def test_filters_low_relevance_scores():
    chunks = sample_chunks()

    index = FakeIndex(
        [0.90, 0.15, 0.10],
        [0, 1, 2]
    )

    result = recommend_papers(
        "Machine learning",
        FakeModel(),
        index,
        chunks,
        min_score=0.20
    )

    assert len(result) == 1
    assert result[0]["document_name"] == "Paper1.pdf"


# Test 7: Sort recommendations by relevance
def test_sorts_recommendations_by_relevance():
    chunks = sample_chunks()

    index = FakeIndex(
        [0.70, 0.95, 0.80],
        [2, 0, 1]
    )

    result = recommend_papers(
        "Machine learning",
        FakeModel(),
        index,
        chunks
    )

    assert result[0]["document_name"] == "Paper1.pdf"
    assert result[1]["document_name"] == "Paper2.pdf"


# Test 8: Return matched page numbers
def test_returns_matched_pages():
    chunks = sample_chunks()

    index = FakeIndex(
        [0.95, 0.85, 0.75],
        [0, 1, 2]
    )

    result = recommend_papers(
        "Machine learning",
        FakeModel(),
        index,
        chunks
    )

    assert result[0]["matched_pages"] == [1, 2]


# Test 9: Respect top_k limit
def test_respects_top_k():
    chunks = sample_chunks()

    index = FakeIndex(
        [0.95, 0.85, 0.75],
        [0, 1, 2]
    )

    result = recommend_papers(
        "Machine learning",
        FakeModel(),
        index,
        chunks,
        top_k=1
    )

    assert len(result) == 1


# Test 10: Ignore invalid FAISS indices
def test_ignores_invalid_indices():
    chunks = sample_chunks()

    index = FakeIndex(
        [0.95, 0.90, 0.80],
        [0, -1, 2]
    )

    result = recommend_papers(
        "Machine learning",
        FakeModel(),
        index,
        chunks
    )

    assert len(result) == 2
    assert all(
        item["document_name"] in ["Paper1.pdf", "Paper2.pdf"]
        for item in result
    )
