
import pytest
import src.reranker as reranker

from src.reranker import (
    rerank_results,
    load_reranker_model,
    MODEL_NAME
)


# Fake model to avoid downloading the real Cross-Encoder
class FakeReranker:

    def predict(self, pairs):
        return [0.2, 0.9, 0.5]


def create_sample_results():

    return [
        {
            "score": 0.80,
            "chunk": {"text": "Python is a programming language."}
        },
        {
            "score": 0.70,
            "chunk": {"text": "Machine learning uses data."}
        },
        {
            "score": 0.60,
            "chunk": {"text": "Deep learning uses neural networks."}
        }
    ]


def test_reranker_sorts_by_relevance():

    results = create_sample_results()

    reranked = rerank_results(
        "Explain AI",
        results,
        model=FakeReranker()
    )

    assert reranked[0]["rerank_score"] == 0.9
    assert reranked[1]["rerank_score"] == 0.5
    assert reranked[2]["rerank_score"] == 0.2


def test_reranker_preserves_original_score():

    results = create_sample_results()

    reranked = rerank_results(
        "Explain AI",
        results,
        model=FakeReranker()
    )

    assert "original_score" in reranked[0]
    assert reranked[0]["original_score"] == 0.70


def test_reranker_respects_top_k():

    results = create_sample_results()

    reranked = rerank_results(
        "Explain AI",
        results,
        top_k=2,
        model=FakeReranker()
    )

    assert len(reranked) == 2


def test_reranker_rejects_empty_query():

    with pytest.raises(ValueError):

        rerank_results(
            "",
            create_sample_results(),
            model=FakeReranker()
        )


def test_reranker_rejects_invalid_top_k():

    with pytest.raises(ValueError):

        rerank_results(
            "Explain AI",
            create_sample_results(),
            top_k=0,
            model=FakeReranker()
        )


def test_reranker_handles_empty_results():

    reranked = rerank_results(
        "Explain AI",
        [],
        model=FakeReranker()
    )

    assert reranked == []


def test_load_reranker_model_uses_cache(monkeypatch):

    import src.reranker as reranker_module

    calls = []

    class FakeCrossEncoder:

        def __init__(self, model_name):
            calls.append(model_name)

    monkeypatch.setattr(
        reranker_module,
        "CrossEncoder",
        FakeCrossEncoder
    )

    monkeypatch.setattr(
        reranker_module,
        "_reranker_model",
        None
    )

    first_model = load_reranker_model()
    second_model = load_reranker_model()

    assert first_model is second_model
    assert calls == [MODEL_NAME]

# Test: Automatically load reranker model when model is None
def test_rerank_results_loads_model_when_none(monkeypatch):

    class MockModel:

        def predict(self, pairs):
            return [0.9]

    mock_model = MockModel()

    monkeypatch.setattr(
        reranker,
        "load_reranker_model",
        lambda: mock_model
    )

    search_results = [
        {
            "chunk": {
                "text": "Machine learning research"
            },
            "score": 0.75
        }
    ]

    result = reranker.rerank_results(
        query="What is machine learning?",
        search_results=search_results,
        model=None
    )

    assert len(result) == 1
    assert result[0]["rerank_score"] == 0.9