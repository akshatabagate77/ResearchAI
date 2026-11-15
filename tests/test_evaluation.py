
import json
import pytest

from unittest.mock import patch, MagicMock

from src.evaluation import (
    load_evaluation_questions,
    calculate_metrics,
    evaluate_single_question,
    evaluate_retrieval_method,
    evaluate_all_retrieval_methods,
    evaluate_retrieval
)


# Sample evaluation question
def sample_question():
    return {
        "question": "What is machine learning?",
        "expected_document": "Paper1.pdf",
        "expected_page": 2
    }


# Sample retrieved results
def sample_results():
    return [
        {
            "chunk": {
                "document_name": "Paper2.pdf",
                "page_number": 1,
                "text": "Deep learning"
            },
            "score": 0.90
        },
        {
            "chunk": {
                "document_name": "Paper1.pdf",
                "page_number": 2,
                "text": "Machine learning"
            },
            "score": 0.85
        }
    ]


# Test 1: Load evaluation questions
def test_load_evaluation_questions(tmp_path):
    questions = [sample_question()]

    file_path = tmp_path / "questions.json"
    file_path.write_text(
        json.dumps(questions),
        encoding="utf-8"
    )

    result = load_evaluation_questions(str(file_path))

    assert result == questions


# Test 2: Raise error when dataset is missing
def test_missing_dataset():
    with pytest.raises(FileNotFoundError):
        load_evaluation_questions("missing.json")


# Test 3: Reject JSON that is not a list
def test_reject_invalid_dataset_format(tmp_path):
    file_path = tmp_path / "questions.json"
    file_path.write_text(
        json.dumps({"question": "Example"}),
        encoding="utf-8"
    )

    with pytest.raises(ValueError):
        load_evaluation_questions(str(file_path))


# Test 4: Calculate retrieval metrics
def test_calculate_metrics():
    result = calculate_metrics(
        question_data=sample_question(),
        retrieved_results=sample_results(),
        top_k=2,
        latency=0.5
    )

    assert result["precision"] == 0.5
    assert result["recall"] == 1.0
    assert result["hit"] is True
    assert result["mrr"] == 0.5
    assert result["latency_ms"] == 500.0


# Test 5: Handle no relevant results
def test_metrics_without_relevant_results():
    results = [
        {
            "chunk": {
                "document_name": "Other.pdf",
                "page_number": 1
            },
            "score": 0.8
        }
    ]

    result = calculate_metrics(
        sample_question(),
        results,
        top_k=5,
        latency=0.1
    )

    assert result["precision"] == 0.0
    assert result["recall"] == 0.0
    assert result["hit"] is False
    assert result["mrr"] == 0.0


# Test 6: Calculate MRR for first result
def test_mrr_for_first_result():
    results = [
        {
            "chunk": {
                "document_name": "Paper1.pdf",
                "page_number": 2
            },
            "score": 0.9
        }
    ]

    result = calculate_metrics(
        sample_question(),
        results,
        top_k=1,
        latency=0.1
    )

    assert result["mrr"] == 1.0


# Test 7: Evaluate using semantic search
@patch("src.evaluation.search_similar_chunks")
def test_semantic_evaluation(mock_search):
    mock_search.return_value = sample_results()

    result = evaluate_single_question(
        sample_question(),
        model=MagicMock(),
        index=MagicMock(),
        chunks=[],
        top_k=2,
        method="semantic"
    )

    mock_search.assert_called_once()
    assert result["hit"] is True


# Test 8: Evaluate using hybrid search
@patch("src.evaluation.search_hybrid_chunks")
def test_hybrid_evaluation(mock_search):
    mock_search.return_value = sample_results()

    result = evaluate_single_question(
        sample_question(),
        model=MagicMock(),
        index=MagicMock(),
        chunks=[],
        top_k=2,
        method="hybrid"
    )

    mock_search.assert_called_once()
    assert result["hit"] is True


# Test 9: Evaluate using hybrid search and reranking
@patch("src.evaluation.rerank_results")
@patch("src.evaluation.search_hybrid_chunks")
def test_hybrid_reranked_evaluation(
    mock_search,
    mock_rerank
):
    mock_search.return_value = sample_results()
    mock_rerank.return_value = sample_results()

    result = evaluate_single_question(
        sample_question(),
        model=MagicMock(),
        index=MagicMock(),
        chunks=[],
        top_k=2,
        method="hybrid_reranked",
        reranker_model=MagicMock()
    )

    mock_search.assert_called_once()
    mock_rerank.assert_called_once()
    assert result["hit"] is True


# Test 10: Reject unsupported evaluation method
def test_reject_unsupported_method():
    with pytest.raises(ValueError):
        evaluate_single_question(
            sample_question(),
            model=MagicMock(),
            index=MagicMock(),
            chunks=[],
            method="invalid"
        )


# Test 11: Reject empty evaluation dataset
def test_reject_empty_questions():
    with pytest.raises(ValueError):
        evaluate_retrieval_method(
            questions=[],
            model=MagicMock(),
            index=MagicMock(),
            chunks=[]
        )


# Test 12: Evaluate multiple questions
@patch("src.evaluation.evaluate_single_question")
def test_evaluate_retrieval_method(mock_evaluate):
    mock_evaluate.return_value = {
        "precision": 0.5,
        "recall": 1.0,
        "hit": True,
        "mrr": 0.5,
        "latency_ms": 100.0
    }

    questions = [
        sample_question(),
        sample_question()
    ]

    result = evaluate_retrieval_method(
        questions=questions,
        model=MagicMock(),
        index=MagicMock(),
        chunks=[],
        top_k=2
    )

    assert result["total_questions"] == 2
    assert result["precision_at_k"] == 0.5
    assert result["recall_at_k"] == 1.0
    assert result["hit_rate"] == 1.0
    assert result["mrr"] == 0.5
    assert result["average_latency_ms"] == 100.0


# Test 13: Reject invalid top_k
def test_reject_invalid_top_k():
    index = MagicMock()
    index.ntotal = 10

    with pytest.raises(ValueError):
        evaluate_all_retrieval_methods(
            dataset_path="questions.json",
            model=MagicMock(),
            index=index,
            chunks=[],
            top_k=0
        )


# Test 14: Reject empty FAISS index
def test_reject_empty_faiss_index():
    index = MagicMock()
    index.ntotal = 0

    with pytest.raises(ValueError):
        evaluate_all_retrieval_methods(
            "questions.json",
            MagicMock(),
            index,
            [],
            top_k=5
        )


# Test 15: Reject mismatched chunks and index
def test_reject_mismatched_chunks():
    index = MagicMock()
    index.ntotal = 5

    with pytest.raises(ValueError):
        evaluate_all_retrieval_methods(
            "questions.json",
            MagicMock(),
            index,
            chunks=[],
            top_k=5
        )


# Test 16: Backward-compatible evaluation
@patch("src.evaluation.evaluate_retrieval_method")
@patch("src.evaluation.load_evaluation_questions")
def test_backward_compatible_evaluation(
    mock_load,
    mock_evaluate
):
    questions = [sample_question()]
    mock_load.return_value = questions
    mock_evaluate.return_value = {"method": "semantic"}

    result = evaluate_retrieval(
        "questions.json",
        MagicMock(),
        MagicMock(),
        []
    )

    assert result == {"method": "semantic"}

    mock_evaluate.assert_called_once()
    assert mock_evaluate.call_args.kwargs["method"] == "semantic"

# Test 17: Reject an empty evaluation dataset
@patch("src.evaluation.load_evaluation_questions")
def test_reject_empty_evaluation_dataset(mock_load):

    mock_load.return_value = []

    index = MagicMock()
    index.ntotal = 2

    with pytest.raises(ValueError, match="Evaluation dataset is empty"):

        evaluate_all_retrieval_methods(
            dataset_path="questions.json",
            model=MagicMock(),
            index=index,
            chunks=[{}, {}],
            top_k=5
        )


# Test 18: Successfully compare all three retrieval methods
@patch("src.evaluation.evaluate_retrieval_method")
@patch("src.evaluation.load_reranker_model")
@patch("src.evaluation.load_evaluation_questions")
def test_evaluate_all_retrieval_methods(
    mock_load_questions,
    mock_load_reranker,
    mock_evaluate
):

    questions = [sample_question()]
    mock_load_questions.return_value = questions

    reranker = MagicMock()
    mock_load_reranker.return_value = reranker

    mock_evaluate.side_effect = [
        {"method": "semantic"},
        {"method": "hybrid"},
        {"method": "hybrid_reranked"}
    ]

    index = MagicMock()
    index.ntotal = 2

    result = evaluate_all_retrieval_methods(
        dataset_path="questions.json",
        model=MagicMock(),
        index=index,
        chunks=[{}, {}],
        top_k=5
    )

    assert result["total_questions"] == 1
    assert result["top_k"] == 5

    assert list(result["comparison"].keys()) == [
        "semantic",
        "hybrid",
        "hybrid_reranked"
    ]

    assert mock_load_reranker.called
    assert mock_evaluate.call_count == 3
