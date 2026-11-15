
import json
import pytest

import src.rag_pipeline as rag


def test_detect_follow_up_question():
    assert rag.is_follow_up_question("Explain more about this")
    assert rag.is_follow_up_question("What about its limitations?")
    assert not rag.is_follow_up_question("Explain machine learning")


def test_get_previous_history_removes_current_query():

    history = [
        {"role": "user", "content": "Explain AI"},
        {"role": "assistant", "content": "AI is..."},
        {"role": "user", "content": "Explain more"}
    ]

    result = rag.get_previous_history(history, "Explain more")

    assert len(result) == 2
    assert result[-1]["role"] == "assistant"


def test_prepare_chat_history_formats_messages():

    history = [
        {"role": "user", "content": "Explain AI"},
        {"role": "assistant", "content": "AI is artificial intelligence."}
    ]

    result = rag.prepare_chat_history(history)

    assert "User: Explain AI" in result
    assert "ResearchAI: AI is artificial intelligence." in result


def test_prepare_chat_history_handles_empty_history():

    result = rag.prepare_chat_history([])

    assert result == "No previous conversation."


def test_filter_relevant_results():

    results = [
        {"score": 0.80, "chunk": {"text": "Relevant passage"}},
        {"score": 0.10, "chunk": {"text": "Weak passage"}},
        {"score": 0.50, "chunk": {"text": "Another relevant passage"}}
    ]

    filtered = rag.filter_relevant_results(results)

    assert len(filtered) == 2
    assert all(item["score"] >= 0.15 for item in filtered)


def test_build_research_context():

    results = [
        {
            "chunk": {
                "document_name": "Research Paper.pdf",
                "page_number": 3,
                "text": "Machine learning learns from data."
            }
        }
    ]

    context = rag.build_research_context(results)

    assert "Research Paper.pdf" in context
    assert "Page: 3" in context
    assert "Machine learning learns from data." in context


def test_build_research_prompt_contains_evidence_rules():

    prompt = rag.build_research_prompt(
        query="Explain machine learning",
        context="Machine learning learns from data."
    )

    assert "Explain machine learning" in prompt
    assert "Machine learning learns from data." in prompt
    assert "Do not invent facts" in prompt
    assert "Use only the retrieved passages as evidence" in prompt


def test_verify_answer_support_accepts_valid_json(monkeypatch):

    response = {
        "supported": True,
        "unsupported_claims": [],
        "explanation": "The answer matches the research context."
    }

    monkeypatch.setattr(
        rag,
        "generate_answer",
        lambda *args, **kwargs: json.dumps(response)
    )

    result = rag.verify_answer_support(
        answer="Machine learning uses data.",
        context="Machine learning uses data."
    )

    assert result["status"] == "supported"
    assert result["supported"] is True
    assert result["unsupported_claims"] == []


def test_verify_answer_support_handles_invalid_json(monkeypatch):

    monkeypatch.setattr(
        rag,
        "generate_answer",
        lambda *args, **kwargs: "This is not valid JSON"
    )

    result = rag.verify_answer_support(
        answer="Some answer",
        context="Some context"
    )

    assert result["status"] == "unavailable"
    assert result["supported"] is None


def test_answer_question_complete_pipeline(monkeypatch):

    sample_results = [
        {
            "score": 0.85,
            "semantic_score": 0.90,
            "keyword_score": 0.75,
            "original_score": 0.85,
            "rerank_score": 0.95,
            "chunk": {
                "document_name": "AI_Paper.pdf",
                "page_number": 2,
                "text": "Machine learning enables computers to learn from data."
            }
        }
    ]

    monkeypatch.setattr(
        rag,
        "rewrite_research_query",
        lambda query, history: query
    )

    monkeypatch.setattr(
        rag,
        "search_hybrid_chunks",
        lambda **kwargs: sample_results
    )

    monkeypatch.setattr(
        rag,
        "rerank_results",
        lambda **kwargs: sample_results
    )

    monkeypatch.setattr(
        rag,
        "generate_answer",
        lambda *args, **kwargs: "Machine learning allows computers to learn from data."
    )

    monkeypatch.setattr(
        rag,
        "verify_answer_support",
        lambda **kwargs: {
            "status": "supported",
            "supported": True,
            "unsupported_claims": [],
            "explanation": "Answer is supported."
        }
    )

    result = rag.answer_question(
        query="Explain machine learning",
        model=None,
        index=None,
        chunks=[]
    )

    assert "Machine learning" in result["answer"]
    assert len(result["sources"]) == 1
    assert result["sources"][0]["document_name"] == "AI_Paper.pdf"
    assert result["verification"]["status"] == "supported"


def test_answer_question_rejects_empty_query():

    result = rag.answer_question(
        query="",
        model=None,
        index=None,
        chunks=[]
    )

    assert "Question cannot be empty" in result["answer"]
    assert result["sources"] == []


def test_answer_question_handles_no_relevant_results(monkeypatch):

    monkeypatch.setattr(
        rag,
        "rewrite_research_query",
        lambda query, history: query
    )

    monkeypatch.setattr(
        rag,
        "search_hybrid_chunks",
        lambda **kwargs: []
    )

    result = rag.answer_question(
        query="Explain quantum computing",
        model=None,
        index=None,
        chunks=[]
    )

    assert result["sources"] == []
    assert result["verification"]["status"] == "not_applicable"
    assert "couldn't find sufficiently relevant" in result["answer"]


def test_answer_question_handles_generation_error(monkeypatch):

    sample_results = [
        {
            "score": 0.85,
            "chunk": {
                "document_name": "AI_Paper.pdf",
                "page_number": 1,
                "text": "Artificial intelligence enables machines to perform tasks."
            }
        }
    ]

    monkeypatch.setattr(
        rag,
        "rewrite_research_query",
        lambda query, history: query
    )

    monkeypatch.setattr(
        rag,
        "search_hybrid_chunks",
        lambda **kwargs: sample_results
    )

    monkeypatch.setattr(
        rag,
        "rerank_results",
        lambda **kwargs: sample_results
    )

    def fake_generate_answer(*args, **kwargs):
        raise RuntimeError("Simulated generation failure")

    monkeypatch.setattr(
        rag,
        "generate_answer",
        fake_generate_answer
    )

    result = rag.answer_question(
        query="Explain artificial intelligence",
        model=None,
        index=None,
        chunks=[]
    )

    assert "Simulated generation failure" in result["answer"]
    assert result["sources"] == []
    assert result["verification"]["status"] == "unavailable"

# Test 14: Successfully rewrite a follow-up question
def test_rewrite_research_query_success(monkeypatch):

    def fake_generate_answer(*args, **kwargs):
        return '"Machine learning model limitations"'

    monkeypatch.setattr(
        rag,
        "generate_answer",
        fake_generate_answer
    )

    history = [
        {"role": "user", "content": "Explain machine learning"}
    ]

    result = rag.rewrite_research_query(
        "What about its limitations?",
        history
    )

    assert result == "Machine learning model limitations"


# Test 15: Use fallback when query rewriting fails
def test_rewrite_research_query_fallback(monkeypatch):

    def fake_generate_answer(*args, **kwargs):
        raise Exception("API error")

    monkeypatch.setattr(
        rag,
        "generate_answer",
        fake_generate_answer
    )

    history = [
        {"role": "user", "content": "Explain machine learning"}
    ]

    result = rag.rewrite_research_query(
        "What about its limitations?",
        history
    )

    assert result == (
        "Explain machine learning What about its limitations?"
    )


# Test 16: Use fallback when rewritten query is invalid
def test_rewrite_research_query_invalid_response(monkeypatch):

    monkeypatch.setattr(
        rag,
        "generate_answer",
        lambda *args, **kwargs: "Invalid\nresponse"
    )

    history = [
        {"role": "user", "content": "Explain machine learning"}
    ]

    result = rag.rewrite_research_query(
        "What about its limitations?",
        history
    )

    assert result == (
        "Explain machine learning What about its limitations?"
    )


# Test 17: Verify answer with Markdown JSON code fences
def test_verify_answer_support_with_code_fences(monkeypatch):

    response = """```json
{
    "supported": true,
    "unsupported_claims": [],
    "explanation": "The context supports the answer."
}
```"""

    monkeypatch.setattr(
        rag,
        "generate_answer",
        lambda *args, **kwargs: response
    )

    result = rag.verify_answer_support(
        "Machine learning uses data.",
        "The paper explains machine learning."
    )

    assert result["status"] == "supported"
    assert result["supported"] is True
    assert result["unsupported_claims"] == []


# Test 18: Handle invalid verification response
def test_verify_answer_support_invalid_response(monkeypatch):

    monkeypatch.setattr(
        rag,
        "generate_answer",
        lambda *args, **kwargs: '{"supported": "yes"}'
    )

    result = rag.verify_answer_support(
        "Example answer",
        "Example context"
    )

    assert result["status"] == "unavailable"
    assert result["supported"] is None


# Test 19: Handle empty results after reranking
def test_answer_question_empty_reranked_results(monkeypatch):

    monkeypatch.setattr(
        rag,
        "search_hybrid_chunks",
        lambda *args, **kwargs: [
            {
                "chunk": {"text": "Relevant information"},
                "score": 0.9
            }
        ]
    )

    monkeypatch.setattr(
        rag,
        "rerank_results",
        lambda *args, **kwargs: []
    )

    result = rag.answer_question(
        query="Explain machine learning",
        model=None,
        index=None,
        chunks=[],
        top_k=3
    )

    assert result["sources"] == []
    assert result["verification"]["status"] == "not_applicable"
    assert "reranking" in result["answer"].lower()


# Test 20: Reject invalid top_k
def test_answer_question_rejects_invalid_top_k():

    result = rag.answer_question(
        query="Explain machine learning",
        model=None,
        index=None,
        chunks=[],
        top_k=0
    )

    assert "top_k must be greater than zero" in result["answer"]

# Test 21: Return original query when no previous user question exists
def test_rewrite_research_query_without_previous_user_question(monkeypatch):

    monkeypatch.setattr(
        rag,
        "generate_answer",
        lambda *args, **kwargs: "Invalid\nresponse"
    )

    history = [
        {"role": "assistant", "content": "How can I help?"}
    ]

    result = rag.rewrite_research_query(
        "What about its limitations?",
        history
    )

    assert result == "What about its limitations?"


# Test 22: Handle verification response that is not a dictionary
def test_verify_answer_support_rejects_non_dictionary(monkeypatch):

    monkeypatch.setattr(
        rag,
        "generate_answer",
        lambda *args, **kwargs: "[]"
    )

    result = rag.verify_answer_support(
        answer="Example answer",
        context="Example context"
    )

    assert result["status"] == "unavailable"
    assert result["supported"] is None


# Test 23: Handle unsupported_claims when it is not a list
def test_verify_answer_support_rejects_invalid_claims_type(monkeypatch):

    response = {
        "supported": True,
        "unsupported_claims": "None",
        "explanation": "Example explanation"
    }

    monkeypatch.setattr(
        rag,
        "generate_answer",
        lambda *args, **kwargs: json.dumps(response)
    )

    result = rag.verify_answer_support(
        answer="Example answer",
        context="Example context"
    )

    assert result["status"] == "unavailable"
    assert result["supported"] is None