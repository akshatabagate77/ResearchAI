
import pytest

import src.research_comparator as comparator


# Sample research paper summaries
def sample_summaries():
    return {
        "Paper1.pdf": "This paper studies machine learning.",
        "Paper2.pdf": "This paper studies deep learning.",
        "Paper3.pdf": "This paper studies neural networks.",
        "Paper4.pdf": "This paper studies artificial intelligence.",
        "Paper5.pdf": "This paper studies robotics."
    }


# Test 1: Reject fewer than two papers
def test_rejects_less_than_two_papers():
    summaries = sample_summaries()

    with pytest.raises(ValueError):
        comparator.compare_research_papers(
            summaries,
            ["Paper1.pdf"]
        )


# Test 2: Reject more than four papers
def test_rejects_more_than_four_papers():
    summaries = sample_summaries()

    with pytest.raises(ValueError):
        comparator.compare_research_papers(
            summaries,
            [
                "Paper1.pdf",
                "Paper2.pdf",
                "Paper3.pdf",
                "Paper4.pdf",
                "Paper5.pdf"
            ]
        )


# Test 3: Reject missing paper summary
def test_rejects_missing_summary():
    summaries = sample_summaries()

    with pytest.raises(ValueError, match="Summary not found"):
        comparator.compare_research_papers(
            summaries,
            ["Paper1.pdf", "Unknown.pdf"]
        )


# Test 4: Generate comparison report
def test_generates_comparison_report(monkeypatch):
    summaries = sample_summaries()

    monkeypatch.setattr(
        comparator,
        "generate_answer",
        lambda prompt: "Research comparison completed."
    )

    result = comparator.compare_research_papers(
        summaries,
        ["Paper1.pdf", "Paper2.pdf"]
    )

    assert result == "Research comparison completed."


# Test 5: Check selected papers are included in prompt
def test_prompt_contains_selected_papers(monkeypatch):
    summaries = sample_summaries()

    captured_prompt = {}

    def fake_generate_answer(prompt):
        captured_prompt["text"] = prompt
        return "Comparison report"

    monkeypatch.setattr(
        comparator,
        "generate_answer",
        fake_generate_answer
    )

    comparator.compare_research_papers(
        summaries,
        ["Paper1.pdf", "Paper2.pdf"]
    )

    assert "Paper1.pdf" in captured_prompt["text"]
    assert "Paper2.pdf" in captured_prompt["text"]
    assert summaries["Paper1.pdf"] in captured_prompt["text"]
    assert summaries["Paper2.pdf"] in captured_prompt["text"]


# Test 6: Exclude unselected papers
def test_prompt_excludes_unselected_papers(monkeypatch):
    summaries = sample_summaries()

    captured_prompt = {}

    def fake_generate_answer(prompt):
        captured_prompt["text"] = prompt
        return "Comparison report"

    monkeypatch.setattr(
        comparator,
        "generate_answer",
        fake_generate_answer
    )

    comparator.compare_research_papers(
        summaries,
        ["Paper1.pdf", "Paper2.pdf"]
    )

    assert "Paper3.pdf" not in captured_prompt["text"]


# Test 7: Accept four papers
def test_accepts_four_papers(monkeypatch):
    summaries = sample_summaries()

    monkeypatch.setattr(
        comparator,
        "generate_answer",
        lambda prompt: "Four paper comparison completed."
    )

    result = comparator.compare_research_papers(
        summaries,
        [
            "Paper1.pdf",
            "Paper2.pdf",
            "Paper3.pdf",
            "Paper4.pdf"
        ]
    )

    assert result == "Four paper comparison completed."


# Test 8: Return generated comparison
def test_returns_generated_answer(monkeypatch):
    summaries = sample_summaries()

    expected_report = "Detailed comparison of research papers."

    monkeypatch.setattr(
        comparator,
        "generate_answer",
        lambda prompt: expected_report
    )

    result = comparator.compare_research_papers(
        summaries,
        ["Paper1.pdf", "Paper2.pdf"]
    )

    assert result == expected_report
