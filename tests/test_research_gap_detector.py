
import pytest

import src.research_gap_detector as gap_detector


# Sample research paper summaries
def sample_summaries():
    return {
        "Paper1.pdf": "This paper studies machine learning.",
        "Paper2.pdf": "This paper studies deep learning.",
        "Paper3.pdf": "This paper studies neural networks.",
        "Paper4.pdf": "This paper studies artificial intelligence.",
        "Paper5.pdf": "This paper studies data science.",
        "Paper6.pdf": "This paper studies robotics."
    }


# Test 1: Reject fewer than two papers
def test_rejects_less_than_two_papers():
    summaries = sample_summaries()

    with pytest.raises(ValueError):
        gap_detector.detect_research_gaps(
            summaries,
            ["Paper1.pdf"]
        )


# Test 2: Reject more than five papers
def test_rejects_more_than_five_papers():
    summaries = sample_summaries()

    with pytest.raises(ValueError):
        gap_detector.detect_research_gaps(
            summaries,
            [
                "Paper1.pdf",
                "Paper2.pdf",
                "Paper3.pdf",
                "Paper4.pdf",
                "Paper5.pdf",
                "Paper6.pdf"
            ]
        )


# Test 3: Reject missing paper summary
def test_rejects_missing_summary():
    summaries = sample_summaries()

    with pytest.raises(ValueError, match="Summary not found"):
        gap_detector.detect_research_gaps(
            summaries,
            ["Paper1.pdf", "Unknown.pdf"]
        )


# Test 4: Accept two papers and generate report
def test_generates_research_gap_report(monkeypatch):
    summaries = sample_summaries()

    monkeypatch.setattr(
        gap_detector,
        "generate_answer",
        lambda prompt: "Research gaps identified successfully."
    )

    result = gap_detector.detect_research_gaps(
        summaries,
        ["Paper1.pdf", "Paper2.pdf"]
    )

    assert result == "Research gaps identified successfully."


# Test 5: Check whether paper summaries are included in prompt
def test_prompt_contains_selected_papers(monkeypatch):
    summaries = sample_summaries()

    captured_prompt = {}

    def fake_generate_answer(prompt):
        captured_prompt["text"] = prompt
        return "Research gap report"

    monkeypatch.setattr(
        gap_detector,
        "generate_answer",
        fake_generate_answer
    )

    gap_detector.detect_research_gaps(
        summaries,
        ["Paper1.pdf", "Paper2.pdf"]
    )

    assert "Paper1.pdf" in captured_prompt["text"]
    assert "Paper2.pdf" in captured_prompt["text"]
    assert summaries["Paper1.pdf"] in captured_prompt["text"]
    assert summaries["Paper2.pdf"] in captured_prompt["text"]


# Test 6: Check that unselected papers are not included
def test_prompt_excludes_unselected_papers(monkeypatch):
    summaries = sample_summaries()

    captured_prompt = {}

    def fake_generate_answer(prompt):
        captured_prompt["text"] = prompt
        return "Research gap report"

    monkeypatch.setattr(
        gap_detector,
        "generate_answer",
        fake_generate_answer
    )

    gap_detector.detect_research_gaps(
        summaries,
        ["Paper1.pdf", "Paper2.pdf"]
    )

    assert "Paper3.pdf" not in captured_prompt["text"]


# Test 7: Accept five papers
def test_accepts_five_papers(monkeypatch):
    summaries = sample_summaries()

    monkeypatch.setattr(
        gap_detector,
        "generate_answer",
        lambda prompt: "Five paper analysis completed."
    )

    result = gap_detector.detect_research_gaps(
        summaries,
        [
            "Paper1.pdf",
            "Paper2.pdf",
            "Paper3.pdf",
            "Paper4.pdf",
            "Paper5.pdf"
        ]
    )

    assert result == "Five paper analysis completed."


# Test 8: Return Gemini response
def test_returns_generated_answer(monkeypatch):
    summaries = sample_summaries()

    expected_report = "Common themes and future research directions."

    monkeypatch.setattr(
        gap_detector,
        "generate_answer",
        lambda prompt: expected_report
    )

    result = gap_detector.detect_research_gaps(
        summaries,
        ["Paper1.pdf", "Paper2.pdf"]
    )

    assert result == expected_report
