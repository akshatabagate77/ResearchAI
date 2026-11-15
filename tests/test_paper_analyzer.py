
import pytest

from src.paper_analyzer import (
    detect_sections,
    analyze_paper,
    analyze_research_collection
)


# Test 1: Detect all research paper sections
def test_detect_all_sections():
    text = """
    Abstract
    Introduction
    Methodology
    Results
    Discussion
    Limitations
    Conclusion
    References
    """

    result = detect_sections(text)

    assert all(result.values())


# Test 2: Detect only available sections
def test_detect_partial_sections():
    text = """
    Abstract
    Introduction
    Conclusion
    """

    result = detect_sections(text)

    assert result["Abstract"] is True
    assert result["Introduction"] is True
    assert result["Conclusion"] is True
    assert result["Methodology"] is False


# Test 3: Detect sections without case sensitivity
def test_detect_sections_case_insensitive():
    text = "ABSTRACT INTRODUCTION CONCLUSION"

    result = detect_sections(text)

    assert result["Abstract"] is True
    assert result["Introduction"] is True
    assert result["Conclusion"] is True


# Test 4: Recognize alternative section names
def test_detect_alternative_section_names():
    text = """
    Background
    Proposed Approach
    Findings
    Bibliography
    """

    result = detect_sections(text)

    assert result["Introduction"] is True
    assert result["Methodology"] is True
    assert result["Results"] is True
    assert result["References"] is True


# Test 5: Reject empty paper text
def test_analyze_paper_rejects_empty_text():
    with pytest.raises(ValueError):
        analyze_paper("")


# Test 6: Reject whitespace-only paper text
def test_analyze_paper_rejects_whitespace():
    with pytest.raises(ValueError):
        analyze_paper("   \n  ")


# Test 7: Calculate completeness score
def test_calculate_completeness_score():
    text = """
    Abstract
    Introduction
    Methodology
    Results
    """

    result = analyze_paper(text)

    assert result["present_sections"] == 4
    assert result["total_sections"] == 8
    assert result["completeness_score"] == 50.0


# Test 8: Identify missing sections
def test_identify_missing_sections():
    text = "Abstract Introduction Conclusion"

    result = analyze_paper(text)

    assert "Abstract" not in result["missing_sections"]
    assert "Introduction" not in result["missing_sections"]
    assert "Conclusion" not in result["missing_sections"]
    assert "Methodology" in result["missing_sections"]


# Test 9: Return complete analysis structure
def test_analyze_paper_result_structure():
    result = analyze_paper("Abstract Introduction")

    assert "completeness_score" in result
    assert "present_sections" in result
    assert "total_sections" in result
    assert "detected_sections" in result
    assert "missing_sections" in result


# Test 10: Analyze multiple research papers
def test_analyze_research_collection():
    chunks = [
        {
            "document_name": "Paper1.pdf",
            "text": "Abstract Introduction Conclusion"
        },
        {
            "document_name": "Paper2.pdf",
            "text": "Abstract Methodology Results"
        }
    ]

    result = analyze_research_collection(chunks)

    assert "Paper1.pdf" in result
    assert "Paper2.pdf" in result

    assert result["Paper1.pdf"]["present_sections"] == 3
    assert result["Paper2.pdf"]["present_sections"] == 3


# Test 11: Skip documents with empty text
def test_skip_empty_documents():
    chunks = [
        {
            "document_name": "Empty.pdf",
            "text": "   "
        },
        {
            "document_name": "Paper1.pdf",
            "text": "Abstract Introduction"
        }
    ]

    result = analyze_research_collection(chunks)

    assert "Empty.pdf" not in result
    assert "Paper1.pdf" in result


# Test 12: Handle empty research collection
def test_empty_research_collection():
    result = analyze_research_collection([])

    assert result == {}
