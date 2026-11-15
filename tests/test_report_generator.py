
import pytest
import pymupdf

import src.report_generator as report_generator


# Sample research paper summaries
def sample_summaries():
    return {
        "Paper1.pdf": "This paper studies machine learning.",
        "Paper2.pdf": "This paper studies deep learning."
    }


# Helper function to extract text from generated PDF
def extract_pdf_text(pdf_bytes):
    document = pymupdf.open(
        stream=pdf_bytes,
        filetype="pdf"
    )

    text = ""

    for page in document:
        text += page.get_text()

    document.close()

    return text


# Test 1: Reject empty summaries
def test_rejects_empty_summaries():
    with pytest.raises(ValueError):
        report_generator.generate_research_report({})


# Test 2: Generate a valid PDF
def test_generates_valid_pdf():
    result = report_generator.generate_research_report(
        sample_summaries()
    )

    assert isinstance(result, bytes)
    assert result.startswith(b"%PDF")


# Test 3: Include paper names in PDF
def test_pdf_contains_paper_names():
    result = report_generator.generate_research_report(
        sample_summaries()
    )

    pdf_text = extract_pdf_text(result)

    assert "Paper1.pdf" in pdf_text
    assert "Paper2.pdf" in pdf_text


# Test 4: Include research summaries in PDF
def test_pdf_contains_summaries():
    result = report_generator.generate_research_report(
        sample_summaries()
    )

    pdf_text = extract_pdf_text(result)

    assert "machine learning" in pdf_text
    assert "deep learning" in pdf_text


# Test 5: Include comparison report
def test_pdf_contains_comparison_report():
    result = report_generator.generate_research_report(
        sample_summaries(),
        comparison_report="Both papers focus on artificial intelligence."
    )

    pdf_text = extract_pdf_text(result)

    assert "Comparative Research Analysis" in pdf_text
    assert "Both papers focus on artificial intelligence." in pdf_text


# Test 6: Include research gap report
def test_pdf_contains_gap_report():
    result = report_generator.generate_research_report(
        sample_summaries(),
        gap_report="More research is needed in model evaluation."
    )

    pdf_text = extract_pdf_text(result)

    assert "Research Gap Analysis" in pdf_text
    assert "More research is needed in model evaluation." in pdf_text


# Test 7: Include both optional reports
def test_pdf_contains_both_reports():
    result = report_generator.generate_research_report(
        sample_summaries(),
        comparison_report="Comparison of both papers.",
        gap_report="Potential research gaps."
    )

    pdf_text = extract_pdf_text(result)

    assert "Comparison of both papers." in pdf_text
    assert "Potential research gaps." in pdf_text


# Test 8: Escape special HTML characters
def test_format_inline_text_escapes_html():
    result = report_generator.format_inline_text(
        "Research & AI <Technology>"
    )

    assert "&amp;" in result
    assert "&lt;" in result
    assert "&gt;" in result


# Test 9: Handle empty Markdown content
def test_add_markdown_content_handles_empty_content():
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import Paragraph

    story = []
    styles = getSampleStyleSheet()

    report_generator.add_markdown_content(
        story,
        "",
        styles
    )

    assert len(story) == 1
    assert isinstance(story[0], Paragraph)



# Test 10: Handle all Markdown formatting types
def test_add_markdown_content_handles_all_markdown_types():

    from reportlab.lib.styles import (
        getSampleStyleSheet,
        ParagraphStyle
    )

    from reportlab.platypus import (
        Paragraph,
        Spacer,
        HRFlowable
    )

    story = []
    styles = getSampleStyleSheet()

    # Add the custom style required by the report generator
    styles.add(
        ParagraphStyle(
            name="BulletText",
            parent=styles["BodyText"],
            leftIndent=18,
            firstLineIndent=-10
        )
    )

    content = """
# Main Heading
## Subheading
### Smaller Heading

- First bullet
* Second bullet

1. Numbered point

---

***

Normal paragraph with **bold** and *italic* text.
"""

    report_generator.add_markdown_content(
        story,
        content,
        styles
    )

    assert any(isinstance(item, Paragraph) for item in story)
    assert any(isinstance(item, Spacer) for item in story)
    assert any(isinstance(item, HRFlowable) for item in story)

    paragraph_text = [
        item.getPlainText()
        for item in story
        if isinstance(item, Paragraph)
    ]

    assert "Main Heading" in paragraph_text
    assert "Subheading" in paragraph_text
    assert "Smaller Heading" in paragraph_text
    assert "First bullet" in " ".join(paragraph_text)
    assert "Second bullet" in " ".join(paragraph_text)
    assert "1. Numbered point" in paragraph_text
    assert "Normal paragraph" in " ".join(paragraph_text)
