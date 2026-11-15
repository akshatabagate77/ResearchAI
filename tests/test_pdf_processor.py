
import pymupdf
import pytest

from src.pdf_processor import extract_pages_from_pdf


def create_sample_pdf():
    """Create a small PDF in memory for testing."""

    document = pymupdf.open()

    page = document.new_page()
    page.insert_text((50, 50), "Artificial Intelligence")

    page = document.new_page()
    page.insert_text((50, 50), "Machine Learning")

    pdf_bytes = document.tobytes()
    document.close()

    return pdf_bytes


def test_extract_pages_returns_all_pages():

    pdf_bytes = create_sample_pdf()

    pages = extract_pages_from_pdf(pdf_bytes)

    assert len(pages) == 2


def test_extract_pages_preserves_page_numbers():

    pdf_bytes = create_sample_pdf()

    pages = extract_pages_from_pdf(pdf_bytes)

    assert pages[0]["page_number"] == 1
    assert pages[1]["page_number"] == 2


def test_extract_pages_extracts_text():

    pdf_bytes = create_sample_pdf()

    pages = extract_pages_from_pdf(pdf_bytes)

    assert "Artificial Intelligence" in pages[0]["text"]
    assert "Machine Learning" in pages[1]["text"]


def test_invalid_pdf_raises_error():

    invalid_data = b"This is not a PDF file."

    with pytest.raises(Exception):
        extract_pages_from_pdf(invalid_data)
