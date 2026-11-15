
from src.citation_extractor import (
    extract_doi,
    extract_urls,
    extract_year,
    extract_references_from_text,
    extract_document_references
)


# Test 1: Extract DOI
def test_extract_doi():
    text = "Research DOI: 10.1234/example.2024"

    result = extract_doi(text)

    assert result == ["10.1234/example.2024"]


# Test 2: Remove duplicate DOIs
def test_remove_duplicate_dois():
    text = (
        "10.1234/example "
        "10.1234/example"
    )

    result = extract_doi(text)

    assert result == ["10.1234/example"]


# Test 3: Remove punctuation after DOI
def test_remove_doi_punctuation():
    text = "DOI: 10.1234/example."

    result = extract_doi(text)

    assert result == ["10.1234/example"]


# Test 4: Return empty list when DOI is missing
def test_no_doi_found():
    result = extract_doi("No DOI available.")

    assert result == []


# Test 5: Extract URLs
def test_extract_urls():
    text = "Visit https://example.com/research"

    result = extract_urls(text)

    assert result == ["https://example.com/research"]


# Test 6: Remove duplicate URLs
def test_remove_duplicate_urls():
    text = (
        "https://example.com "
        "https://example.com"
    )

    result = extract_urls(text)

    assert result == ["https://example.com"]


# Test 7: Remove punctuation after URL
def test_remove_url_punctuation():
    text = "Website: https://example.com."

    result = extract_urls(text)

    assert result == ["https://example.com"]


# Test 8: Extract publication year
def test_extract_year():
    text = "This research was published in 2024."

    result = extract_year(text)

    assert result == "2024"


# Test 9: Return None when year is missing
def test_no_year_found():
    result = extract_year("Publication year unavailable.")

    assert result is None


# Test 10: Extract numbered references
def test_extract_references():
    text = """
    Introduction
    This is a research paper.

    References
    [1] Smith, J. Machine Learning. 2022.
    [2] John, A. Deep Learning. 2023.
    """

    result = extract_references_from_text(text)

    assert len(result) == 2
    assert "Machine Learning" in result[0]["reference"]
    assert "Deep Learning" in result[1]["reference"]


# Test 11: Extract DOI and URL from references
def test_extract_doi_and_url_from_references():
    text = """
    References
    [1] Smith. AI Research. 2024.
    DOI: 10.1234/example
    Website: https://example.com
    """

    result = extract_references_from_text(text)

    assert len(result) == 1
    assert result[0]["year"] == "2024"
    assert result[0]["dois"] == ["10.1234/example"]
    assert result[0]["urls"] == ["https://example.com"]


# Test 12: Support Bibliography heading
def test_support_bibliography_heading():
    text = """
    Bibliography
    1. Smith. AI Research. 2023.
    """

    result = extract_references_from_text(text)

    assert len(result) == 1
    assert result[0]["year"] == "2023"


# Test 13: Return empty list without reference heading
def test_no_reference_heading():
    text = """
    This is a research paper.
    [1] Smith. AI Research. 2023.
    """

    result = extract_references_from_text(text)

    assert result == []


# Test 14: Group references by document
def test_extract_document_references():
    chunks = [
        {
            "document_name": "Paper1.pdf",
            "text": """
            References
            [1] Smith. Machine Learning. 2022.
            """
        },
        {
            "document_name": "Paper2.pdf",
            "text": """
            References
            [1] John. Deep Learning. 2023.
            """
        }
    ]

    result = extract_document_references(chunks)

    assert "Paper1.pdf" in result
    assert "Paper2.pdf" in result
    assert result["Paper1.pdf"][0]["year"] == "2022"
    assert result["Paper2.pdf"][0]["year"] == "2023"


# Test 15: Handle empty document chunks
def test_empty_document_chunks():
    result = extract_document_references([])

    assert result == {}


# Test 16: Handle empty or whitespace-only reference text
def test_empty_or_whitespace_reference_text():

    assert extract_references_from_text("") == []
    assert extract_references_from_text("   ") == []
    assert extract_references_from_text("\n\t") == []
    assert extract_references_from_text(None) == []
