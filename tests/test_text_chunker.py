
from src.text_chunker import chunk_pages


def test_chunk_pages_returns_chunks():
    pages = [
        {
            "page_number": 1,
            "text": "Artificial intelligence is transforming technology."
        }
    ]

    chunks = chunk_pages(pages, chunk_size=20, overlap=5)

    assert len(chunks) > 0


def test_chunk_pages_preserves_page_number():
    pages = [
        {
            "page_number": 3,
            "text": "Machine learning is useful."
        }
    ]

    chunks = chunk_pages(pages)

    assert chunks[0]["page_number"] == 3


def test_chunk_ids_are_sequential():
    pages = [
        {
            "page_number": 1,
            "text": "Artificial intelligence and machine learning are important technologies."
        }
    ]

    chunks = chunk_pages(pages, chunk_size=25, overlap=5)

    for number, chunk in enumerate(chunks, start=1):
        assert chunk["chunk_id"] == number


def test_empty_pages_return_empty_list():
    chunks = chunk_pages([])

    assert chunks == []


def test_invalid_chunk_size_raises_error():
    pages = []

    try:
        chunk_pages(pages, chunk_size=0)
        assert False, "Expected ValueError"
    except ValueError:
        assert True


def test_invalid_overlap_raises_error():
    pages = []

    try:
        chunk_pages(pages, chunk_size=100, overlap=100)
        assert False, "Expected ValueError"
    except ValueError:
        assert True
