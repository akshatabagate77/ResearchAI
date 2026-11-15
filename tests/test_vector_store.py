
import numpy as np
import pytest

from src.vector_store import create_vector_store


def test_vector_store_creation():

    chunks = [
        {"chunk_id": 1, "text": "Artificial Intelligence"},
        {"chunk_id": 2, "text": "Machine Learning"}
    ]

    embeddings = np.array([
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6]
    ])

    index, stored_chunks = create_vector_store(
        chunks,
        embeddings
    )

    assert index.ntotal == 2
    assert index.d == 3
    assert stored_chunks == chunks


def test_vector_store_preserves_chunks():

    chunks = [
        {"chunk_id": 1, "text": "Deep Learning"},
        {"chunk_id": 2, "text": "Neural Networks"}
    ]

    embeddings = np.array([
        [0.1, 0.2],
        [0.3, 0.4]
    ])

    index, stored_chunks = create_vector_store(
        chunks,
        embeddings
    )

    assert stored_chunks[0]["text"] == "Deep Learning"
    assert stored_chunks[1]["text"] == "Neural Networks"


def test_empty_chunks_raise_error():

    with pytest.raises(ValueError):
        create_vector_store([], np.array([]))


def test_mismatched_embeddings_raise_error():

    chunks = [
        {"chunk_id": 1, "text": "Artificial Intelligence"},
        {"chunk_id": 2, "text": "Machine Learning"}
    ]

    embeddings = np.array([
        [0.1, 0.2, 0.3]
    ])

    with pytest.raises(ValueError):
        create_vector_store(chunks, embeddings)


def test_embeddings_are_stored_as_float32():

    chunks = [
        {"chunk_id": 1, "text": "Artificial Intelligence"}
    ]

    embeddings = np.array([
        [0.1, 0.2, 0.3]
    ], dtype=np.float64)

    index, stored_chunks = create_vector_store(
        chunks,
        embeddings
    )

    assert index.ntotal == 1
    assert index.d == 3
