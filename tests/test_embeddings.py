
import numpy as np
from unittest.mock import patch

from src.embeddings import (
    load_embedding_model,
    generate_embeddings,
    MODEL_NAME
)


# A fake model for testing
class FakeModel:

    def encode(self, texts, normalize_embeddings, show_progress_bar):

        return np.array([
            [0.1, 0.2, 0.3]
            for text in texts
        ])


def test_load_embedding_model():

    with patch("src.embeddings.SentenceTransformer") as mock_model:

        load_embedding_model()

        mock_model.assert_called_once_with(MODEL_NAME)


def test_generate_embeddings():

    chunks = [
        {"text": "Artificial Intelligence"},
        {"text": "Machine Learning"}
    ]

    model = FakeModel()

    embeddings = generate_embeddings(chunks, model)

    assert embeddings.shape == (2, 3)


def test_generate_embeddings_passes_correct_text():

    chunks = [
        {"text": "Artificial Intelligence"},
        {"text": "Deep Learning"}
    ]

    model = FakeModel()

    with patch.object(model, "encode", wraps=model.encode) as mock_encode:

        generate_embeddings(chunks, model)

        mock_encode.assert_called_once_with(
            ["Artificial Intelligence", "Deep Learning"],
            normalize_embeddings=True,
            show_progress_bar=True
        )


def test_generate_embeddings_with_empty_chunks():

    model = FakeModel()

    embeddings = generate_embeddings([], model)

    assert embeddings.shape == (0,)
