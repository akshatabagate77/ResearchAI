
from sentence_transformers import SentenceTransformer


# Name of our pretrained embedding model
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def load_embedding_model():

    model = SentenceTransformer(MODEL_NAME)

    return model


def generate_embeddings(chunks, model):

    texts = []

    # Extract text from every chunk
    for chunk in chunks:

        texts.append(chunk["text"])

    # Convert text into numerical vectors
    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    return embeddings
