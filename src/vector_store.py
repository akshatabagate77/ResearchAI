
import faiss
import numpy as np


def create_vector_store(chunks, embeddings):

    # Step 1: Check whether we have any chunks
    if len(chunks) == 0:
        raise ValueError("No text chunks available to store.")

    # Step 2: Convert embeddings into a NumPy array
    vectors = np.asarray(
        embeddings,
        dtype=np.float32
    )

    # Step 3: Check that every chunk has an embedding
    if len(chunks) != vectors.shape[0]:
        raise ValueError(
            "The number of chunks and embeddings must match."
        )

    # Step 4: Ensure vectors are stored in contiguous memory
    vectors = np.ascontiguousarray(vectors)

    # Step 5: Get the number of dimensions in each vector
    dimension = vectors.shape[1]

    # Step 6: Create a FAISS index using inner product
    index = faiss.IndexFlatIP(dimension)

    # Step 7: Add all embeddings to the FAISS index
    index.add(vectors)

    # Step 8: Return the index and original chunks
    return index, chunks
