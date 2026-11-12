
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ---------------------------------------------------------
# STEP 1: EXISTING SEMANTIC SEARCH
# ---------------------------------------------------------

def search_similar_chunks(
    query,
    model,
    index,
    chunks,
    top_k=3
):
    """Retrieve passages using FAISS semantic similarity."""

    if not query or not query.strip():
        raise ValueError("Please enter a question.")

    if index.ntotal == 0:
        raise ValueError("The FAISS index is empty.")

    query_embedding = model.encode(
        [query],
        normalize_embeddings=True
    )

    query_vector = np.asarray(
        query_embedding,
        dtype=np.float32
    )

    query_vector = np.ascontiguousarray(query_vector)

    top_k = min(top_k, index.ntotal)

    scores, indices = index.search(
        query_vector,
        top_k
    )

    results = []

    for score, chunk_index in zip(scores[0], indices[0]):

        if chunk_index == -1:
            continue

        results.append({
            "score": float(score),
            "chunk": chunks[int(chunk_index)]
        })

    return results


# ---------------------------------------------------------
# STEP 2: NEW KEYWORD SEARCH
# ---------------------------------------------------------

def search_keyword_chunks(
    query,
    chunks,
    top_k=10
):
    """
    Retrieve passages using TF-IDF keyword similarity.

    This helps find exact technical terms and phrases.
    """

    if not query or not query.strip():
        raise ValueError("Please enter a question.")

    if not chunks:
        return []

    documents = [
        chunk.get("text", "")
        for chunk in chunks
    ]

    try:

        vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            sublinear_tf=True
        )

        document_vectors = vectorizer.fit_transform(
            documents
        )

        query_vector = vectorizer.transform(
            [query]
        )

        scores = cosine_similarity(
            query_vector,
            document_vectors
        )[0]

    except ValueError:
        # Handles empty vocabulary or empty PDF text.
        return []

    ranked_indices = np.argsort(scores)[::-1]

    results = []

    for chunk_index in ranked_indices[:top_k]:

        score = float(scores[chunk_index])

        if score <= 0:
            continue

        results.append({
            "score": score,
            "chunk": chunks[int(chunk_index)]
        })

    return results


# ---------------------------------------------------------
# STEP 3: NEW HYBRID SEARCH
# ---------------------------------------------------------

def search_hybrid_chunks(
    query,
    model,
    index,
    chunks,
    top_k=5,
    semantic_weight=0.7
):
    """
    Combine semantic and keyword search.

    Final score:
        70% semantic similarity
        30% keyword similarity

    Returns ranked passages with individual scores.
    """

    if not query or not query.strip():
        raise ValueError("Please enter a question.")

    if not 0 <= semantic_weight <= 1:
        raise ValueError(
            "semantic_weight must be between 0 and 1."
        )

    if top_k <= 0:
        raise ValueError("top_k must be greater than zero.")

    if index.ntotal == 0:
        raise ValueError("The FAISS index is empty.")

    if len(chunks) != index.ntotal:
        raise ValueError(
            "The number of chunks does not match "
            "the number of FAISS vectors."
        )

    candidate_k = min(
        top_k * 4,
        index.ntotal
    )

    # Get semantic search results.
    semantic_results = search_similar_chunks(
        query=query,
        model=model,
        index=index,
        chunks=chunks,
        top_k=candidate_k
    )

    # Get keyword search results.
    keyword_results = search_keyword_chunks(
        query=query,
        chunks=chunks,
        top_k=candidate_k
    )

    # Store scores using chunk positions.
    semantic_scores = {
        id(result["chunk"]): result["score"]
        for result in semantic_results
    }

    keyword_scores = {
        id(result["chunk"]): result["score"]
        for result in keyword_results
    }

    # Combine candidates from both search methods.
    candidate_chunks = {}

    for result in semantic_results + keyword_results:

        chunk = result["chunk"]

        candidate_chunks[id(chunk)] = chunk

    hybrid_results = []

    for chunk_id, chunk in candidate_chunks.items():

        semantic_score = semantic_scores.get(
            chunk_id,
            0.0
        )

        keyword_score = keyword_scores.get(
            chunk_id,
            0.0
        )

        hybrid_score = (
            semantic_weight * semantic_score
            + (1 - semantic_weight) * keyword_score
        )

        hybrid_results.append({

            "score": float(hybrid_score),

            "semantic_score": float(semantic_score),

            "keyword_score": float(keyword_score),

            "chunk": chunk

        })

    # Sort from highest to lowest combined score.
    hybrid_results.sort(
        key=lambda result: result["score"],
        reverse=True
    )

    return hybrid_results[:top_k]

