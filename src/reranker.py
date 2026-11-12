
from sentence_transformers import CrossEncoder


# ---------------------------------------------------------
# STEP 1: LOAD CROSS-ENCODER MODEL
# ---------------------------------------------------------

_reranker_model = None

MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def load_reranker_model():
    """
    Load the Cross-Encoder model only once.
    """

    global _reranker_model

    if _reranker_model is None:
        print("ResearchAI: Loading Cross-Encoder model...")

        _reranker_model = CrossEncoder(
            MODEL_NAME
        )

        print("ResearchAI: Reranker loaded successfully.")

    return _reranker_model


# ---------------------------------------------------------
# STEP 2: RERANK SEARCH RESULTS
# ---------------------------------------------------------

def rerank_results(
    query,
    search_results,
    top_k=5,
    model=None
):
    """
    Rerank retrieved passages using a Cross-Encoder.

    Each question and passage is evaluated together.

    Original retrieval scores are preserved.
    """

    if not query or not query.strip():
        raise ValueError("Query cannot be empty.")

    if top_k <= 0:
        raise ValueError("top_k must be greater than zero.")

    if not search_results:
        return []

    if model is None:
        model = load_reranker_model()

    # Create question-passage pairs.
    pairs = []

    for result in search_results:

        chunk = result.get("chunk", {})

        passage = chunk.get("text", "")

        pairs.append([
            query,
            passage
        ])

    # Predict relevance scores.
    scores = model.predict(pairs)

    # Attach reranking scores.
    reranked_results = []

    for result, score in zip(search_results, scores):

        updated_result = result.copy()

        updated_result["original_score"] = float(
            result.get("score", 0.0)
        )

        updated_result["rerank_score"] = float(score)

        reranked_results.append(updated_result)

    # Sort by Cross-Encoder relevance.
    reranked_results.sort(
        key=lambda item: item["rerank_score"],
        reverse=True
    )

    return reranked_results[:top_k]
