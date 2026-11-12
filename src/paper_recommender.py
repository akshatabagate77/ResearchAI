
import numpy as np


def recommend_papers(
    query,
    model,
    index,
    chunks,
    top_k=10,
    min_score=0.20
):
    """
    Recommend research papers based on semantic similarity
    between the user's topic and uploaded PDF chunks.

    Returns a ranked list of recommended papers.
    """

    if not query or not query.strip():
        raise ValueError("Please enter a research topic.")

    if index is None or not chunks:
        return []

    # 1. Convert the user's topic into an embedding
    query_embedding = model.encode(
        [query],
        normalize_embeddings=True
    )

    query_embedding = np.asarray(
        query_embedding,
        dtype="float32"
    )

    # 2. Search FAISS for relevant text chunks
    search_count = min(top_k * 3, len(chunks))

    scores, indices = index.search(
        query_embedding,
        search_count
    )

    # 3. Group matching chunks by research paper
    paper_results = {}

    for score, chunk_index in zip(scores[0], indices[0]):

        if chunk_index < 0:
            continue

        score = float(score)

        if score < min_score:
            continue

        chunk = chunks[int(chunk_index)]

        document_name = chunk.get(
            "document_name",
            "Unknown document"
        )

        if document_name not in paper_results:
            paper_results[document_name] = {
                "document_name": document_name,
                "best_score": score,
                "matched_chunks": [],
                "pages": set()
            }

        paper = paper_results[document_name]

        paper["best_score"] = max(
            paper["best_score"],
            score
        )

        paper["matched_chunks"].append({
            "text": chunk.get("text", ""),
            "page_number": chunk.get("page_number", 0),
            "score": score
        })

        paper["pages"].add(
            chunk.get("page_number", 0)
        )

    # 4. Prepare final recommendations
    recommendations = []

    for paper in paper_results.values():

        paper["matched_chunks"].sort(
            key=lambda item: item["score"],
            reverse=True
        )

        recommendations.append({
            "document_name": paper["document_name"],
            "relevance_score": round(
                paper["best_score"] * 100, 2
            ),
            "matched_pages": sorted(paper["pages"]),
            "matched_chunks": paper["matched_chunks"][:3]
        })

    # 5. Rank papers by relevance
    recommendations.sort(
        key=lambda item: item["relevance_score"],
        reverse=True
    )

    return recommendations[:top_k]
