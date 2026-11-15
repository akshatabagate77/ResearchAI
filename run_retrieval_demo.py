
from pathlib import Path

from src.pdf_processor import extract_pages_from_pdf
from src.text_chunker import chunk_pages
from src.embeddings import load_embedding_model, generate_embeddings
from src.vector_store import create_vector_store

from src.semantic_search import (
    search_similar_chunks,
    search_hybrid_chunks
)

from src.reranker import (
    load_reranker_model,
    rerank_results
)


PDF_FILES = [
    Path(r"E:\AI research assistant\test_papers\sample_research_paper_1_student_risk (1).pdf"),
    Path(r"E:\AI research assistant\test_papers\sample_research_paper_2_early_support (1).pdf")
]


def main():
    print("\n========== RESEARCHAI RETRIEVAL TEST ==========\n")

    # Step 1: Extract text from both PDFs
    all_pages = []

    for pdf_path in PDF_FILES:
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        print(f"Extracting: {pdf_path.name}")

        with open(pdf_path, "rb") as pdf_file:
            pages = extract_pages_from_pdf(pdf_file.read())

        for page in pages:
            page["document_name"] = pdf_path.name

        all_pages.extend(pages)

    print(f"\nTotal pages extracted: {len(all_pages)}")

    # Step 2: Chunk the extracted text
    chunks = chunk_pages(all_pages)
    print(f"Total chunks created: {len(chunks)}")

    if not chunks:
        raise ValueError("No text chunks were created.")

    # Step 3: Generate embeddings
    print("\nLoading embedding model...")
    embedding_model = load_embedding_model()

    print("Generating embeddings...")
    embeddings = generate_embeddings(chunks, embedding_model)

    # Step 4: Create FAISS index
    index, chunks = create_vector_store(chunks, embeddings)

    print(f"FAISS index contains {index.ntotal} vectors.")

    # Step 5: Test query
    query = input(
        "\nEnter a research question "
        "(example: How can machine learning identify students at risk?): "
    ).strip()

    if not query:
        raise ValueError("Please enter a research question.")

    # Step 6: Semantic search
    print("\n========== SEMANTIC SEARCH ==========")
    semantic_results = search_similar_chunks(
        query, embedding_model, index, chunks, top_k=5
    )

    for i, result in enumerate(semantic_results, 1):
        print(f"\n{i}. Score: {result['score']:.4f}")
        print(result["chunk"].get("text", "")[:400])

    # Step 7: Hybrid search
    print("\n========== HYBRID SEARCH ==========")
    hybrid_results = search_hybrid_chunks(
        query, embedding_model, index, chunks, top_k=5
    )

    for i, result in enumerate(hybrid_results, 1):
        print(f"\n{i}. Combined score: {result['score']:.4f}")
        print(result["chunk"].get("text", "")[:400])

    # Step 8: Hybrid search + Cross-Encoder reranking
    print("\n========== RERANKED SEARCH ==========")
    reranker_model = load_reranker_model()

    reranked_results = rerank_results(
        query,
        hybrid_results,
        top_k=5,
        model=reranker_model
    )

    for i, result in enumerate(reranked_results, 1):
        print(f"\n{i}. Rerank score: {result['rerank_score']:.4f}")
        print(result["chunk"].get("text", "")[:400])

    print("\n========== ALL TESTS COMPLETED ==========")


if __name__ == "__main__":
    main()
