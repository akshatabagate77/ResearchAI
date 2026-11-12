
import json
from pathlib import Path

from src.pdf_processor import extract_pages_from_pdf
from src.text_chunker import chunk_pages
from src.embeddings import load_embedding_model, generate_embeddings
from src.vector_store import create_vector_store
from src.evaluation import evaluate_all_retrieval_methods


PDF_FILES = [
    Path(r"E:\AI research assistant\test_papers\sample_research_paper_1_student_risk (1).pdf"),
    Path(r"E:\AI research assistant\test_papers\sample_research_paper_2_early_support (1).pdf")
]

DATASET_PATH = "evaluation_questions.json"


def main():
    print("\n========== RESEARCHAI EVALUATION ==========\n")

    # 1. Extract PDF pages
    all_pages = []

    for pdf_path in PDF_FILES:
        print(f"Reading: {pdf_path.name}")

        with open(pdf_path, "rb") as pdf_file:
            pages = extract_pages_from_pdf(pdf_file.read())

        for page in pages:
            page["document_name"] = pdf_path.name

        all_pages.extend(pages)

    # 2. Create chunks
    chunks = chunk_pages(all_pages)
    print(f"\nTotal chunks: {len(chunks)}")

    if not chunks:
        raise ValueError("No chunks found in the PDFs.")

    # 3. Load embedding model
    print("\nLoading embedding model...")
    model = load_embedding_model()

    # 4. Generate embeddings
    print("Generating embeddings...")
    embeddings = generate_embeddings(chunks, model)

    # 5. Create vector database
    index, chunks = create_vector_store(chunks, embeddings)
    print(f"FAISS vectors: {index.ntotal}")

    # 6. Run all three evaluation methods
    print("\nRunning retrieval benchmark...")
    results = evaluate_all_retrieval_methods(
        dataset_path=DATASET_PATH,
        model=model,
        index=index,
        chunks=chunks,
        top_k=5
    )

    # 7. Display results
    print("\n========== BENCHMARK RESULTS ==========\n")
    print(json.dumps(results, indent=4))

    # 8. Save results
    output_path = Path("retrieval_evaluation_results.json")

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=4)

    print(f"\nResults saved to: {output_path.resolve()}")
    print("\n========== EVALUATION COMPLETED ==========")


if __name__ == "__main__":
    main()
