
from src.reranker import rerank_results

sample_results = [
    {
        "score": 0.75,
        "chunk": {
            "document_name": "paper1.pdf",
            "page_number": 2,
            "text": "CNN models use convolutional layers to extract image features."
        }
    },
    {
        "score": 0.80,
        "chunk": {
            "document_name": "paper2.pdf",
            "page_number": 5,
            "text": "The proposed CNN achieved 94 percent accuracy on the test dataset."
        }
    },
    {
        "score": 0.60,
        "chunk": {
            "document_name": "paper3.pdf",
            "page_number": 1,
            "text": "Cloud computing provides scalable storage infrastructure."
        }
    }
]

query = "What accuracy did the CNN model achieve?"

results = rerank_results(
    query=query,
    search_results=sample_results,
    top_k=3
)

print("\nRERANKED RESULTS\n")

for result in results:
    print("Document:", result["chunk"]["document_name"])
    print("Score:", result["rerank_score"])
    print("Text:", result["chunk"]["text"])
    print("-" * 50)
