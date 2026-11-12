
from src.embeddings import load_embedding_model
from src.vector_store import create_vector_store
from src.rag_pipeline import answer_question

# Load embedding model
model = load_embedding_model()

# Sample research paper content
chunks = [
    {
        "chunk_id": 1,
        "page_number": 1,
        "text": (
            "Machine learning is a branch of artificial "
            "intelligence that enables computers to learn "
            "patterns from data and make predictions."
        )
    },
    {
        "chunk_id": 2,
        "page_number": 2,
        "text": (
            "Deep learning uses neural networks with "
            "multiple layers to learn complex patterns."
        )
    }
]

# Generate embeddings
embeddings = model.encode(
    [chunk["text"] for chunk in chunks],
    normalize_embeddings=True
)

# Create FAISS vector store
index, chunks = create_vector_store(
    chunks,
    embeddings
)

# Ask a question
question = "What is machine learning?"

# Generate RAG answer
result = answer_question(
    query=question,
    model=model,
    index=index,
    chunks=chunks,
    top_k=2
)

# Display the result
print("\nAI ANSWER:\n")
print(result["answer"])

print("\nSOURCES:\n")

for source in result["sources"]:
    print("Page:", source["page_number"])
    print("Similarity:", source["score"])
