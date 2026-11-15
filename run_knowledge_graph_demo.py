
from src.knowledge_graph import (
    build_knowledge_graph,
    get_graph_statistics
)

sample_chunks = [
    {
        "document_name": "AI_Healthcare.pdf",
        "page_number": 1,
        "text": (
            "Artificial intelligence and machine learning "
            "are transforming healthcare through medical "
            "diagnosis and predictive analytics."
        )
    },
    {
        "document_name": "Deep_Learning.pdf",
        "page_number": 2,
        "text": (
            "Deep learning uses neural networks for "
            "medical image analysis and disease prediction."
        )
    },
    {
        "document_name": "AI_Healthcare.pdf",
        "page_number": 3,
        "text": (
            "Machine learning models support healthcare "
            "professionals in clinical decision making."
        )
    }
]

graph = build_knowledge_graph(sample_chunks)

statistics = get_graph_statistics(graph)

print("Knowledge Graph Test")
print("--------------------")
print("Concepts:", statistics["total_concepts"])
print("Connections:", statistics["total_connections"])
print("Connected Groups:", statistics["connected_groups"])

print("\nSample Connections:")

for first, second, data in list(graph.edges(data=True))[:5]:
    print(first, "<-->", second, "| Weight:", data["weight"])
