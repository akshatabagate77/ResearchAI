
import networkx as nx

from src.knowledge_graph import (
    build_knowledge_graph,
    get_graph_statistics
)


# Sample research paper chunks
def sample_chunks():
    return [
        {
            "document_name": "Paper1.pdf",
            "text": (
                "Machine learning uses algorithms "
                "to learn from data."
            )
        },
        {
            "document_name": "Paper2.pdf",
            "text": (
                "Deep learning uses neural networks "
                "to process data."
            )
        }
    ]


# Test 1: Build a knowledge graph
def test_build_knowledge_graph():
    graph = build_knowledge_graph(sample_chunks())

    assert isinstance(graph, nx.Graph)
    assert graph.number_of_nodes() > 0


# Test 2: Return empty graph for empty chunks
def test_empty_chunks_return_empty_graph():
    graph = build_knowledge_graph([])

    assert isinstance(graph, nx.Graph)
    assert graph.number_of_nodes() == 0
    assert graph.number_of_edges() == 0


# Test 3: Handle blank text
def test_blank_text_returns_empty_graph():
    chunks = [
        {"document_name": "Paper1.pdf", "text": "   "},
        {"document_name": "Paper2.pdf", "text": ""}
    ]

    graph = build_knowledge_graph(chunks)

    assert graph.number_of_nodes() == 0


# Test 4: Every node has paper information
def test_nodes_contain_paper_information():
    graph = build_knowledge_graph(sample_chunks())

    for node, data in graph.nodes(data=True):
        assert "paper_count" in data
        assert "papers" in data
        assert data["paper_count"] >= 1


# Test 5: Nodes track papers containing concepts
def test_nodes_track_document_names():
    graph = build_knowledge_graph(sample_chunks())

    all_papers = []

    for _, data in graph.nodes(data=True):
        all_papers.extend(data["papers"])

    assert "Paper1.pdf" in all_papers
    assert "Paper2.pdf" in all_papers


# Test 6: Edges contain weights
def test_edges_have_weights():
    graph = build_knowledge_graph(sample_chunks())

    for first, second, data in graph.edges(data=True):
        assert "weight" in data
        assert data["weight"] >= 1


# Test 7: Apply minimum edge weight
def test_min_edge_weight_filters_connections():
    graph = build_knowledge_graph(
        sample_chunks(),
        min_edge_weight=100
    )

    assert graph.number_of_edges() == 0


# Test 8: Respect top keywords per chunk
def test_respects_top_keywords_per_chunk():
    graph = build_knowledge_graph(
        sample_chunks(),
        top_keywords_per_chunk=1
    )

    assert graph.number_of_nodes() <= 2


# Test 9: Calculate graph statistics
def test_get_graph_statistics():
    graph = nx.Graph()

    graph.add_edge("machine learning", "neural network")
    graph.add_node("deep learning")

    result = get_graph_statistics(graph)

    assert result["total_concepts"] == 3
    assert result["total_connections"] == 1
    assert result["connected_groups"] == 2


# Test 10: Handle empty graph statistics
def test_empty_graph_statistics():
    graph = nx.Graph()

    result = get_graph_statistics(graph)

    assert result["total_concepts"] == 0
    assert result["total_connections"] == 0
    assert result["connected_groups"] == 0


# Test 11: Handle text containing only stop words
def test_stop_words_only_returns_empty_graph():

    chunks = [
        {
            "document_name": "Paper1.pdf",
            "text": "the is and are"
        }
    ]

    graph = build_knowledge_graph(chunks)

    assert isinstance(graph, nx.Graph)
    assert graph.number_of_nodes() == 0
    assert graph.number_of_edges() == 0


# Test 12: Skip chunks with no meaningful TF-IDF features
def test_skip_chunk_with_no_tfidf_features():

    chunks = [
        {
            "document_name": "Paper1.pdf",
            "text": "the is and"
        },
        {
            "document_name": "Paper2.pdf",
            "text": "machine learning algorithms process data"
        }
    ]

    graph = build_knowledge_graph(chunks)

    assert isinstance(graph, nx.Graph)
    assert graph.number_of_nodes() > 0
    assert "machine learning" in graph.nodes
