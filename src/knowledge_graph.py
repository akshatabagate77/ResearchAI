
from collections import Counter
from itertools import combinations

import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer


def build_knowledge_graph(
    chunks,
    top_keywords_per_chunk=5,
    min_edge_weight=1
):
    """
    Build a concept graph from uploaded research paper chunks.

    Nodes represent important concepts.
    Edges represent concepts appearing together.
    """

    graph = nx.Graph()

    if not chunks:
        return graph

    texts = [
        chunk.get("text", "").strip()
        for chunk in chunks
    ]

    valid_chunks = [
        (chunk, text)
        for chunk, text in zip(chunks, texts)
        if text
    ]

    if not valid_chunks:
        return graph

    documents = [text for _, text in valid_chunks]

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        max_features=3000
    )

    try:
        matrix = vectorizer.fit_transform(documents)
    except ValueError:
        return graph

    terms = vectorizer.get_feature_names_out()

    edge_counts = Counter()
    node_documents = {}

    for row_number, (chunk, _) in enumerate(valid_chunks):

        row = matrix.getrow(row_number)

        if row.nnz == 0:
            continue

        ranked_indices = row.data.argsort()[::-1][
            :top_keywords_per_chunk
        ]

        keywords = [
            terms[row.indices[index]]
            for index in ranked_indices
        ]

        document_name = chunk.get(
            "document_name",
            "Unknown document"
        )

        for keyword in keywords:

            if keyword not in node_documents:
                node_documents[keyword] = set()

            node_documents[keyword].add(document_name)

        for first, second in combinations(
            sorted(set(keywords)), 2
        ):
            edge_counts[(first, second)] += 1

    # Add concept nodes
    for keyword, documents in node_documents.items():

        graph.add_node(
            keyword,
            paper_count=len(documents),
            papers=list(documents)
        )

    # Add connections between concepts
    for (first, second), weight in edge_counts.items():

        if weight >= min_edge_weight:

            graph.add_edge(
                first,
                second,
                weight=weight
            )

    return graph


def get_graph_statistics(graph):

    return {
        "total_concepts": graph.number_of_nodes(),
        "total_connections": graph.number_of_edges(),
        "connected_groups": nx.number_connected_components(graph)
        if graph.number_of_nodes() > 0 else 0
    }
