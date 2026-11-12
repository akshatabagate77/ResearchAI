
import json
import os
import time

from src.semantic_search import (
    search_similar_chunks,
    search_hybrid_chunks
)

from src.reranker import (
    rerank_results,
    load_reranker_model
)


# ---------------------------------------------------------
# STEP 1: LOAD EVALUATION DATASET
# ---------------------------------------------------------

def load_evaluation_questions(dataset_path):

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(
            f"Evaluation dataset not found: {dataset_path}"
        )

    with open(
        dataset_path,
        "r",
        encoding="utf-8"
    ) as file:

        questions = json.load(file)

    if not isinstance(questions, list):
        raise ValueError(
            "The evaluation dataset must contain a JSON list."
        )

    return questions


# ---------------------------------------------------------
# STEP 2: CALCULATE METRICS
# ---------------------------------------------------------

def calculate_metrics(
    question_data,
    retrieved_results,
    top_k,
    latency
):
    """
    Calculate Precision@K, Recall@K, Hit Rate,
    MRR and retrieval latency for one question.
    """

    expected_document = question_data["expected_document"]
    expected_page = question_data["expected_page"]

    relevant_count = 0
    first_relevant_rank = 0

    retrieved_details = []

    for rank, result in enumerate(
        retrieved_results,
        start=1
    ):

        chunk = result["chunk"]

        document_name = chunk.get(
            "document_name",
            "Unknown document"
        )

        page_number = chunk.get(
            "page_number",
            "Unknown"
        )

        is_relevant = (
            document_name == expected_document
            and page_number == expected_page
        )

        if is_relevant:

            relevant_count += 1

            if first_relevant_rank == 0:
                first_relevant_rank = rank

        retrieved_details.append({

            "rank": rank,

            "document": document_name,

            "page": page_number,

            "score": float(
                result.get("score", 0)
            ),

            "rerank_score": result.get(
                "rerank_score"
            ),

            "relevant": is_relevant

        })

    # Precision@K
    precision = relevant_count / top_k

    # Binary Recall@K for one labelled document-page pair.
    recall = 1.0 if relevant_count > 0 else 0.0

    # Reciprocal rank of the first relevant result.
    mrr = (
        1.0 / first_relevant_rank
        if first_relevant_rank > 0
        else 0.0
    )

    return {

        "question": question_data["question"],

        "expected_document": expected_document,

        "expected_page": expected_page,

        "precision": precision,

        "recall": recall,

        "hit": relevant_count > 0,

        "mrr": mrr,

        "latency_ms": latency * 1000,

        "retrieved_results": retrieved_details

    }


# ---------------------------------------------------------
# STEP 3: EVALUATE ONE RETRIEVAL METHOD
# ---------------------------------------------------------

def evaluate_single_question(
    question_data,
    model,
    index,
    chunks,
    top_k=5,
    method="semantic",
    reranker_model=None
):
    """
    Evaluate one question using the selected method.

    Supported methods:
        semantic
        hybrid
        hybrid_reranked
    """

    question = question_data["question"]

    start_time = time.perf_counter()

    # Method 1: Semantic Search
    if method == "semantic":

        retrieved_results = search_similar_chunks(
            query=question,
            model=model,
            index=index,
            chunks=chunks,
            top_k=top_k
        )

    # Method 2: Hybrid Search
    elif method == "hybrid":

        retrieved_results = search_hybrid_chunks(
            query=question,
            model=model,
            index=index,
            chunks=chunks,
            top_k=top_k,
            semantic_weight=0.7
        )

    # Method 3: Hybrid Search + Reranking
    elif method == "hybrid_reranked":

        candidate_results = search_hybrid_chunks(
            query=question,
            model=model,
            index=index,
            chunks=chunks,
            top_k=top_k * 4,
            semantic_weight=0.7
        )

        retrieved_results = rerank_results(
            query=question,
            search_results=candidate_results,
            top_k=top_k,
            model=reranker_model
        )

    else:

        raise ValueError(
            f"Unsupported evaluation method: {method}"
        )

    latency = time.perf_counter() - start_time

    return calculate_metrics(
        question_data=question_data,
        retrieved_results=retrieved_results,
        top_k=top_k,
        latency=latency
    )


# ---------------------------------------------------------
# STEP 4: EVALUATE AN ENTIRE DATASET
# ---------------------------------------------------------

def evaluate_retrieval_method(
    questions,
    model,
    index,
    chunks,
    top_k=5,
    method="semantic",
    reranker_model=None
):
    """
    Evaluate one retrieval method across all questions.
    """

    if not questions:
        raise ValueError("Evaluation dataset is empty.")

    results = []

    for number, question_data in enumerate(
        questions,
        start=1
    ):

        print(
            f"[{method}] Evaluating "
            f"{number}/{len(questions)}"
        )

        result = evaluate_single_question(
            question_data=question_data,
            model=model,
            index=index,
            chunks=chunks,
            top_k=top_k,
            method=method,
            reranker_model=reranker_model
        )

        results.append(result)

    total = len(results)

    return {

        "method": method,

        "total_questions": total,

        "top_k": top_k,

        "precision_at_k": sum(
            item["precision"] for item in results
        ) / total,

        "recall_at_k": sum(
            item["recall"] for item in results
        ) / total,

        "hit_rate": sum(
            item["hit"] for item in results
        ) / total,

        "mrr": sum(
            item["mrr"] for item in results
        ) / total,

        "average_latency_ms": sum(
            item["latency_ms"] for item in results
        ) / total,

        "results": results

    }


# ---------------------------------------------------------
# STEP 5: COMPARE ALL THREE METHODS
# ---------------------------------------------------------

def evaluate_all_retrieval_methods(
    dataset_path,
    model,
    index,
    chunks,
    top_k=5
):
    """
    Compare:
        1. Semantic Search
        2. Hybrid Search
        3. Hybrid Search + Cross-Encoder Reranking
    """

    if top_k <= 0:
        raise ValueError(
            "top_k must be greater than zero."
        )

    if index.ntotal == 0:
        raise ValueError(
            "The FAISS index is empty."
        )

    if len(chunks) != index.ntotal:
        raise ValueError(
            "Chunks and FAISS index sizes do not match."
        )

    questions = load_evaluation_questions(
        dataset_path
    )

    if not questions:
        raise ValueError(
            "Evaluation dataset is empty."
        )

    # Load model before timing individual retrieval queries.
    print("Loading Cross-Encoder for evaluation...")

    reranker_model = load_reranker_model()

    methods = [
        "semantic",
        "hybrid",
        "hybrid_reranked"
    ]

    comparison = {}

    for method in methods:

        print("\n" + "=" * 50)
        print(f"Evaluating: {method}")
        print("=" * 50)

        comparison[method] = evaluate_retrieval_method(
            questions=questions,
            model=model,
            index=index,
            chunks=chunks,
            top_k=top_k,
            method=method,
            reranker_model=reranker_model
        )

    return {

        "total_questions": len(questions),

        "top_k": top_k,

        "comparison": comparison

    }


# ---------------------------------------------------------
# STEP 6: BACKWARD-COMPATIBLE EVALUATION FUNCTION
# ---------------------------------------------------------

def evaluate_retrieval(
    dataset_path,
    model,
    index,
    chunks,
    top_k=5
):
    """
    Preserve the original evaluation function.

    Existing Streamlit code can continue using it.
    """

    questions = load_evaluation_questions(
        dataset_path
    )

    return evaluate_retrieval_method(
        questions=questions,
        model=model,
        index=index,
        chunks=chunks,
        top_k=top_k,
        method="semantic"
    )
