
from src.llm import generate_answer


def compare_research_papers(summaries, selected_documents):

    # Validate the selected documents

    if len(selected_documents) < 2:
        raise ValueError(
            "Please select at least two research papers to compare."
        )

    if len(selected_documents) > 4:
        raise ValueError(
            "Please select a maximum of four papers at a time."
        )

    # Collect summaries of selected documents

    comparison_context = ""

    for document_name in selected_documents:

        if document_name not in summaries:
            raise ValueError(
                f"Summary not found for {document_name}. "
                "Generate its summary first."
            )

        comparison_context += (
            f"\n\n{'=' * 50}\n"
            f"RESEARCH PAPER: {document_name}\n"
            f"{'=' * 50}\n\n"
            f"{summaries[document_name]}\n"
        )

    # Prepare Gemini prompt

    prompt = f"""
    You are an expert academic research analyst.

    Your task is to compare multiple research papers.

    Use ONLY the summaries provided below.

    Do not invent experimental results, algorithms,
    datasets, numerical values, or research findings.

    If information is missing, clearly mention:
    "Not specified in the available summary."

    RESEARCH PAPERS:

    {comparison_context}

    Prepare a detailed comparison report using
    the following sections:

    1. OVERVIEW
       Briefly explain the focus of each paper.

    2. RESEARCH OBJECTIVES
       Compare the problems and objectives.

    3. METHODOLOGY COMPARISON
       Explain the approaches used by each paper.

    4. TECHNOLOGIES AND ALGORITHMS
       Compare the algorithms, models, and technologies.

    5. KEY FINDINGS
       Compare the findings reported in the summaries.

    6. SIMILARITIES
       Identify common approaches, objectives, or findings.

    7. DIFFERENCES
       Explain the major differences between the papers.

    8. LIMITATIONS
       Compare the limitations identified by the authors.

    9. POTENTIAL RESEARCH GAPS
       Suggest possible research directions based on
       the limitations and differences.

       Clearly label these as your analysis, not as
       findings claimed by the authors.

    10. FINAL COMPARISON
        Give a concise conclusion explaining what each
        paper contributes and where they differ.

    Use clear headings and tables where appropriate.

    COMPARISON REPORT:
    """

    # Generate comparison using Gemini

    comparison = generate_answer(prompt)

    return comparison
