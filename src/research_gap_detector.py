
import time

from src.llm import generate_answer


def detect_research_gaps(summaries, selected_documents):

    # STEP 1: Validate selected research papers

    if len(selected_documents) < 2:
        raise ValueError(
            "Please select at least two research papers."
        )

    if len(selected_documents) > 5:
        raise ValueError(
            "Please select a maximum of five papers."
        )

    # STEP 2: Collect summaries

    research_context = ""

    for document_name in selected_documents:

        if document_name not in summaries:
            raise ValueError(
                f"Summary not found for {document_name}. "
                "Generate its summary first."
            )

        research_context += (
            f"\n\nRESEARCH PAPER: {document_name}\n"
            f"{summaries[document_name]}\n"
        )

    # STEP 3: Prepare the AI prompt

    prompt = f"""
    You are an expert academic researcher and research analyst.

    Your task is to identify research gaps by analysing
    the summaries of multiple research papers.

    Use only the information provided below.

    Do not invent datasets, results, experiments,
    limitations, or claims made by the authors.

    Clearly distinguish:
    A. Gaps explicitly mentioned by the authors.
    B. Potential gaps inferred through your analysis.

    RESEARCH PAPERS:

    {research_context}

    Prepare a detailed research gap analysis using
    these sections:

    1. COMMON RESEARCH THEMES
       Identify the main topics shared by the papers.

    2. EXISTING LIMITATIONS
       List the limitations reported by the authors.

    3. IDENTIFIED RESEARCH GAPS
       Explain what remains unexplored or insufficiently
       addressed across the papers.

    4. METHODOLOGICAL GAPS
       Identify possible weaknesses or missing comparisons
       in the methods, experiments, or evaluation approaches.

    5. DATASET AND EVALUATION GAPS
       Identify missing dataset diversity, evaluation
       methods, or validation opportunities, if supported
       by the available summaries.

    6. FUTURE RESEARCH DIRECTIONS
       Suggest practical research directions based on
       the identified gaps.

    7. POTENTIAL RESEARCH QUESTIONS
       Generate five specific research questions that
       a student or researcher could investigate.

    8. FINAL RECOMMENDATION
       Recommend the most promising research direction
       and explain why.

    IMPORTANT:
    - Do not present assumptions as established facts.
    - Mention which paper supports each observation.
    - If information is unavailable, say so.
    - Label your own deductions as potential research gaps.

    RESEARCH GAP ANALYSIS:
    """

    # STEP 4: Generate the research gap report and measure time

    print("Research gap detection started.")
    print("Prompt characters:", len(prompt))

    start_time = time.perf_counter()

    gap_report = generate_answer(prompt)

    end_time = time.perf_counter()

    print(
        f"Gemini response time: {end_time - start_time:.2f} seconds"
    )

    return gap_report
