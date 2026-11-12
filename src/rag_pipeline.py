
import re
import json

from src.semantic_search import search_hybrid_chunks
from src.reranker import rerank_results
from src.llm import generate_answer


# --------------------------------------------------
# STEP 1: DETECT FOLLOW-UP QUESTIONS
# --------------------------------------------------

def is_follow_up_question(query):

    follow_up_patterns = [
        r"\bwhat about\b",
        r"\bhow about\b",
        r"\bexplain more\b",
        r"\btell me more\b",
        r"\belaborate\b",
        r"\bwhy is that\b",
        r"\bhow does it\b",
        r"\bwhat does it mean\b",
        r"\bcan you explain\b",
        r"\bcontinue\b",
        r"\bcompare it\b",
        r"\bwhat are its\b",
        r"\bwhat is its\b",
        r"\bthose results\b",
        r"\bthese findings\b",
    ]

    query_lower = query.lower().strip()

    return any(
        re.search(pattern, query_lower)
        for pattern in follow_up_patterns
    )


# --------------------------------------------------
# STEP 2: PREPARE CONVERSATION HISTORY
# --------------------------------------------------

def get_previous_history(chat_history, current_query):

    if not chat_history:
        return []

    history = list(chat_history)

    if (
        history
        and history[-1].get("role") == "user"
        and history[-1].get("content", "").strip()
        == current_query.strip()
    ):
        history = history[:-1]

    return history


def prepare_chat_history(chat_history, max_messages=6):

    if not chat_history:
        return "No previous conversation."

    recent_history = chat_history[-max_messages:]

    formatted_history = []

    for message in recent_history:

        role = message.get("role", "user")
        content = message.get("content", "")

        if role == "assistant":
            role = "ResearchAI"

        elif role == "user":
            role = "User"

        formatted_history.append(
            f"{role}: {content}"
        )

    return "\n".join(formatted_history)


# --------------------------------------------------
# STEP 3: REWRITE FOLLOW-UP QUESTIONS
# --------------------------------------------------

def rewrite_research_query(query, chat_history):

    if not chat_history or not is_follow_up_question(query):
        return query

    history_text = prepare_chat_history(chat_history)

    rewrite_prompt = f"""
You are a research search-query rewriting assistant.

Rewrite the user's latest question into a clear,
standalone search query using the conversation history.

Rules:
1. Preserve the user's original intent.
2. Include relevant paper topics or methods when needed.
3. Do not answer the question.
4. Return only the rewritten search query.
5. Keep it concise.

CONVERSATION:
{history_text}

LATEST QUESTION:
{query}

STANDALONE SEARCH QUERY:
"""

    try:

        rewritten_query = generate_answer(
            rewrite_prompt,
            max_retries=0,
            temperature=0.1,
            max_output_tokens=150
        )

        rewritten_query = (
            rewritten_query.strip()
            .strip('"')
            .strip("'")
        )

        if (
            rewritten_query
            and len(rewritten_query) <= 500
            and "\n" not in rewritten_query
        ):
            return rewritten_query

    except Exception:
        pass

    # Safe fallback using the previous user question.
    previous_questions = [
        message.get("content", "")
        for message in chat_history
        if message.get("role") == "user"
    ]

    if previous_questions:

        previous_question = previous_questions[-1][:300]

        return f"{previous_question} {query}"

    return query


# --------------------------------------------------
# STEP 4: FILTER RELEVANT RESULTS
# --------------------------------------------------

def filter_relevant_results(search_results, min_score=0.15):
    """
    Filter weak hybrid-search results.

    Hybrid scores combine semantic and keyword similarity.
    This threshold is an initial setting and should be
    tuned using the retrieval evaluation dataset.
    """

    relevant_results = []

    for result in search_results:

        score = float(result.get("score", 0))

        if score >= min_score:
            relevant_results.append(result)

    return relevant_results


# --------------------------------------------------
# STEP 5: BUILD RESEARCH CONTEXT
# --------------------------------------------------

def build_research_context(search_results):

    context_parts = []

    for result in search_results:

        chunk = result.get("chunk", {})

        document_name = chunk.get(
            "document_name",
            "Unknown document"
        )

        page_number = chunk.get(
            "page_number",
            "Unknown"
        )

        text = chunk.get("text", "")

        context_parts.append(
            f"""
Document: {document_name}
Page: {page_number}
Passage:
{text}
"""
        )

    return "\n".join(context_parts)


# --------------------------------------------------
# STEP 6: BUILD FINAL ANSWER PROMPT
# --------------------------------------------------

def build_research_prompt(query, context, chat_history=None):

    history_text = prepare_chat_history(chat_history)

    prompt = f"""
You are ResearchAI, an intelligent research assistant.

Your job is to answer questions using the uploaded
research papers and retrieved passages.

IMPORTANT RULES:

1. Use the provided research context as your primary
   source of information.
2. Do not invent facts, results, statistics, or citations.
3. If the answer is not present in the context,
   clearly say that the uploaded papers do not provide
   enough information.
4. Explain technical concepts in simple language.
5. Use headings and bullet points where appropriate.
6. Mention document names and page numbers when useful.
7. Distinguish between what the paper states and
   your own interpretation.
8. Do not claim that a paper proves something if it
   only suggests or discusses it.
9. Use only the retrieved passages as evidence.
10. Never invent a page number or document name.

CONVERSATION HISTORY:
{history_text}

RESEARCH PAPER CONTEXT:
{context}

USER QUESTION:
{query}

Provide a clear, accurate, evidence-based answer.
"""

    return prompt


# --------------------------------------------------
# STEP 7: VERIFY ANSWER SUPPORT
# --------------------------------------------------

def verify_answer_support(answer, context):
    """
    Verify whether the generated answer is supported
    by the retrieved research-paper passages.

    This uses an additional Gemini API request.
    """

    verification_prompt = f"""
You are a strict research fact-checker.

Check whether the generated answer is supported
by the supplied research-paper context.

RULES:
1. Use only the supplied context.
2. Check factual claims against the context.
3. Do not use outside knowledge.
4. Mark unsupported claims clearly.
5. If evidence is insufficient, mark the answer
   as not fully supported.
6. Return ONLY valid JSON. No Markdown.

RESEARCH CONTEXT:
{context}

GENERATED ANSWER:
{answer}

Return JSON in exactly this format:
{{
    "supported": true,
    "unsupported_claims": [],
    "explanation": "Brief explanation"
}}
"""

    try:

        response = generate_answer(
            verification_prompt,
            max_retries=0,
            temperature=0.0,
            max_output_tokens=500
        )

        response = response.strip()

        # Remove optional Markdown code fences.
        if response.startswith("```"):

            response = re.sub(
                r"^```(?:json)?\s*",
                "",
                response
            )

            response = re.sub(
                r"\s*```$",
                "",
                response
            )

        result = json.loads(response)

        if not isinstance(result, dict):
            raise ValueError("Invalid verification response.")

        if not isinstance(result.get("supported"), bool):
            raise ValueError("Missing supported status.")

        if not isinstance(result.get("unsupported_claims"), list):
            raise ValueError("Invalid unsupported claims.")

        return {
            "status": (
                "supported"
                if result["supported"]
                else "warning"
            ),
            "supported": result["supported"],
            "unsupported_claims": result["unsupported_claims"],
            "explanation": str(
                result.get("explanation", "")
            )
        }

    except Exception:

        return {
            "status": "unavailable",
            "supported": None,
            "unsupported_claims": [],
            "explanation": (
                "Verification could not be completed. "
                "The answer has not been independently verified."
            )
        }


# --------------------------------------------------
# STEP 8: MAIN RAG PIPELINE
# --------------------------------------------------

def answer_question(
    query,
    model,
    index,
    chunks,
    top_k=5,
    chat_history=None
):
    """
    Advanced Retrieval-Augmented Generation pipeline.

    Flow:
    1. Rewrite follow-up questions.
    2. Retrieve passages using hybrid search.
    3. Filter weak hybrid matches.
    4. Rerank passages using Cross-Encoder.
    5. Generate a grounded answer.
    6. Verify answer against retrieved evidence.
    7. Return answer, sources and verification.
    """

    try:

        if not isinstance(query, str) or not query.strip():
            raise ValueError("Question cannot be empty.")

        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")

        # STEP 1: Prepare conversation history.

        previous_history = get_previous_history(
            chat_history,
            query
        )

        # STEP 2: Rewrite follow-up questions.

        retrieval_query = rewrite_research_query(
            query,
            previous_history
        )

        # STEP 3: Retrieve candidates using hybrid search.

        search_results = search_hybrid_chunks(
            query=retrieval_query,
            model=model,
            index=index,
            chunks=chunks,
            top_k=top_k * 4,
            semantic_weight=0.7
        )

        # STEP 4: Filter weak hybrid matches.

        relevant_results = filter_relevant_results(
            search_results,
            min_score=0.15
        )

        if not relevant_results:

            return {
                "answer": (
                    "I couldn't find sufficiently relevant "
                    "information in your uploaded research papers. "
                    "Try rephrasing your question or uploading "
                    "a more relevant paper."
                ),
                "sources": [],
                "retrieval_query": retrieval_query,
                "verification": {
                    "status": "not_applicable",
                    "supported": None,
                    "unsupported_claims": [],
                    "explanation": (
                        "No sufficiently relevant passages "
                        "were retrieved."
                    )
                }
            }

        # STEP 5: Rerank candidates using Cross-Encoder.

        reranked_results = rerank_results(
            query=retrieval_query,
            search_results=relevant_results,
            top_k=top_k
        )

        if not reranked_results:

            return {
                "answer": (
                    "No relevant passages remained after "
                    "reranking. Try modifying your question."
                ),
                "sources": [],
                "retrieval_query": retrieval_query,
                "verification": {
                    "status": "not_applicable",
                    "supported": None,
                    "unsupported_claims": [],
                    "explanation": (
                        "The reranker did not return "
                        "any relevant passages."
                    )
                }
            }

        # STEP 6: Build research context.

        context = build_research_context(
            reranked_results
        )

        # STEP 7: Build final research prompt.

        prompt = build_research_prompt(
            query=query,
            context=context,
            chat_history=previous_history
        )

        # STEP 8: Generate answer.

        answer = generate_answer(
            prompt,
            max_retries=2,
            temperature=0.3,
            max_output_tokens=2000
        )

        # STEP 9: Verify answer against evidence.

        verification = verify_answer_support(
            answer=answer,
            context=context
        )

        # STEP 10: Prepare source information.

        sources = []

        for result in reranked_results:

            chunk = result.get("chunk", {})

            sources.append({

                "document_name": chunk.get(
                    "document_name",
                    "Unknown document"
                ),

                "page_number": chunk.get(
                    "page_number",
                    "Unknown"
                ),

                "text": chunk.get("text", ""),

                "score": float(
                    result.get("score", 0)
                ),

                "semantic_score": float(
                    result.get("semantic_score", 0)
                ),

                "keyword_score": float(
                    result.get("keyword_score", 0)
                ),

                "original_score": float(
                    result.get("original_score", 0)
                ),

                "rerank_score": float(
                    result.get("rerank_score", 0)
                )

            })

        # STEP 11: Return everything to Streamlit.

        return {
            "answer": answer,
            "sources": sources,
            "retrieval_query": retrieval_query,
            "verification": verification
        }

    except Exception as error:

        return {
            "answer": (
                f"An error occurred while processing "
                f"your question: {str(error)}"
            ),
            "sources": [],
            "retrieval_query": query,
            "verification": {
                "status": "unavailable",
                "supported": None,
                "unsupported_claims": [],
                "explanation": (
                    "The pipeline could not complete "
                    "the verification process."
                )
            }
        }
