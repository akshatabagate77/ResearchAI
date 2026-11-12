
import hashlib
from pathlib import Path

import streamlit as st
import networkx as nx
import plotly.graph_objects as go

from src.knowledge_graph import (
    build_knowledge_graph,
    get_graph_statistics
)
import pandas as pd

from src.citation_extractor import extract_document_references
from src.pdf_processor import extract_pages_from_pdf
from src.text_chunker import chunk_pages
from src.embeddings import load_embedding_model, generate_embeddings
from src.vector_store import create_vector_store
from src.rag_pipeline import answer_question
from src.research_summarizer import summarize_research_papers
from src.research_comparator import compare_research_papers
from src.research_gap_detector import detect_research_gaps
from src.report_generator import generate_research_report
from src.evaluation import (
    evaluate_retrieval,
    evaluate_all_retrieval_methods
)
from src.paper_recommender import recommend_papers
from src.paper_analyzer import analyze_research_collection

# NEW: Topic extraction module
from src.topic_extractor import (
    extract_document_keywords,
    extract_common_keywords
)


# ---------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------

st.set_page_config(
    page_title="ResearchAI | Intelligent Research Workspace",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ---------------------------------------------------------
# CUSTOM CSS
# ---------------------------------------------------------

st.markdown("""
<style>

.stApp {
    background-color: #0b1020;
    color: #e8ecf5;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1500px;
}

section[data-testid="stSidebar"] {
    background-color: #10182b;
    border-right: 1px solid #27344d;
}

h1, h2, h3 {
    color: #f4f6ff !important;
    font-weight: 700 !important;
}

p, label, li {
    color: #c3cce0;
}

.stButton > button {
    background: linear-gradient(90deg, #6956e8, #3987f5);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 0.65rem 1.1rem;
    font-weight: 600;
    transition: 0.2s;
}

.stButton > button:hover {
    border: 1px solid #a99cff;
    color: white;
    transform: translateY(-2px);
}

.stDownloadButton > button {
    background-color: #202d45;
    color: white;
    border: 1px solid #52658a;
    border-radius: 9px;
}

.stTextInput input,
.stTextArea textarea {
    background-color: #151f33;
    color: white;
    border: 1px solid #35445f;
    border-radius: 9px;
}

.stChatInput textarea {
    background-color: #151f33;
    color: white;
}

[data-testid="stFileUploader"] {
    background-color: #151f33;
    border: 1px dashed #7566e8;
    border-radius: 12px;
    padding: 12px;
}

[data-testid="stMetric"] {
    background-color: #151f33;
    border: 1px solid #2b3a55;
    padding: 18px;
    border-radius: 14px;
}

[data-testid="stMetricValue"] {
    color: #a99cff;
}

button[data-baseweb="tab"] {
    color: #c5cce0;
    font-weight: 600;
}

button[data-baseweb="tab"][aria-selected="true"] {
    color: #a99cff;
}

[data-testid="stExpander"] {
    background-color: #121c30;
    border: 1px solid #293954;
    border-radius: 10px;
}

hr {
    border-color: #293954;
}

.stAlert {
    border-radius: 10px;
}

</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# SESSION STATE
# ---------------------------------------------------------

DEFAULT_STATE = {
    "pdf_data": [],
    "all_chunks": [],
    "index": None,
    "embedding_model": None,
    "chat_history": [],
    "summaries": {},
    "comparison_result": None,
    "gap_report": None,
    "generated_pdf": None,
    "evaluation_report": None,
    "upload_signature": None,
    "processing_complete": False,
    "paper_recommendations": [],
"last_recommendation_query": "",
"knowledge_graph": None,
"graph_settings": {},
"citation_results": {},
"citation_extraction_complete": False,
"paper_quality_results": {},
"paper_quality_complete": False,

    # NEW: Topic Explorer state
    "document_keywords": {},
    "common_keywords": []
}

for key, value in DEFAULT_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ---------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------

def display_sources(sources):
    """Display the passages used to generate an AI answer."""

    st.markdown("### 📚 Sources and Evidence")

    if not sources:
        st.info("No relevant sources were found.")
        return

    for i, source in enumerate(sources, start=1):

        document_name = source.get(
            "document_name",
            "Unknown document"
        )

        page_number = source.get(
            "page_number",
            "Unknown"
        )

        score = source.get("score", 0)
        passage = source.get("text", "")

        with st.expander(
            f"Source {i}: {document_name} | Page {page_number}"
        ):

            st.write("**Retrieved passage:**")
            st.write(passage)

            st.caption(
                f"Similarity score: {score:.4f}"
            )


def display_verification(verification):
    """Display the answer evidence-verification result."""

    if not verification:
        return

    status = verification.get("status")

    if status == "supported":

        st.success(
            "Evidence check: Answer is supported by retrieved passages."
        )

    elif status == "warning":

        st.warning(
            "Evidence check: Some claims may not be supported."
        )

        for claim in verification.get("unsupported_claims", []):
            st.write(f"• {claim}")

    elif status == "unavailable":

        st.info(
            "Evidence check unavailable. "
            "The answer has not been independently verified."
        )

    elif status == "not_applicable":

        st.info(
            verification.get(
                "explanation",
                "No relevant evidence was retrieved."
            )
        )

    explanation = verification.get("explanation", "")

    if explanation:
        st.caption(explanation)


def create_upload_signature(files):
    """Create a unique signature for uploaded PDFs."""

    hasher = hashlib.sha256()

    for file in files:
        hasher.update(file.name.encode("utf-8"))
        hasher.update(file.getvalue())

    return hasher.hexdigest()


def clear_project_results():
    """Clear results when uploaded PDFs change."""

    st.session_state.pdf_data = []
    st.session_state.all_chunks = []
    st.session_state.index = None
    st.session_state.chat_history = []
    st.session_state.summaries = {}
    st.session_state.comparison_result = None
    st.session_state.gap_report = None
    st.session_state.generated_pdf = None
    st.session_state.evaluation_report = None
    st.session_state.processing_complete = False

    # Clear Topic Explorer results too
    st.session_state.document_keywords = {}
    st.session_state.common_keywords = []


def get_document_names():

    return sorted({
        chunk.get("document_name", "Unknown document")
        for chunk in st.session_state.all_chunks
    })


def require_documents():

    if not st.session_state.processing_complete:

        st.info(
            "Upload and process your research papers "
            "from the sidebar to use this feature."
        )

        return False

    if (
        not st.session_state.all_chunks
        or st.session_state.index is None
    ):

        st.warning("No searchable document data is available.")

        return False

    return True


# ---------------------------------------------------------
# PDF PROCESSING
# ---------------------------------------------------------

def process_uploaded_pdfs(uploaded_files):
    

    if not uploaded_files:

        st.warning("Please upload at least one PDF.")
        return

    try:

        with st.status(
            "Processing your research papers...",
            expanded=True
        ) as status:

            # Load embedding model
            st.write("Loading embedding model...")

            if st.session_state.embedding_model is None:

                st.session_state.embedding_model = (
                    load_embedding_model()
                )

            model = st.session_state.embedding_model

            all_chunks = []
            pdf_data = []

            # Process each PDF
            for file_number, uploaded_file in enumerate(
                uploaded_files,
                start=1
            ):

                st.write(
                    f"Extracting text from {uploaded_file.name}..."
                )

                pdf_bytes = uploaded_file.getvalue()

                pages = extract_pages_from_pdf(pdf_bytes)

                if not pages:

                    st.warning(
                        f"No pages were extracted from "
                        f"{uploaded_file.name}."
                    )

                    continue

                chunks = chunk_pages(pages)

                for chunk in chunks:
                    chunk["document_name"] = uploaded_file.name

                all_chunks.extend(chunks)

                pdf_data.append({
                    "name": uploaded_file.name,
                    "pages": len(pages),
                    "chunks": len(chunks)
                })

            if not all_chunks:

                raise ValueError(
                    "No readable text was extracted "
                    "from the uploaded PDFs."
                )

            # Generate embeddings
            st.write("Generating embeddings...")

            embeddings = generate_embeddings(
                all_chunks,
                model
            )

            # Create FAISS index
            st.write("Building FAISS vector database...")

            index, stored_chunks = create_vector_store(
                all_chunks,
                embeddings
            )

            # Save everything in session state
            st.session_state.pdf_data = pdf_data
            st.session_state.all_chunks = stored_chunks
            st.session_state.index = index
            st.session_state.knowledge_graph = None
            st.session_state.graph_settings = {}
            st.session_state.processing_complete = True
            st.session_state.citation_results = {}
            st.session_state.citation_extraction_complete = False
            st.session_state.paper_quality_results = {}
            st.session_state.paper_quality_complete = False

            # Reset old analysis
            st.session_state.summaries = {}
            st.session_state.chat_history = []
            st.session_state.comparison_result = None
            st.session_state.gap_report = None
            st.session_state.generated_pdf = None
            st.session_state.evaluation_report = None

            # Reset Topic Explorer
            st.session_state.document_keywords = {}
            st.session_state.common_keywords = []

            status.update(
                label="Research papers processed successfully!",
                state="complete",
                expanded=False
            )

        st.success(
            f"Successfully processed {len(pdf_data)} research paper(s)."
        )

    except Exception as error:

        st.error(f"Processing failed: {error}")
        st.exception(error)


# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------

with st.sidebar:

    st.title("🔬 ResearchAI")

    st.caption("Intelligent Research Workspace")

    st.divider()

    st.subheader("📚 Your Research Library")

    uploaded_files = st.file_uploader(
        "Add research papers",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload one or multiple research papers in PDF format."
    )

    if uploaded_files:

        current_signature = create_upload_signature(
            uploaded_files
        )

        if current_signature != st.session_state.upload_signature:

            clear_project_results()

            st.session_state.upload_signature = current_signature

        st.caption(
            f"{len(uploaded_files)} PDF file(s) selected"
        )

        if st.button(
            "⚡ Process Research Papers",
            use_container_width=True
        ):

            process_uploaded_pdfs(uploaded_files)

    else:

        if st.session_state.processing_complete:

            st.info(
                "Your previously processed papers "
                "are available in this session."
            )

        else:

            st.caption("Your uploaded papers will appear here.")

    st.divider()

    st.subheader("📄 Processed Documents")

    if st.session_state.pdf_data:

        for document in st.session_state.pdf_data:

            with st.expander(document["name"]):

                st.write(f"Pages: {document['pages']}")
                st.write(f"Text chunks: {document['chunks']}")

    else:

        st.caption("No processed documents yet.")

    st.divider()

    st.caption(
        "ResearchAI | AI-powered research assistant"
    )

    st.caption(
        "Built with Python, Streamlit, FAISS and Gemini"
    )


# ---------------------------------------------------------
# MAIN DASHBOARD
# ---------------------------------------------------------

st.title("Think deeper. Research smarter.")

st.markdown(
    "Your AI-powered research companion. Explore papers, ask grounded "
    "questions, compare findings, discover research gaps, and generate "
    "structured research reports."
)

st.divider()


# ---------------------------------------------------------
# DASHBOARD METRICS
# ---------------------------------------------------------

document_count = len(st.session_state.pdf_data)
chunk_count = len(st.session_state.all_chunks)
summary_count = len(st.session_state.summaries)
question_count = len(st.session_state.chat_history) // 2

col1, col2, col3, col4 = st.columns(4)

with col1:

    st.metric(
        "Research Papers",
        document_count,
        help="Number of processed research papers."
    )

with col2:

    st.metric(
        "Text Chunks",
        chunk_count,
        help="Number of text chunks stored in FAISS."
    )

with col3:

    st.metric(
        "AI Summaries",
        summary_count,
        help="Number of generated paper summaries."
    )

with col4:

    st.metric(
        "Questions Asked",
        question_count,
        help="Number of questions asked in this session."
    )

st.divider()


# ---------------------------------------------------------
# APPLICATION TABS
# ---------------------------------------------------------

tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10, tab11 = st.tabs([
    "Ask AI",
    "Summarize",
    "Compare",
    "Research Gaps",
    "Reports",
    "Evaluation",
    "Topic Explorer",
    "Paper Recommendations",
    "Knowledge Graph",
    "Citation Manager",
    "Paper Quality Analyzer"
])


# =========================================================
# TAB 1: ASK AI
# =========================================================

with tab1:

    st.header("Ask your research papers")

    st.write(
        "Ask questions about your uploaded research papers. "
        "Answers are generated using retrieved document content."
    )

    if require_documents():

        # Display previous conversation
        for message in st.session_state.chat_history:

            with st.chat_message(message["role"]):

                st.write(message["content"])

                if message.get("sources"):

                    with st.expander("📚 View previous answer sources"):

                        for source in message["sources"]:

                            st.markdown(
                                f"**{source['document_name']}** "
                                f"| Page {source['page_number']}"
                            )

                            st.caption(
                                source.get("text", "")[:700]
                            )

        query = st.chat_input(
            "Ask something about your research papers..."
        )

        if query:

            st.session_state.chat_history.append({
                "role": "user",
                "content": query
            })

            with st.chat_message("user"):
                st.write(query)

            with st.chat_message("assistant"):

                with st.spinner(
                    "ResearchAI is analyzing your papers..."
                ):

                    try:

                        result = answer_question(
                            query=query,
                            model=st.session_state.embedding_model,
                            index=st.session_state.index,
                            chunks=st.session_state.all_chunks,
                            top_k=5,
                            chat_history=st.session_state.chat_history
                        )

                        answer = result["answer"]
                        sources = result.get("sources", [])
                        verification = result.get("verification", {})

                        # Display AI answer
                        st.markdown(answer)

                        # Display verification result
                        display_verification(verification)

                        # Display sources once
                        display_sources(sources)

                        # Save assistant response
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": answer,
                            "sources": sources,
                            "verification": verification
                        })

                    except Exception as error:

                        st.error(
                            f"Question answering failed: {error}"
                        )

                        st.exception(error)

        if st.button("Clear Chat History"):

            st.session_state.chat_history = []
            st.rerun()


# =========================================================
# TAB 2: SUMMARIZE
# =========================================================

with tab2:

    st.header("Research Paper Summarization")

    st.write(
        "Generate structured AI summaries of your uploaded research papers."
    )

    if require_documents():

        if st.button(
            "✨ Generate Research Summaries",
            use_container_width=True
        ):

            try:

                with st.spinner(
                    "Analyzing research papers and generating summaries..."
                ):

                    summaries = summarize_research_papers(
                        st.session_state.all_chunks,
                        force_refresh=False
                    )

                    st.session_state.summaries = summaries
                    st.session_state.comparison_result = None
                    st.session_state.gap_report = None
                    st.session_state.generated_pdf = None

                st.success(
                    "Research summaries generated successfully."
                )

            except Exception as error:

                st.error(f"Summarization failed: {error}")
                st.exception(error)

        if st.session_state.summaries:

            st.divider()

            for document_name, summary in (
                st.session_state.summaries.items()
            ):

                with st.expander(
                    f"📄 {document_name}",
                    expanded=True
                ):

                    st.markdown(summary)


# =========================================================
# TAB 3: COMPARE
# =========================================================

with tab3:

    st.header("Compare Research Papers")

    st.write(
        "Compare research methodologies, findings, approaches and limitations."
    )

    if require_documents():

        if not st.session_state.summaries:

            st.info(
                "Generate research summaries first before comparing papers."
            )

        else:

            available_documents = list(
                st.session_state.summaries.keys()
            )

            selected_documents = st.multiselect(
                "Select 2–4 research papers",
                options=available_documents,
                default=available_documents[:min(2, len(available_documents))]
            )

            if st.button(
                "⚖️ Compare Selected Papers",
                use_container_width=True
            ):

                if (
                    len(selected_documents) < 2
                    or len(selected_documents) > 4
                ):

                    st.warning(
                        "Please select between 2 and 4 research papers."
                    )

                else:

                    try:

                        with st.spinner("Comparing research papers..."):

                            comparison = compare_research_papers(
                                summaries=st.session_state.summaries,
                                selected_documents=selected_documents
                            )

                            st.session_state.comparison_result = comparison
                            st.session_state.generated_pdf = None

                        st.success("Comparison completed.")

                    except Exception as error:

                        st.error(f"Comparison failed: {error}")
                        st.exception(error)

            if st.session_state.comparison_result:

                st.divider()

                st.subheader("Comparison Results")

                st.markdown(
                    st.session_state.comparison_result
                )


# =========================================================
# TAB 4: RESEARCH GAPS
# =========================================================

with tab4:

    st.header("Research Gap Detection")

    st.write(
        "Identify missing research areas, limitations and possible "
        "directions for future research."
    )

    if require_documents():

        if not st.session_state.summaries:

            st.info(
                "Generate research summaries first to detect research gaps."
            )

        else:

            available_documents = list(
                st.session_state.summaries.keys()
            )

            selected_documents = st.multiselect(
                "Select 2–5 research papers",
                options=available_documents,
                default=available_documents[:min(2, len(available_documents))],
                key="gap_documents"
            )

            if st.button(
                "🔎 Detect Research Gaps",
                use_container_width=True
            ):

                if (
                    len(selected_documents) < 2
                    or len(selected_documents) > 5
                ):

                    st.warning(
                        "Please select between 2 and 5 research papers."
                    )

                else:

                    try:

                        with st.spinner(
                            "Analyzing research limitations and gaps..."
                        ):

                            gap_report = detect_research_gaps(
                                summaries=st.session_state.summaries,
                                selected_documents=selected_documents
                            )

                            st.session_state.gap_report = gap_report
                            st.session_state.generated_pdf = None

                        st.success(
                            "Research gap analysis completed."
                        )

                    except Exception as error:

                        st.error(f"Gap detection failed: {error}")
                        st.exception(error)

            if st.session_state.gap_report:

                st.divider()

                st.subheader("Identified Research Gaps")

                st.markdown(
                    st.session_state.gap_report
                )


# =========================================================
# TAB 5: REPORTS
# =========================================================

with tab5:

    st.header("Research Report Generator")

    st.write(
        "Convert your AI-generated research analysis into a downloadable PDF."
    )

    if require_documents():

        if not st.session_state.summaries:

            st.info(
                "Generate research summaries before creating a report."
            )

        else:

            st.write("Your report can include:")

            st.markdown("""
            - Research paper summaries
            - Comparative research analysis
            - Identified research gaps
            """)

            if st.button(
                "📑 Generate Research Report",
                use_container_width=True
            ):

                try:

                    with st.spinner("Creating your PDF report..."):

                        pdf_buffer = generate_research_report(
                            summaries=st.session_state.summaries,
                            comparison_report=st.session_state.comparison_result,
                            gap_report=st.session_state.gap_report
                        )

                        if hasattr(pdf_buffer, "getvalue"):

                            pdf_bytes = pdf_buffer.getvalue()

                        else:

                            pdf_bytes = pdf_buffer

                        st.session_state.generated_pdf = pdf_bytes

                    st.success("Your research report is ready.")

                except Exception as error:

                    st.error(f"Report generation failed: {error}")
                    st.exception(error)

            if st.session_state.generated_pdf:

                st.download_button(
                    label="⬇️ Download Research Report",
                    data=st.session_state.generated_pdf,
                    file_name="ResearchAI_Report.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )



# =========================================================
# TAB 6: ADVANCED RETRIEVAL EVALUATION DASHBOARD
# =========================================================

with tab6:

    st.header("Retrieval Evaluation Dashboard")

    st.write(
        "Compare Semantic Search, Hybrid Search and "
        "Cross-Encoder Reranking using measurable retrieval metrics."
    )

    if require_documents():

        st.markdown("### Evaluation Configuration")

        top_k = st.slider(
            "Number of retrieved chunks (Top-K)",
            min_value=1,
            max_value=10,
            value=5,
            key="evaluation_top_k"
        )

        possible_paths = [
            Path("evaluation_questions.json"),
            Path("data/evaluation_questions.json"),
            Path("tests/evaluation_questions.json")
        ]

        dataset_path = next(
            (
                path for path in possible_paths
                if path.exists()
            ),
            None
        )

        if dataset_path is None:

            st.error(
                "evaluation_questions.json was not found. "
                "Place it in your project root."
            )

        else:

            st.success(
                f"Evaluation dataset found: {dataset_path}"
            )

            if st.button(
                "Run Advanced Retrieval Evaluation",
                use_container_width=True,
                key="run_advanced_evaluation"
            ):

                try:

                    with st.spinner(
                        "Comparing all three retrieval methods. "
                        "This may take some time on the first run..."
                    ):

                        evaluation_report = (
                            evaluate_all_retrieval_methods(
                                dataset_path=str(dataset_path),
                                model=st.session_state.embedding_model,
                                index=st.session_state.index,
                                chunks=st.session_state.all_chunks,
                                top_k=top_k
                            )
                        )

                        st.session_state.evaluation_report = (
                            evaluation_report
                        )

                    st.success(
                        "Advanced retrieval evaluation completed!"
                    )

                except Exception as error:

                    st.error(
                        f"Advanced evaluation failed: {error}"
                    )

                    st.exception(error)

        report = st.session_state.evaluation_report

        if report and report.get("comparison"):

            comparison = report["comparison"]

            st.divider()

            st.subheader("Overall Evaluation Summary")

            st.caption(
                f"Questions evaluated: {report['total_questions']} "
                f"| Top-K: {report['top_k']}"
            )

            method_labels = {
                "semantic": "Semantic Search",
                "hybrid": "Hybrid Search",
                "hybrid_reranked": "Hybrid + Reranking"
            }

            # -----------------------------------------
            # PERFORMANCE COMPARISON TABLE
            # -----------------------------------------

            st.subheader("Retrieval Method Comparison")

            metric_rows = []

            for method, results in comparison.items():

                metric_rows.append({
                    "Retrieval Method": method_labels.get(
                        method, method
                    ),
                    "Precision@K": results.get(
                        "precision_at_k", 0
                    ),
                    "Recall@K": results.get(
                        "recall_at_k", 0
                    ),
                    "Hit Rate": results.get(
                        "hit_rate", 0
                    ),
                    "MRR": results.get(
                        "mrr", 0
                    ),
                    "Average Latency (ms)": results.get(
                        "average_latency_ms", 0
                    )
                })

            metrics_df = pd.DataFrame(metric_rows)

            st.dataframe(
                metrics_df.style.format({
                    "Precision@K": "{:.2%}",
                    "Recall@K": "{:.2%}",
                    "Hit Rate": "{:.2%}",
                    "MRR": "{:.4f}",
                    "Average Latency (ms)": "{:.2f}"
                }),
                use_container_width=True,
                hide_index=True
            )

            st.divider()

            # -----------------------------------------
            # BEST METHOD IDENTIFICATION
            # -----------------------------------------

            st.subheader("Performance Highlights")

            best_hit_method = max(
                comparison,
                key=lambda method: comparison[method].get(
                    "hit_rate", 0
                )
            )

            best_mrr_method = max(
                comparison,
                key=lambda method: comparison[method].get(
                    "mrr", 0
                )
            )

            fastest_method = min(
                comparison,
                key=lambda method: comparison[method].get(
                    "average_latency_ms", float("inf")
                )
            )

            highlight1, highlight2, highlight3 = st.columns(3)

            with highlight1:

                st.metric(
                    "Best Hit Rate",
                    method_labels.get(best_hit_method, best_hit_method)
                )

            with highlight2:

                st.metric(
                    "Best MRR",
                    method_labels.get(best_mrr_method, best_mrr_method)
                )

            with highlight3:

                st.metric(
                    "Fastest Retrieval",
                    method_labels.get(fastest_method, fastest_method)
                )

            st.caption(
                "These highlights are based on the evaluation dataset, "
                "not a guarantee of performance on every research paper."
            )

            st.divider()

            # -----------------------------------------
            # PERFORMANCE CHARTS
            # -----------------------------------------

            st.subheader("Retrieval Performance Charts")

            chart_col1, chart_col2 = st.columns(2)

            with chart_col1:

                st.markdown("**Precision@K Comparison**")

                precision_df = metrics_df.set_index(
                    "Retrieval Method"
                )[["Precision@K"]]

                st.bar_chart(precision_df)

                st.markdown("**Hit Rate Comparison**")

                hit_df = metrics_df.set_index(
                    "Retrieval Method"
                )[["Hit Rate"]]

                st.bar_chart(hit_df)

            with chart_col2:

                st.markdown("**MRR Comparison**")

                mrr_df = metrics_df.set_index(
                    "Retrieval Method"
                )[["MRR"]]

                st.bar_chart(mrr_df)

                st.markdown("**Average Retrieval Latency**")

                latency_df = metrics_df.set_index(
                    "Retrieval Method"
                )[["Average Latency (ms)"]]

                st.bar_chart(latency_df)

            st.divider()

            # -----------------------------------------
            # QUESTION-WISE RESULTS
            # -----------------------------------------

            st.subheader("Detailed Question-wise Evaluation")

            selected_method = st.selectbox(
                "Choose retrieval method",
                options=list(comparison.keys()),
                format_func=lambda method: method_labels.get(
                    method, method
                ),
                key="evaluation_selected_method"
            )

            selected_results = comparison[selected_method]

            for number, result in enumerate(
                selected_results.get("results", []),
                start=1
            ):

                with st.expander(
                    f"Question {number}: {result['question']}"
                ):

                    st.write(
                        f"**Expected document:** "
                        f"{result['expected_document']}"
                    )

                    st.write(
                        f"**Expected page:** "
                        f"{result['expected_page']}"
                    )

                    col1, col2, col3 = st.columns(3)

                    col1.metric(
                        "Precision",
                        f"{result['precision']:.2%}"
                    )

                    col2.metric(
                        "Recall",
                        f"{result['recall']:.2%}"
                    )

                    col3.metric(
                        "MRR",
                        f"{result.get('mrr', 0):.4f}"
                    )

                    st.write(
                        "**Relevant result found:**",
                        "Yes" if result["hit"] else "No"
                    )

                    st.markdown("**Retrieved passages**")

                    for retrieved in result["retrieved_results"]:

                        st.write(
                            f"Document: {retrieved['document']} "
                            f"| Page: {retrieved['page']} "
                            f"| Score: {retrieved['score']}"
                        )

                        if retrieved["relevant"]:

                            st.success("Relevant result")

                        else:

                            st.caption(
                                "Not the expected document/page"
                            )

            st.divider()

            # -----------------------------------------
            # DOWNLOAD EVALUATION RESULTS
            # -----------------------------------------

            st.subheader("Export Evaluation Results")

            import json

            json_data = json.dumps(
                report,
                indent=4
            )

            st.download_button(
                label="Download Complete Evaluation Results (JSON)",
                data=json_data,
                file_name="ResearchAI_Retrieval_Evaluation.json",
                mime="application/json",
                use_container_width=True
            )


# =========================================================
# TAB 7: TOPIC EXPLORER - NEW
# =========================================================

with tab7:

    st.header("🔍 Research Topic Explorer")

    st.write(
        "Discover important keywords and common research themes "
        "across your uploaded research papers."
    )

    if require_documents():

        # Extract keywords once and preserve them in session state.
        if not st.session_state.document_keywords:

            try:

                with st.spinner(
                    "Extracting research keywords and topics..."
                ):

                    document_keywords = extract_document_keywords(
                        st.session_state.all_chunks,
                        top_n=10
                    )

                    common_keywords = extract_common_keywords(
                        document_keywords,
                        top_n=15
                    )

                    st.session_state.document_keywords = (
                        document_keywords
                    )

                    st.session_state.common_keywords = (
                        common_keywords
                    )

            except Exception as error:

                st.error(
                    f"Topic extraction failed: {error}"
                )

                st.exception(error)

        document_keywords = st.session_state.document_keywords
        common_keywords = st.session_state.common_keywords

        if not document_keywords:

            st.warning(
                "No keywords could be extracted from your papers."
            )

        else:
            

            # -----------------------------------------
            # SECTION 1: OVERVIEW METRICS
            # -----------------------------------------

            st.subheader("📊 Topic Analysis Overview")

            topic_col1, topic_col2, topic_col3 = st.columns(3)

            with topic_col1:

                st.metric(
                    "Papers Analyzed",
                    len(document_keywords)
                )

            with topic_col2:

                total_keywords = sum(
                    len(keywords)
                    for keywords in document_keywords.values()
                )

                st.metric(
                    "Extracted Keywords",
                    total_keywords
                )

            with topic_col3:

                st.metric(
                    "Common Themes",
                    len(common_keywords)
                )

            st.divider()

            # -----------------------------------------
            # SECTION 2: COMMON RESEARCH THEMES
            # -----------------------------------------

            st.subheader("🌐 Common Research Themes")

            if common_keywords:

                chart_data = {
                    "Keyword": [
                        item["keyword"]
                        for item in common_keywords
                    ],
                    "Number of Papers": [
                        item["paper_count"]
                        for item in common_keywords
                    ]
                }

                st.bar_chart(
                    chart_data,
                    x="Keyword",
                    y="Number of Papers"
                )

                st.caption(
                    "A keyword is counted once per paper when "
                    "it appears among that paper's extracted "
                    "top keywords."
                )

                st.markdown("**Common keyword details**")

                for item in common_keywords:

                    st.write(
                        f"• **{item['keyword']}** — "
                        f"Found in {item['paper_count']} paper(s)"
                    )

            else:

                st.info(
                    "No common keywords found across the papers."
                )

            st.divider()

            # -----------------------------------------
            # SECTION 3: PAPER-WISE KEYWORDS
            # -----------------------------------------

            st.subheader("📄 Keywords by Research Paper")

            selected_document = st.selectbox(
                "Select a research paper",
                options=list(document_keywords.keys()),
                key="topic_selected_document"
            )

            selected_keywords = document_keywords.get(
                selected_document,
                []
            )

            if selected_keywords:

                st.markdown(
                    f"**Important keywords in {selected_document}**"
                )

                keyword_col1, keyword_col2 = st.columns(2)

                for i, item in enumerate(selected_keywords):

                    target_col = (
                        keyword_col1 if i % 2 == 0
                        else keyword_col2
                    )

                    with target_col:

                        st.write(
                            f"**{item['keyword']}**"
                        )

                        st.progress(
                            min(float(item["score"]), 1.0)
                        )

                        st.caption(
                            f"TF-IDF score: {item['score']:.4f}"
                        )

            else:

                st.info(
                    "No keywords available for this paper."
                )

            st.divider()

            # -----------------------------------------
            # SECTION 4: ALL PAPER KEYWORDS
            # -----------------------------------------

            st.subheader("📚 Complete Keyword Overview")

            for document_name, keywords in document_keywords.items():

                with st.expander(document_name):

                    if keywords:

                        for item in keywords:

                            st.write(
                                f"• **{item['keyword']}** "
                                f"({item['score']:.4f})"
                            )

                    else:

                        st.write(
                            "No keywords extracted."
                        )

            # -----------------------------------------
            # SECTION 5: REANALYZE
            # -----------------------------------------

            if st.button(
                "🔄 Refresh Topic Analysis",
                use_container_width=True
            ):

                st.session_state.document_keywords = {}
                st.session_state.common_keywords = []

                st.rerun()

# =====================================================
# TAB 8: PAPER RECOMMENDATIONS
# =====================================================

with tab8:

    st.header("Research Paper Recommendation Engine")

    st.write(
        "Find the most relevant research papers from your "
        "uploaded collection using semantic similarity."
    )

    if (
        not st.session_state.processing_complete
        or st.session_state.index is None
        or not st.session_state.all_chunks
    ):

        st.info(
            "Please upload and process research papers "
            "from the sidebar before using recommendations."
        )

    else:

        research_query = st.text_input(
            "Enter your research topic",
            placeholder="Example: Artificial Intelligence in Healthcare",
            key="paper_recommendation_query"
        )

        number_of_recommendations = st.slider(
            "Number of recommendations",
            min_value=1,
            max_value=10,
            value=5,
            key="recommendation_count"
        )

        if st.button(
            "Find Relevant Research Papers",
            key="find_recommendations"
        ):

            if not research_query.strip():

                st.warning("Please enter a research topic.")

            else:

                with st.spinner(
                    "Searching your research collection..."
                ):

                    try:

                        recommendations = recommend_papers(
                            query=research_query,
                            model=st.session_state.embedding_model,
                            index=st.session_state.index,
                            chunks=st.session_state.all_chunks,
                            top_k=number_of_recommendations
                        )

                        st.session_state.paper_recommendations = (
                            recommendations
                        )

                        st.session_state.last_recommendation_query = (
                            research_query
                        )

                    except Exception as e:

                        st.error(
                            f"Recommendation error: {str(e)}"
                        )

        recommendations = st.session_state.get(
            "paper_recommendations",
            []
        )

        previous_query = st.session_state.get(
            "last_recommendation_query",
            ""
        )

        if recommendations:

            st.success(
                f"Found {len(recommendations)} relevant papers "
                f"for: {previous_query}"
            )

            for rank, paper in enumerate(
                recommendations,
                start=1
            ):

                with st.expander(
                    f"{rank}. {paper['document_name']} "
                    f"— {paper['relevance_score']}% similarity",
                    expanded=(rank == 1)
                ):

                    st.write(
                        "**Relevant pages:**",
                        ", ".join(
                            str(page)
                            for page in paper["matched_pages"]
                        )
                    )

                    st.write("**Matching passages:**")

                    for match_number, match in enumerate(
                        paper["matched_chunks"],
                        start=1
                    ):

                        st.markdown(
                            f"**Passage {match_number} "
                            f"(Page {match['page_number']})**"
                        )

                        st.write(match["text"])

                        st.divider()

        elif previous_query:

            st.warning(
                "No sufficiently relevant papers were found. "
                "Try a different or broader research topic."
            )

# =====================================================
# TAB 9: RESEARCH KNOWLEDGE GRAPH
# =====================================================

with tab9:

    st.header("Research Knowledge Graph")

    st.write(
        "Explore concepts and their relationships across "
        "your uploaded research papers."
    )

    if (
        not st.session_state.processing_complete
        or not st.session_state.all_chunks
    ):

        st.info(
            "Please upload and process research papers "
            "from the sidebar first."
        )

    else:

        st.markdown("### Graph Settings")

        col1, col2 = st.columns(2)

        with col1:
            top_keywords = st.slider(
                "Concepts per text chunk",
                min_value=3,
                max_value=10,
                value=5,
                key="graph_top_keywords"
            )

        with col2:
            min_connections = st.slider(
                "Minimum connection strength",
                min_value=1,
                max_value=5,
                value=1,
                key="graph_min_connections"
            )

        if st.button(
            "Generate Knowledge Graph",
            key="generate_knowledge_graph"
        ):

            with st.spinner(
                "Building your research knowledge graph..."
            ):

                try:

                    graph = build_knowledge_graph(
                        st.session_state.all_chunks,
                        top_keywords_per_chunk=top_keywords,
                        min_edge_weight=min_connections
                    )

                    st.session_state.knowledge_graph = graph

                    st.session_state.graph_settings = {
                        "top_keywords": top_keywords,
                        "min_connections": min_connections
                    }

                except Exception as e:

                    st.error(
                        f"Could not generate graph: {str(e)}"
                    )

        graph = st.session_state.get(
            "knowledge_graph",
            None
        )

        saved_settings = st.session_state.get(
            "graph_settings",
            {}
        )

        settings_match = (
            saved_settings.get("top_keywords") == top_keywords
            and saved_settings.get("min_connections") == min_connections
        )

        if graph is not None and settings_match:

            statistics = get_graph_statistics(graph)

            st.markdown("### Knowledge Graph Overview")

            metric1, metric2, metric3 = st.columns(3)

            metric1.metric(
                "Total Concepts",
                statistics["total_concepts"]
            )

            metric2.metric(
                "Connections",
                statistics["total_connections"]
            )

            metric3.metric(
                "Connected Groups",
                statistics["connected_groups"]
            )

            if graph.number_of_nodes() == 0:

                st.warning(
                    "No concepts could be extracted. "
                    "Try uploading PDFs with more readable text."
                )

            else:

                # Calculate node positions
                positions = nx.spring_layout(
                    graph,
                    seed=42,
                    k=0.8,
                    iterations=80
                )

                # Prepare edge coordinates
                edge_x = []
                edge_y = []

                for source, target, data in graph.edges(data=True):

                    x0, y0 = positions[source]
                    x1, y1 = positions[target]

                    edge_x.extend([x0, x1, None])
                    edge_y.extend([y0, y1, None])

                edge_trace = go.Scatter(
                    x=edge_x,
                    y=edge_y,
                    mode="lines",
                    line=dict(
                        width=1,
                        color="rgba(130, 150, 180, 0.45)"
                    ),
                    hoverinfo="none"
                )

                # Prepare node information
                node_x = []
                node_y = []
                node_text = []
                node_sizes = []
                node_colors = []
                node_hover = []

                for node in graph.nodes():

                    x, y = positions[node]

                    attributes = graph.nodes[node]

                    papers = attributes.get("papers", [])

                    paper_count = attributes.get(
                        "paper_count",
                        1
                    )

                    degree = graph.degree[node]

                    node_x.append(x)
                    node_y.append(y)
                    node_text.append(node)
                    node_sizes.append(16 + degree * 3)
                    node_colors.append(paper_count)

                    node_hover.append(
                        f"<b>{node}</b>"
                        f"<br>Papers mentioning concept: {paper_count}"
                        f"<br>Connections: {degree}"
                        f"<br>Research papers: "
                        f"{', '.join(papers)}"
                    )

                node_trace = go.Scatter(
                    x=node_x,
                    y=node_y,
                    mode="markers+text",
                    text=node_text,
                    textposition="top center",
                    hovertext=node_hover,
                    hoverinfo="text",
                    marker=dict(
                        size=node_sizes,
                        color=node_colors,
                        colorscale="Viridis",
                        showscale=True,
                        colorbar=dict(
                            title="Paper Count"
                        ),
                        line=dict(
                            width=1,
                            color="white"
                        )
                    )
                )

                figure = go.Figure(
                    data=[edge_trace, node_trace]
                )

                figure.update_layout(
                    title="Research Concept Network",
                    showlegend=False,
                    hovermode="closest",
                    height=700,
                    margin=dict(
                        b=20,
                        l=10,
                        r=10,
                        t=60
                    ),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    xaxis=dict(
                        showgrid=False,
                        zeroline=False,
                        showticklabels=False
                    ),
                    yaxis=dict(
                        showgrid=False,
                        zeroline=False,
                        showticklabels=False
                    )
                )

                st.plotly_chart(
                    figure,
                    use_container_width=True
                )

                st.markdown("### How to read this graph")

                st.write(
                    "- **Nodes:** Important concepts extracted "
                    "from your research papers."
                )

                st.write(
                    "- **Lines:** Concepts appearing together "
                    "in the same text chunk."
                )

                st.write(
                    "- **Node size:** Number of connections "
                    "to other concepts."
                )

                st.write(
                    "- **Node colour:** Number of papers "
                    "containing that concept."
                )

                st.caption(
                    "These relationships indicate co-occurrence "
                    "in text, not proven scientific or causal relationships."
                )

        elif graph is not None:

            st.info(
                "You changed the graph settings. Click "
                "'Generate Knowledge Graph' to rebuild it."
            )

# =====================================================
# TAB 10: CITATION & REFERENCE MANAGER
# =====================================================

with tab10:

    st.header("Citation & Reference Manager")

    st.write(
        "Extract and organize references, publication years, "
        "DOIs and URLs from your uploaded research papers."
    )

    if (
        not st.session_state.processing_complete
        or not st.session_state.all_chunks
    ):

        st.info(
            "Please upload and process research papers "
            "from the sidebar first."
        )

    else:

        if st.button(
            "Extract References",
            key="extract_citations"
        ):

            with st.spinner(
                "Extracting references from your research papers..."
            ):

                try:

                    results = extract_document_references(
                        st.session_state.all_chunks
                    )

                    st.session_state.citation_results = results
                    st.session_state.citation_extraction_complete = True

                except Exception as e:

                    st.error(
                        f"Reference extraction failed: {str(e)}"
                    )

        if st.session_state.citation_extraction_complete:

            citation_results = st.session_state.citation_results

            total_references = sum(
                len(references)
                for references in citation_results.values()
            )

            total_dois = sum(
                len(reference["dois"])
                for references in citation_results.values()
                for reference in references
            )

            total_urls = sum(
                len(reference["urls"])
                for references in citation_results.values()
                for reference in references
            )

            st.markdown("### Extraction Overview")

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Total References",
                total_references
            )

            col2.metric(
                "DOI Identifiers",
                total_dois
            )

            col3.metric(
                "URLs Found",
                total_urls
            )

            st.divider()

            if total_references == 0:

                st.warning(
                    "No numbered references were detected. "
                    "Check whether your PDFs contain a recognizable "
                    "References or Bibliography section."
                )

            else:

                selected_document = st.selectbox(
                    "Select a research paper",
                    options=list(citation_results.keys()),
                    key="selected_citation_document"
                )

                selected_references = citation_results[
                    selected_document
                ]

                st.subheader(
                    f"References in {selected_document}"
                )

                if not selected_references:

                    st.warning(
                        "No references were detected in this paper."
                    )

                else:

                    table_rows = []

                    for number, reference in enumerate(
                        selected_references,
                        start=1
                    ):

                        table_rows.append({
                            "No.": number,
                            "Reference": reference["reference"],
                            "Year": reference["year"] or "Not found",
                            "DOI": "; ".join(reference["dois"]) or "Not found",
                            "URLs": "; ".join(reference["urls"]) or "Not found"
                        })

                    reference_df = pd.DataFrame(table_rows)

                    st.dataframe(
                        reference_df,
                        use_container_width=True,
                        hide_index=True
                    )

                    # Export selected paper references
                    csv_data = reference_df.to_csv(
                        index=False
                    ).encode("utf-8-sig")

                    st.download_button(
                        label="Download Selected Paper References (CSV)",
                        data=csv_data,
                        file_name="research_references.csv",
                        mime="text/csv",
                        key="download_selected_references"
                    )

                # Export all extracted references
                all_rows = []

                for document_name, references in citation_results.items():

                    for number, reference in enumerate(
                        references,
                        start=1
                    ):

                        all_rows.append({
                            "Document": document_name,
                            "Reference Number": number,
                            "Reference": reference["reference"],
                            "Year": reference["year"] or "",
                            "DOI": "; ".join(reference["dois"]),
                            "URLs": "; ".join(reference["urls"])
                        })

                if all_rows:

                    all_references_df = pd.DataFrame(all_rows)

                    all_csv = all_references_df.to_csv(
                        index=False
                    ).encode("utf-8-sig")

                    st.divider()

                    st.download_button(
                        label="Download All References (CSV)",
                        data=all_csv,
                        file_name="all_research_references.csv",
                        mime="text/csv",
                        key="download_all_references"
                    )


# =====================================================
# TAB 11: PAPER QUALITY ANALYZER
# =====================================================

with tab11:

    st.header("AI Research Paper Quality Analyzer")

    st.write(
        "Analyze the structural completeness of your "
        "research papers and identify missing sections."
    )

    st.info(
        "This version checks paper structure using text patterns. "
        "It does not judge scientific correctness or research quality."
    )

    if (
        not st.session_state.processing_complete
        or not st.session_state.all_chunks
    ):

        st.warning(
            "Please upload and process research papers "
            "from the sidebar first."
        )

    else:

        if st.button(
            "Analyze Research Papers",
            key="analyze_paper_quality"
        ):

            with st.spinner(
                "Analyzing research paper structures..."
            ):

                try:

                    results = analyze_research_collection(
                        st.session_state.all_chunks
                    )

                    st.session_state.paper_quality_results = results
                    st.session_state.paper_quality_complete = True

                except Exception as e:

                    st.error(
                        f"Paper analysis failed: {str(e)}"
                    )

        if st.session_state.paper_quality_complete:

            quality_results = st.session_state.paper_quality_results

            if not quality_results:

                st.warning(
                    "No readable paper text was available for analysis."
                )

            else:

                # Overall statistics
                scores = [
                    result["completeness_score"]
                    for result in quality_results.values()
                ]

                average_score = round(
                    sum(scores) / len(scores),
                    2
                )

                complete_papers = sum(
                    1 for score in scores if score == 100
                )

                st.markdown("### Collection Overview")

                col1, col2, col3 = st.columns(3)

                col1.metric(
                    "Papers Analyzed",
                    len(quality_results)
                )

                col2.metric(
                    "Average Completeness",
                    f"{average_score}%"
                )

                col3.metric(
                    "All Sections Detected",
                    complete_papers
                )

                st.divider()

                # Individual paper analysis
                st.markdown("### Individual Paper Analysis")

                for document_name, result in quality_results.items():

                    score = result["completeness_score"]

                    with st.expander(
                        f"{document_name} — {score}% completeness"
                    ):

                        st.progress(
                            min(score / 100, 1.0)
                        )

                        st.write(
                            f"**Detected sections:** "
                            f"{result['present_sections']} / "
                            f"{result['total_sections']}"
                        )

                        st.markdown("#### Section Checklist")

                        for section, present in (
                            result["detected_sections"].items()
                        ):

                            if present:
                                st.write(f"✅ {section}")
                            else:
                                st.write(f"❌ {section}")

                        st.markdown("#### Recommendations")

                        if result["missing_sections"]:

                            for section in result["missing_sections"]:

                                st.write(
                                    f"- Check whether a "
                                    f"**{section}** section is present "
                                    f"or needs clearer labeling."
                                )

                        else:

                            st.success(
                                "All eight tracked section labels "
                                "were detected."
                            )

                # Comparison across papers
                st.divider()

                st.markdown("### Paper Structure Comparison")

                comparison_rows = []

                for document_name, result in quality_results.items():

                    comparison_rows.append({
                        "Research Paper": document_name,
                        "Completeness (%)": result["completeness_score"],
                        "Sections Present": (
                            f"{result['present_sections']}/"
                            f"{result['total_sections']}"
                        ),
                        "Missing Sections": (
                            ", ".join(result["missing_sections"])
                            if result["missing_sections"]
                            else "None detected"
                        )
                    })

                comparison_df = pd.DataFrame(comparison_rows)

                st.dataframe(
                    comparison_df,
                    use_container_width=True,
                    hide_index=True
                )

                # CSV download
                csv_data = comparison_df.to_csv(
                    index=False
                ).encode("utf-8-sig")

                st.download_button(
                    "Download Paper Quality Analysis (CSV)",
                    data=csv_data,
                    file_name="paper_quality_analysis.csv",
                    mime="text/csv",
                    key="download_paper_quality"
                )





# ---------------------------------------------------------
# FOOTER
# ---------------------------------------------------------

st.divider()

st.caption(
    "ResearchAI | Intelligent Research Workspace | "
    "Powered by Python, Streamlit, FAISS and Gemini"
)

