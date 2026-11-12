# ResearchAI — AI-Powered Research Paper Assistant

ResearchAI is an AI-powered research assistant designed to help students and researchers understand, analyze, compare, and explore academic research papers using Retrieval-Augmented Generation (RAG), semantic search, and large language models.

## Features

- **AI Research Chat:** Ask questions about uploaded research papers and receive context-aware answers with source references.
- **Research Summarization:** Generate concise summaries of academic papers.
- **Paper Comparison:** Compare multiple research papers.
- **Research Gap Detection:** Identify potential research gaps from uploaded papers.
- **PDF Report Generation:** Generate downloadable research reports.
- **Retrieval Evaluation:** Evaluate semantic, hybrid, and reranked retrieval using Precision@K, Recall@K, Hit Rate, MRR, and latency.
- **Topic Explorer:** Explore important research topics.
- **Paper Recommendations:** Discover related research papers.
- **Interactive Knowledge Graph:** Visualize relationships between research concepts.
- **Citation Manager:** Extract citations and references.
- **Paper Quality Analyzer:** Analyze research paper quality.

## Technology Stack

- **Programming Language:** Python
- **Frontend:** Streamlit
- **Embeddings:** Sentence Transformers
- **Vector Database:** FAISS
- **Retrieval:** Semantic Search, Hybrid Search, TF-IDF
- **Reranking:** Cross-Encoder
- **LLM:** Google Gemini
- **PDF Processing:** PyMuPDF
- **Data Processing:** Pandas, NumPy
- **Visualization:** Plotly

## Project Architecture

```text
AI research assistant/
│
├── app.py
├── requirements.txt
├── README.md
├── evaluation_questions.json
│
├── src/
│   ├── pdf_processor.py
│   ├── text_chunker.py
│   ├── embeddings.py
│   ├── vector_store.py
│   ├── semantic_search.py
│   ├── reranker.py
│   ├── rag_pipeline.py
│   ├── research_summarizer.py
│   ├── research_comparator.py
│   ├── research_gap_detector.py
│   ├── report_generator.py
│   ├── evaluation.py
│   ├── topic_extractor.py
│   ├── paper_recommender.py
│   ├── knowledge_graph.py
│   ├── citation_extractor.py
│   └── paper_analyzer.py
│
└── test_papers/
```

## Installation and Setup

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd "AI research assistant"
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

### 3. Activate the environment

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Configure your Gemini API key

Set up your Gemini API key using the configuration method implemented in the project. Never commit API keys or other secrets to GitHub.

### 6. Run the application

```bash
streamlit run app.py
```

## Retrieval Evaluation

ResearchAI includes an evaluation dashboard that compares three retrieval approaches:

1. Semantic Retrieval
2. Hybrid Retrieval
3. Hybrid Retrieval with Cross-Encoder Reranking

Evaluation metrics include:

- Precision@K
- Recall@K
- Hit Rate
- Mean Reciprocal Rank (MRR)
- Average Retrieval Latency

The initial evaluation dataset is intended for development testing and can be expanded for more rigorous benchmarking.

## Future Improvements

- Support for larger research-paper collections.
- Improved retrieval evaluation with a larger ground-truth dataset.
- Persistent paper storage and user accounts.
- More advanced citation analysis.
- Cloud deployment and public access.
- Improved research recommendations.

## Project Status

Core application features implemented and tested locally. Further improvements and deployment are planned.

## Author

B.Tech Computer Science and Engineering (AI & ML) Student

**ResearchAI — Making academic research easier to explore.**