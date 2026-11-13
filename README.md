# ResearchAI — AI-Powered Research Paper Assistant

**ResearchAI** is an AI-powered research workspace designed to help students and researchers understand, analyze, compare, and explore academic research papers.

Built using **Retrieval-Augmented Generation (RAG), semantic search, FAISS, Sentence Transformers, and Google Gemini**, ResearchAI transforms research papers into an interactive knowledge experience.

🌐 **Live Demo:** https://researchai-c2n8fwuammsbp5ujkcrtpt.streamlit.app/

💻 **GitHub Repository:** https://github.com/akshatabagate77/ResearchAI


## 📸 Application Preview

![ResearchAI Dashboard](screenshots/dashboard.png)


---

## ✨ Features

ResearchAI provides 11 integrated research tools:

| Feature | Description |
|---|---|
| **AI Research Chat** | Ask questions about uploaded research papers and receive context-aware answers with source references. |
| **Research Summarization** | Generate structured summaries of academic papers. |
| **Paper Comparison** | Compare multiple research papers and understand their similarities and differences. |
| **Research Gap Detection** | Identify potential research gaps across selected papers. |
| **PDF Report Generation** | Generate downloadable research reports. |
| **Retrieval Evaluation** | Evaluate retrieval approaches using standard information retrieval metrics. |
| **Topic Explorer** | Explore important topics and concepts from research papers. |
| **Paper Recommendations** | Discover related research papers. |
| **Interactive Knowledge Graph** | Visualize relationships between research concepts. |
| **Citation Manager** | Extract citations and references from research papers. |
| **Paper Quality Analyzer** | Analyze research paper quality and key characteristics. |

---

## 🛠️ Technology Stack

| Component | Technology |
|---|---|
| Programming Language | Python |
| Frontend | Streamlit |
| Embedding Model | Sentence Transformers |
| Vector Database | FAISS |
| Retrieval | Semantic Search, Hybrid Search, TF-IDF |
| Reranking | Cross-Encoder |
| Large Language Model | Google Gemini |
| PDF Processing | PyMuPDF |
| Data Processing | Pandas, NumPy |
| Visualization | Plotly |
| Knowledge Graph | NetworkX |

---

## 🏗️ Project Architecture

```text
ResearchAI/
│
├── app.py
├── requirements.txt
├── README.md
├── evaluation_questions.json
│
└── src/
    ├── pdf_processor.py
    ├── text_chunker.py
    ├── embeddings.py
    ├── vector_store.py
    ├── semantic_search.py
    ├── reranker.py
    ├── rag_pipeline.py
    ├── llm.py
    ├── research_summarizer.py
    ├── research_comparator.py
    ├── research_gap_detector.py
    ├── report_generator.py
    ├── evaluation.py
    ├── topic_extractor.py
    ├── paper_recommender.py
    ├── knowledge_graph.py
    ├── citation_extractor.py
    └── paper_analyzer.py
```

### How ResearchAI Works

1. **PDF Upload:** Users upload academic research papers.
2. **Text Extraction:** Text is extracted from PDF documents.
3. **Text Chunking:** Extracted text is divided into manageable chunks.
4. **Embedding Generation:** Sentence Transformers convert text chunks into vector embeddings.
5. **Vector Storage:** FAISS stores embeddings for efficient similarity search.
6. **Retrieval:** Relevant passages are retrieved using semantic or hybrid retrieval.
7. **Reranking:** A Cross-Encoder can improve the ranking of retrieved passages.
8. **AI Generation:** Google Gemini generates responses using the retrieved research context.

---

## 🚀 Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/akshatabagate77/ResearchAI.git
cd ResearchAI
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

### 3. Activate the environment

**Windows PowerShell:**

```powershell
.venv\Scripts\Activate.ps1
```

### 4. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 5. Configure the Gemini API key

ResearchAI uses Google Gemini for AI-powered research analysis.

Create a `.env` file in the project root and configure:

```env
GEMINI_API_KEY=your_gemini_api_key
GEMINI_MODEL=your_supported_gemini_model
```

Use a Gemini model supported by your API account and project configuration.

For Streamlit Community Cloud, configure the same values through the app's **Secrets** settings.

**Security:** Never commit `.env`, API keys, or Streamlit secrets to GitHub.

### 6. Run the application

```bash
streamlit run app.py
```

The application will open in your browser.

---

## 📊 Retrieval Evaluation

ResearchAI includes a retrieval evaluation module to compare three retrieval approaches:

1. Semantic Retrieval
2. Hybrid Retrieval
3. Hybrid Retrieval with Cross-Encoder Reranking

### Evaluation Metrics

- Precision@K
- Recall@K
- Hit Rate
- Mean Reciprocal Rank (MRR)
- Average Retrieval Latency

The initial evaluation dataset is intended for development testing and can be expanded for more rigorous benchmarking.

---

## 🔮 Future Improvements

- Support for larger research-paper collections.
- Expanded retrieval evaluation datasets.
- Persistent paper storage and user accounts.
- More advanced citation and reference analysis.
- Improved research recommendations.
- Further performance and user experience improvements.

---

## 📌 Project Status

**Core application implemented, tested, and deployed.**

ResearchAI currently includes 11 integrated research features and is available through Streamlit Community Cloud.

The project is actively evolving with a focus on improving performance, usability, and research capabilities.

---

## 👩‍💻 Author

B.Tech Computer Science and Engineering (AI & ML) Student

---

**ResearchAI — Making academic research easier to explore.**
