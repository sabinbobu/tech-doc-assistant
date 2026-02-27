# 📚 Technical Documentation Assistant

A RAG (Retrieval-Augmented Generation) system that ingests technical PDF documents and enables natural language Q&A with cited answers.

## 🎯 What This Does

Feed it technical manuals, standards, or specifications → Ask questions in plain English → Get accurate answers with source citations.

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    User Interface (Streamlit)            │
│                  "How does X work?"                      │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│                  Retrieval System                        │
│  1. Embed the question                                  │
│  2. Search vector DB for similar chunks                 │
│  3. Return top-k relevant passages                      │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│               LLM Answer Generation                     │
│  "Based on the following context, answer..."            │
│  → Generates answer with [Source: doc, page] citations  │
└─────────────────────────────────────────────────────────┘

===== Offline Pipeline (runs once per document) =====

┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
│  PDF     │ →  │  Text    │ →  │  Chunk   │ →  │  Embed   │
│  Ingestion│    │  Cleaning│    │  Splitting│    │  & Store │
└──────────┘    └──────────┘    └──────────┘    └──────────┘
```

## 📁 Project Structure

```
tech-doc-assistant/
├── src/                    # All source code
│   ├── ingestion/          # PDF reading & text extraction
│   ├── chunking/           # Text splitting strategies
│   ├── embedding/          # Vector generation & storage
│   ├── retrieval/          # Search & ranking
│   ├── generation/         # LLM prompting & answer generation
│   └── ui/                 # Streamlit web interface
├── tests/                  # Unit & integration tests
├── data/                   # Sample documents (gitignored for large files)
│   ├── raw/                # Original PDFs
│   └── processed/          # Extracted & chunked text
├── vectorstore/            # Persistent vector database (gitignored)
├── evaluation/             # Eval scripts and results
├── .github/workflows/      # CI/CD pipeline
├── pyproject.toml          # Project metadata & dependencies
├── Dockerfile              # Container definition
└── README.md               # This file
```

## 🚀 Quick Start

```bash
# Clone and setup
git clone https://github.com/YOUR_USERNAME/tech-doc-assistant.git
cd tech-doc-assistant
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -e ".[dev]"

# Add your documents
cp your-document.pdf data/raw/

# Ingest documents
python -m src.ingestion.pipeline

# Run the assistant
streamlit run src/ui/app.py
```

## 🛠️ Tech Stack

- **Python 3.11+** — Core language
- **LangChain** — LLM orchestration framework
- **ChromaDB** — Vector database (local, no server needed)
- **OpenAI / Anthropic API** — LLM for answer generation
- **Sentence-Transformers** — Embedding models (optional local alternative)
- **Streamlit** — Web UI
- **PyMuPDF (fitz)** — PDF text extraction
- **Pytest** — Testing
- **Docker** — Containerization
- **GitHub Actions** — CI/CD

## 📊 Evaluation

See `evaluation/` for retrieval precision, answer faithfulness, and relevance metrics.

## 📝 License

MIT
