# 📚 Technical Documentation Assistant

A production-structured RAG (Retrieval-Augmented Generation) system that ingests technical PDF documents and enables natural language Q&A with cited answers.

Built as a portfolio project demonstrating: RAG pipeline engineering, LLM API integration (OpenAI + Anthropic), vector databases, MCP server implementation, evaluation metrics, and software engineering best practices.

---

## What It Does

Feed it any technical PDF → ask questions in plain English → get accurate answers with page-level citations.

```
User:    "What is a deviation in MISRA compliance?"
System:  "A deviation is a formal process that permits a project to use a guideline
          in a manner different from that specified. Deviations must be documented
          and authorized before the software is released.

          Sources:
            - MISRA-Compliance-2020.pdf, page 14
            - MISRA-Compliance-2020.pdf, page 15"
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                 Interfaces                                       │
│   Streamlit UI  ──────────────────────────────────────────────  │
│   MCP Server  (Claude Desktop / any MCP client)                 │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                 RAG Pipeline                                     │
│                                                                  │
│   Query  ──►  Embed Query  ──►  Vector Search  ──►  LLM  ──►   │
│                                  (ChromaDB)     (cited answer)  │
└───────────────────────────┬─────────────────────────────────────┘
                            │
            Offline ingestion (runs once per document)
                            │
                            ▼
┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────────────┐
│  PDF     │──►│  Clean   │──►│  Chunk   │──►│  Embed & Store   │
│  Reader  │   │ Headers/ │   │ (overlap)│   │  (ChromaDB)      │
│(PyMuPDF) │   │ Footers  │   │          │   │                  │
└──────────┘   └──────────┘   └──────────┘   └──────────────────┘
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| PDF extraction | PyMuPDF |
| LLM orchestration | LangChain |
| Embeddings | OpenAI `text-embedding-3-small` |
| Vector database | ChromaDB (local persistent) |
| LLM providers | OpenAI GPT-4o-mini / Anthropic Claude |
| Web UI | Streamlit |
| MCP interface | FastMCP (stdio transport) |
| Evaluation | RAGAS (faithfulness, relevancy, context precision) |
| Testing | pytest + unittest.mock |
| CI/CD | GitHub Actions |
| Containerization | Docker + Docker Compose |

---

## Project Structure

```
tech-doc-assistant/
├── src/
│   ├── config.py              # Centralized settings via pydantic-settings
│   ├── ingestion/             # PDF reading and data models
│   ├── chunking/              # Header/footer cleaning + text splitting
│   ├── embedding/             # Vector generation and ChromaDB storage
│   ├── generation/            # Prompt templates and LLM answer generation
│   ├── ui/                    # Streamlit web interface
│   └── mcp_server/            # MCP server exposing RAG as tools
├── tests/                     # Unit tests with mocking (pytest)
├── evaluation/                # RAGAS evaluation pipeline + ground truth dataset
├── scripts/                   # Manual pipeline verification scripts
├── data/raw/                  # PDF documents (gitignored)
├── vectorstore/               # ChromaDB persistent storage (gitignored)
├── .github/workflows/         # CI/CD pipeline
├── Dockerfile                 # Two-stage production image
└── docker-compose.yml         # Local development orchestration
```

---

## Quick Start

### Option 1: Local (recommended for development)

```bash
# Clone
git clone https://github.com/YOUR_USERNAME/tech-doc-assistant.git
cd tech-doc-assistant

# Setup environment
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"

# Configure API keys
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY

# Add a PDF document
cp your-document.pdf data/raw/

# Run the UI
uv run streamlit run src/ui/app.py
```

Open http://localhost:8501, upload your PDF, and start asking questions.

### Option 2: Docker

```bash
# Configure API keys
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY

# Build and run
docker compose up --build
```

Open http://localhost:8501.

---

## MCP Integration

This project exposes three MCP tools that any MCP-compatible client (Claude Desktop, VS Code, AI agents) can call:

| Tool | Description |
|---|---|
| `docs_status` | Check if documents are indexed and ready |
| `docs_search` | Retrieve raw relevant passages (no LLM call) |
| `docs_ask` | Full RAG Q&A with cited answer |

### Connect to Claude Desktop

**Step 1:** Find your Claude Desktop config file.

- macOS: `~/Library/Application Support/Claude/claude_desktop_config.json`
- Windows: `%APPDATA%\Claude\claude_desktop_config.json`

**Step 2:** Add the MCP server entry, using the absolute path to this project:

```json
{
  "mcpServers": {
    "tech-docs": {
      "command": "uv",
      "args": [
        "run",
        "--directory", "/absolute/path/to/tech-doc-assistant",
        "python", "src/mcp_server/server.py"
      ],
      "env": {
        "OPENAI_API_KEY": "your-key-here",
        "ANTHROPIC_API_KEY": "your-key-here"
      }
    }
  }
}
```

**Step 3:** Fully quit and reopen Claude Desktop. You should see a 🔧 icon in the chat interface
indicating MCP tools are available.

**Step 4:** Test it — type in Claude Desktop:

```
Check if my documentation assistant is ready, then ask it what a deviation is in MISRA compliance.
```

Claude will automatically call `docs_status` then `docs_ask` and return a cited answer from your
local document.

### Testing the server manually

```bash
uv run python src/mcp_server/server.py
```

You should see `MCP server starting — connecting to ChromaDB...` followed by
`ChromaDB ready — XXXX chunks indexed`. Press Ctrl+C to stop.

---

## Evaluation

The system is evaluated using RAGAS with gpt-4o as the judge model:

```bash
uv pip install -e ".[eval]"
uv run python evaluation/evaluate.py
```

**Metrics measured:**

| Metric | What it measures |
|---|---|
| Faithfulness | Does the answer stay within retrieved context? |
| Answer Relevancy | Does the answer address the question asked? |
| Context Precision | Are the retrieved chunks relevant to the query? |

Results are saved with timestamps to `evaluation/results/` for tracking improvement over time.

---

## Running Tests

```bash
# All tests
uv run pytest tests/ -v

# With coverage report
uv run pytest tests/ -v --cov=src --cov-report=term-missing
```

All unit tests use mocking — no API calls, no cost, fully deterministic.

---

## Key Engineering Decisions

**Why ChromaDB over Pinecone/Weaviate?**
Local persistence with zero infrastructure setup. Swappable — all vector DB logic is isolated in `src/embedding/vector_store.py`.

**Why recursive character splitting over fixed-size?**
Preserves semantic boundaries (paragraph > sentence > word). Fixed-size splitting cuts mid-sentence, destroying context coherence.

**Why content-based header/footer detection over coordinate-based?**
More robust across different PDF layouts. Detects repetition statistically rather than assuming fixed coordinates.

**Why two MCP tools (search + ask) instead of one?**
`docs_search` returns raw chunks for agents that want to do their own synthesis. `docs_ask` returns a finished answer for agents or humans that want a direct response. Different callers need different interfaces.

**Why `temperature=0` for generation?**
RAG answers should be deterministic and factual. Same question should give same answer. Creativity adds no value here.

---

## Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `OPENAI_API_KEY` | Yes | — | OpenAI API key (also used for embeddings regardless of `LLM_PROVIDER`) |
| `ANTHROPIC_API_KEY` | No | — | Anthropic API key (if using Claude) |
| `OPENROUTER_API_KEY` | No | — | OpenRouter API key (if using `LLM_PROVIDER=openrouter`) |
| `LLM_PROVIDER` | No | `openai` | `openai`, `anthropic`, or `openrouter` |
| `LLM_MODEL` | No | `gpt-4o-mini` | Model name (an OpenRouter model slug, e.g. `deepseek/deepseek-chat-v3.1:free`, when `LLM_PROVIDER=openrouter`) |
| `EMBEDDING_MODEL` | No | `text-embedding-3-small` | Embedding model |
| `CHUNK_SIZE` | No | `1000` | Characters per chunk |
| `CHUNK_OVERLAP` | No | `200` | Overlap between chunks |
| `RETRIEVAL_TOP_K` | No | `8` | Chunks retrieved per query |
