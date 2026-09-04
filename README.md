<div align="center">

# Technical Documentation Assistant

**Ask questions about technical PDFs — datasheets, safety standards, service manuals — and get answers with page-level citations.**

A RAG pipeline where every component earns its place with a measured number, not a blog post.

[![CI](https://github.com/sabinbobu/tech-doc-assistant/actions/workflows/ci.yml/badge.svg)](https://github.com/sabinbobu/tech-doc-assistant/actions/workflows/ci.yml)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![Tests](https://img.shields.io/badge/tests-168%20passing-brightgreen.svg)](#testing)
[![recall@8](https://img.shields.io/badge/recall%408-1.000-brightgreen.svg)](#benchmarks)
[![License: MIT](https://img.shields.io/badge/license-MIT-yellow.svg)](LICENSE)

[Quick start](#quick-start) · [Benchmarks](#benchmarks) · [Using the module](#using-the-module) · [Behavior under load and failure](#behavior-under-load-and-failure) · [Architecture](#architecture) · [Why not a stock RAG stack](#why-not-a-stock-rag-stack)

</div>

---

Real output against a 120-chunk Infineon power-switch datasheet — abridged only in length, not edited:

```
You:  Guide me step by step on how to diagnose a short to ground on the BTS7200-2EPA.

Assistant:  [mode=procedural  has_answer=True]

  1. Connect the necessary components: a pull-down resistor and a switchable
     pull-up current source on the load, so the device can measure the output
     voltage and compare it with the threshold voltage
     [Source: infineon-bts7200-2epa-datasheet-en.pdf, page 45].

  2. Set the DEN pin "high" to enable device diagnosis. This also clears the
     protection counter for the selected channel
     [Source: infineon-bts7200-2epa-datasheet-en.pdf, page 6].

  3. Select the channel with the DSEL pin ...
     [Source: infineon-bts7200-2epa-datasheet-en.pdf, page 6].

  4. Monitor the IS pin for the SENSE current output. A short to ground shows
     IIS(FAULT) once the internal counter is greater than zero
     [Source: infineon-bts7200-2epa-datasheet-en.pdf, page 42].

  5. (Inferred) Compare the sensed current ...
```

Four pages, none adjacent — pin configuration (p.6), sense circuitry (p.45), fault current (p.42).
A single top-8 retrieval finds one of those neighborhoods, not all four. Note the `(Inferred)`
marker on step 5: the pipeline flags where it connected two passages rather than quoting one.

Answers are grounded **exclusively** in the indexed documents. When the corpus doesn't cover the
question, the pipeline returns `has_answer=False` and says so, rather than filling the gap.

---

## Table of contents

- [What this is](#what-this-is)
- [Why not a stock RAG stack](#why-not-a-stock-rag-stack)
- [Benchmarks](#benchmarks)
- [Specifications](#specifications)
- [Quick start](#quick-start)
- [Using the module](#using-the-module)
  - [Python API](#1-python-api)
  - [MCP server](#2-mcp-server)
  - [Streamlit UI](#3-streamlit-ui)
- [Behavior under load and failure](#behavior-under-load-and-failure)
- [Architecture](#architecture)
- [Configuration](#configuration)
- [Evaluation](#evaluation)
- [Testing](#testing)
- [Engineering decisions](#engineering-decisions)
- [Limitations](#limitations)

---

## What this is

A self-contained retrieval-augmented generation pipeline for technical documentation, with three
interfaces over one core:

| Interface | Use it for |
|---|---|
| **Python API** | Embedding the pipeline in your own code — three public functions, no framework buy-in |
| **MCP server** | Exposing your document corpus as tools to Claude Desktop, an IDE, or any MCP client |
| **Streamlit UI** | Uploading a PDF and asking questions in a browser |

Everything runs locally except the embedding and generation API calls. The vector store is a file,
the reranker is a 22M-parameter model on your CPU, and the keyword index is built in-process.
No Docker-compose-of-six-services, no managed vector DB, no per-query reranking bill.

---

## Why not a stock RAG stack

The default RAG tutorial — chunk the text, embed it, cosine-similarity the query, stuff the top-k
into a prompt — works until you point it at a datasheet. Then it fails on part numbers, on spec
tables, on "list the table of contents", and on any question whose answer is scattered across four
non-adjacent pages.

Each row below is a default this project replaced, and the number that justified it. Every figure
was produced by `evaluation/compare.py` against the live index, not estimated.

| Common default | What this does instead | Measured effect |
|---|---|---|
| Vector search only | **Hybrid BM25 + vector**, fused with Reciprocal Rank Fusion (`k=60`) | Identifier-heavy questions (`BTS7200-2EPA`, `Rule 8.4`, `RSENSE`) are exactly where cosine similarity blurs and exact-term matching wins |
| Embed the bare chunk text | Embed **`{source} > {section} > page {n}\n\n{text}`** so both the dense vector and BM25 see the breadcrumb | MRR 0.772 → **0.818**, nDCG@8 0.774 → **0.811**. Free — it's a string concatenation |
| `query.lower().split()` for BM25 | **Regex tokenizer** that strips only leading/trailing punctuation and preserves internal `-` `.` `'` (so `bts7200-2epa` and `8.4` survive) | A question ending in `?` glued the punctuation onto its last token, so it never matched the corpus. Fixing it took the target chunk from BM25 **rank 61 → rank 3**; recall@8 **28/30 → 30/30** |
| One retrieval strategy for every question | **Route** each question to `lookup` / `procedural` / `structural` | "List the table of contents" has no chunk-retrieval answer at all and used to be refused outright. Procedural questions are decomposed into sub-queries and their results unioned |
| Rerank through a hosted API (Cohere/Jina/Voyage) | **Local cross-encoder**, warmed at process start | 296 ms warm for 20 candidates, **$0 per query**, works offline, no extra API key |
| Ship every feature that sounds like an improvement | **Query rewriting is shipped OFF** | It measured a wash on retrieval and dropped RAGAS faithfulness **0.815 → 0.682** by stripping question framing. The module and its 12 tests stay; the feature doesn't ship active |
| "It seems better" as the acceptance criterion | **Deterministic retrieval eval**: 36 gold-labeled questions, recall@k / MRR / nDCG@k, ~60 s, **$0**, no LLM judge | RAGAS runs on top as the slower end-to-end gate, not as the inner loop |

The honest version of the pitch: this is not a faster LangChain. It is a **reference implementation
of a RAG pipeline where the defaults were tested rather than assumed**, on a corpus (technical
datasheets) that punishes the naive ones.

---

## Benchmarks

### Retrieval quality

Deterministic, no LLM judge. `n=36` gold-labeled questions across three documents, scored against
manually verified gold pages. Reproduce with `python -m evaluation.compare`.

| Stage | recall@8 | MRR | nDCG@8 |
|---|---|---|---|
| Vector + BM25 hybrid alone | 0.972 | 0.810 | 0.816 |
| **+ cross-encoder rerank** *(shipped default)* | **1.000** | **0.852** | **0.842** |
| + query rewriting *(disabled)* | 1.000 | 0.852 | 0.842 |

Query rewriting is byte-identical to rerank-alone on retrieval and measurably worse downstream —
which is why it ships off.

### Retrieval quality by question type

The hypothesis behind hybrid search is that BM25 rescues questions hinging on exact identifiers.
The split tests it:

| Subset | n | recall@8 | MRR | nDCG@8 |
|---|---|---|---|---|
| Identifier-heavy (part numbers, rule IDs, spec values) | 19 | 1.000 | 0.840 | 0.838 |
| General prose questions | 17 | 1.000 | 0.865 | 0.847 |

Before the tokenizer fix, identifier-heavy scored **0.857** recall@8 against general's 1.000 — the
gap the hybrid design existed to close, and didn't, until the BM25 input was actually inspected.

### End-to-end answer quality

RAGAS, `n=36`, default config (`evaluation/results/eval_results_20260902_215850.json`):

| Metric | Score | Threshold | Baseline before retrieval fixes |
|---|---|---|---|
| Faithfulness | **0.894** | ≥ 0.70 | 0.757 |
| Answer relevancy | **0.849** | ≥ 0.70 | 0.832 |
| Context precision | **0.862** | ≥ 0.70 | 0.763 |

Faithfulness moved 0.757 → 0.894 **without a single change to `src/generation/`**. It was suppressed
by retrieval surfacing the wrong page, not by prompt quality — the kind of conclusion you can only
draw when retrieval and generation are measured separately.

### Latency

Measured on an Intel Mac, CPU-only torch, via Opik traces:

| Operation | Time |
|---|---|
| Reranker model load — **once per process** | 12,411 ms |
| Rerank, 20 candidates, warm | 296 ms |
| Rerank, 10 candidates, warm | 259 ms |
| Rerank, warm, after ~12 s process idle | 1,386 ms |

Read that first row carefully. A trace taken on a cold process shows a ~9 s `rerank` span that is
almost entirely model load — reading it as steady state makes the reranker look like **54%** of
end-to-end latency when warm it is closer to **6%**. The fix was `warm_up()` at process start
(`src/retrieval/rerank.py:74`), not a smaller model and not a hosted API.

---

## Specifications

| Parameter | Value | Notes |
|---|---|---|
| Python | 3.11 | Pinned; `torch` has no x86_64 macOS wheel past 2.2.2 (`cp312` cap) |
| PDF extraction | PyMuPDF (`fitz`) | C-based; 10–50× faster than PyPDF2 |
| Chunk size / overlap | 1000 / 200 chars | Recursive character splitting on paragraph → sentence → word |
| Embedding model | `text-embedding-3-small` | 1536-dim |
| Vector store | ChromaDB, local persistent | Single SQLite file, no server |
| Keyword index | BM25 (`rank-bm25`), in-process | Rebuilt per query over the filtered set |
| Fusion | Reciprocal Rank Fusion, `k=60` | Standard RRF constant |
| Candidate pool → top-k | 20 → 8 | Deliberately over-fetched: RRF's top-8 by rank ≠ the reranker's top-8 by relevance |
| Reranker | `cross-encoder/ms-marco-MiniLM-L6-v2` | 22M params, local, CPU |
| Generation | OpenAI / Anthropic / OpenRouter | `temperature=0` |
| Answer token budget | 1024 lookup / 3000 procedural | Procedural answers truncated mid-step at 1024 |
| Reference corpus | 565 chunks / 3 PDFs | Epson service manual 338, Infineon datasheet 120, MISRA standard 107. The PDFs themselves are gitignored — bring your own |
| Eval set | 36 gold-labeled questions | 19 identifier-heavy, 17 general |
| Test suite | 168 tests, ~14 s | Fully mocked — no API calls, no cost |
| Observability | Opik tracing on every stage | Route / retrieve / rerank / generate, with latency + tokens + cost |

---

## Quick start

### Local

```bash
git clone https://github.com/sabinbobu/tech-doc-assistant.git
cd tech-doc-assistant

uv venv && source .venv/bin/activate
uv sync --extra dev

cp .env.example .env          # add your OPENAI_API_KEY
cp your-document.pdf data/raw/

python scripts/reingest.py --wipe    # build the index
streamlit run src/ui/app.py          # http://localhost:8501
```

`data/raw/` ships empty — the PDFs used for the benchmarks are gitignored, so bring your own.
`OPENAI_API_KEY` is required regardless of `LLM_PROVIDER`: embeddings always go through OpenAI.

### Docker

```bash
cp .env.example .env          # add your OPENAI_API_KEY
docker compose up --build     # http://localhost:8501
```

---

## Using the module

### 1. Python API

Three public entry points. Everything else is an implementation detail.

**Ingest a corpus** — offline, run once per document set:

```python
from src.ingestion.pdf_reader import extract_all_pdfs
from src.chunking.cleaner import clean_document
from src.chunking.chunker import chunk_document
from src.embedding.vector_store import embed_chunks, save_document_outline

for doc in extract_all_pdfs("data/raw"):
    cleaned = clean_document(doc)          # statistical header/footer removal
    embed_chunks(chunk_document(cleaned))  # embeds "{source} > {section} > page {n}\n\n{text}"
    save_document_outline(cleaned)         # PDF outline, for structural questions
```

**Retrieve passages** — hybrid search, no LLM call, no cost:

```python
from src.embedding.vector_store import query_collection

chunks = query_collection(
    "absolute maximum VS rating",
    n_results=8,
    source_filenames=["infineon-bts7200-2epa-datasheet-en.pdf"],  # None = search everything
)
# -> [{"text": ..., "metadata": {"source_filename", "page_number", "section", ...}, "distance": ...}]
```

**Answer a question** — the full pipeline: route → retrieve → rerank → generate:

```python
from src.generation.generator import generate_answer

answer = generate_answer(
    "What are the four protection functions?",
    n_results=8,               # defaults to settings.retrieval_top_k
    source_filenames=None,     # scope to specific documents
)

answer.answer            # str  — the generated text, with inline [Source: ...] markers
answer.has_answer        # bool — False when the corpus doesn't cover the question
answer.mode              # "lookup" | "procedural" | "structural" — how it was routed
answer.sources           # list[RetrievedContext] — text, citation, page_number, rerank_score
print(answer.formatted_sources)
```

`Answer` is a Pydantic model (`src/generation/models.py`). Citations are structurally enforced —
there is no code path that returns an answer without its sources.

**Warm the reranker** if you're running a long-lived process:

```python
from src.retrieval.rerank import warm_up
warm_up()   # absorbs the 12.4 s model load at startup instead of on the first user query
```

### 2. MCP server

Three read-only tools for any MCP client:

| Tool | Does | LLM call? |
|---|---|---|
| `docs_status` | Reports index readiness and chunk count — call this first | No |
| `docs_search` | Returns reranked raw passages for your own reasoning chain | No |
| `docs_ask` | Full RAG Q&A with citations | Yes |

Add to `~/Library/Application Support/Claude/claude_desktop_config.json` (macOS) or
`%APPDATA%\Claude\claude_desktop_config.json` (Windows):

```json
{
  "mcpServers": {
    "tech-docs": {
      "command": "uv",
      "args": [
        "run", "--directory", "/absolute/path/to/tech-doc-assistant",
        "python", "src/mcp_server/server.py"
      ],
      "env": { "OPENAI_API_KEY": "sk-..." }
    }
  }
}
```

Fully quit and reopen Claude Desktop, then:

> *Check if my documentation assistant is ready, then ask it what a deviation is in MISRA compliance.*

Claude calls `docs_status` then `docs_ask` and returns a cited answer from your local corpus.

Test the server standalone with `uv run python src/mcp_server/server.py` — it should print
`ChromaDB ready — N chunks indexed`.

### 3. Streamlit UI

```bash
streamlit run src/ui/app.py
```

Sidebar for document upload and per-document retrieval scoping; main area shows the transcript with
per-source rerank scores and the routing mode used for each answer.

---

## Behavior under load and failure

The design rule: **every optional stage degrades to the next-simpler strategy. None of them can take
down an answer.** Generation is the deliberate exception — there is nothing simpler behind it, so it
raises loudly rather than silently switching providers.

| Condition | Behavior | Cost |
|---|---|---|
| **Cold process, first query** | `warm_up()` at startup absorbs the 12.4 s model load | Without it, the first user query pays all 12.4 s |
| **Reranker missing** (no torch, offline, `RERANKING_ENABLED=false`) | Falls back to hybrid RRF order | recall@8 1.000 → 0.972, MRR 0.852 → 0.810 |
| **Process idle > ~12 s** | Next rerank takes ~1,386 ms, then returns to ~300 ms | Steady state in the real pipeline is the ~1.3 s figure — `rerank` always runs right after the router's LLM call has blocked. Don't read the gap as a regression |
| **Query classifier returns unparseable JSON** | Falls back to `lookup` mode with the raw question | Observed live and handled with no code change when OpenRouter routed the call to a content-safety classifier model |
| **Question is `procedural`** | Decomposed into sub-queries, results unioned, 3000-token answer budget | One query only searches one neighborhood of a document |
| **Question is `structural`** ("list the TOC") | Answered from the stored PDF outline, no chunk retrieval at all | No chunk boundary can reconstruct a whole TOC |
| **Answer isn't in the corpus** | `has_answer=False`, explicit statement of the gap | No hallucinated fill — the prompt forbids outside knowledge |
| **Index empty** | `query_collection` returns `[]` and logs a warning | No crash |
| **Multi-document corpus** | `source_filenames` scopes retrieval per query | Without it, an unrelated document's chunks leak into the top-k |
| **OpenRouter free-tier rate limit** | `LLM_MODEL=openrouter/free` spreads calls across ~24 free models | Verified: 15/15 sequential calls, 0 errors, 8 different models, under the exact burst that 429'd a single pinned model. Tradeoff: answers are no longer deterministic across runs even at `temperature=0` |
| **LLM provider fails outright** | Raises; the caller converts it to a clear user-facing error | Intentionally *not* a silent fallback |
| **CI and test runs** | `OPIK_TRACK_DISABLE=true` turns every `@opik.track` into a pass-through | No spans, no network, no API key needed |
| **Re-ingesting a shorter revision of a PDF** | ⚠️ `chunk_index` restarts per document and upsert is not delete — orphaned chunks remain | Always `scripts/reingest.py --wipe` |

---

## Architecture

### Query path

```
             Streamlit UI    ·    MCP server    ·    Python API
                               │
                               ▼
                       generate_answer()
                               │
            ┌──────────────────┴──────────────────┐
            │  ROUTE            query_plan.py     │
            └──────────────────┬──────────────────┘
       ┌───────────────────────┼─────────────────────────────────────┐
       ▼                       ▼                                     ▼
    lookup                procedural                            structural
   one query       decompose → N sub-queries                    PDF outline
       │                       │                                     │
       └───────────────────────┤                                     │
                               ▼                                     │
            ┌──────────────────┴──────────────────┐                  │
            │  RETRIEVE         BM25 + vector     │ → 20 candidates  │
            │                   RRF fusion, k=60  │                  │
            └──────────────────┬──────────────────┘                  │
                               ▼                                     │
            ┌──────────────────┴──────────────────┐                  │
            │  RERANK           cross-encoder     │ → top 8          │
            └──────────────────┬──────────────────┘                  │
                               ▼                                     │
            ┌──────────────────┴──────────────────┐◄─────────────────┘
            │  GENERATE         LLM + citations   │
            └──────────────────┬──────────────────┘
                               ▼
             Answer(text, sources, mode, has_answer)
```

Every span above is traced to Opik — route, retrieve, rerank and generate each carry their own
latency, token count and cost.

### Ingestion path — offline, once per document

```
   PDF  ──►  extract  ──►  clean  ──►  chunk  ──►  embed  ──►  ChromaDB
            (PyMuPDF)      repeated     1000 /      "{source} > {section} > page {n}
            + outline      header /     200 chars    \n\n{text}"
                           footer       + "Parent    — the breadcrumb is what BM25
                           removal      topic:"        and the dense vector both see
                                        pre-split
```

### Module map

```
src/
├── config.py                  Pydantic settings — one source of truth, env-overridable.
│                              Read the comments here before tuning anything; they carry
│                              the measurements behind each default.
├── ingestion/
│   ├── pdf_reader.py          PyMuPDF extraction + PDF outline capture
│   └── models.py              PageContent / DocumentContent / OutlineEntry
├── chunking/
│   ├── cleaner.py             Statistical header/footer detection (repetition, not coordinates)
│   ├── chunker.py             Recursive splitting + "Parent topic:" pre-split for RoboHelp exports
│   └── models.py              Chunk, carrying source / page / section metadata
├── embedding/
│   └── vector_store.py        ChromaDB, BM25, RRF fusion, tokenizer, per-document filtering
├── retrieval/
│   ├── query_plan.py          lookup / procedural / structural routing + decomposition
│   ├── rerank.py              Local cross-encoder + warm_up()
│   ├── structure.py           TOC / outline answering
│   └── query_rewrite.py       Shipped disabled — kept with its tests as a negative result
├── generation/
│   ├── generator.py           The public pipeline: generate_answer()
│   ├── llm_client.py          Provider gateway (OpenAI / Anthropic / OpenRouter)
│   ├── prompts.py             Prompt templates — logic, not strings
│   └── models.py              Answer / RetrievedContext
├── ui/                        Streamlit app, components, formatting
└── mcp_server/server.py       FastMCP stdio server — docs_status / docs_search / docs_ask

evaluation/
├── dataset.py                 36 gold-labeled questions; gold_pages verified against the PDFs
├── retrieval_metrics.py       recall@k, MRR, nDCG@k
├── compare.py                 The inner loop — deterministic, ~60 s, $0
└── evaluate.py                RAGAS end-to-end gate — slow, costs API credits

scripts/reingest.py            Wipe + rebuild the index from data/raw/ only
```

---

## Configuration

All settings live in `src/config.py` and are overridable by environment variable or `.env`.

| Variable | Required | Default | Description |
|---|---|---|---|
| `OPENAI_API_KEY` | **Yes** | — | Used for embeddings regardless of `LLM_PROVIDER` |
| `ANTHROPIC_API_KEY` | No | — | Required when `LLM_PROVIDER=anthropic` |
| `OPENROUTER_API_KEY` | No | — | Required when `LLM_PROVIDER=openrouter` |
| `LLM_PROVIDER` | No | `openai` | `openai` \| `anthropic` \| `openrouter` |
| `LLM_MODEL` | No | `gpt-4o-mini` | An OpenRouter slug when using OpenRouter; prefer `openrouter/free` over pinning one `:free` model |
| `EMBEDDING_MODEL` | No | `text-embedding-3-small` | |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | No | `1000` / `200` | Characters |
| `RETRIEVAL_TOP_K` | No | `8` | Chunks passed to the LLM |
| `RETRIEVAL_CANDIDATE_K` | No | `20` | Pool fed to the reranker; must be ≥ `RETRIEVAL_TOP_K` |
| `RERANKING_ENABLED` | No | `true` | Kill switch — retrieval still works when off |
| `RERANKER_MODEL` | No | `cross-encoder/ms-marco-MiniLM-L6-v2` | |
| `QUERY_REWRITE_ENABLED` | No | `false` | Off by measurement — read `src/config.py` before enabling |
| `QUERY_PLANNING_ENABLED` | No | `true` | Question routing |
| `OPIK_TRACK_DISABLE` | No | unset | `true` makes all tracing a pass-through |

Opik credentials live in `~/.opik.config` (written by `opik configure`), not in `.env`.

---

## Evaluation

Two loops, deliberately separate.

**Inner loop** — deterministic, no LLM judge, no API cost, ~60 s. Run this on every retrieval change:

```bash
python -m evaluation.compare
```

Scores `recall@k`, `MRR` and `nDCG@k` for each pipeline stage against `evaluation/dataset.py`, where
every question carries a `gold_doc` and manually verified `gold_pages`. A chunk counts as relevant
only if it comes from the gold document *and* overlaps a gold page — a deliberately strict bar.

**Outer loop** — RAGAS end-to-end, slow, costs API credits. Run before handing off a unit of work:

```bash
uv pip install -e ".[eval]"
python evaluation/evaluate.py
```

Results are timestamped into `evaluation/results/`. **No metric claim in this repo exists without
its before/after JSON in that directory.**

Inspect the live index:

```bash
python -c "import sqlite3;print(sqlite3.connect('vectorstore/chroma.sqlite3').execute(\"select string_value,count(*) from embedding_metadata where key='source_filename' group by 1\").fetchall())"
```

---

## Testing

```bash
pytest tests/ -q                                    # 168 tests, ~14 s
pytest tests/ -v --cov=src --cov-report=term-missing
```

Every test mocks its LLM and embedding calls — no API keys, no cost, fully deterministic.

CI runs exactly this, plus:

```bash
ruff check src/ tests/ && ruff format --check src/ tests/
mypy src/ --ignore-missing-imports
```

Dependencies install from the committed `uv.lock` via `uv sync --locked`, not a fresh resolve.
`pyproject.toml` declares open lower bounds, so `pip install -e .` silently type-checked a different
dependency set than any developer had locally — `--locked` is what makes CI and local reproduce each
other.

---

## Engineering decisions

**Hybrid search over vector-only.** Technical corpora are dense with exact identifiers — part
numbers, rule IDs, register names — and cosine similarity blurs precisely those. RRF (`k=60`) fuses
the two rankings without needing a tuned weight between incomparable score scales.

**A 22M-param reranker, not a 568M one.** `BAAI/bge-reranker-v2-m3` benchmarked at 11.7 s per 30
candidates on this machine — unusable interactively. `ms-marco-MiniLM-L6-v2` does the same job in
~296 ms warm with no measurable quality loss on English short-passage technical text. Bigger
cross-encoders earn their cost on harder ranking problems; this isn't one.

**Local reranking, not an API reranker.** Cohere/Jina/Voyage would *add* network latency, a key, and
a per-query bill to an operation that is currently local, free and offline-capable. Strictly worse on
every axis this project optimizes for.

**Reranking stays on, despite an early read saying otherwise.** An `n=8` RAGAS run said flat-to-worse;
the `n=36` deterministic run says a consistent win on all three metrics. The larger, deterministic,
free measurement wins.

**Query rewriting ships disabled.** It's a wash on retrieval and dropped faithfulness 0.815 → 0.682,
flipping a correct answer ("Mandatory/Advisory") to a wrong-but-plausible one ("Rules/Directives") by
stripping question framing. The code and its 12 tests stay in the repo as a documented negative
result rather than being deleted or quietly enabled.

**Content-based header/footer detection, not coordinate-based.** Detects repetition statistically
across pages instead of assuming fixed geometry — robust across PDF layouts that differ wildly
between a standards document and a RoboHelp export.

**Two MCP retrieval tools, not one.** `docs_search` returns raw passages for agents that want to do
their own synthesis; `docs_ask` returns a finished cited answer. Different callers, different
interfaces.

**`temperature=0`.** RAG answers should be reproducible and factual. Creativity adds nothing here.

**ChromaDB over Pinecone/Weaviate.** Local persistence, zero infrastructure. All vector-DB logic is
isolated in `src/embedding/vector_store.py` and is swappable if scale ever demands it.

---

## Limitations

Stated plainly, because a benchmark table without them is marketing:

- **The eval set is 36 questions.** Enough to catch a broken retriever, not enough to resolve
  differences smaller than ~0.03 on any metric. Treat small movements as noise.
- **`recall@8 = 1.000` is on this corpus.** Three documents, 565 chunks, questions written against
  them. It is a regression gate, not a claim about your PDFs.
- **Single-machine, single-user.** ChromaDB local persistence, no auth, no multi-tenancy, no
  horizontal scaling story.
- **Table extraction is text-based.** Complex multi-column spec tables are chunked as text; there is
  no structured table parser. The `Parent topic:` pre-split is specific to RoboHelp exports.
- **`openrouter/free` is not reproducible.** Model identity varies per call, so answer phrasing and
  quality differ across runs even at `temperature=0`. Pin `openai` or `anthropic` for eval runs.
- **Intel-Mac Python 3.11 pin.** `torch` never published an x86_64 macOS wheel past 2.2.2, which
  caps at `cp312`. On other platforms the floor could be raised.

---

## License

MIT — see [LICENSE](LICENSE).
