"""Generate a PDF report of the Tech Doc Assistant project implementation."""

from fpdf import FPDF


class Report(FPDF):
    DARK = (30, 30, 30)
    ACCENT = (0, 90, 156)
    LIGHT_ACCENT = (230, 242, 250)
    GRAY = (100, 100, 100)
    LIGHT_GRAY = (245, 245, 245)
    WHITE = (255, 255, 255)

    def header(self):
        if self.page_no() > 1:
            self.set_font("Helvetica", "I", 8)
            self.set_text_color(*self.GRAY)
            self.cell(0, 6, "Tech Doc Assistant - Implementation Report", align="R")
            self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(*self.GRAY)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", align="C")

    def chapter_title(self, num, title):
        self.set_font("Helvetica", "B", 16)
        self.set_text_color(*self.ACCENT)
        self.cell(0, 12, f"{num}. {title}", new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(*self.ACCENT)
        self.set_line_width(0.6)
        self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
        self.ln(6)

    def section_title(self, title):
        self.set_font("Helvetica", "B", 12)
        self.set_text_color(*self.DARK)
        self.cell(0, 10, title, new_x="LMARGIN", new_y="NEXT")
        self.ln(2)

    def subsection_title(self, title):
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(*self.ACCENT)
        self.cell(0, 8, title, new_x="LMARGIN", new_y="NEXT")
        self.ln(1)

    def body_text(self, text):
        self.set_font("Helvetica", "", 10)
        self.set_text_color(*self.DARK)
        self.multi_cell(0, 5.5, text)
        self.ln(2)

    def bullet(self, text, indent=10):
        x = self.get_x()
        self.set_font("Helvetica", "", 10)
        self.set_text_color(*self.DARK)
        self.set_x(x + indent)
        self.cell(5, 5.5, "-")
        self.multi_cell(0, 5.5, text)
        self.ln(1)

    def code_block(self, text):
        self.set_fill_color(*self.LIGHT_GRAY)
        self.set_font("Courier", "", 8.5)
        self.set_text_color(50, 50, 50)
        x = self.get_x()
        w = self.w - self.l_margin - self.r_margin
        lines = text.split("\n")
        h = len(lines) * 4.5 + 6
        if self.get_y() + h > self.h - 25:
            self.add_page()
        y_start = self.get_y()
        self.rect(x, y_start, w, h, "F")
        self.set_xy(x + 3, y_start + 3)
        for line in lines:
            self.cell(0, 4.5, line, new_x="LMARGIN", new_y="NEXT")
            self.set_x(x + 3)
        self.set_y(y_start + h + 3)

    def highlight_box(self, title, text):
        self.set_fill_color(*self.LIGHT_ACCENT)
        self.set_draw_color(*self.ACCENT)
        w = self.w - self.l_margin - self.r_margin
        x = self.get_x()
        y_start = self.get_y()
        self.set_line_width(0.4)
        # estimate height
        self.set_font("Helvetica", "", 10)
        n_lines = len(self.multi_cell(w - 10, 5.5, text, dry_run=True, output="LINES"))
        h = 8 + n_lines * 5.5 + 6
        if y_start + h > self.h - 25:
            self.add_page()
            y_start = self.get_y()
        self.rect(x, y_start, w, h, "DF")
        self.set_xy(x + 4, y_start + 2)
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(*self.ACCENT)
        self.cell(0, 6, title, new_x="LMARGIN", new_y="NEXT")
        self.set_x(x + 4)
        self.set_font("Helvetica", "", 10)
        self.set_text_color(*self.DARK)
        self.multi_cell(w - 10, 5.5, text)
        self.set_y(y_start + h + 4)

    def table_row(self, cells, widths, bold=False):
        style = "B" if bold else ""
        self.set_font("Helvetica", style, 9)
        h = 7
        x_start = self.get_x()
        if bold:
            self.set_fill_color(*self.ACCENT)
            self.set_text_color(*self.WHITE)
        else:
            self.set_fill_color(*self.WHITE)
            self.set_text_color(*self.DARK)
        for i, (cell, w) in enumerate(zip(cells, widths)):
            self.cell(w, h, cell, border=1, fill=True)
        self.ln(h)
        self.set_text_color(*self.DARK)


def build_report():
    pdf = Report()
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.set_left_margin(15)
    pdf.set_right_margin(15)

    # ── COVER PAGE ──
    pdf.add_page()
    pdf.ln(50)
    pdf.set_font("Helvetica", "B", 28)
    pdf.set_text_color(*Report.ACCENT)
    pdf.cell(0, 15, "Technical Documentation Assistant", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)
    pdf.set_font("Helvetica", "", 16)
    pdf.set_text_color(*Report.GRAY)
    pdf.cell(0, 10, "Implementation Report", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(8)
    pdf.set_draw_color(*Report.ACCENT)
    pdf.set_line_width(0.8)
    mid = pdf.w / 2
    pdf.line(mid - 40, pdf.get_y(), mid + 40, pdf.get_y())
    pdf.ln(12)
    pdf.set_font("Helvetica", "", 12)
    pdf.set_text_color(*Report.DARK)
    pdf.cell(0, 8, "RAG-Based Q&A System for Technical Standards", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 8, "Built with Python, OpenAI, ChromaDB, LangChain, MCP, Streamlit", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(30)
    pdf.set_font("Helvetica", "I", 10)
    pdf.set_text_color(*Report.GRAY)
    pdf.cell(0, 8, "Portfolio Project for Software Engineering Interview Preparation", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 8, "April 2026", align="C", new_x="LMARGIN", new_y="NEXT")

    # ── TABLE OF CONTENTS ──
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 20)
    pdf.set_text_color(*Report.ACCENT)
    pdf.cell(0, 12, "Table of Contents", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(6)
    toc = [
        ("1", "Project Overview & Architecture"),
        ("2", "Configuration Management"),
        ("3", "PDF Ingestion Pipeline"),
        ("4", "Header/Footer Cleaning"),
        ("5", "Text Chunking Strategy"),
        ("6", "Embedding & Vector Storage"),
        ("7", "Answer Generation (RAG Core)"),
        ("8", "MCP Server (Claude Desktop Integration)"),
        ("9", "Streamlit Web Interface"),
        ("10", "Testing Strategy"),
        ("11", "Evaluation Framework (RAGAS)"),
        ("12", "Containerization (Docker)"),
        ("13", "CI/CD Pipeline"),
        ("14", "Technologies Summary"),
    ]
    for num, title in toc:
        pdf.set_font("Helvetica", "", 11)
        pdf.set_text_color(*Report.DARK)
        pdf.cell(12, 7, num + ".")
        pdf.cell(0, 7, title, new_x="LMARGIN", new_y="NEXT")

    # ── 1. PROJECT OVERVIEW ──
    pdf.add_page()
    pdf.chapter_title("1", "Project Overview & Architecture")

    pdf.body_text(
        "The Technical Documentation Assistant is a production-structured RAG (Retrieval-Augmented Generation) "
        "system that enables natural language Q&A over technical PDF documents. Users upload PDFs, the system "
        "ingests, chunks, and embeds them into a vector database, then answers questions with cited, grounded "
        "responses."
    )

    pdf.section_title("Core Pipeline Flow")
    pdf.code_block(
        "PDF Document\n"
        "    |  extract_text_from_pdf()\n"
        "    v\n"
        "DocumentContent (structured pages)\n"
        "    |  clean_document()\n"
        "    v\n"
        "Cleaned Document (headers/footers removed)\n"
        "    |  chunk_document()\n"
        "    v\n"
        "List[Chunk] (overlapping text segments)\n"
        "    |  embed_chunks()\n"
        "    v\n"
        "ChromaDB Vector Store (persistent vectors)\n"
        "    |  query_collection() + generate_answer()\n"
        "    v\n"
        "Cited Answer with Sources"
    )

    pdf.section_title("Three Access Interfaces")
    pdf.bullet("Streamlit Web UI - document upload, real-time Q&A with source display")
    pdf.bullet("MCP Server - Claude Desktop integration via stdio transport (3 tools: search, ask, status)")
    pdf.bullet("Direct Pipeline - scripts for testing and batch processing")

    pdf.section_title("Key Architectural Decisions")
    pdf.bullet("ChromaDB for vector storage: local persistence, zero infra, easily swappable")
    pdf.bullet("Recursive character splitting: preserves semantic boundaries over naive fixed-size cuts")
    pdf.bullet("Content-based header/footer detection: frequency analysis, works on any PDF layout")
    pdf.bullet("Temperature=0 for LLM calls: RAG answers must be deterministic and factual")
    pdf.bullet("Pydantic models at all boundaries: runtime type validation prevents silent failures")
    pdf.bullet("Dual LLM support (OpenAI + Anthropic): abstraction layer makes providers swappable")

    # ── 2. CONFIGURATION ──
    pdf.add_page()
    pdf.chapter_title("2", "Configuration Management")

    pdf.body_text(
        "All settings are centralized in src/config.py using Pydantic Settings. This provides type-validated "
        "configuration with a clear priority hierarchy: environment variables > .env file > defaults. "
        "No hardcoded secrets or magic strings anywhere in the codebase."
    )

    pdf.subsection_title("Configuration Fields")
    pdf.code_block(
        "class Settings(BaseSettings):\n"
        "    # Paths (OS-agnostic via pathlib.Path)\n"
        "    project_root, data_raw_dir, vectorstore_dir\n"
        "\n"
        "    # LLM Configuration\n"
        "    openai_api_key: str        # From .env\n"
        "    anthropic_api_key: str     # From .env\n"
        "    llm_provider: str          # 'openai' or 'anthropic'\n"
        "    llm_model: str             # default: 'gpt-4o-mini'\n"
        "\n"
        "    # Embedding\n"
        "    embedding_model: str       # 'text-embedding-3-small'\n"
        "\n"
        "    # Chunking\n"
        "    chunk_size: int = 1000     # Target chars per chunk\n"
        "    chunk_overlap: int = 200   # Sliding window overlap\n"
        "\n"
        "    # Retrieval\n"
        "    retrieval_top_k: int = 5   # Results per query"
    )

    pdf.highlight_box(
        "Design Pattern: Singleton",
        "A single global 'settings' instance is created on module import and shared across the entire "
        "application. This ensures consistent configuration without passing config objects through every "
        "function call."
    )

    # ── 3. PDF INGESTION ──
    pdf.add_page()
    pdf.chapter_title("3", "PDF Ingestion Pipeline")

    pdf.body_text(
        "The ingestion module (src/ingestion/) extracts text from PDF documents using PyMuPDF (fitz), "
        "a C-based library that is 10-50x faster than pure-Python alternatives like PyPDF2."
    )

    pdf.section_title("Data Models (src/ingestion/models.py)")
    pdf.bullet("PageContent: Represents one page (page_number + text). Page numbers enable citations.")
    pdf.bullet("DocumentContent: Full document with metadata (filename, filepath, total_pages, pages list). "
               "Provides full_text property for concatenation and get_page_text() for safe page access.")

    pdf.section_title("Extraction Process (src/ingestion/pdf_reader.py)")
    pdf.bullet("Opens PDF with context manager (RAII pattern - resource cleanup guaranteed)")
    pdf.bullet("Iterates pages with 1-indexed page numbers for human-readable citations")
    pdf.bullet("Applies _clean_page_text() to each page: collapses excessive blank lines, fixes "
               "hyphenation (auto-\\nmatic -> automatic), strips whitespace")
    pdf.bullet("Skips empty pages to avoid wasting downstream processing on noise")
    pdf.bullet("Error handling: FileNotFoundError for missing files, ValueError for non-PDFs")

    pdf.section_title("Batch Processing")
    pdf.body_text(
        "extract_all_pdfs(directory) processes every PDF in a directory with per-file error handling. "
        "A single corrupt PDF won't crash the entire batch - it logs the error and continues."
    )

    # ── 4. HEADER/FOOTER CLEANING ──
    pdf.add_page()
    pdf.chapter_title("4", "Header/Footer Cleaning")

    pdf.body_text(
        "Technical PDFs like MISRA standards have repeated headers and footers on every page. "
        "These pollute retrieved context and waste tokens. The cleaner module (src/chunking/cleaner.py) "
        "uses a content-based frequency analysis approach that works on any PDF layout."
    )

    pdf.section_title("Detection Algorithm")
    pdf.bullet("1. Scan all pages and count unique lines per page")
    pdf.bullet("2. For each unique line, count how many pages it appears on")
    pdf.bullet("3. If a line appears on >= threshold% of pages AND is < 200 chars, flag it")
    pdf.bullet("4. Remove all flagged lines from all pages")

    pdf.subsection_title("Why Content-Based Over Coordinate-Based?")
    pdf.bullet("Works on any PDF layout (header position varies between documents)")
    pdf.bullet("Self-adapting - learns from the document itself, no manual configuration")
    pdf.bullet("Handles multi-line headers naturally")
    pdf.bullet("200-char limit ensures long content lines are never mistakenly removed")

    pdf.highlight_box(
        "Threshold Parameter (default: 0.4 = 40%)",
        "Controls aggressiveness of detection. Lower threshold = more aggressive removal. "
        "Higher = more conservative. A line appearing on 40%+ of pages is almost certainly "
        "a header/footer, not meaningful content."
    )

    pdf.section_title("Immutability Pattern")
    pdf.body_text(
        "clean_document() returns a NEW DocumentContent object - it never mutates the input. "
        "This functional approach makes the pipeline easier to reason about and debug, "
        "since each step produces a fresh output without side effects."
    )

    # ── 5. CHUNKING ──
    pdf.add_page()
    pdf.chapter_title("5", "Text Chunking Strategy")

    pdf.body_text(
        "Documents can be hundreds of pages, but LLMs have limited context windows. "
        "The chunker (src/chunking/chunker.py) splits documents into overlapping segments "
        "that preserve semantic coherence using LangChain's RecursiveCharacterTextSplitter."
    )

    pdf.section_title("Recursive Character Splitting")
    pdf.body_text("The splitter tries separators in order of preference:")
    pdf.code_block(
        'separators = ["\\n\\n", "\\n", ". ", " ", ""]\n'
        "\n"
        "1. Paragraph breaks (\\n\\n)  - most natural split point\n"
        "2. Single newlines (\\n)     - good fallback\n"
        "3. Sentence boundaries (. ) - preserve sentences\n"
        "4. Word boundaries ( )      - last resort\n"
        "5. Characters ()            - only for single words > chunk_size"
    )

    pdf.highlight_box(
        "Why Not Fixed-Size Splitting?",
        "Fixed-size cuts mid-sentence, destroying context. Example: 'Rule 8.4 states that a "
        "compatible decla-' (cut mid-word). Recursive splitting prefers natural boundaries, "
        "producing chunks that are semantically coherent."
    )

    pdf.section_title("Configuration")
    pdf.code_block(
        "chunk_size: int = 1000     # Target characters per chunk\n"
        "chunk_overlap: int = 200   # Overlap between consecutive chunks\n"
        "min_chunk_size: int = 100  # Discard chunks smaller than this"
    )

    pdf.body_text(
        "Overlap (200 chars) ensures that if context spans a chunk boundary, both adjacent chunks "
        "contain the relevant passage. min_chunk_size filters out noise like stray headers or "
        "lone page numbers that survived cleaning."
    )

    pdf.section_title("Metadata Tracking")
    pdf.body_text(
        "Each Chunk carries full provenance: source_filename, page_number, chunk_index, "
        "start_char, end_char. This enables precise citations and debugging. "
        "The citation property (e.g., 'MISRA-2020.pdf, page 14') is computed automatically."
    )

    # ── 6. EMBEDDING ──
    pdf.add_page()
    pdf.chapter_title("6", "Embedding & Vector Storage")

    pdf.body_text(
        "The embedding module (src/embedding/vector_store.py) converts text chunks into "
        "high-dimensional vectors and stores them in ChromaDB for similarity search."
    )

    pdf.section_title("Embedding Model: OpenAI text-embedding-3-small")
    pdf.bullet("Converts text into 1536-dimensional vectors")
    pdf.bullet("Same model used for both chunks and queries (vector space consistency)")
    pdf.bullet("Cost: ~$0.02 per 1M tokens (very cheap)")

    pdf.section_title("Vector Database: ChromaDB")
    pdf.bullet("Local persistent storage (vectorstore/ directory) - survives process restarts")
    pdf.bullet("Cosine similarity for nearest-neighbor search")
    pdf.bullet("Each chunk stored with metadata: source_filename, page_number, citation, chunk_index")
    pdf.bullet("Idempotent upsert: re-running embed_chunks doesn't duplicate vectors")

    pdf.section_title("Key Functions")
    pdf.bullet("get_chroma_client() - creates or connects to persistent ChromaDB instance")
    pdf.bullet("get_or_create_collection(client) - gets/creates 'tech_docs' collection with OpenAI embeddings")
    pdf.bullet("embed_chunks(chunks) - batch upsert in groups of 100 for efficiency")
    pdf.bullet("query_collection(query, n_results=5) - returns top-k most similar chunks by cosine similarity")

    pdf.highlight_box(
        "Abstraction Layer Design",
        "All vector DB logic is isolated in one module. To swap ChromaDB for Pinecone, Weaviate, "
        "or pgvector, you only need to reimplement these 4 functions. The rest of the pipeline "
        "doesn't know or care which vector database is being used."
    )

    # ── 7. ANSWER GENERATION ──
    pdf.add_page()
    pdf.chapter_title("7", "Answer Generation (RAG Core)")

    pdf.body_text(
        "The generation module (src/generation/) is where retrieval meets generation. "
        "It takes a user question, retrieves relevant chunks, and uses an LLM to synthesize "
        "a cited answer grounded exclusively in the retrieved context."
    )

    pdf.section_title("The RAG Pipeline (generate_answer)")
    pdf.code_block(
        "def generate_answer(question, n_results=5):\n"
        "    # 1. RETRIEVE: find relevant chunks\n"
        "    raw_chunks = query_collection(question, n_results)\n"
        "\n"
        "    # 2. BUILD PROMPT: format context + question\n"
        "    user_prompt = build_user_prompt(question, raw_chunks)\n"
        "\n"
        "    # 3. CALL LLM: generate answer\n"
        "    answer_text = _call_llm(user_prompt)\n"
        "\n"
        "    # 4. PACKAGE: return Answer object with metadata\n"
        "    return Answer(question, answer_text, sources, has_answer)"
    )

    pdf.section_title("Prompt Engineering")
    pdf.body_text("The system prompt enforces strict constraints to prevent hallucination:")
    pdf.bullet("Rule 1: Use ONLY information from provided context (no general knowledge)")
    pdf.bullet("Rule 2: After each claim, cite source: [Source: filename, page X]")
    pdf.bullet("Rule 3: If context insufficient, say 'documentation does not contain information'")
    pdf.bullet("Rule 4: Be concise and technical")
    pdf.bullet("Temperature=0: deterministic responses - same question always gives same answer")

    pdf.section_title("Multi-Provider LLM Support")
    pdf.body_text(
        "The generator supports both OpenAI and Anthropic through an abstraction layer. "
        "The _call_llm() function dispatches to _call_openai() or _call_anthropic() based on "
        "settings.llm_provider. Switching providers requires only a config change, no code changes."
    )

    pdf.section_title("Hallucination Prevention (5 Layers)")
    pdf.bullet("1. System prompt explicitly forbids using knowledge outside context")
    pdf.bullet("2. Only retrieved chunks are fed to the LLM (not the full document)")
    pdf.bullet("3. Citation requirement forces verifiable claims")
    pdf.bullet("4. has_answer flag: honest 'I don't know' beats confident hallucination")
    pdf.bullet("5. Evaluation metrics (faithfulness) detect hallucinations quantitatively")

    # ── 8. MCP SERVER ──
    pdf.add_page()
    pdf.chapter_title("8", "MCP Server (Claude Desktop Integration)")

    pdf.body_text(
        "The MCP server (src/mcp_server/server.py) exposes the RAG pipeline as standardized tools "
        "that any MCP-compatible client can call. It uses FastMCP with stdio transport - Claude Desktop "
        "spawns this script as a child process and communicates over stdin/stdout."
    )

    pdf.section_title("Three Exposed Tools")

    pdf.subsection_title("1. docs_status - Check Index Status")
    pdf.body_text("Returns whether documents are loaded and how many chunks are indexed. "
                  "Agents should call this first to verify readiness before querying.")

    pdf.subsection_title("2. docs_search - Raw Retrieval (No LLM)")
    pdf.body_text(
        "Returns raw passages with citations and relevance scores. No LLM synthesis. "
        "Use case: agents that want to inspect source material directly or feed passages "
        "into their own reasoning chain."
    )

    pdf.subsection_title("3. docs_ask - Full RAG Q&A")
    pdf.body_text(
        "Runs the complete RAG pipeline and returns a cited answer. "
        "Use case: agents/humans wanting a finished, grounded answer."
    )

    pdf.section_title("Input Validation")
    pdf.body_text(
        "All tool inputs are validated with Pydantic models before execution. "
        "SearchInput requires query (3-500 chars) and n_results (1-10). "
        "AskInput requires question (10-500 chars). Invalid inputs are rejected immediately "
        "with clear error messages."
    )

    pdf.section_title("Lifespan Management")
    pdf.body_text(
        "ChromaDB connection is initialized once at server startup via an async context manager "
        "and reused across all tool calls. This avoids reconnecting on every request."
    )

    # ── 9. STREAMLIT UI ──
    pdf.add_page()
    pdf.chapter_title("9", "Streamlit Web Interface")

    pdf.body_text(
        "The Streamlit app (src/ui/app.py) provides a web-based interface for document upload "
        "and interactive Q&A. It is intentionally thin - all logic is delegated to pipeline modules."
    )

    pdf.section_title("Layout")
    pdf.bullet("Left column: document status indicator, file upload, processing results")
    pdf.bullet("Right column: chat interface with message history, answer display with expandable sources")

    pdf.section_title("Key Features")
    pdf.bullet("Real-time document status: shows indexed chunk count, warns if no documents loaded")
    pdf.bullet("PDF upload: saves BytesIO to temp file (PyMuPDF needs real path), runs full pipeline")
    pdf.bullet("Processing feedback: displays page count, chunk count, avg chunk size after ingestion")
    pdf.bullet("Chat interface: maintains message history via st.session_state across reruns")
    pdf.bullet("Source display: expandable panel showing citations, relevance scores, and chunk previews")

    # ── 10. TESTING ──
    pdf.add_page()
    pdf.chapter_title("10", "Testing Strategy")

    pdf.body_text(
        "The project has 33 unit tests across 4 test modules, all designed to be deterministic, "
        "fast, and cost-free. No real API calls are made during testing."
    )

    pdf.section_title("Testing Philosophy")
    pdf.bullet("100% mocking of external services (OpenAI API, ChromaDB) - no network calls")
    pdf.bullet("Tests verify interface contracts: what we call, what we expect back")
    pdf.bullet("Each pipeline module is tested in isolation")
    pdf.bullet("Fixtures provide realistic sample data without touching real PDFs or APIs")

    pdf.section_title("Test Modules")

    pdf.subsection_title("test_pdf_reader.py - Data Models & I/O")
    pdf.bullet("PageContent construction and type validation")
    pdf.bullet("DocumentContent: full text concatenation, page lookup, metadata")
    pdf.bullet("_clean_page_text: whitespace handling, hyphenation fixing, edge cases")
    pdf.bullet("extract_text_from_pdf: file not found, non-PDF rejection")

    pdf.subsection_title("test_chunker.py - Cleaning & Splitting")
    pdf.bullet("detect_repeating_lines: footer detection, threshold sensitivity, empty documents")
    pdf.bullet("clean_document: footer removal, content preservation, immutability verification")
    pdf.bullet("chunk_document: chunk production, metadata correctness, size limits, citation format")
    pdf.bullet("get_chunk_stats: empty list handling, correct statistics calculation")

    pdf.subsection_title("test_vector_store.py - Vector Operations")
    pdf.bullet("embed_chunks: correct upsert calls, metadata inclusion, empty list handling")
    pdf.bullet("query_collection: result structure, distance calculation, empty collection")
    pdf.bullet("Uses @patch to replace ChromaDB client and collection with MagicMock")

    pdf.subsection_title("test_generator.py - Answer Generation")
    pdf.bullet("build_user_prompt: question inclusion, chunk text, citations, ordering")
    pdf.bullet("generate_answer: return type, has_answer detection, source mapping, empty retrieval")
    pdf.bullet("Answer model: source deduplication in formatted output")

    pdf.section_title("Running Tests")
    pdf.code_block(
        "# All tests\n"
        "pytest tests/ -v\n"
        "\n"
        "# With coverage\n"
        "pytest tests/ -v --cov=src --cov-report=term-missing\n"
        "\n"
        "# Result: 33 passed in ~13s"
    )

    # ── 11. EVALUATION ──
    pdf.add_page()
    pdf.chapter_title("11", "Evaluation Framework (RAGAS)")

    pdf.highlight_box(
        "Why This Matters",
        "Any engineer can build a RAG system. Few can evaluate one rigorously. "
        "Showing up with metrics like 'faithfulness: 0.73, answer relevancy: 0.86' "
        "signals production engineering mindset, not just prototype thinking."
    )

    pdf.section_title("Three RAGAS Metrics")

    pdf.subsection_title("Faithfulness (0.0 - 1.0)")
    pdf.body_text(
        "Does the answer stay within retrieved context? Detects hallucinations - claims the LLM "
        "made that are not supported by the provided chunks. Computed by an LLM judge that checks "
        "each statement in the answer against the context."
    )

    pdf.subsection_title("Answer Relevancy (0.0 - 1.0)")
    pdf.body_text(
        "Does the answer address the actual question? Detects off-topic responses. "
        "The LLM judge generates questions from the answer and measures how similar they are "
        "to the original question using embeddings."
    )

    pdf.subsection_title("Context Precision (0.0 - 1.0)")
    pdf.body_text(
        "Are the retrieved chunks actually relevant to the question? Measures retrieval quality. "
        "Low scores indicate the vector search is returning noisy, irrelevant chunks."
    )

    pdf.section_title("Evaluation Dataset (evaluation/dataset.py)")
    pdf.body_text(
        "8 hand-crafted question/ground-truth pairs across categories: definition, classification, "
        "and process questions about MISRA compliance. Hand-crafting (vs auto-generation) demonstrates "
        "domain understanding and evaluation rigor."
    )

    pdf.section_title("Evaluation Pipeline (evaluation/evaluate.py)")
    pdf.bullet("Step 1: collect_rag_outputs() - run each question through full RAG pipeline, "
               "store question + answer + contexts + ground_truth")
    pdf.bullet("Step 2: run_ragas_evaluation() - convert to HuggingFace Dataset format, "
               "score with RAGAS using LLM-as-judge (gpt-4o-mini)")
    pdf.bullet("Step 3: save_results() - write timestamped JSON to evaluation/results/")
    pdf.bullet("Step 4: print_report() - display scores with threshold warnings")

    pdf.section_title("Implementation Challenges & Solutions")

    pdf.body_text(
        "The RAGAS evaluation pipeline required significant debugging due to breaking API changes "
        "across RAGAS versions. This section documents the issues encountered and solutions applied."
    )

    pdf.subsection_title("Challenge 1: RAGAS Metric Import Changes (v0.4.x)")
    pdf.body_text(
        "RAGAS v0.4.x introduced a parallel metrics API (ragas.metrics.collections) alongside the "
        "legacy API (ragas.metrics.base). The new 'collections' classes (Faithfulness, AnswerRelevancy, "
        "ContextPrecision) inherit from a different base class (BaseMetric) than what evaluate() expects "
        "(Metric from ragas.metrics.base)."
    )
    pdf.body_text(
        "Solution: Use the legacy singleton metrics from the private modules "
        "(ragas.metrics._faithfulness.faithfulness, etc.) which pass the isinstance(m, Metric) check "
        "that evaluate() performs. These are pre-instantiated instances, not classes to call."
    )

    pdf.subsection_title("Challenge 2: Environment Variable Propagation")
    pdf.body_text(
        "RAGAS internally creates its own OpenAI() client for LLM-as-judge evaluation. "
        "The project uses pydantic-settings to read .env files, but pydantic-settings only "
        "populates its own Settings object - it does not set os.environ. RAGAS's OpenAI() "
        "client checks os.environ['OPENAI_API_KEY'] and failed with 'api_key must be set'."
    )
    pdf.body_text(
        "Solution: Added load_dotenv() from python-dotenv at the top of evaluate.py to "
        "populate os.environ before any RAGAS code runs."
    )

    pdf.subsection_title("Challenge 3: Result Format Change")
    pdf.body_text(
        "RAGAS v0.4.x changed the EvaluationResult format. Previously result['faithfulness'] "
        "returned a single float. Now it returns a list of per-question scores. The save_results() "
        "and print_report() functions called float(scores['faithfulness']), which failed with "
        "'float() argument must be a string or real number, not list'."
    )
    pdf.body_text(
        "Solution: Added a _mean_score() helper that checks the type and computes the average "
        "if the value is a list, or returns the float directly if it's already scalar."
    )

    pdf.subsection_title("Challenge 4: Explicit LLM/Embeddings Requirement")
    pdf.body_text(
        "RAGAS v0.4.x deprecated auto-detection of the LLM client. The evaluate() function "
        "accepts optional llm and embeddings parameters. Without them, RAGAS tries to create "
        "OpenAI() internally. With them, you have full control over which model judges your system."
    )
    pdf.body_text(
        "Solution: Explicitly created judge LLM and embeddings using LangchainLLMWrapper "
        "and LangchainEmbeddingsWrapper, passing them to evaluate(). This also allows using "
        "a different model for judging (gpt-4o-mini) than for generation."
    )

    pdf.section_title("Latest Evaluation Results")
    pdf.code_block(
        "RAG EVALUATION REPORT\n"
        "Questions evaluated:  8\n"
        "Questions answered:   7\n"
        "\n"
        "SCORES (0.0 - 1.0, higher is better):\n"
        "  Faithfulness:       0.726\n"
        "  Answer Relevancy:   0.864\n"
        "  Context Precision:  0.312\n"
        "\n"
        "ISSUES DETECTED:\n"
        "  Low context precision - retrieval returning noisy chunks"
    )

    pdf.body_text(
        "The low context precision (0.312) indicates the retrieval step is returning chunks that "
        "aren't highly relevant to the questions. Potential improvements: tune chunk size/overlap, "
        "use a better embedding model, add metadata filtering, or implement hybrid search "
        "(vector similarity + keyword BM25)."
    )

    # ── 12. DOCKER ──
    pdf.add_page()
    pdf.chapter_title("12", "Containerization (Docker)")

    pdf.section_title("Two-Stage Build Strategy")
    pdf.body_text(
        "The Dockerfile uses a two-stage build to minimize the final image size. Stage 1 (builder) "
        "installs all dependencies using uv. Stage 2 (runtime) copies only the installed packages "
        "and application source code."
    )

    pdf.code_block(
        "Stage 1 (builder):\n"
        "  FROM python:3.11-slim\n"
        "  Install uv, copy pyproject.toml\n"
        "  uv pip install --system --no-cache '.[eval]'\n"
        "\n"
        "Stage 2 (runtime):\n"
        "  FROM python:3.11-slim\n"
        "  Copy site-packages from builder\n"
        "  Copy src/, evaluation/, pyproject.toml\n"
        "  pip install --no-deps -e .  (register package)\n"
        "  Configure Streamlit, expose port 8501\n"
        "  HEALTHCHECK every 30s"
    )

    pdf.section_title("docker-compose.yml")
    pdf.bullet("Port mapping: 8501:8501 (Streamlit)")
    pdf.bullet("Volume mounts: ./data:/app/data and ./vectorstore:/app/vectorstore for persistence")
    pdf.bullet("Environment: API keys from .env, configurable LLM provider and model")
    pdf.bullet("Restart policy: unless-stopped (auto-recover from crashes)")

    pdf.section_title("Key Docker Decisions")
    pdf.bullet("Slim base image (python:3.11-slim): ~200MB vs ~1GB for full image")
    pdf.bullet("uv for dependency installation: 10-100x faster than pip in build stage")
    pdf.bullet("Volume mounts for data: PDFs and vectors persist across container restarts")
    pdf.bullet("Health check: Docker restarts container if Streamlit stops responding")

    # ── 13. CI/CD ──
    pdf.add_page()
    pdf.chapter_title("13", "CI/CD Pipeline")

    pdf.body_text(
        "GitHub Actions runs on every push and PR to main, enforcing code quality before merge."
    )

    pdf.section_title("Three Automated Checks")

    pdf.subsection_title("1. Lint with Ruff")
    pdf.body_text("Checks for style issues, import ordering, naming violations, and common bugs. "
                  "Also verifies consistent formatting.")

    pdf.subsection_title("2. Type Check with MyPy")
    pdf.body_text("Static type checking catches type mismatches at build time rather than runtime.")

    pdf.subsection_title("3. Run Tests with pytest")
    pdf.body_text("All 33 tests must pass. Generates coverage report. No code reaches main "
                  "without passing all checks.")

    # ── 14. TECHNOLOGIES ──
    pdf.add_page()
    pdf.chapter_title("14", "Technologies Summary")

    widths = [50, 55, 75]
    pdf.table_row(["Technology", "Version", "Purpose"], widths, bold=True)
    rows = [
        ("PyMuPDF", "1.24+", "PDF text extraction (C-based, fast)"),
        ("LangChain", "0.3+", "Text splitter abstractions"),
        ("OpenAI API", "GPT-4o-mini", "LLM generation + embeddings"),
        ("Anthropic API", "Claude", "Alternative LLM provider"),
        ("ChromaDB", "0.5+", "Local vector database"),
        ("Pydantic", "2.0+", "Data validation + settings"),
        ("Streamlit", "1.38+", "Web UI framework"),
        ("FastMCP", "1.0+", "MCP server (Claude Desktop)"),
        ("RAGAS", "0.4+", "RAG evaluation metrics"),
        ("pytest", "8.0+", "Unit testing framework"),
        ("Ruff", "0.6+", "Linter + formatter"),
        ("MyPy", "1.11+", "Static type checking"),
        ("Docker", "24+", "Containerization"),
        ("GitHub Actions", "-", "CI/CD automation"),
        ("uv", "latest", "Fast dependency management"),
    ]
    
    for row in rows:
        pdf.table_row(list(row), widths)

    # ── OUTPUT ──
    output_path = "/Users/bobus/Projects/tech-doc-assistant/Tech_Doc_Assistant_Report.pdf"
    pdf.output(output_path)
    print(f"Report generated: {output_path}")
    print(f"Pages: {pdf.page_no()}")


if __name__ == "__main__":
    build_report()
