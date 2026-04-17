"""
Streamlit web interface for the Technical Documentation Assistant.

WHY STREAMLIT:
Streamlit turns Python scripts into web apps with minimal boilerplate.
No HTML, no JavaScript, no REST API needed.
For a portfolio project, it lets you focus on the AI logic rather than
frontend engineering. In a real BMW project you'd have a proper frontend —
but for demonstrating RAG capabilities, Streamlit is the industry standard.

RUN WITH:
    uv run streamlit run src/ui/app.py

ARCHITECTURE:
This file is intentionally thin — it only handles:
  1. UI rendering and state management
  2. Calling the pipeline functions you already built

All heavy logic stays in src/ingestion/, src/chunking/, src/embedding/,
src/generation/. The UI is just a face on top of the pipeline.

ANALOGY:
This is like your diagnostic display in an automotive tool —
it shows state and accepts commands, but the real work happens
in the ECU firmware underneath.
"""

import logging
import tempfile
from pathlib import Path

import streamlit as st

from src.chunking.cleaner import clean_document
from src.chunking.chunker import chunk_document, get_chunk_stats
from src.embedding.vector_store import embed_chunks, get_chroma_client, get_or_create_collection
from src.generation.generator import generate_answer
from src.ingestion.pdf_reader import extract_text_from_pdf

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ── Page configuration ──
# Must be the first Streamlit call in the script
st.set_page_config(
    page_title="Technical Documentation Assistant",
    page_icon="📚",
    layout="wide",
)


# ── Helper functions ──

def get_document_count() -> int:
    """
    Check how many vectors are stored in ChromaDB.

    This is our "is the system ready?" signal.
    0 vectors = nothing ingested = user can't query yet.
    """
    try:
        client = get_chroma_client()
        collection = get_or_create_collection(client)
        return collection.count()
    except Exception:
        return 0


def run_ingestion_pipeline(pdf_path: str) -> dict:
    """
    Run the full ingestion pipeline on a PDF and return stats.

    Returns a dict with processing statistics for display in the UI.
    Separating this from the UI code makes it easier to test and reuse.
    """
    doc = extract_text_from_pdf(pdf_path)
    cleaned = clean_document(doc)
    chunks = chunk_document(cleaned)
    stats = get_chunk_stats(chunks)
    embed_chunks(chunks)
    return {
        "filename": doc.filename,
        "pages": len(doc.pages),
        "chunks": stats.get("total_chunks", 0),
        "avg_chunk_size": stats.get("avg_chars", 0),
    }


# ── Session state initialization ──
# Streamlit reruns the entire script on every interaction.
# st.session_state persists values across reruns — like static variables
# in an embedded interrupt handler.
if "processing" not in st.session_state:
    st.session_state.processing = False

if "ingestion_result" not in st.session_state:
    st.session_state.ingestion_result = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # list of (question, answer) tuples


# ── Layout ──
st.title("📚 Technical Documentation Assistant")
st.caption("Upload technical PDFs and ask questions in plain English.")

# Two-column layout: sidebar for ingestion, main area for Q&A
col_sidebar, col_main = st.columns([1, 2])


# ══════════════════════════════════════════
# LEFT COLUMN: Document Management
# ══════════════════════════════════════════
with col_sidebar:
    st.subheader("📄 Documents")

    # ── System status indicator ──
    doc_count = get_document_count()

    if doc_count == 0:
        st.error("⚠️ No documents loaded. Upload a PDF to get started.")
    else:
        st.success(f"✅ Ready — {doc_count:,} chunks indexed")

    st.divider()

    # ── PDF upload ──
    st.markdown("**Upload a new document**")
    uploaded_file = st.file_uploader(
        label="Choose a PDF",
        type=["pdf"],
        help="Upload a technical manual, standard, or specification.",
        disabled=st.session_state.processing,
    )

    if uploaded_file is not None:
        st.info(f"Selected: `{uploaded_file.name}` ({uploaded_file.size / 1024:.1f} KB)")

        if st.button(
            "⚙️ Process Document",
            disabled=st.session_state.processing,
            use_container_width=True,
        ):
            st.session_state.processing = True
            st.session_state.ingestion_result = None

            # Save uploaded file to a temp location
            # Streamlit gives us a BytesIO object — we need a real file path
            # for PyMuPDF to open. tempfile handles cleanup automatically.
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=".pdf"
            ) as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name

            with st.spinner(f"Processing {uploaded_file.name}..."):
                try:
                    result = run_ingestion_pipeline(tmp_path)
                    st.session_state.ingestion_result = result
                    st.session_state.processing = False
                    st.rerun()  # Refresh to update the status indicator
                except Exception as e:
                    st.error(f"Processing failed: {e}")
                    st.session_state.processing = False

    # Show ingestion result if available
    if st.session_state.ingestion_result:
        r = st.session_state.ingestion_result
        st.success("✅ Document processed successfully!")
        st.markdown(f"""
        **Results:**
        - File: `{r['filename']}`
        - Pages extracted: `{r['pages']}`
        - Chunks created: `{r['chunks']}`
        - Avg chunk size: `{r['avg_chunk_size']} chars`
        """)


# ══════════════════════════════════════════
# RIGHT COLUMN: Q&A Interface
# ══════════════════════════════════════════
with col_main:
    st.subheader("💬 Ask a Question")

    # Show chat history — previous Q&A pairs in this session
    if st.session_state.chat_history:
        for past_question, past_answer in st.session_state.chat_history:
            with st.chat_message("user"):
                st.write(past_question)
            with st.chat_message("assistant"):
                st.write(past_answer.answer)
                if past_answer.has_answer:
                    with st.expander("📎 Sources"):
                        st.text(past_answer.formatted_sources)
                        # Show retrieved chunks with their distances
                        for source in past_answer.sources:
                            st.markdown(
                                f"**{source.citation}** "
                                f"*(relevance: {1 - source.distance:.2f})*"
                            )
                            st.caption(source.text[:300] + "...")

    # ── Question input ──
    # Disable if no documents are loaded — user can't query an empty system
    question = st.chat_input(
        placeholder="e.g. What is a deviation in MISRA compliance?",
        disabled=(doc_count == 0),
    )

    if question:
        # Show the user's question immediately
        with st.chat_message("user"):
            st.write(question)

        # Generate and display the answer
        with st.chat_message("assistant"):
            with st.spinner("Searching documentation..."):
                answer = generate_answer(question)

            st.write(answer.answer)

            if answer.has_answer:
                with st.expander("📎 Sources", expanded=True):
                    st.text(answer.formatted_sources)
                    for source in answer.sources:
                        st.markdown(
                            f"**{source.citation}** "
                            f"*(relevance: {1 - source.distance:.2f})*"
                        )
                        st.caption(source.text[:300] + "...")
            else:
                st.warning(
                    "The loaded documents don't contain information about this topic. "
                    "Try uploading more relevant documentation."
                )

        # Save to chat history for display on next rerun
        st.session_state.chat_history.append((question, answer))
