"""
Technical Documentation Assistant — Streamlit web UI.

A RAG chatbot for technical PDFs. Upload documents, ask questions, get cited answers.

LAYOUT: Real st.sidebar (documents, scope, config) + full-width main area (transcript).
st.chat_input is top-level, so Streamlit pins it to the viewport bottom.

SESSION STATE:
- turns: list[ChatTurn] — conversation history (question, answer, timing, scope)
- processing: bool — ingestion in progress (disables upload)
- ingestion_result: dict | None — last successful upload details
"""

import logging
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import streamlit as st

from src.chunking.chunker import chunk_document
from src.chunking.cleaner import clean_document
from src.embedding.vector_store import (
    embed_chunks,
    get_chroma_client,
    get_or_create_collection,
    list_indexed_documents,
    save_document_outline,
)
from src.generation.generator import generate_answer
from src.ingestion.pdf_reader import extract_text_from_pdf
from src.ui.components import (
    render_empty_state,
    render_sidebar,
    render_turn,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


@dataclass
class ChatTurn:
    """
    A single turn in the conversation.

    A mutable dataclass, not a NamedTuple — render_turn's thumbs up/down
    buttons mutate `feedback` in place on the instance stored in
    st.session_state.turns.
    """

    question: str
    answer: any  # Answer pydantic model
    duration_s: float
    scope: list[str] | None
    feedback: int | None = None


def init_session_state() -> None:
    """Initialize session state keys if not present."""
    if "turns" not in st.session_state:
        st.session_state.turns = []
    if "processing" not in st.session_state:
        st.session_state.processing = False
    if "ingestion_result" not in st.session_state:
        st.session_state.ingestion_result = None


@st.cache_data(ttl=60, show_spinner=False)
def get_index_snapshot() -> dict:
    """
    Get the current index state: document count and per-document chunk counts.

    Cached with 60s TTL and no spinner. Invalidated explicitly after ingestion
    (see run_ingestion_pipeline). The ttl is belt-and-braces since st.cache_data
    is process-global and another browser tab's upload wouldn't otherwise
    invalidate it.
    """
    collection = get_or_create_collection(get_chroma_client())
    chunk_count = collection.count()

    if chunk_count == 0:
        return {"chunk_count": 0, "documents": []}

    docs = list_indexed_documents()
    return {"chunk_count": chunk_count, "documents": docs}


def run_ingestion_pipeline(pdf_bytes: bytes, display_name: str) -> dict:
    """
    Ingest a PDF: extract, clean, chunk, embed, and store.

    Args:
        pdf_bytes: Raw PDF file content.
        display_name: User-facing filename (e.g., "user-upload.pdf").

    Returns:
        {"filename": str, "pages": int, "chunks": int, "avg_chunk_size": float}

    Raises:
        Any exception from the pipeline (will be caught and displayed as st.error).
    """
    tmp_path = None
    try:
        # Write to a temporary file — the pipeline expects a file path.
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(pdf_bytes)
            tmp_path = tmp.name

        # Run the pipeline.
        doc = extract_text_from_pdf(tmp_path, display_name=display_name)
        cleaned = clean_document(doc)
        chunks = chunk_document(cleaned)

        # Compute stats before embedding.
        total_chunks = len(chunks)
        avg_size = sum(len(c.text) for c in chunks) / total_chunks if total_chunks > 0 else 0

        # Embed and store.
        embed_chunks(chunks)

        # Persist the document's TOC (pymupdf outline) as a JSON sidecar so
        # "structural" mode (see src/retrieval/structure.py) can answer
        # table-of-contents questions for this document without a chunk
        # retrieval that would never reconstruct a whole TOC from fragments.
        save_document_outline(cleaned)

        # Invalidate the cache so the sidebar updates.
        get_index_snapshot.clear()

        return {
            "filename": display_name,
            "pages": len(doc.pages),
            "chunks": total_chunks,
            "avg_chunk_size": avg_size,
        }

    finally:
        # Clean up temp file — the comment in the old code was wrong about
        # tempfile.NamedTemporaryFile(delete=False) handling cleanup.
        if tmp_path:
            Path(tmp_path).unlink(missing_ok=True)


def main() -> None:
    """Streamlit app entry point."""
    st.set_page_config(
        page_title="Technical Documentation Assistant",
        page_icon="📚",
        layout="wide",
    )

    st.title("📚 Technical Documentation Assistant")
    st.caption("Upload technical PDFs and ask questions in plain English.")

    init_session_state()

    snapshot = get_index_snapshot()
    source_filenames = render_sidebar(snapshot)

    st.markdown("## Conversation")

    # Replay history.
    for turn in st.session_state.turns:
        render_turn(turn)

    # Empty state or example questions.
    if not st.session_state.turns:
        example_question = render_empty_state(snapshot)
        if example_question:
            st.session_state.pending_question = example_question
            st.rerun()

    # New question input (top-level, so Streamlit pins it to viewport bottom).
    question = st.chat_input(
        "Ask a question...",
        disabled=(snapshot["chunk_count"] == 0),
    )

    # st.chat_input can't be pre-filled programmatically, so an example-question
    # click (render_empty_state above) stashes its text in pending_question
    # instead and we treat it as this run's submitted question here.
    if not question and "pending_question" in st.session_state:
        question = st.session_state.pop("pending_question")

    if question:
        # Echo the user's question.
        st.chat_message("user").write(question)

        # Generate answer.
        start = time.time()
        try:
            with st.spinner("Searching documentation..."):
                answer = generate_answer(question, source_filenames=source_filenames)
            duration_s = time.time() - start

            # Append to history and re-run so the new turn renders via render_turn.
            turn = ChatTurn(
                question=question,
                answer=answer,
                duration_s=duration_s,
                scope=source_filenames,
            )
            st.session_state.turns.append(turn)
            st.rerun()

        except Exception as e:
            st.error(f"Failed to generate answer: {e}")
            logger.exception("Answer generation failed")


if __name__ == "__main__":
    main()
