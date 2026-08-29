"""
Rendering components for the Streamlit UI.

High-level functions that render UI sections (sidebar, turns, sources, etc).
Each function is responsible for one visual section and handles its own state.
"""

import streamlit as st

from src.config import settings
from src.generation.models import Answer
from src.ui.formatting import (
    MODE_COLORS,
    MODE_LABELS,
    format_scores,
    style_inference_markers,
    unique_citations,
)


def render_sidebar(snapshot: dict) -> list[str] | None:
    """
    Render the sidebar: document library, upload, scope picker, pipeline config, clear chat.

    Args:
        snapshot: {"chunk_count": int, "documents": list[{filename, chunk_count}]}

    Returns:
        The selected source_filenames for scope filtering, or None if all/empty selected.
    """
    with st.sidebar:
        st.subheader("📚 Library")

        if snapshot["chunk_count"] == 0:
            st.error("❌ No documents indexed")
            indexed_documents = []
        else:
            st.success(f"✅ Ready — {snapshot['chunk_count']:,} chunks indexed")
            indexed_documents = snapshot.get("documents", [])
            for doc in indexed_documents:
                st.caption(f"📄 {doc['filename']} — {doc['chunk_count']:,} chunks")

        st.divider()

        with st.expander("📥 Add a document", expanded=False):
            uploaded_file = st.file_uploader(
                "Choose a PDF",
                type=["pdf"],
                disabled=st.session_state.processing,
                help="Max 200MB per file",
            )
            if uploaded_file:
                col1, col2 = st.columns(2)
                with col1:
                    st.caption(f"{uploaded_file.name} • {uploaded_file.size // 1024} KB")
                with col2:
                    if st.button(
                        "⚙️ Process",
                        key="process_button",
                        use_container_width=True,
                        disabled=st.session_state.processing,
                    ):
                        st.session_state.processing = True
                        st.session_state.ingestion_result = None
                        st.rerun()

                if st.session_state.processing and uploaded_file:
                    with st.spinner("Extracting, cleaning, chunking, embedding..."):
                        from src.ui.app import run_ingestion_pipeline

                        try:
                            result = run_ingestion_pipeline(
                                uploaded_file.getvalue(), display_name=uploaded_file.name
                            )
                            st.session_state.ingestion_result = result
                            msg = (
                                f"✅ Indexed {result['filename']}: "
                                f"{result['pages']} pages, {result['chunks']} chunks"
                            )
                            st.success(msg)
                        except Exception as e:
                            st.error(f"❌ Failed: {e}")
                        finally:
                            st.session_state.processing = False
                            st.rerun()

        st.divider()

        st.markdown("**Scope**")
        if indexed_documents:
            filenames = [doc["filename"] for doc in indexed_documents]
            selected = st.pills(
                "Search within:",
                options=filenames,
                default=filenames,
                selection_mode="multi",
                key="scope_pills",
            )
            source_filenames = selected if selected and len(selected) < len(filenames) else None
        else:
            st.caption("(no documents indexed yet)")
            source_filenames = None

        st.divider()

        with st.popover("⚙️ Pipeline config"):
            st.caption("Read-only settings:")
            st.code(
                f"""LLM Provider: {settings.llm_provider}
Model: {settings.llm_model}
Retrieval top-k: {settings.retrieval_top_k}
Candidate pool: {settings.retrieval_candidate_k}
Reranking: {"✅" if settings.reranking_enabled else "❌"}
Query planning: {"✅" if settings.query_planning_enabled else "❌"}""",
                language="text",
            )

        if st.button("🗑️ Clear conversation", use_container_width=True):
            st.session_state.turns = []
            st.rerun()

    return source_filenames


def render_turn(turn) -> None:
    """
    Render a single conversation turn: user question + assistant answer + evidence.

    Args:
        turn: ChatTurn(question, answer: Answer, duration_s, scope)
    """
    st.chat_message("user").write(turn.question)

    with st.chat_message("assistant"):
        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            st.badge(
                MODE_LABELS.get(turn.answer.mode, turn.answer.mode),
                color=MODE_COLORS.get(turn.answer.mode, "gray"),
            )
        with col2:
            st.badge(f"{len(turn.answer.sources)} sources")
        with col3:
            st.caption(f"{turn.duration_s:.1f}s")

        st.markdown(style_inference_markers(turn.answer.answer))
        render_sources(turn.answer)

        with st.container():
            col1, col2 = st.columns(2)
            with col1:
                if st.button("👍", key=f"thumb_up_{id(turn)}", use_container_width=True):
                    turn.feedback = 1
            with col2:
                if st.button("👎", key=f"thumb_down_{id(turn)}", use_container_width=True):
                    turn.feedback = 0
            if turn.answer.sources:
                with st.popover("📋 Copy"):
                    st.code(turn.answer.answer, language=None)


def render_sources(answer: Answer) -> None:
    """
    Render source chunks for an answer.

    For `structural` mode (no chunk retrieval), show a caption explaining
    the answer came from the document outline instead of searching.

    For other modes, show citation badges and an expander of bordered evidence cards.
    """
    if answer.mode == "structural":
        st.caption("Answered from the stored document outline — no chunk retrieval was performed.")
        return

    if not answer.sources:
        return

    citations = unique_citations(answer)
    if citations:
        st.write("**Sources:**")
        cols = st.columns(min(3, len(citations)))
        for idx, citation in enumerate(citations):
            with cols[idx % len(cols)]:
                st.badge(citation)

    with st.expander(f"📎 Evidence — {len(answer.sources)} chunks", expanded=False):
        for idx, source in enumerate(answer.sources):
            with st.container(border=True):
                st.markdown(f"**{source.citation}**")
                st.caption(format_scores(source))
                st.text(source.text[:300] + "..." if len(source.text) > 300 else source.text)
                with st.popover(f"📖 Full chunk {idx + 1}"):
                    st.text(source.text)


def render_no_answer() -> None:
    """
    Render a message when the LLM has no answer.

    Provides concrete next steps instead of a bare warning.
    """
    st.warning(
        "**No information found.** Try:\n"
        "- Widening your document scope (check the Scope selector in the sidebar)\n"
        "- Uploading more documents\n"
        "- Rephrasing your question more specifically"
    )


def render_empty_state(snapshot: dict) -> str | None:
    """
    Render an empty state when no turns exist.

    If no documents are indexed, prompt to upload.
    If documents exist but no conversation, show example questions.

    Returns:
        The clicked example question, or None.
    """
    if snapshot["chunk_count"] == 0:
        st.info("📤 Upload a PDF to get started. Use the 'Add a document' section in the sidebar.")
        return None

    st.info("Try one of these to get started:")
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("🔍 Lookup Example", use_container_width=True):
            return "What is the key concept in this document?"

    with col2:
        if st.button("🛠️ Procedural Example", use_container_width=True):
            return "Guide me step by step on how to configure something."

    with col3:
        if st.button("📑 Structural Example", use_container_width=True):
            return "What is the table of contents?"

    return None
