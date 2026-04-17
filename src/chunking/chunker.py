"""
Text chunking strategies for RAG pipelines.

We implement two strategies here:
  1. RecursiveCharacterTextSplitter — the production default
  2. SimpleCharacterSplitter — a naive baseline for comparison

WHY TWO STRATEGIES?
In embedded, you'd benchmark two different filter algorithms before committing
to one. Same here. Having a naive baseline lets you measure whether the
"smart" strategy actually performs better during evaluation.

RECURSIVE CHARACTER SPLITTER — HOW IT WORKS:
It tries to split on natural boundaries in order of preference:
  1. Paragraph breaks (\n\n)  ← most natural split point
  2. Single newlines (\n)
  3. Spaces
  4. Individual characters (last resort — should rarely happen)

This preserves semantic coherence much better than "cut every 1000 chars."
A rule description in MISRA won't get cut in the middle of a sentence —
the splitter will prefer to cut at the nearest paragraph break.

ANALOGY: Like packetizing a CAN message — you prefer natural frame boundaries
over arbitrary byte counts, but you fall back to byte-level splitting
if the message is too long regardless.
"""

import logging
from dataclasses import dataclass

from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.chunking.models import Chunk
from src.ingestion.models import DocumentContent

logger = logging.getLogger(__name__)


@dataclass
class ChunkingConfig:
    """
    Configuration for the chunker.

    Separate from the global Settings because we might want to experiment
    with different chunking configs without changing the app-wide config.
    Like having per-module #defines instead of global constants.
    """
    chunk_size: int = 1000       # Target characters per chunk
    chunk_overlap: int = 200     # Overlap between consecutive chunks
    min_chunk_size: int = 100    # Discard chunks smaller than this (noise)


def chunk_document(
    document: DocumentContent,
    config: ChunkingConfig | None = None,
) -> list[Chunk]:
    """
    Split a document into overlapping chunks ready for embedding.

    This is the main entry point for this module.
    Input: a cleaned DocumentContent
    Output: a flat list of Chunk objects

    The pipeline is:
      for each page → split into chunks → attach metadata → collect all chunks

    Args:
        document: Cleaned DocumentContent (run clean_document first!)
        config: Chunking configuration. Uses defaults if not provided.

    Returns:
        List of Chunk objects with full metadata for citations.
    """
    if config is None:
        config = ChunkingConfig()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
        # These are the separators tried in order — preference for natural boundaries
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )

    all_chunks: list[Chunk] = []
    chunk_index = 0  # Global chunk index across the whole document

    for page in document.pages:
        if not page.text.strip():
            continue

        # Split this page's text into raw string chunks
        raw_chunks = splitter.split_text(page.text)

        for raw_chunk in raw_chunks:
            # Skip chunks that are too small — they're usually noise
            # (a stray header that survived cleaning, a lone page number, etc.)
            if len(raw_chunk.strip()) < config.min_chunk_size:
                logger.debug(f"Skipping tiny chunk ({len(raw_chunk)} chars) on page {page.page_number}")
                continue

            # Find where this chunk sits within the page text
            # This lets us reconstruct exact position for debugging
            start_char = page.text.find(raw_chunk)
            end_char = start_char + len(raw_chunk) if start_char != -1 else -1

            all_chunks.append(
                Chunk(
                    text=raw_chunk.strip(),
                    source_filename=document.filename,
                    source_filepath=document.filepath,
                    page_number=page.page_number,
                    chunk_index=chunk_index,
                    start_char=max(start_char, 0),
                    end_char=max(end_char, 0),
                )
            )
            chunk_index += 1

    logger.info(
        f"Chunked '{document.filename}' into {len(all_chunks)} chunks "
        f"(avg {sum(c.char_count for c in all_chunks) // max(len(all_chunks), 1)} chars each)"
    )

    return all_chunks


def get_chunk_stats(chunks: list[Chunk]) -> dict:
    """
    Compute statistics about a set of chunks.

    WHY THIS EXISTS:
    In embedded, after running your filter algorithm you'd check the output
    signal characteristics before moving to the next stage.
    Same here — inspect your chunks before feeding them to the embedder.
    Call this and log the output whenever you change chunking config.

    Returns a dict with min/max/avg chunk sizes and total chunk count.
    """
    if not chunks:
        return {"total": 0}

    sizes = [c.char_count for c in chunks]
    return {
        "total_chunks": len(chunks),
        "min_chars": min(sizes),
        "max_chars": max(sizes),
        "avg_chars": sum(sizes) // len(sizes),
        "total_chars": sum(sizes),
    }
