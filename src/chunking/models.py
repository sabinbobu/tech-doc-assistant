"""
Data models for the chunking module.

A chunk is the atomic unit of our RAG system — everything downstream
(embedding, retrieval, citation generation) operates on chunks.

WHY WE TRACK METADATA PER CHUNK:
When the LLM answers "What does Rule 8.4 say?", we need to tell the user
WHERE that answer came from: which document, which page.
Without metadata, we have answers but no citations — useless in production.

Think of it like a CAN frame: the payload (text) is useless without
the identifier (metadata) telling you what system sent it and what it means.
"""

from pydantic import BaseModel, Field


class Chunk(BaseModel):
    """
    A single chunk of text ready for embedding.

    This is the core data unit that flows from chunking → embedding → retrieval.
    Every downstream module speaks in terms of Chunk objects.
    """

    # The actual text content — what gets embedded and retrieved
    text: str = Field(description="The chunk text content")

    # Source traceability — critical for citations
    source_filename: str = Field(description="Filename of the source document")
    source_filepath: str = Field(description="Full path to source document")
    page_number: int = Field(description="Page where this chunk starts")

    # Position within the document — useful for debugging and re-ranking
    chunk_index: int = Field(description="Index of this chunk within the document (0-based)")

    # Character-level position — lets us reconstruct context if needed
    start_char: int = Field(description="Start character position within the page text")
    end_char: int = Field(description="End character position within the page text")

    @property
    def citation(self) -> str:
        """
        Human-readable citation string for this chunk.

        When the LLM answers a question, we append this to the answer:
        "Source: MISRA-Compliance-2020.pdf, page 23"
        """
        return f"{self.source_filename}, page {self.page_number}"

    @property
    def char_count(self) -> int:
        """Length of this chunk in characters."""
        return len(self.text)
