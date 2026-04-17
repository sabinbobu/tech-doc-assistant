"""
Data models for the answer generation module.

The Answer model is the final output of our entire RAG pipeline.
Everything before this — ingestion, chunking, embedding, retrieval —
was building up to producing this object.
"""

from pydantic import BaseModel, Field


class RetrievedContext(BaseModel):
    """
    A single retrieved chunk with its relevance score.
    Bridges the retrieval module output into the generation module.
    """
    text: str = Field(description="The chunk text")
    citation: str = Field(description="Human-readable source citation")
    page_number: int = Field(description="Source page number")
    source_filename: str = Field(description="Source document filename")
    distance: float = Field(description="Similarity distance — lower is more relevant")


class Answer(BaseModel):
    """
    The final output of the RAG pipeline.

    Contains both the generated answer AND the sources it was based on.
    Never return an answer without its sources — citations are not optional.

    In an interview, if asked "how do you prevent hallucinations in RAG?",
    this model is part of your answer: you structurally enforce that every
    answer carries its source context.
    """
    question: str = Field(description="The original user question")
    answer: str = Field(description="LLM-generated answer based on retrieved context")
    sources: list[RetrievedContext] = Field(description="Chunks used to generate the answer")
    has_answer: bool = Field(
        description="False if the retrieved context didn't contain enough information"
    )

    @property
    def formatted_sources(self) -> str:
        """
        Format sources as a readable list for display in the UI.

        Example output:
            Sources:
            - MISRA-Compliance-2020.pdf, page 12
            - MISRA-Compliance-2020.pdf, page 23
        """
        unique_citations = list(dict.fromkeys(s.citation for s in self.sources))
        lines = ["Sources:"] + [f"  - {c}" for c in unique_citations]
        return "\n".join(lines)
