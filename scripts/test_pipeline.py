"""
Manual pipeline verification script — full RAG loop.

Run from the project root with:
    uv run python scripts/test_pipeline.py

Requires OPENAI_API_KEY in your .env file.
NOTE: If you already ran this script before, ChromaDB has your chunks cached.
      The embed_chunks() call will upsert (safe to re-run, no duplicates).
"""

import logging
logging.basicConfig(level=logging.INFO, format="%(name)s - %(message)s")

from src.ingestion.pdf_reader import extract_text_from_pdf
from src.chunking.cleaner import clean_document
from src.chunking.chunker import chunk_document, get_chunk_stats
from src.embedding.vector_store import embed_chunks
from src.generation.generator import generate_answer

# ── Steps 1-3: Ingest, Clean, Chunk ──
print("\n=== INGESTION + CHUNKING ===")
doc = extract_text_from_pdf("data/raw/MISRA-Compliance-2020.pdf")
cleaned = clean_document(doc)
chunks = chunk_document(cleaned)
print(get_chunk_stats(chunks))

# ── Step 4: Embed (safe to re-run — upsert is idempotent) ──
print("\n=== EMBEDDING ===")
embed_chunks(chunks)

# ── Step 5: Full RAG queries ──
print("\n=== RAG QUERIES ===")

questions = [
    "What is a deviation in MISRA compliance?",
    "What are the categories of MISRA guidelines?",
    "How should violations be documented?",
    "What is the boiling point of water?"
]


for question in questions:
    print(f"\nQ: {question}")
    print("-" * 60)

    answer = generate_answer(question)

    print(f"A: {answer.answer}")
    print(f"\n{answer.formatted_sources}")
    print(f"has_answer: {answer.has_answer}")
