"""
Manual pipeline verification script — Steps 1 through 4.

Run from the project root with:
    uv run python scripts/test_pipeline.py

Requires OPENAI_API_KEY in your .env file.
This script will cost a few cents in OpenAI API calls (embedding ~100 chunks).
"""

import logging
logging.basicConfig(level=logging.INFO, format="%(name)s - %(message)s")

from src.ingestion.pdf_reader import extract_text_from_pdf
from src.chunking.cleaner import clean_document
from src.chunking.chunker import chunk_document, get_chunk_stats
from src.embedding.vector_store import embed_chunks, query_collection

# ── Step 1: Ingest ──
print("\n=== STEP 1: INGESTION ===")
doc = extract_text_from_pdf("data/raw/MISRA-Compliance-2020.pdf")
print(f"Ingested: {doc.filename}, {len(doc.pages)} pages")

# ── Step 2: Clean ──
print("\n=== STEP 2: CLEANING ===")
cleaned = clean_document(doc)

# ── Step 3: Chunk ──
print("\n=== STEP 3: CHUNKING ===")
chunks = chunk_document(cleaned)
stats = get_chunk_stats(chunks)
print(f"Stats: {stats}")
print(f"\nSample chunk:\n{chunks[10].text}")
print(f"Citation: {chunks[10].citation}")

# ── Step 4: Embed ──
print("\n=== STEP 4: EMBEDDING ===")
print("Sending chunks to OpenAI for embedding (costs ~$0.01)...")
embed_chunks(chunks)

# ── Step 4b: Query ──
print("\n=== STEP 4b: TEST QUERY ===")
question = "How do I achieve compliance?"
print(f"Query: '{question}'")
results = query_collection(question, n_results=3)

for i, result in enumerate(results, 1):
    print(f"\n--- Result {i} (distance: {result['distance']:.4f}) ---")
    print(f"Citation: {result['metadata']['citation']}")
    print(f"Text: {result['text'][:200]}...")
