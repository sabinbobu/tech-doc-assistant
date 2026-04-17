"""
Manual pipeline verification script.

Run from the project root with:
    python scripts/test_pipeline.py

NOT a unit test — this is for manual inspection during development.
"""

from src.ingestion.pdf_reader import extract_text_from_pdf
from src.chunking.cleaner import clean_document
from src.chunking.chunker import chunk_document, get_chunk_stats

doc = extract_text_from_pdf("data/raw/MISRA-Compliance-2020.pdf")
print(f"Ingested: {doc.filename}, {len(doc.pages)} pages")

cleaned = clean_document(doc)

chunks = chunk_document(cleaned)
print(get_chunk_stats(chunks))

print("\n--- Sample chunk ---")
print(chunks[10].text)
print(f"\nCitation: {chunks[10].citation}")