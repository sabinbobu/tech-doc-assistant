"""
Rebuild the vector store from scratch, from data/raw/ only.

WHY THIS SCRIPT EXISTS:
embed_chunks() upserts — safe for re-running the same document, but it never
removes anything. Documents ingested through the Streamlit UI (uploads saved
to a tempfile) stay in the collection forever, even after the browser tab is
closed. Over time the collection accumulates chunks from documents that were
only ever meant to be a one-off test upload, alongside your real corpus in
data/raw/. There's no "list what's actually in there and prune the rest" tool
except this one.

This is the fix for that: wipe the ChromaDB collection completely, then
re-ingest only the PDFs that live in data/raw/ — the durable, intentional
corpus. Anything ingested via the UI and not also saved to data/raw/ is gone
after this runs. That's the point.

RUN WITH:
    uv run python scripts/reingest.py --wipe

Requires OPENAI_API_KEY in your .env (embedding calls hit OpenAI).
"""

import argparse
import logging

from src.chunking.chunker import chunk_document, get_chunk_stats
from src.chunking.cleaner import clean_document
from src.config import settings
from src.embedding.vector_store import (
    COLLECTION_NAME,
    embed_chunks,
    get_chroma_client,
    list_indexed_documents,
)
from src.ingestion.pdf_reader import extract_all_pdfs

logging.basicConfig(level=logging.INFO, format="%(name)s - %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--wipe",
        action="store_true",
        help="Actually delete the existing collection before re-ingesting. "
        "Without this flag, the script only reports what it WOULD do.",
    )
    args = parser.parse_args()

    client = get_chroma_client()
    existing = list_indexed_documents()

    if existing:
        print(f"\nCurrent collection ({sum(d['chunk_count'] for d in existing)} chunks):")
        for indexed in existing:
            print(f"  {indexed['filename']}: {indexed['chunk_count']} chunks")
    else:
        print("\nCurrent collection: empty")

    pdf_files = sorted(settings.data_raw_dir.glob("*.pdf"))
    print(f"\ndata/raw/ contains {len(pdf_files)} PDF(s):")
    for pdf in pdf_files:
        print(f"  {pdf.name}")

    if not args.wipe:
        print(
            "\nDry run — no changes made. Re-run with --wipe to delete the "
            "existing collection and rebuild it from data/raw/ only."
        )
        return

    if existing:
        logger.info(f"Deleting collection '{COLLECTION_NAME}'...")
        client.delete_collection(name=COLLECTION_NAME)

    documents = extract_all_pdfs(settings.data_raw_dir)
    if not documents:
        logger.error(f"No PDFs found in {settings.data_raw_dir} — nothing to ingest.")
        return

    for doc in documents:
        cleaned = clean_document(doc)
        chunks = chunk_document(cleaned)
        stats = get_chunk_stats(chunks)
        logger.info(f"{doc.filename}: {stats}")
        embed_chunks(chunks)

    print("\nDone. Final collection:")
    for indexed in list_indexed_documents():
        print(f"  {indexed['filename']}: {indexed['chunk_count']} chunks")


if __name__ == "__main__":
    main()
