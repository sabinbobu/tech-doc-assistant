"""
Embedding generation and vector storage using ChromaDB.

WHAT HAPPENS HERE:
1. Take each Chunk's text
2. Send it to an embedding model → get back a vector of ~1536 numbers
3. Store that vector + original text + metadata in ChromaDB

WHY CHROMADB FOR DEVELOPMENT:
ChromaDB runs fully locally — no server, no cloud account, no cost.
It persists to disk (our vectorstore/ directory) so you don't re-embed
every time you restart. Think of it like an SQLite for vectors.

In production at BMW you'd likely use Pinecone, Weaviate, or pgvector —
but the interface is identical. Swapping is trivial because we isolate
all vector DB logic here, in one module.

ANALOGY:
The embedding model is your ADC — converts raw signal (text) into
a numerical representation your system can process.
ChromaDB is your data logger — stores those readings with timestamps
(metadata) so you can query them later.
"""

import logging
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions
from langchain_openai import OpenAIEmbeddings

from src.chunking.models import Chunk
from src.config import settings

logger = logging.getLogger(__name__)

# Name of the collection inside ChromaDB
# Think of a collection like a table in SQL — groups related vectors together
COLLECTION_NAME = "tech_docs"


def get_chroma_client() -> chromadb.PersistentClient:
    """
    Create or connect to a persistent ChromaDB instance.

    PersistentClient saves data to disk — vectors survive process restarts.
    This means you only embed your documents ONCE, not on every run.
    Like flashing config to EEPROM vs. recalculating every boot.
    """
    vectorstore_path = settings.vectorstore_dir
    vectorstore_path.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(path=str(vectorstore_path))
    logger.info(f"ChromaDB connected at: {vectorstore_path}")
    return client


def get_or_create_collection(
    client: chromadb.PersistentClient,
) -> chromadb.Collection:
    """
    Get existing collection or create a new one.

    We use OpenAI's embedding function directly in ChromaDB so that
    queries are automatically embedded with the same model as the chunks.
    This guarantees vector space consistency — same model in, same model out.
    """
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=settings.openai_api_key,
        model_name=settings.embedding_model,  # "text-embedding-3-small"
    )

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=openai_ef,
        metadata={"hnsw:space": "cosine"},  # cosine similarity for semantic search
    )

    logger.info(
        f"Collection '{COLLECTION_NAME}' ready "
        f"({collection.count()} vectors stored)"
    )
    return collection


def embed_chunks(chunks: list[Chunk]) -> None:
    """
    Embed all chunks and store them in ChromaDB.

    This is the OFFLINE step — run once per document ingestion.
    Not called during query time.

    ChromaDB expects three parallel lists:
    - ids: unique string identifier per chunk
    - documents: the raw text (stored alongside vector for retrieval)
    - metadatas: dict of metadata per chunk (source, page, etc.)

    ChromaDB handles the actual embedding call internally using
    the embedding_function we attached to the collection.

    Args:
        chunks: List of Chunk objects from the chunking module.
    """
    if not chunks:
        logger.warning("No chunks to embed — skipping")
        return

    client = get_chroma_client()
    collection = get_or_create_collection(client)

    # Build the three parallel lists ChromaDB expects
    ids = [f"chunk_{chunk.chunk_index}_{chunk.source_filename}" for chunk in chunks]
    documents = [chunk.text for chunk in chunks]
    metadatas = [
        {
            "source_filename": chunk.source_filename,
            "source_filepath": chunk.source_filepath,
            "page_number": chunk.page_number,
            "chunk_index": chunk.chunk_index,
            "citation": chunk.citation,
        }
        for chunk in chunks
    ]

    # Batch upsert — "upsert" means insert or update if already exists
    # This makes re-ingestion safe: running twice won't duplicate vectors
    # Like an idempotent CAN message handler
    logger.info(f"Embedding {len(chunks)} chunks...")

    # ChromaDB recommends batching large inserts
    batch_size = 100
    for i in range(0, len(chunks), batch_size):
        batch_ids = ids[i:i + batch_size]
        batch_docs = documents[i:i + batch_size]
        batch_meta = metadatas[i:i + batch_size]

        collection.upsert(
            ids=batch_ids,
            documents=batch_docs,
            metadatas=batch_meta,
        )
        logger.info(f"  Embedded batch {i // batch_size + 1} ({len(batch_ids)} chunks)")

    logger.info(f"Done. Collection now has {collection.count()} total vectors.")


def query_collection(query_text: str, n_results: int = 5) -> list[dict]:
    """
    Search the vector store for chunks similar to the query.

    This is the ONLINE step — called at query time.

    ChromaDB automatically embeds query_text using the same
    embedding function attached to the collection, then returns
    the n_results most similar chunks by cosine similarity.

    Args:
        query_text: The user's question as a plain string.
        n_results: How many chunks to retrieve (our top-k).

    Returns:
        List of dicts, each containing:
        - text: the chunk text
        - metadata: source, page, citation info
        - distance: similarity score (lower = more similar for cosine)
    """
    client = get_chroma_client()
    collection = get_or_create_collection(client)

    if collection.count() == 0:
        logger.warning("Collection is empty — run embed_chunks first")
        return []

    results = collection.query(
        query_texts=[query_text],
        n_results=min(n_results, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    # ChromaDB returns nested lists (one per query) — we only send one query
    # so we unwrap the first element of each list
    retrieved = []
    for text, metadata, distance in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        retrieved.append({
            "text": text,
            "metadata": metadata,
            "distance": distance,
        })

    logger.info(
        f"Retrieved {len(retrieved)} chunks for query: '{query_text[:60]}...'"
    )
    return retrieved
