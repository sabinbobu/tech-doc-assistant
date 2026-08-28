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
from rank_bm25 import BM25Okapi

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

    logger.info(f"Collection '{COLLECTION_NAME}' ready ({collection.count()} vectors stored)")
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
        batch_ids = ids[i : i + batch_size]
        batch_docs = documents[i : i + batch_size]
        batch_meta = metadatas[i : i + batch_size]

        collection.upsert(
            ids=batch_ids,
            documents=batch_docs,
            metadatas=batch_meta,
        )
        logger.info(f"  Embedded batch {i // batch_size + 1} ({len(batch_ids)} chunks)")

    logger.info(f"Done. Collection now has {collection.count()} total vectors.")


def _build_source_filter(source_filenames: list[str] | None) -> dict | None:
    """Build a ChromaDB `where` clause scoping a query to specific documents."""
    if not source_filenames:
        return None
    if len(source_filenames) == 1:
        return {"source_filename": source_filenames[0]}
    return {"source_filename": {"$in": source_filenames}}


def list_indexed_documents() -> list[dict]:
    """
    List every distinct document currently stored in the collection.

    All uploaded PDFs share one ChromaDB collection, so this is the only
    way to see what's actually in there and how many chunks each one
    contributed — used by the UI to show per-document status and to
    populate the "search within" document picker.

    Returns:
        List of dicts sorted by filename: [{"filename": ..., "chunk_count": ...}, ...]
    """
    client = get_chroma_client()
    collection = get_or_create_collection(client)

    if collection.count() == 0:
        return []

    result = collection.get(include=["metadatas"])
    counts: dict[str, int] = {}
    for metadata in result["metadatas"]:
        filename = metadata.get("source_filename", "Unknown")
        counts[filename] = counts.get(filename, 0) + 1

    return [
        {"filename": filename, "chunk_count": count} for filename, count in sorted(counts.items())
    ]


def _vector_search(
    collection: chromadb.Collection,
    query_text: str,
    n_results: int,
    where: dict | None,
) -> list[dict]:
    """Cosine-similarity search — finds chunks that are semantically similar."""
    results = collection.query(
        query_texts=[query_text],
        n_results=min(n_results, collection.count()),
        include=["documents", "metadatas", "distances"],
        **({"where": where} if where else {}),
    )

    # ChromaDB returns nested lists (one per query) — we only send one query
    # so we unwrap the first element of each list
    retrieved = []
    for text, metadata, distance in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
        strict=True,
    ):
        retrieved.append(
            {
                "text": text,
                "metadata": metadata,
                "distance": distance,
            }
        )
    return retrieved


def _keyword_search(
    collection: chromadb.Collection,
    query_text: str,
    n_results: int,
    where: dict | None,
) -> list[dict]:
    """
    BM25 keyword search — finds chunks that share exact terms with the query.

    Vector search alone can miss document-specific keywords (part numbers,
    rule IDs) whose meaning doesn't shift much in embedding space. BM25
    complements it by rewarding exact/near-exact term overlap.

    Rebuilds the BM25 index from the collection's stored text on every call —
    fine at this project's scale (hundreds to low thousands of chunks) since
    there's no persistent index to keep in sync with ingestion.
    """
    corpus = collection.get(
        include=["documents", "metadatas"],
        **({"where": where} if where else {}),
    )
    documents = corpus["documents"]
    metadatas = corpus["metadatas"]

    if not documents:
        return []

    tokenized_corpus = [doc.lower().split() for doc in documents]
    bm25 = BM25Okapi(tokenized_corpus)
    scores = bm25.get_scores(query_text.lower().split())

    ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    top_indices = [i for i in ranked_indices if scores[i] > 0][:n_results]

    return [{"text": documents[i], "metadata": metadatas[i], "distance": None} for i in top_indices]


def _reciprocal_rank_fusion(
    vector_results: list[dict],
    keyword_results: list[dict],
    n_results: int,
    k: int = 60,
) -> list[dict]:
    """
    Merge vector and keyword rankings via Reciprocal Rank Fusion (RRF).

    Each chunk's fused score is sum(1 / (k + rank)) across whichever
    list(s) it appears in (rank is 1-indexed). k=60 is the standard RRF
    constant — it flattens the influence of any single rank so a chunk
    doesn't need to be #1 in either list to surface, just consistently
    present. This lets a chunk that's a strong keyword match but a weak
    vector match (or vice versa) still make the final cut.
    """

    def chunk_key(chunk: dict) -> tuple:
        metadata = chunk["metadata"]
        return (metadata.get("source_filename"), metadata.get("chunk_index"))

    scores: dict[tuple, float] = {}
    chunks_by_key: dict[tuple, dict] = {}

    for result_list in (vector_results, keyword_results):
        for rank, chunk in enumerate(result_list, start=1):
            key = chunk_key(chunk)
            scores[key] = scores.get(key, 0.0) + 1 / (k + rank)
            # Prefer whichever version of the chunk we saw first (the vector
            # list is processed first, so its real `distance` wins over a
            # keyword-only hit's placeholder).
            chunks_by_key.setdefault(key, chunk)

    ranked_keys = sorted(scores, key=lambda key: scores[key], reverse=True)

    fused = []
    for key in ranked_keys[:n_results]:
        chunk = dict(chunks_by_key[key])
        if chunk["distance"] is None:
            # Keyword-only hit — no cosine distance available. Fall back to
            # a neutral mid-point so the UI's "relevance = 1 - distance"
            # display stays in a sane 0-1 range instead of crashing on None.
            chunk["distance"] = 0.5
        fused.append(chunk)
    return fused


def query_collection(
    query_text: str,
    n_results: int = 5,
    source_filenames: list[str] | None = None,
) -> list[dict]:
    """
    Search the vector store for chunks relevant to the query.

    This is the ONLINE step — called at query time. It runs two searches
    in parallel and fuses the results:
      1. Vector search: ChromaDB embeds query_text with the same model
         used for the chunks, then returns the most semantically similar
         chunks by cosine similarity.
      2. Keyword search: BM25 over the same (optionally filtered) chunks,
         which catches exact-term matches vector search can miss.
    See _reciprocal_rank_fusion for how the two rankings are merged.

    Args:
        query_text: The user's question as a plain string.
        n_results: How many chunks to retrieve (our top-k) after fusion.
        source_filenames: If given, restrict the search to chunks whose
            source_filename metadata is in this list. Every uploaded PDF
            lives in the same collection, so without this a query can
            pull in chunks from an unrelated document. None searches
            everything (the previous, unscoped behavior).

    Returns:
        List of dicts, each containing:
        - text: the chunk text
        - metadata: source, page, citation info
        - distance: similarity score (lower = more similar for cosine;
          keyword-only hits get a neutral placeholder — see
          _reciprocal_rank_fusion)
    """
    client = get_chroma_client()
    collection = get_or_create_collection(client)

    if collection.count() == 0:
        logger.warning("Collection is empty — run embed_chunks first")
        return []

    where = _build_source_filter(source_filenames)

    # Over-fetch candidates from each search before fusing, so a chunk
    # that's merely "good" (not top-1) in one ranking still has a chance
    # to surface if the other ranking likes it too.
    candidate_pool = max(n_results * 3, n_results)
    vector_results = _vector_search(collection, query_text, candidate_pool, where)
    keyword_results = _keyword_search(collection, query_text, candidate_pool, where)

    retrieved = _reciprocal_rank_fusion(vector_results, keyword_results, n_results)

    logger.info(f"Retrieved {len(retrieved)} chunks (hybrid) for query: '{query_text[:60]}...'")
    return retrieved
