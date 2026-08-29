"""
Application configuration using pydantic-settings.

WHY THIS MATTERS:
In embedded C, you'd have a config.h with #define macros for hardware parameters.
This is the Python equivalent — but smarter. It:
  1. Reads from environment variables (for secrets like API keys)
  2. Has type validation (catches misconfiguration early)
  3. Has defaults (so the app works out of the box for development)
  4. Is centralized (one source of truth, not scattered magic strings)

In production at BMW, you'd never hardcode an API key. This pattern is industry standard.
"""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    All application settings in one place.

    Values are loaded in this priority order (highest wins):
    1. Environment variables (e.g., export OPENAI_API_KEY=sk-...)
    2. .env file in project root
    3. Default values defined here

    This is like having a config.h with fallback defaults,
    but environment variables can override at runtime without recompiling.
    """

    model_config = SettingsConfigDict(
        env_file=".env",  # Load from .env file if present
        env_file_encoding="utf-8",
        extra="ignore",  # Don't crash on unknown env vars
    )

    # ── Paths ──
    # Using Path objects instead of strings — they handle OS differences
    # (like forward vs backslash) automatically, similar to how you'd use
    # platform-agnostic path macros in a cross-platform C project.
    project_root: Path = Path(__file__).parent.parent
    data_raw_dir: Path = project_root / "data" / "raw"
    data_processed_dir: Path = project_root / "data" / "processed"
    vectorstore_dir: Path = project_root / "vectorstore"

    # ── LLM Configuration ──
    # We support OpenAI, Anthropic, and OpenRouter (a proxy exposing many
    # providers, including free-tier models, behind one OpenAI-compatible API —
    # see src/generation/llm_client.py).
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    openrouter_api_key: str = ""

    # Which LLM to use for answer generation
    # "gpt-4o-mini" is cheap and good for development
    # "claude-sonnet-4-5-20250514" for Anthropic
    # For "openrouter", llm_model must be an OpenRouter model slug, e.g.
    # "deepseek/deepseek-chat-v3.1:free" — see https://openrouter.ai/models
    llm_provider: str = "openai"  # "openai", "anthropic", or "openrouter"
    llm_model: str = "gpt-4o-mini"

    # ── Embedding Configuration ──
    # The embedding model converts text → vectors
    # "text-embedding-3-small" is OpenAI's efficient model
    # Good balance of quality vs cost for development
    embedding_model: str = "text-embedding-3-small"

    # ── Chunking Configuration ──
    # These values will make more sense when we build the chunker.
    # For now, think of them as buffer sizes:
    # - chunk_size: how many characters per chunk (like a buffer size)
    # - chunk_overlap: how many characters overlap between chunks
    #   (like a sliding window with overlap to avoid cutting mid-sentence)
    chunk_size: int = 1000
    chunk_overlap: int = 200

    # ── Retrieval Configuration ──
    # How many chunks to retrieve per query
    # Like "top N search results" — too few misses context, too many adds noise
    retrieval_top_k: int = 8

    # How many candidates to pull from hybrid search BEFORE reranking.
    # Must be >= retrieval_top_k — the reranker can only pick from what it's
    # handed. Wider than top_k on purpose: RRF's top-8 by rank isn't
    # necessarily the reranker's top-8 by actual relevance, so we give it a
    # bigger pool to re-sort. Like oversampling an ADC before decimating —
    # you want more raw samples than your final output rate.
    retrieval_candidate_k: int = 20

    # Cross-encoder model for reranking. Runs locally via sentence-transformers,
    # no API key, no network call per query (one-time download on first use).
    #
    # Benchmarked on this project's dev machine (Intel Mac, CPU-only torch —
    # see pyproject.toml's required-environments): BAAI/bge-reranker-v2-m3
    # (568M params) took 11.7s to score 30 candidates — unusable in an
    # interactive app. ms-marco-MiniLM-L6-v2 (22M params) does the same job
    # in ~1s with no meaningful quality loss for this corpus's English,
    # short-passage technical documents. Bigger cross-encoders earn their
    # cost on harder ranking problems; this one doesn't need it.
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L6-v2"

    # Kill switch — if the reranker model fails to load (e.g. torch not
    # installed, offline environment), retrieval must still work. Set False
    # to skip reranking and use the hybrid search order as-is.
    reranking_enabled: bool = True

    # Rewrite the user's question into a cleaner search query before it hits
    # the index — see src/retrieval/query_rewrite.py.
    #
    # Defaults OFF, unlike reranking_enabled. Measured on this project's own
    # RAGAS eval (see CLAUDE.md-equivalent history / eval_results_*.json):
    # enabling it dropped faithfulness 0.815 -> 0.682 and flipped a
    # previously-correct answer ("what are the two categories of MISRA
    # guidelines" -> Mandatory/Advisory) to a wrong-but-plausible one
    # (Rules/Directives) by stripping question-framing context the rewrite
    # prompt didn't preserve. The code and tests are real and working; it
    # just doesn't have evidence to justify shipping active. Re-evaluate once
    # a larger eval set (Phase 2) can resolve small-sample swings properly.
    query_rewrite_enabled: bool = False

    # Model used for the rewrite call. None reuses llm_model (same provider,
    # same model as answer generation) — rewriting is a much simpler task
    # than generation, so a cheaper/faster model of the same provider is a
    # reasonable override once you have real latency numbers to justify it.
    rewrite_model: str | None = None

    # Max output tokens for a generated answer. 1024 is enough for a lookup
    # answer with citations but truncates a multi-step configuration
    # procedure mid-sentence — synthesis mode (src/retrieval/query_plan.py)
    # requests a larger budget explicitly rather than raising this default,
    # since most questions are still simple lookups that don't need it.
    max_answer_tokens: int = 1024
    synthesis_answer_tokens: int = 3000

    # Classify each question as lookup / procedural / structural and, for
    # procedural questions, decompose it into sub-queries (src/retrieval/
    # query_plan.py). Unlike query_rewrite_enabled, this defaults ON — it's
    # not an optional optimization measured to regress something that
    # already worked, it's the actual fix for two concrete, verified
    # failures ("guide me step by step to configure X" and "list the table
    # of contents" both used to refuse outright). Degrades to lookup mode
    # with the raw question on any failure — routing must never break
    # retrieval, same philosophy as reranking_enabled and query_rewrite_enabled.
    query_planning_enabled: bool = True

    # Model for the classification/decomposition call. None reuses llm_model.
    query_plan_model: str | None = None


# Singleton pattern — one settings instance for the whole app
# In C terms, this is like a global config struct initialized once at startup
settings = Settings()
