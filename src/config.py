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
        env_file=".env",        # Load from .env file if present
        env_file_encoding="utf-8",
        extra="ignore",         # Don't crash on unknown env vars
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
    # We support both OpenAI and Anthropic — the job description asks for both.
    openai_api_key: str = ""
    anthropic_api_key: str = ""

    # Which LLM to use for answer generation
    # "gpt-4o-mini" is cheap and good for development
    # "claude-sonnet-4-5-20250514" for Anthropic
    llm_provider: str = "openai"  # "openai" or "anthropic"
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
    retrieval_top_k: int = 5


# Singleton pattern — one settings instance for the whole app
# In C terms, this is like a global config struct initialized once at startup
settings = Settings()
