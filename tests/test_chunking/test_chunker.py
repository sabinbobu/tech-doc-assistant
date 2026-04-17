"""
Tests for the chunking module.

We test three things:
  1. Header/footer detection logic
  2. Page cleaning
  3. Chunk generation and metadata correctness
"""

import pytest

from src.chunking.chunker import ChunkingConfig, chunk_document, get_chunk_stats
from src.chunking.cleaner import clean_document, detect_repeating_lines
from src.chunking.models import Chunk
from src.ingestion.models import DocumentContent, PageContent


# ── Fixtures ──
# Reusable test data — like test vectors in embedded unit testing

@pytest.fixture
def document_with_footers() -> DocumentContent:
    """A document where every page has the same footer."""
    pages = [
        PageContent(
            page_number=i,
            text=f"This is the content of page {i}.\nIt has some meaningful text.\n\nMISRA Compliance 2020\n© MISRA Ltd\n{i}",
        )
        for i in range(1, 11)  # 10 pages
    ]
    return DocumentContent(
        filename="test.pdf",
        filepath="/tmp/test.pdf",
        total_pages=10,
        pages=pages,
    )


@pytest.fixture
def clean_document_fixture() -> DocumentContent:
    """A simple clean document with no headers/footers."""
    pages = [
        PageContent(
            page_number=1,
            text="Rule 8.4: A compatible declaration shall be visible.\n\n"
                 "This rule requires that when an object or function is defined, "
                 "a compatible declaration is visible at the point of definition.",
        ),
        PageContent(
            page_number=2,
            text="Rule 8.5: An external object or function shall be declared once.\n\n"
                 "Declaring an external object or function in more than one file "
                 "can lead to inconsistencies between the declarations.",
        ),
    ]
    return DocumentContent(
        filename="misra.pdf",
        filepath="/tmp/misra.pdf",
        total_pages=2,
        pages=pages,
    )


# ── Tests for header/footer detection ──

class TestDetectRepeatingLines:

    def test_detects_footer_on_all_pages(self, document_with_footers):
        """Lines present on every page should be detected as headers/footers."""
        repeating = detect_repeating_lines(document_with_footers, threshold=0.4)
        assert "MISRA Compliance 2020" in repeating
        assert "© MISRA Ltd" in repeating

    def test_does_not_flag_unique_content(self, document_with_footers):
        """Content unique to one page should NOT be flagged."""
        repeating = detect_repeating_lines(document_with_footers, threshold=0.4)
        # "content of page 1" only appears on page 1
        assert "This is the content of page 1." not in repeating

    def test_empty_document_returns_empty_set(self):
        """Should handle empty documents gracefully."""
        empty_doc = DocumentContent(
            filename="empty.pdf", filepath="/tmp/empty.pdf",
            total_pages=0, pages=[]
        )
        result = detect_repeating_lines(empty_doc)
        assert result == set()

    def test_threshold_controls_sensitivity(self, document_with_footers):
        """Higher threshold = less aggressive cleaning."""
        # With threshold=0.99, almost nothing should be flagged
        # (nothing appears on 99% of 10 pages = 9.9 pages)
        strict = detect_repeating_lines(document_with_footers, threshold=0.99)
        # With threshold=0.1, almost everything should be flagged
        aggressive = detect_repeating_lines(document_with_footers, threshold=0.1)
        assert len(strict) <= len(aggressive)


# ── Tests for document cleaning ──

class TestCleanDocument:

    def test_removes_footers(self, document_with_footers):
        """Cleaning should remove detected footers from all pages."""
        cleaned = clean_document(document_with_footers, threshold=0.4)
        for page in cleaned.pages:
            assert "MISRA Compliance 2020" not in page.text
            assert "© MISRA Ltd" not in page.text

    def test_preserves_unique_content(self, document_with_footers):
        """Meaningful content unique to each page should survive cleaning."""
        cleaned = clean_document(document_with_footers, threshold=0.4)
        # Page 1's unique content should still be there
        assert "content of page 1" in cleaned.pages[0].text

    def test_preserves_total_pages(self, document_with_footers):
        """Cleaning should not remove pages, only noise within pages."""
        cleaned = clean_document(document_with_footers)
        assert cleaned.total_pages == document_with_footers.total_pages
        assert len(cleaned.pages) == len(document_with_footers.pages)

    def test_returns_new_document_not_mutated(self, document_with_footers):
        """Original document should be unchanged — immutability."""
        original_text = document_with_footers.pages[0].text
        clean_document(document_with_footers)
        assert document_with_footers.pages[0].text == original_text


# ── Tests for chunker ──

class TestChunkDocument:

    def test_produces_chunks(self, clean_document_fixture):
        """Should produce at least one chunk from a non-empty document."""
        chunks = chunk_document(clean_document_fixture)
        assert len(chunks) > 0

    def test_chunk_metadata_is_correct(self, clean_document_fixture):
        """Every chunk should have correct source metadata."""
        chunks = chunk_document(clean_document_fixture)
        for chunk in chunks:
            assert chunk.source_filename == "misra.pdf"
            assert chunk.page_number in [1, 2]
            assert chunk.chunk_index >= 0

    def test_chunk_text_is_not_empty(self, clean_document_fixture):
        """No chunk should have empty text."""
        chunks = chunk_document(clean_document_fixture)
        for chunk in chunks:
            assert chunk.text.strip() != ""

    def test_chunk_size_respects_config(self, clean_document_fixture):
        """Chunks should not exceed the configured max size by much.

        We allow some tolerance because the splitter respects sentence
        boundaries — it may slightly exceed chunk_size to avoid
        cutting mid-sentence.
        """
        config = ChunkingConfig(chunk_size=200, chunk_overlap=20)
        chunks = chunk_document(clean_document_fixture, config)
        tolerance = 50  # allow 50 chars over limit for natural boundaries
        for chunk in chunks:
            assert chunk.char_count <= config.chunk_size + tolerance, \
                f"Chunk too large: {chunk.char_count} chars"

    def test_citation_format(self, clean_document_fixture):
        """Citation property should return readable source string."""
        chunks = chunk_document(clean_document_fixture)
        assert "misra.pdf" in chunks[0].citation
        assert "page" in chunks[0].citation


# ── Tests for chunk stats ──

class TestGetChunkStats:

    def test_stats_on_empty_list(self):
        """Should handle empty chunk list gracefully."""
        stats = get_chunk_stats([])
        assert stats["total"] == 0

    def test_stats_correct(self, clean_document_fixture):
        """Stats should correctly reflect the chunk set."""
        chunks = chunk_document(clean_document_fixture)
        stats = get_chunk_stats(chunks)
        assert stats["total_chunks"] == len(chunks)
        assert stats["min_chars"] <= stats["avg_chars"] <= stats["max_chars"]
