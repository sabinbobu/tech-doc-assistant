"""
Tests for PDF ingestion module.

WHY WRITE TESTS FOR A PORTFOLIO PROJECT?
1. BMW explicitly requires "unit testing" experience
2. Tests ARE documentation — they show exactly how your code should be used
3. They catch regressions when you refactor (and you WILL refactor)
4. In an interview, tests signal engineering maturity

ANALOGY: In embedded, you'd run HIL (Hardware-in-the-Loop) tests
before flashing to the ECU. These unit tests are the Python equivalent —
they verify your logic works correctly before integration.

We use pytest because it's the Python standard (like Unity for embedded C testing).
"""

import pytest

from src.ingestion.models import DocumentContent, PageContent
from src.ingestion.pdf_reader import _clean_page_text


# ── Tests for data models ──
# These test that our "structs" behave correctly

class TestPageContent:
    """Tests for the PageContent model."""

    def test_create_valid_page(self):
        """Basic construction — like testing struct initialization."""
        page = PageContent(page_number=1, text="Hello world")
        assert page.page_number == 1
        assert page.text == "Hello world"

    def test_page_number_must_be_int(self):
        """Pydantic should reject invalid types — runtime type safety."""
        # This would crash in C if you passed a string to a uint8_t field.
        # Pydantic catches it at construction time.
        with pytest.raises(Exception):
            PageContent(page_number="not a number", text="Hello")


class TestDocumentContent:
    """Tests for the DocumentContent model."""

    def test_create_valid_document(self):
        """Test full document construction."""
        pages = [
            PageContent(page_number=1, text="Page one content"),
            PageContent(page_number=2, text="Page two content"),
        ]
        doc = DocumentContent(
            filename="test.pdf",
            filepath="/path/to/test.pdf",
            total_pages=2,
            pages=pages,
        )
        assert doc.filename == "test.pdf"
        assert doc.total_pages == 2
        assert len(doc.pages) == 2

    def test_full_text_concatenation(self):
        """full_text property should join all pages with double newlines."""
        pages = [
            PageContent(page_number=1, text="First page"),
            PageContent(page_number=2, text="Second page"),
        ]
        doc = DocumentContent(
            filename="test.pdf",
            filepath="/path/to/test.pdf",
            total_pages=2,
            pages=pages,
        )
        assert doc.full_text == "First page\n\nSecond page"

    def test_get_page_text_existing_page(self):
        """Should return text for a valid page number."""
        pages = [PageContent(page_number=3, text="Page three")]
        doc = DocumentContent(
            filename="test.pdf",
            filepath="/path/to/test.pdf",
            total_pages=5,
            pages=pages,
        )
        assert doc.get_page_text(3) == "Page three"

    def test_get_page_text_missing_page(self):
        """Should return None for a page that doesn't exist.

        Defensive programming — like checking NULL pointers.
        """
        pages = [PageContent(page_number=1, text="Page one")]
        doc = DocumentContent(
            filename="test.pdf",
            filepath="/path/to/test.pdf",
            total_pages=1,
            pages=pages,
        )
        assert doc.get_page_text(99) is None


# ── Tests for text cleaning ──
# These verify our "noise filtering" works correctly

class TestCleanPageText:
    """Tests for the _clean_page_text helper function."""

    def test_removes_excessive_blank_lines(self):
        """Should collapse 3+ blank lines into max 2."""
        text = "Line one\n\n\n\n\nLine two"
        result = _clean_page_text(text)
        # Should have at most 2 blank lines between content
        assert "\n\n\n" not in result
        assert "Line one" in result
        assert "Line two" in result

    def test_fixes_hyphenation(self):
        """Words split across lines with hyphens should be rejoined.

        PDFs often break "automatic" into "auto-\\nmatic" due to
        line wrapping in the original document.
        """
        text = "This is auto-\nmatic control"
        result = _clean_page_text(text)
        assert "automatic" in result

    def test_strips_leading_trailing_whitespace(self):
        """Clean text should not start or end with whitespace."""
        text = "   \n\n  Some content  \n\n   "
        result = _clean_page_text(text)
        assert result == "Some content"

    def test_preserves_meaningful_content(self):
        """Cleaning should not destroy actual content."""
        text = "Section 1: Introduction\nThis is important text.\nIt has multiple lines."
        result = _clean_page_text(text)
        assert "Section 1: Introduction" in result
        assert "This is important text." in result

    def test_empty_input(self):
        """Should handle empty strings gracefully — no crashes."""
        result = _clean_page_text("")
        assert result == ""


# ── Tests for PDF reader function ──
# These test the actual file I/O

class TestExtractTextFromPdf:
    """Tests for extract_text_from_pdf function.

    Note: We test error handling here without needing actual PDF files.
    For integration tests with real PDFs, see tests/integration/.
    """

    def test_file_not_found(self):
        """Should raise FileNotFoundError for missing files."""
        from src.ingestion.pdf_reader import extract_text_from_pdf

        with pytest.raises(FileNotFoundError):
            extract_text_from_pdf("/nonexistent/path/fake.pdf")

    def test_non_pdf_file(self, tmp_path):
        """Should reject non-PDF files.

        tmp_path is a pytest fixture that gives us a temporary directory —
        cleaned up automatically after the test. Like a RAM disk for testing.
        """
        from src.ingestion.pdf_reader import extract_text_from_pdf

        # Create a fake .txt file
        fake_file = tmp_path / "not_a_pdf.txt"
        fake_file.write_text("I am not a PDF")

        with pytest.raises(ValueError, match="Expected a PDF file"):
            extract_text_from_pdf(fake_file)
