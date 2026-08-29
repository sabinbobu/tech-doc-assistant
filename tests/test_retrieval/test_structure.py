"""
Tests for document structure lookups.

We mock load_document_outline/list_indexed_documents, same pattern as the
rest of tests/test_retrieval/ — this tests the rendering/lookup LOGIC, not
Chroma or filesystem persistence (vector_store's own tests cover the
save/load round-trip).
"""

from unittest.mock import patch

from src.ingestion.models import OutlineEntry
from src.retrieval.structure import find_sections, render_toc, render_toc_for_all_documents


def _entry(level: int, title: str, page: int) -> OutlineEntry:
    return OutlineEntry(level=level, title=title, page=page)


class TestRenderToc:
    def test_renders_nested_outline_with_indentation(self):
        outline = [
            _entry(1, "1 Overview", 1),
            _entry(1, "2 Block Diagram", 3),
            _entry(2, "2.1 Details", 3),
        ]
        with patch("src.retrieval.structure.load_document_outline", return_value=outline):
            result = render_toc("ds.pdf")

        assert result is not None
        assert "ds.pdf:" in result
        assert "1 Overview (page 1)" in result
        assert "2.1 Details (page 3)" in result
        # The nested entry must be indented further than its parent.
        parent_indent = result.index("2 Block Diagram")
        child_indent = result.index("2.1 Details")
        parent_line_start = result.rfind("\n", 0, parent_indent) + 1
        child_line_start = result.rfind("\n", 0, child_indent) + 1
        assert (child_indent - child_line_start) > (parent_indent - parent_line_start)

    def test_returns_none_when_document_has_no_outline(self):
        with patch("src.retrieval.structure.load_document_outline", return_value=None):
            assert render_toc("scanned.pdf") is None

    def test_returns_none_for_empty_outline(self):
        with patch("src.retrieval.structure.load_document_outline", return_value=[]):
            assert render_toc("empty.pdf") is None


class TestRenderTocForAllDocuments:
    def test_combines_multiple_documents(self):
        with (
            patch(
                "src.retrieval.structure.list_indexed_documents",
                return_value=[{"filename": "a.pdf"}, {"filename": "b.pdf"}],
            ),
            patch(
                "src.retrieval.structure.load_document_outline",
                side_effect=lambda name: [_entry(1, f"Chapter in {name}", 1)],
            ),
        ):
            result = render_toc_for_all_documents()

        assert "a.pdf:" in result
        assert "b.pdf:" in result

    def test_skips_documents_with_no_outline_instead_of_erroring(self):
        with (
            patch(
                "src.retrieval.structure.list_indexed_documents",
                return_value=[{"filename": "a.pdf"}, {"filename": "scanned.pdf"}],
            ),
            patch(
                "src.retrieval.structure.load_document_outline",
                side_effect=lambda name: None if name == "scanned.pdf" else [_entry(1, "X", 1)],
            ),
        ):
            result = render_toc_for_all_documents()

        assert "a.pdf:" in result
        assert "scanned.pdf" not in result

    def test_respects_source_filenames_filter(self):
        with (
            patch(
                "src.retrieval.structure.list_indexed_documents",
                return_value=[{"filename": "a.pdf"}, {"filename": "b.pdf"}],
            ),
            patch(
                "src.retrieval.structure.load_document_outline",
                side_effect=lambda name: [_entry(1, "X", 1)],
            ),
        ):
            result = render_toc_for_all_documents(source_filenames=["a.pdf"])

        assert "a.pdf:" in result
        assert "b.pdf:" not in result

    def test_no_documents_have_outlines_returns_a_message_not_empty_string(self):
        with (
            patch(
                "src.retrieval.structure.list_indexed_documents",
                return_value=[{"filename": "scanned.pdf"}],
            ),
            patch("src.retrieval.structure.load_document_outline", return_value=None),
        ):
            result = render_toc_for_all_documents()

        assert result  # not empty/falsy
        assert "scanned.pdf" not in result


class TestFindSections:
    def test_case_insensitive_substring_match(self):
        outline = [
            _entry(1, "9 Diagnosis", 41),
            _entry(2, "9.3 Diagnosis in OFF state", 45),
            _entry(1, "10 Package", 55),
        ]
        with patch("src.retrieval.structure.load_document_outline", return_value=outline):
            result = find_sections("ds.pdf", "DIAGNOS")

        assert len(result) == 2
        assert result[0].title == "9 Diagnosis"

    def test_no_match_returns_empty_list_not_none(self):
        with patch(
            "src.retrieval.structure.load_document_outline",
            return_value=[_entry(1, "1 Overview", 1)],
        ):
            result = find_sections("ds.pdf", "nonexistent term")

        assert result == []

    def test_document_with_no_outline_returns_empty_list(self):
        with patch("src.retrieval.structure.load_document_outline", return_value=None):
            assert find_sections("scanned.pdf", "anything") == []
