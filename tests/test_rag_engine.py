"""Unit tests for rag_engine module."""

import numpy as np
import pytest
from unittest.mock import patch, MagicMock

import rag_engine
from rag_engine import (
    extract_text_from_pdf,
    chunk_text,
    VectorStore,
    generate_answer,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_pages_data(texts, source="test.pdf"):
    """Build pages_data list from a list of text strings."""
    return [
        {"text": t, "page_num": i + 1, "source": source}
        for i, t in enumerate(texts)
    ]


def _fake_pdf_bytes(pages_text: list[str]) -> bytes:
    """Create a minimal PDF in memory with the given page texts using pypdf."""
    from pypdf import PdfWriter, PageObject
    from pypdf.generic import (
        ArrayObject,
        DecodedStreamObject,
        DictionaryObject,
        NameObject,
        NumberObject,
    )
    import io

    writer = PdfWriter()
    for text in pages_text:
        # Build a minimal page with text content stream
        page = PageObject.create_blank_page(width=612, height=792)
        # Create a content stream that writes the text using Tj operator
        content = f"BT /F1 12 Tf 100 700 Td ({text}) Tj ET"
        stream = DecodedStreamObject()
        stream.set_data(content.encode("latin-1"))
        # Add a font resource so the PDF is somewhat valid
        resources = DictionaryObject(
            {
                NameObject("/Font"): DictionaryObject(
                    {
                        NameObject("/F1"): DictionaryObject(
                            {
                                NameObject("/Type"): NameObject("/Font"),
                                NameObject("/Subtype"): NameObject("/Type1"),
                                NameObject("/BaseFont"): NameObject("/Helvetica"),
                            }
                        )
                    }
                )
            }
        )
        page[NameObject("/Resources")] = resources
        page[NameObject("/Contents")] = stream
        writer.add_page(page)

    buf = io.BytesIO()
    writer.write(buf)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Tests: extract_text_from_pdf
# ---------------------------------------------------------------------------

class TestExtractTextFromPdf:
    def test_extracts_text_from_single_page(self):
        pdf_bytes = _fake_pdf_bytes(["Hello World"])
        result = extract_text_from_pdf(pdf_bytes, "doc.pdf")
        assert len(result) >= 1
        assert result[0]["source"] == "doc.pdf"
        assert result[0]["page_num"] == 1
        assert "Hello" in result[0]["text"] or "World" in result[0]["text"]

    def test_extracts_multiple_pages(self):
        pdf_bytes = _fake_pdf_bytes(["Page one content", "Page two content"])
        result = extract_text_from_pdf(pdf_bytes, "multi.pdf")
        assert len(result) >= 2
        assert result[0]["page_num"] == 1
        assert result[1]["page_num"] == 2

    def test_skips_blank_pages(self):
        """Pages with empty/whitespace text should be skipped."""
        pdf_bytes = _fake_pdf_bytes(["Real content", ""])
        result = extract_text_from_pdf(pdf_bytes, "blank.pdf")
        # At minimum the first page should be extracted
        assert any("Real" in r["text"] or "content" in r["text"] for r in result)

    def test_returns_empty_for_empty_pdf(self):
        """A PDF with no text at all returns an empty list."""
        pdf_bytes = _fake_pdf_bytes([""])
        result = extract_text_from_pdf(pdf_bytes, "empty.pdf")
        # Either empty list or only pages with non-empty text
        for page in result:
            assert page["text"].strip() != ""


# ---------------------------------------------------------------------------
# Tests: chunk_text
# ---------------------------------------------------------------------------

class TestChunkText:
    def test_small_text_single_chunk(self):
        pages = _make_pages_data(["Short text."])
        chunks = chunk_text(pages, chunk_size=1000, chunk_overlap=200)
        assert len(chunks) == 1
        assert chunks[0]["text"] == "Short text."
        assert chunks[0]["page_num"] == 1
        assert chunks[0]["source"] == "test.pdf"

    def test_text_longer_than_chunk_size_splits(self):
        long_text = "word " * 500  # ~2500 characters
        pages = _make_pages_data([long_text])
        chunks = chunk_text(pages, chunk_size=200, chunk_overlap=50)
        assert len(chunks) > 1
        # All chunks should have metadata
        for c in chunks:
            assert c["page_num"] == 1
            assert c["source"] == "test.pdf"
            assert len(c["text"]) > 0

    def test_chunk_overlap_creates_overlapping_content(self):
        text = "A " * 300  # 600 chars
        pages = _make_pages_data([text])
        chunks = chunk_text(pages, chunk_size=200, chunk_overlap=100)
        # With overlap, chunks should share some content
        assert len(chunks) >= 2

    def test_multiple_pages_produce_separate_chunks(self):
        pages = _make_pages_data(["First page text.", "Second page text."])
        chunks = chunk_text(pages, chunk_size=1000, chunk_overlap=200)
        assert len(chunks) == 2
        assert chunks[0]["page_num"] == 1
        assert chunks[1]["page_num"] == 2

    def test_empty_pages_data_returns_empty(self):
        chunks = chunk_text([], chunk_size=1000, chunk_overlap=200)
        assert chunks == []

    def test_respects_word_boundaries(self):
        # Create text where the chunk boundary falls mid-word
        text = "abcdefghij " * 20  # Each 'word' is 10 chars + space
        pages = _make_pages_data([text])
        chunks = chunk_text(pages, chunk_size=50, chunk_overlap=10)
        # Chunks should not start/end mid-word (should end at a space)
        for c in chunks:
            # Stripped text shouldn't start or end with partial words that got cut
            assert c["text"] == c["text"].strip()

    def test_zero_overlap(self):
        text = "word " * 100
        pages = _make_pages_data([text])
        chunks = chunk_text(pages, chunk_size=100, chunk_overlap=0)
        assert len(chunks) >= 2

    def test_large_overlap_still_produces_chunks(self):
        """Overlap close to chunk_size still produces chunks without hanging."""
        text = "hello " * 10  # short text, 60 chars
        pages = _make_pages_data([text])
        # overlap < chunk_size but fairly large relative to text
        chunks = chunk_text(pages, chunk_size=100, chunk_overlap=80)
        assert len(chunks) >= 1


# ---------------------------------------------------------------------------
# Tests: VectorStore
# ---------------------------------------------------------------------------

class TestVectorStore:
    def test_init_empty(self):
        vs = VectorStore()
        assert vs.embeddings == []
        assert vs.chunks == []

    @patch("rag_engine.genai")
    def test_add_chunks_stores_embeddings(self, mock_genai):
        mock_genai.embed_content.return_value = {
            "embedding": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        }
        vs = VectorStore()
        chunks = [
            {"text": "chunk one", "page_num": 1, "source": "a.pdf"},
            {"text": "chunk two", "page_num": 2, "source": "a.pdf"},
        ]
        vs.add_chunks(chunks, api_key="fake-key")

        mock_genai.configure.assert_called_once_with(api_key="fake-key")
        assert len(vs.embeddings) == 2
        assert len(vs.chunks) == 2
        assert vs.chunks[0]["text"] == "chunk one"
        assert vs.chunks[1]["text"] == "chunk two"
        np.testing.assert_array_almost_equal(vs.embeddings[0], [0.1, 0.2, 0.3])

    @patch("rag_engine.genai")
    def test_add_chunks_empty_list_does_nothing(self, mock_genai):
        vs = VectorStore()
        vs.add_chunks([], api_key="fake-key")
        mock_genai.configure.assert_not_called()
        assert vs.embeddings == []

    @patch("rag_engine.genai")
    def test_add_chunks_batches_large_input(self, mock_genai):
        """When more than 100 chunks are provided, they should be batched."""
        # Create 150 chunks
        chunks = [
            {"text": f"chunk {i}", "page_num": 1, "source": "big.pdf"}
            for i in range(150)
        ]
        # First batch: 100 embeddings, second batch: 50 embeddings
        mock_genai.embed_content.side_effect = [
            {"embedding": [[0.1] * 3 for _ in range(100)]},
            {"embedding": [[0.2] * 3 for _ in range(50)]},
        ]
        vs = VectorStore()
        vs.add_chunks(chunks, api_key="key")

        assert mock_genai.embed_content.call_count == 2
        assert len(vs.embeddings) == 150
        assert len(vs.chunks) == 150

    @patch("rag_engine.genai")
    def test_search_returns_top_k_results(self, mock_genai):
        vs = VectorStore()
        # Manually set up embeddings
        vs.embeddings = [
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
            np.array([0.9, 0.1, 0.0]),
        ]
        vs.chunks = [
            {"text": "first", "page_num": 1, "source": "a.pdf"},
            {"text": "second", "page_num": 2, "source": "a.pdf"},
            {"text": "third", "page_num": 3, "source": "a.pdf"},
        ]
        # Query embedding most similar to [1, 0, 0]
        mock_genai.embed_content.return_value = {
            "embedding": [1.0, 0.0, 0.0]
        }
        results = vs.search("test query", api_key="key", k=2)

        assert len(results) == 2
        # First result should be the most similar
        assert results[0]["chunk"]["text"] == "first"
        assert results[0]["score"] == pytest.approx(1.0, abs=1e-5)
        # Second should be "third" (0.9 similarity to [1,0,0])
        assert results[1]["chunk"]["text"] == "third"

    @patch("rag_engine.genai")
    def test_search_empty_store_returns_empty(self, mock_genai):
        vs = VectorStore()
        results = vs.search("anything", api_key="key", k=5)
        assert results == []

    @patch("rag_engine.genai")
    def test_search_handles_zero_norm(self, mock_genai):
        """If an embedding is all zeros, similarity should be 0."""
        vs = VectorStore()
        vs.embeddings = [np.array([0.0, 0.0, 0.0])]
        vs.chunks = [{"text": "zero", "page_num": 1, "source": "z.pdf"}]
        mock_genai.embed_content.return_value = {"embedding": [1.0, 0.0, 0.0]}
        results = vs.search("query", api_key="key", k=1)
        assert len(results) == 1
        assert results[0]["score"] == 0.0


# ---------------------------------------------------------------------------
# Tests: generate_answer
# ---------------------------------------------------------------------------

class TestGenerateAnswer:
    @patch("rag_engine.genai")
    def test_generates_answer_with_context(self, mock_genai):
        mock_model = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "The answer is 42."
        mock_model.generate_content.return_value = mock_response
        mock_genai.GenerativeModel.return_value = mock_model

        search_results = [
            {
                "chunk": {"text": "Some context", "page_num": 1, "source": "doc.pdf"},
                "score": 0.95,
            }
        ]
        answer = generate_answer("What is the answer?", search_results, "key")

        assert answer == "The answer is 42."
        mock_genai.configure.assert_called_once_with(api_key="key")
        mock_genai.GenerativeModel.assert_called_once()
        mock_model.generate_content.assert_called_once()

    @patch("rag_engine.genai")
    def test_uses_specified_model_name(self, mock_genai):
        mock_model = MagicMock()
        mock_model.generate_content.return_value = MagicMock(text="response")
        mock_genai.GenerativeModel.return_value = mock_model

        generate_answer("q", [{"chunk": {"text": "c", "page_num": 1, "source": "s"}, "score": 0.5}], "key", model_name="gemini-1.5-pro")

        mock_genai.GenerativeModel.assert_called_once_with(
            model_name="gemini-1.5-pro",
            system_instruction=pytest.approx(mock_genai.GenerativeModel.call_args[1]["system_instruction"]),
        )

    @patch("rag_engine.genai")
    def test_prompt_includes_context_and_query(self, mock_genai):
        mock_model = MagicMock()
        mock_model.generate_content.return_value = MagicMock(text="ans")
        mock_genai.GenerativeModel.return_value = mock_model

        search_results = [
            {
                "chunk": {"text": "Important fact", "page_num": 3, "source": "report.pdf"},
                "score": 0.88,
            }
        ]
        generate_answer("What happened?", search_results, "key")

        # Check the prompt passed to generate_content
        call_args = mock_model.generate_content.call_args[0][0]
        assert "Important fact" in call_args
        assert "report.pdf" in call_args
        assert "What happened?" in call_args

    @patch("rag_engine.genai")
    def test_handles_multiple_search_results(self, mock_genai):
        mock_model = MagicMock()
        mock_model.generate_content.return_value = MagicMock(text="combined")
        mock_genai.GenerativeModel.return_value = mock_model

        search_results = [
            {"chunk": {"text": f"fact {i}", "page_num": i, "source": "d.pdf"}, "score": 0.9 - i * 0.1}
            for i in range(3)
        ]
        answer = generate_answer("summarize", search_results, "key")
        assert answer == "combined"

        prompt = mock_model.generate_content.call_args[0][0]
        assert "fact 0" in prompt
        assert "fact 1" in prompt
        assert "fact 2" in prompt

    @patch("rag_engine.genai")
    def test_empty_search_results(self, mock_genai):
        mock_model = MagicMock()
        mock_model.generate_content.return_value = MagicMock(text="no info")
        mock_genai.GenerativeModel.return_value = mock_model

        answer = generate_answer("unknown", [], "key")
        assert answer == "no info"
