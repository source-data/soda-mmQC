"""Tests for WordExample document loading."""

from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from mmqc_utils.exceptions import DocumentConversionError
from soda_mmqc.core.examples import EXAMPLE_FACTORY, WordExample

MMQC_UTILS_FIXTURES = (
    Path(__file__).resolve().parents[1] / "mmqc_utils" / "tests" / "fixtures"
)
DOCX_FIXTURE = MMQC_UTILS_FIXTURES / "doc.docx"
FAKE_HTML = "<html><body><p>Test manuscript</p></body></html>"


class TestWordExample(unittest.TestCase):
    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp())
        self.example_dir = self.temp_dir / "test-doc"
        self.content_dir = self.example_dir / "content"
        self.content_dir.mkdir(parents=True)

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def _copy_fixture_docx(self, name: str = "manuscript.docx") -> Path:
        if not DOCX_FIXTURE.exists():
            target = self.content_dir / name
            target.write_bytes(b"PK\x03\x04")
            return target
        target = self.content_dir / name
        shutil.copy(DOCX_FIXTURE, target)
        return target

    @patch("soda_mmqc.core.examples.document_to_html", return_value=FAKE_HTML)
    def test_load_from_source_returns_html(self, mock_document_to_html):
        docx_path = self._copy_fixture_docx()
        with patch("soda_mmqc.core.examples.EXAMPLES_DIR", self.temp_dir):
            example = WordExample("test-doc")
            example.load_from_source()

        mock_document_to_html.assert_called_once_with(docx_path)
        self.assertEqual(example.doc_id, "test-doc")
        self.assertEqual(example.content, FAKE_HTML)
        self.assertEqual(example.word_file_path, docx_path)

    @patch("soda_mmqc.core.examples.document_to_html", return_value=FAKE_HTML)
    def test_factory_create_word_example(self, mock_document_to_html):
        self._copy_fixture_docx()
        with patch("soda_mmqc.core.examples.EXAMPLES_DIR", self.temp_dir):
            example = EXAMPLE_FACTORY.create("test-doc", "word")

        self.assertEqual(example.example_class_name, "word")
        self.assertEqual(example.doc_id, "test-doc")
        self.assertEqual(example.content, FAKE_HTML)
        mock_document_to_html.assert_called_once()

    def test_raises_when_no_docx(self):
        with patch("soda_mmqc.core.examples.EXAMPLES_DIR", self.temp_dir):
            example = WordExample("test-doc")
            with self.assertRaises(FileNotFoundError):
                example.load_from_source()

    @patch("soda_mmqc.core.examples.document_to_html")
    def test_raises_when_multiple_docx(self, mock_document_to_html):
        self._copy_fixture_docx("a.docx")
        self._copy_fixture_docx("b.docx")

        with patch("soda_mmqc.core.examples.EXAMPLES_DIR", self.temp_dir):
            example = WordExample("test-doc")
            with self.assertRaises(ValueError) as ctx:
                example.load_from_source()

        self.assertIn("Expected exactly one", str(ctx.exception))
        mock_document_to_html.assert_not_called()

    @patch("soda_mmqc.core.examples.document_to_html", return_value=FAKE_HTML)
    def test_prepare_model_input_includes_html_content(self, _mock_document_to_html):
        self._copy_fixture_docx()
        with patch("soda_mmqc.core.examples.EXAMPLES_DIR", self.temp_dir):
            example = EXAMPLE_FACTORY.create("test-doc", "word")

        model_input = example.prepare_model_input("Find URLs.")
        text = model_input["content"][0]["text"]
        self.assertEqual(text, f"Find URLs.\n\nContent:\n{FAKE_HTML}")

    @patch("soda_mmqc.core.examples.document_to_html", return_value=FAKE_HTML)
    def test_content_hash_is_stable(self, _mock_document_to_html):
        self._copy_fixture_docx()
        with patch("soda_mmqc.core.examples.EXAMPLES_DIR", self.temp_dir):
            example = EXAMPLE_FACTORY.create("test-doc", "word")

        first_hash = example.get_content_hash()
        second_hash = example.get_content_hash()
        self.assertEqual(first_hash, second_hash)
        self.assertEqual(len(first_hash), 64)

    @patch(
        "soda_mmqc.core.examples.document_to_html",
        side_effect=DocumentConversionError("Failed to convert manuscript.docx to HTML"),
    )
    def test_raises_when_conversion_fails(self, _mock_document_to_html):
        self._copy_fixture_docx()
        with patch("soda_mmqc.core.examples.EXAMPLES_DIR", self.temp_dir):
            example = WordExample("test-doc")
            with self.assertRaises(ValueError) as ctx:
                example.load_from_source()

        self.assertIn("Error converting Word file", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
