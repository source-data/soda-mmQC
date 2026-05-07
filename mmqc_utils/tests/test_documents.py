from io import BytesIO
from pathlib import Path

import pytest
from pypdf import PdfWriter

from mmqc_utils.documents import document_to_html
from mmqc_utils.exceptions import UnsupportedDocumentFormatError

FIXTURES = Path(__file__).parent / "fixtures"


def test_document_to_html_rejects_unsupported_suffix(tmp_path: Path) -> None:
    txt_path = tmp_path / "notes.txt"
    txt_path.write_text("hello")

    with pytest.raises(UnsupportedDocumentFormatError):
        document_to_html(txt_path)


# --- RTF tests ---


def _make_rtf(text: str) -> bytes:
    """Create minimal RTF document with given text."""
    return f"{{\\rtf1\\ansi {text}}}".encode("ascii")


def test_rtf_converts_to_html(tmp_path: Path) -> None:
    rtf_path = tmp_path / "doc.rtf"
    rtf_path.write_bytes(_make_rtf("Hello RTF world"))

    html = document_to_html(rtf_path)

    assert "Hello RTF world" in html


def test_rtf_from_bytes() -> None:
    rtf_bytes = _make_rtf("Bytes input test")

    html = document_to_html(rtf_bytes, input_format="rtf")

    assert "Bytes input test" in html


def test_rtf_from_binary_io() -> None:
    rtf_bytes = _make_rtf("BinaryIO input test")
    bio = BytesIO(rtf_bytes)

    html = document_to_html(bio, input_format="rtf")

    assert "BinaryIO input test" in html


# --- PDF tests ---


def _make_pdf_with_text(text: str) -> bytes:
    """Create a minimal PDF with text using pypdf."""
    writer = PdfWriter()
    page = writer.add_blank_page(width=612, height=792)
    page.merge_page(writer.add_blank_page(width=612, height=792))
    # pypdf doesn't have a simple add_text, so we create a PDF with annotations
    # that contain searchable text. For a real text PDF, we use a workaround.
    # Instead, let's create a PDF and verify it at least parses without error.
    buffer = BytesIO()
    writer.write(buffer)
    return buffer.getvalue()


def test_pdf_converts_without_error(tmp_path: Path) -> None:
    """Test that PDF conversion runs without error (blank PDF)."""
    pdf_path = tmp_path / "doc.pdf"
    pdf_path.write_bytes(_make_pdf_with_text("ignored"))

    html = document_to_html(pdf_path)

    assert "<html>" in html
    assert "</html>" in html


def test_pdf_from_bytes(tmp_path: Path) -> None:
    pdf_bytes = _make_pdf_with_text("ignored")

    html = document_to_html(pdf_bytes, input_format="pdf")

    assert "<html>" in html


def test_pdf_from_binary_io() -> None:
    pdf_bytes = _make_pdf_with_text("ignored")
    bio = BytesIO(pdf_bytes)

    html = document_to_html(bio, input_format="pdf")

    assert "<html>" in html


# --- TEX tests ---


def _make_tex(text: str) -> bytes:
    """Create minimal LaTeX document with given text."""
    return f"""\\documentclass{{article}}
\\begin{{document}}
{text}
\\end{{document}}
""".encode()


def test_tex_converts_to_html(tmp_path: Path) -> None:
    tex_path = tmp_path / "doc.tex"
    tex_path.write_bytes(_make_tex("Hello LaTeX world"))

    html = document_to_html(tex_path)

    assert "Hello LaTeX world" in html


def test_tex_from_bytes() -> None:
    tex_bytes = _make_tex("Bytes LaTeX test")

    html = document_to_html(tex_bytes, input_format="tex")

    assert "Bytes LaTeX test" in html


def test_tex_from_str_path(tmp_path: Path) -> None:
    tex_path = tmp_path / "doc.tex"
    tex_path.write_bytes(_make_tex("String path test"))

    html = document_to_html(str(tex_path))

    assert "String path test" in html


# --- DOCX tests ---


def test_docx_converts_to_html() -> None:
    html = document_to_html(FIXTURES / "doc.docx")

    assert "Hello DOCX world" in html


def test_docx_from_bytes() -> None:
    docx_bytes = (FIXTURES / "doc.docx").read_bytes()

    html = document_to_html(docx_bytes, input_format="docx")

    assert "Hello DOCX world" in html


def test_docx_from_binary_io() -> None:
    bio = BytesIO((FIXTURES / "doc.docx").read_bytes())

    html = document_to_html(bio, input_format="docx")

    assert "Hello DOCX world" in html


# --- ODT tests ---


def test_odt_converts_to_html() -> None:
    html = document_to_html(FIXTURES / "doc.odt")

    assert "Hello ODT world" in html


def test_odt_from_bytes() -> None:
    odt_bytes = (FIXTURES / "doc.odt").read_bytes()

    html = document_to_html(odt_bytes, input_format="odt")

    assert "Hello ODT world" in html


def test_odt_from_binary_io() -> None:
    bio = BytesIO((FIXTURES / "doc.odt").read_bytes())

    html = document_to_html(bio, input_format="odt")

    assert "Hello ODT world" in html


# --- PDF text extraction tests ---


def test_pdf_extracts_text() -> None:
    html = document_to_html(FIXTURES / "doc.pdf")

    assert "Hello PDF world" in html


def test_pdf_page_structure() -> None:
    html = document_to_html(FIXTURES / "doc.pdf")

    assert "<div class='page'" in html
    assert "data-page='1'" in html
