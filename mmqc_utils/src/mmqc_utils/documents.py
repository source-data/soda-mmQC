"""Document conversion helpers."""

from __future__ import annotations

from html import escape
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import BinaryIO

import pypandoc
from pypdf import PdfReader

from .exceptions import DocumentConversionError, UnsupportedDocumentFormatError

PathLikeOrFile = str | Path | bytes | bytearray | BinaryIO
_PANDOC_EXTENSIONS = {".docx", ".rtf", ".odt", ".tex"}
_ALLOWED_EXTENSIONS = _PANDOC_EXTENSIONS | {".pdf"}


def _materialize_source(
    source: PathLikeOrFile,
    input_format: str | None,
) -> tuple[Path, bool, str | None]:
    if isinstance(source, Path):
        return source, False, input_format or source.suffix.lower().lstrip(".")
    if isinstance(source, str):
        path = Path(source)
        return path, False, input_format or path.suffix.lower().lstrip(".")

    raw = bytes(source) if isinstance(source, (bytes, bytearray)) else source.read()
    suffix = f".{input_format.lower().lstrip('.')}" if input_format else ""
    handle = NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        handle.write(raw)
        handle.flush()
    finally:
        handle.close()
    return Path(handle.name), True, input_format


def _pdf_to_html(path: Path) -> str:
    html_content = ["<html><body>"]
    with path.open("rb") as file_handle:
        reader = PdfReader(file_handle)
        for page_num, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            if not text.strip():
                continue
            text_html = escape(text).replace("\n", "<br/>")
            html_content.append(f"<div class='page' data-page='{page_num}'>")
            html_content.append(f"<p>{text_html}</p>")
            html_content.append("</div>")
    html_content.append("</body></html>")
    return "\n".join(html_content)


def document_to_html(
    source: PathLikeOrFile,
    *,
    input_format: str | None = None,
) -> str:
    """Convert a supported document to HTML."""
    path, is_temporary, detected_format = _materialize_source(source, input_format)
    try:
        suffix = f".{detected_format.lower().lstrip('.')}" if detected_format else path.suffix.lower()
        if suffix not in _ALLOWED_EXTENSIONS:
            raise UnsupportedDocumentFormatError(f"Unsupported document format: {suffix or '[unknown]'}")

        if suffix in _PANDOC_EXTENSIONS:
            result = pypandoc.convert_file(str(path), "html", format=suffix.lstrip("."))
            return str(result) if result else ""

        try:
            result = pypandoc.convert_file(str(path), "html", format="pdf")
            if result:
                return str(result)
        except Exception:
            pass
        return _pdf_to_html(path)
    except UnsupportedDocumentFormatError:
        raise
    except Exception as exc:
        raise DocumentConversionError(f"Failed to convert {path.name} to HTML") from exc
    finally:
        if is_temporary:
            path.unlink(missing_ok=True)
