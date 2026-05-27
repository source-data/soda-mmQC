"""HTML and plain-text normalization helpers."""

from __future__ import annotations

import re
from html import unescape
from html.parser import HTMLParser
from unicodedata import category

_SEPARATING_TAGS = frozenset(
    {
        "address",
        "article",
        "aside",
        "blockquote",
        "body",
        "br",
        "caption",
        "dd",
        "details",
        "dialog",
        "div",
        "dl",
        "dt",
        "fieldset",
        "figcaption",
        "figure",
        "footer",
        "form",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "header",
        "hgroup",
        "hr",
        "html",
        "li",
        "main",
        "nav",
        "ol",
        "p",
        "pre",
        "section",
        "summary",
        "table",
        "tbody",
        "td",
        "tfoot",
        "th",
        "thead",
        "tr",
        "ul",
    }
)
"""HTML tags that should separate adjacent visible text."""

_IGNORED_CONTENT_TAGS = frozenset({"head", "script", "style", "template"})
"""HTML tags whose content is not browser-visible document text."""

_WHITESPACE_PATTERN = re.compile(r"\s+")
"""Pattern to match any sequence of whitespace characters."""


def _remove_control_characters(text: str) -> str:
    """Remove non-whitespace control and format characters from visible text."""
    return "".join(ch for ch in text if ch.isspace() or category(ch) not in {"Cc", "Cf"})


class _PlainTextParser(HTMLParser):
    """Extract browser-visible text while preserving inline text joins."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._ignored_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self._handle_tag(tag)

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in _IGNORED_CONTENT_TAGS:
            if self._ignored_depth:
                self._ignored_depth -= 1
            return
        self._handle_tag(tag)

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() in _IGNORED_CONTENT_TAGS:
            return
        self._handle_tag(tag)

    def handle_data(self, data: str) -> None:
        if self._ignored_depth:
            return
        if data:
            self.parts.append(_remove_control_characters(data))

    def _handle_tag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in _IGNORED_CONTENT_TAGS:
            self._ignored_depth += 1
            return
        if self._ignored_depth:
            return
        if tag in _SEPARATING_TAGS:
            self.parts.append(" ")


def html_to_text(html: str) -> str:
    """Strip HTML tags and normalize whitespace."""
    if not html:
        return ""

    # Decode first so escaped pandoc fragments like "&lt;p&gt;..." are handled the same way as real tags.
    parser = _PlainTextParser()
    parser.feed(unescape(html))
    parser.close()
    text = "".join(parser.parts)

    # Normalize multiple consecutive whitespace to a single space and trim leading/trailing whitespace.
    text = _WHITESPACE_PATTERN.sub(" ", text)
    return text.strip()


def compute_plain_text(html: str) -> str:
    """
    Compute the normalized plain-text representation of HTML content.

    Specifically:
    - HTML tags are stripped. Block-level tags are replaced by a single space, inline
      tags are removed.
    - All whitespace runs (including the spaces left by tag removal) are collapsed to a
      single space and the result is stripped.
    """
    return html_to_text(html)
