"""HTML and plain-text normalization helpers."""

from __future__ import annotations

import re
from html import unescape

_BLOCK_TAG_PATTERN = re.compile(
    r"</?(?:p|br|div|h[1-6]|li|tr|blockquote|ul|ol|table|section|article)(?:\s[^>]*)?/?>",
    flags=re.IGNORECASE,
)
"""Pattern to match block-level HTML tags."""

_TAG_PATTERN = re.compile(r"<[^>]+>")
"""Pattern to match any HTML tag."""

_WHITESPACE_PATTERN = re.compile(r"\s+")
"""Pattern to match any sequence of whitespace characters."""


def html_to_text(html: str) -> str:
    """Strip HTML tags and normalize whitespace."""
    if not html:
        return ""

    # Unescape HTML entities first (e.g., &amp; -> &)
    text = unescape(html)

    # Replace block-level tags with spaces. This preserves the usual visual separation of block elements in the
    # resulting text.
    text = _BLOCK_TAG_PATTERN.sub(" ", text)

    # Strip remaining HTML tags (inline elements like <i>, <b>, <em>, <strong>, <sub>, <sup>, etc.).
    text = _TAG_PATTERN.sub("", text)

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
