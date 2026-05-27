"""Reusable conversion utilities for MMQC projects."""

from .documents import document_to_html
from .images import compress_to_bounded_jpeg, convert_to_bounded_jpeg
from .quote import AlignmentGap, AlignmentStatus, CharInterval, QuoteAlignment, align_quote
from .text import compute_plain_text, html_to_text

__all__ = [
    "AlignmentGap",
    "AlignmentStatus",
    "CharInterval",
    "QuoteAlignment",
    "align_quote",
    "compress_to_bounded_jpeg",
    "compute_plain_text",
    "convert_to_bounded_jpeg",
    "document_to_html",
    "html_to_text",
]
