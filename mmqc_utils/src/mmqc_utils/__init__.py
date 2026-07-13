"""Reusable conversion utilities for MMQC projects."""

from .documents import document_to_html
from .html_cleanup import postprocess_html
from .images import compress_to_bounded_jpeg, convert_to_bounded_jpeg
from .quote import (
    AlignmentGap,
    AlignmentStatus,
    CharInterval,
    QuoteAlignment,
    align_quote,
    align_quotes,
    render_alignment_view,
    render_coverage_view,
)
from .text import compute_plain_text, html_to_text

__all__ = [
    "AlignmentGap",
    "AlignmentStatus",
    "CharInterval",
    "QuoteAlignment",
    "align_quote",
    "align_quotes",
    "compress_to_bounded_jpeg",
    "compute_plain_text",
    "convert_to_bounded_jpeg",
    "document_to_html",
    "html_to_text",
    "postprocess_html",
    "render_alignment_view",
    "render_coverage_view",
]
