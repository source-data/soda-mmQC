"""Literal quote scoring and span matching helpers."""

from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
from enum import Enum
from html import escape as _html_escape

from rapidfuzz import fuzz

Span = tuple[int, int]
"""Half-open character span."""


class AlignmentStatus(Enum):
    """How a quote aligned to the input text.

    MATCH_EXACT   — verbatim contiguous substring; ``char_intervals`` has one entry,
                    ``score`` is 1.0.
    MATCH_GREATER — all quote chars found across ordered word-boundary spans in
                    ``text``; source has extra text between spans (``source_gaps``);
                    ``score`` is 1.0.
    MATCH_LESSER  — greedy span search matched only a subset of quote chars
                    (skipped chars recorded in ``quote_gaps``);
                    ``score`` = matched chars / len(quote).
    MATCH_FUZZY   — no literal alignment; RapidFuzz partial-ratio fallback;
                    ``char_intervals`` is None, ``approximate_char_intervals`` is set
                    when the score clears the confidence threshold.
    """

    MATCH_EXACT = "match_exact"
    MATCH_GREATER = "match_greater"
    MATCH_LESSER = "match_lesser"
    MATCH_FUZZY = "match_fuzzy"


@dataclass(frozen=True)
class CharInterval:
    """Half-open character interval."""

    start_pos: int | None = None
    end_pos: int | None = None


@dataclass(frozen=True)
class AlignmentGap:
    """Unmatched text interval on one side of an alignment."""

    text: str
    char_interval: CharInterval


@dataclass(frozen=True)
class QuoteAlignment:
    """Structured quote alignment result.

    Attributes:
        score: 1.0 for MATCH_EXACT / MATCH_GREATER; matched_chars / len(quote) for
            MATCH_LESSER; RapidFuzz partial_ratio / 100 for MATCH_FUZZY.
        char_intervals: exact half-open spans in ``text``; populated for
            MATCH_EXACT, MATCH_GREATER, and MATCH_LESSER; None for MATCH_FUZZY.
        approximate_char_intervals: best-effort region in ``text`` for MATCH_FUZZY
            matches that clear the confidence threshold; None otherwise.
        matched_text: concatenation of the matched text spans; set for MATCH_EXACT
            and MATCH_GREATER/MATCH_LESSER greedy matches; None for MATCH_FUZZY.
        source_gaps: text present in ``text`` between matched spans but absent from
            the quote; non-None for MATCH_GREATER and some MATCH_FUZZY results.
        quote_gaps: text present in ``quote`` that was skipped to produce a literal
            match; non-None for MATCH_LESSER.
        occurrence_count: number of exact occurrences of ``quote`` in ``text``;
            set for MATCH_EXACT only; None otherwise.
    """

    quote: str
    score: float
    char_intervals: list[CharInterval] | None = None
    approximate_char_intervals: list[CharInterval] | None = None
    alignment_status: AlignmentStatus | None = None
    matched_text: str | None = None
    source_gaps: list[AlignmentGap] | None = None
    quote_gaps: list[AlignmentGap] | None = None
    occurrence_count: int | None = None


@dataclass(frozen=True)
class _SegmentMatch:
    text_span: Span
    quote_span: Span


@dataclass(frozen=True)
class _GreedyMatch:
    segments: list[_SegmentMatch]
    matched_quote_chars: int
    skipped_quote_chars: int


@dataclass(frozen=True)
class _SearchState:
    segments: list[_SegmentMatch]
    quote_pos: int
    search_start: int
    matched_quote_chars: int
    skipped_quote_chars: int


@dataclass(frozen=True)
class _FuzzyMatch:
    score: float
    text_span: Span
    segments: list[_SegmentMatch]


def _find_exact_match(text: str, pattern: str, start: int = 0) -> Span | None:
    index = text.find(pattern, start)
    if index < 0:
        return None
    return index, index + len(pattern)


def _is_word_char(ch: str) -> bool:
    return ch.isalnum() or ch == "_"


def _has_word_boundaries(s: str, start: int, end: int) -> bool:
    start_ok = start == 0 or not (_is_word_char(s[start - 1]) and _is_word_char(s[start]))
    end_ok = end == len(s) or not (_is_word_char(s[end - 1]) and _is_word_char(s[end]))
    return start_ok and end_ok


def _find_occurrences(text: str, substring: str, start: int, limit: int) -> list[int]:
    indexes: list[int] = []
    index = text.find(substring, start)
    while index >= 0 and len(indexes) < limit:
        indexes.append(index)
        index = text.find(substring, index + 1)
    return indexes


def _candidate_segments(
    text: str,
    quote: str,
    *,
    quote_pos: int,
    search_start: int,
    min_match_len: int,
    max_gap_chars: int,
    length_slack: int = 5,
    max_occurrences_per_length: int = 12,
) -> list[_SegmentMatch]:
    candidates: list[_SegmentMatch] = []
    for skip in range(max_gap_chars + 1):
        quote_start = quote_pos + skip
        if quote_start >= len(quote):
            break

        longest_len: int | None = None
        for quote_end in range(len(quote), quote_start + min_match_len - 1, -1):
            match_len = quote_end - quote_start
            if longest_len is not None and longest_len - match_len > length_slack:
                break

            if not _has_word_boundaries(quote, quote_start, quote_end):
                continue

            substring = quote[quote_start:quote_end]
            occurrences = _find_occurrences(
                text,
                substring,
                search_start,
                limit=max_occurrences_per_length,
            )
            valid_occurrences = [
                text_start
                for text_start in occurrences
                if _has_word_boundaries(text, text_start, text_start + match_len)
            ]
            if not valid_occurrences:
                continue

            if longest_len is None:
                longest_len = match_len

            candidates.extend(
                _SegmentMatch(
                    text_span=(text_start, text_start + match_len),
                    quote_span=(quote_start, quote_end),
                )
                for text_start in valid_occurrences
            )
    return candidates


def _source_gap_chars_from_segments(segments: list[_SegmentMatch]) -> int:
    total = 0
    for left, right in zip(segments, segments[1:], strict=False):
        total += max(0, right.text_span[0] - left.text_span[1])
    return total


def _state_key(state: _SearchState) -> tuple[int, int, int, int, int]:
    return (
        state.matched_quote_chars,
        -state.skipped_quote_chars,
        -_source_gap_chars_from_segments(state.segments),
        -len(state.segments),
        -state.search_start,
    )


def _find_greedy_quote_spans(
    text: str,
    quote: str,
    *,
    min_match_len: int,
    max_gap_chars: int,
    max_segments: int,
    search_start: int = 0,
) -> _GreedyMatch | None:
    beam_size = 40
    states = [
        _SearchState(
            segments=[],
            quote_pos=0,
            search_start=search_start,
            matched_quote_chars=0,
            skipped_quote_chars=0,
        )
    ]
    completed: list[_SearchState] = []

    for _ in range(max_segments + 1):
        next_states: list[_SearchState] = []
        for state in states:
            if state.quote_pos >= len(quote):
                completed.append(state)
                continue

            candidates = _candidate_segments(
                text,
                quote,
                quote_pos=state.quote_pos,
                search_start=state.search_start,
                min_match_len=min_match_len,
                max_gap_chars=max_gap_chars,
            )
            if not candidates:
                remaining_quote_chars = len(quote) - state.quote_pos
                if remaining_quote_chars <= max_gap_chars and state.segments:
                    completed.append(
                        _SearchState(
                            segments=state.segments,
                            quote_pos=len(quote),
                            search_start=state.search_start,
                            matched_quote_chars=state.matched_quote_chars,
                            skipped_quote_chars=state.skipped_quote_chars + remaining_quote_chars,
                        )
                    )
                continue

            for candidate in candidates:
                if len(state.segments) + 1 > max_segments:
                    continue

                skipped_quote_chars = candidate.quote_span[0] - state.quote_pos
                next_state = _SearchState(
                    segments=[*state.segments, candidate],
                    quote_pos=candidate.quote_span[1],
                    search_start=candidate.text_span[1],
                    matched_quote_chars=state.matched_quote_chars + candidate.quote_span[1] - candidate.quote_span[0],
                    skipped_quote_chars=state.skipped_quote_chars + skipped_quote_chars,
                )
                if next_state.quote_pos >= len(quote):
                    completed.append(next_state)
                else:
                    next_states.append(next_state)

        if not next_states:
            break
        states = sorted(next_states, key=_state_key, reverse=True)[:beam_size]

    if not completed:
        return None
    best_state = max(completed, key=_state_key)
    return _GreedyMatch(
        segments=best_state.segments,
        matched_quote_chars=best_state.matched_quote_chars,
        skipped_quote_chars=best_state.skipped_quote_chars,
    )


def _char_intervals(spans: list[Span]) -> list[CharInterval]:
    return [CharInterval(start_pos=start, end_pos=end) for start, end in spans]


def _find_fuzzy_match(
    text: str,
    quote: str,
    *,
    min_score: float,
    min_block_len: int,
) -> _FuzzyMatch | None:
    alignment = fuzz.partial_ratio_alignment(quote, text)
    if alignment is None:
        return None
    score = float(alignment.score) / 100.0
    if score < min_score:
        return None

    quote_start = int(alignment.src_start)
    quote_end = int(alignment.src_end)
    text_start = int(alignment.dest_start)
    text_end = int(alignment.dest_end)
    quote_window = quote[quote_start:quote_end]
    text_window = text[text_start:text_end]
    matcher = SequenceMatcher(None, quote_window, text_window, autojunk=False)

    segments: list[_SegmentMatch] = []
    for block in matcher.get_matching_blocks():
        if block.size < min_block_len:
            continue
        segments.append(
            _SegmentMatch(
                text_span=(text_start + block.b, text_start + block.b + block.size),
                quote_span=(quote_start + block.a, quote_start + block.a + block.size),
            )
        )

    return _FuzzyMatch(
        score=score,
        text_span=(text_start, text_end),
        segments=segments,
    )


def _non_empty_gaps(gaps: list[AlignmentGap]) -> list[AlignmentGap] | None:
    if not gaps:
        return None
    return gaps


def _source_gaps(text: str, segments: list[_SegmentMatch]) -> list[AlignmentGap] | None:
    gaps: list[AlignmentGap] = []
    for left, right in zip(segments, segments[1:], strict=False):
        start = left.text_span[1]
        end = right.text_span[0]
        if start < end:
            gaps.append(
                AlignmentGap(
                    text=text[start:end],
                    char_interval=CharInterval(start_pos=start, end_pos=end),
                )
            )
    return _non_empty_gaps(gaps)


def _quote_gaps(quote: str, segments: list[_SegmentMatch]) -> list[AlignmentGap] | None:
    gaps: list[AlignmentGap] = []
    previous_end = 0
    for segment in segments:
        start = segment.quote_span[0]
        if previous_end < start:
            gaps.append(
                AlignmentGap(
                    text=quote[previous_end:start],
                    char_interval=CharInterval(start_pos=previous_end, end_pos=start),
                )
            )
        previous_end = segment.quote_span[1]

    if previous_end < len(quote):
        gaps.append(
            AlignmentGap(
                text=quote[previous_end:],
                char_interval=CharInterval(start_pos=previous_end, end_pos=len(quote)),
            )
        )
    return _non_empty_gaps(gaps)


def align_quote(
    quote: str,
    text: str,
    *,
    min_match_len: int = 5,
    max_gap_chars: int = 2,
    max_segments: int = 3,
    min_fuzzy_span_score: float = 0.75,
    search_start: int = 0,
) -> QuoteAlignment:
    """Align *quote* against *text* and return spans, score, and alignment status.

    Tries four strategies in order; uses the first that succeeds:

    1. **MATCH_EXACT** — ``quote`` is a verbatim contiguous substring of ``text``.
    2. **MATCH_GREATER** — all quote chars located across up to *max_segments*
       ordered, word-boundary-respecting spans; ``score`` 1.0.
    3. **MATCH_LESSER** — same greedy search, but some quote chars (up to
       *max_gap_chars* per step) are skipped; ``score`` = matched / len(quote).
    4. **MATCH_FUZZY** — RapidFuzz ``partial_ratio`` fallback; ``char_intervals``
       is None; ``approximate_char_intervals`` set when score ≥ *min_fuzzy_span_score*.

    No normalization or plain-text conversion is performed. Returned intervals
    index into exactly the ``text`` string passed by the caller.

    Args:
        quote: String to locate within *text*.
        text: Source text to search.
        min_match_len: Minimum chars for a greedy segment anchor (word-boundary
            aligned).
        max_gap_chars: Maximum chars that may be skipped in the quote per greedy
            step before a segment is considered incomplete.
        max_segments: Maximum disjoint spans the greedy search may use.
        min_fuzzy_span_score: RapidFuzz score threshold below which
            ``approximate_char_intervals`` is left as None.
        search_start: Earliest position in *text* from which to begin matching.
            Text before this offset is ignored by MATCH_EXACT and greedy strategies.
            Has no effect on the MATCH_FUZZY fallback.
    """
    if not quote or not text:
        return QuoteAlignment(quote=quote, score=0.0)

    exact_match = _find_exact_match(text, quote, search_start)
    if exact_match is not None:
        return QuoteAlignment(
            quote=quote,
            score=1.0,
            char_intervals=_char_intervals([exact_match]),
            alignment_status=AlignmentStatus.MATCH_EXACT,
            matched_text=quote,
            occurrence_count=text.count(quote),
        )

    greedy_match = _find_greedy_quote_spans(
        text,
        quote,
        min_match_len=min_match_len,
        max_gap_chars=max_gap_chars,
        max_segments=max_segments,
        search_start=search_start,
    )
    if greedy_match is not None:
        text_spans = [segment.text_span for segment in greedy_match.segments]
        matched_text = "".join(text[start:end] for start, end in text_spans)
        if greedy_match.skipped_quote_chars:
            score = greedy_match.matched_quote_chars / len(quote)
            status = AlignmentStatus.MATCH_LESSER
        else:
            score = 1.0
            status = AlignmentStatus.MATCH_GREATER
        return QuoteAlignment(
            quote=quote,
            score=max(0.0, min(1.0, score)),
            char_intervals=_char_intervals(text_spans),
            alignment_status=status,
            matched_text=matched_text,
            source_gaps=_source_gaps(text, greedy_match.segments),
            quote_gaps=_quote_gaps(quote, greedy_match.segments),
        )

    fuzzy_match = _find_fuzzy_match(
        text,
        quote,
        min_score=min_fuzzy_span_score,
        min_block_len=min_match_len,
    )
    if fuzzy_match is not None:
        return QuoteAlignment(
            quote=quote,
            score=max(0.0, min(1.0, fuzzy_match.score)),
            approximate_char_intervals=_char_intervals([fuzzy_match.text_span]),
            alignment_status=AlignmentStatus.MATCH_FUZZY,
            source_gaps=_source_gaps(text, fuzzy_match.segments) if fuzzy_match.segments else None,
            quote_gaps=_quote_gaps(quote, fuzzy_match.segments) if fuzzy_match.segments else None,
        )

    score = float(fuzz.partial_ratio(quote, text)) / 100.0
    return QuoteAlignment(
        quote=quote,
        score=max(0.0, min(1.0, score)),
        alignment_status=AlignmentStatus.MATCH_FUZZY,
    )


def _spans_overlap(a: Span, b: Span) -> bool:
    return a[0] < b[1] and b[0] < a[1]


def _claim_spans(alignment: QuoteAlignment) -> list[Span]:
    if alignment.char_intervals:
        return [(ci.start_pos or 0, ci.end_pos or 0) for ci in alignment.char_intervals]
    if alignment.approximate_char_intervals:
        return [(ci.start_pos or 0, ci.end_pos or 0) for ci in alignment.approximate_char_intervals]
    return []


def align_quotes(
    quotes: list[str],
    text: str,
    *,
    min_match_len: int = 5,
    max_gap_chars: int = 2,
    max_segments: int = 3,
    min_fuzzy_span_score: float = 0.75,
) -> list[QuoteAlignment]:
    """Align multiple quotes against *text*, resolving duplicate-occurrence ambiguity.

    Calls :func:`align_quote` for each entry in *quotes* in order.  When the
    result has ``char_intervals`` and any of those intervals overlaps a span
    already claimed by an earlier quote, the function retries :func:`align_quote`
    with ``search_start`` set to the end of the latest overlapping claimed span.
    The retry result is used only if its score is at least as good as the
    original; otherwise the original alignment is kept unchanged.

    All resolved spans are added to a claimed set so that subsequent quotes
    avoid them.  Quotes are processed in the order given, so callers should
    pass them in the natural reading order of the source (e.g. panel A before
    panel E) to ensure earlier panels get priority on the first occurrence.
    """
    kwargs: dict = dict(
        min_match_len=min_match_len,
        max_gap_chars=max_gap_chars,
        max_segments=max_segments,
        min_fuzzy_span_score=min_fuzzy_span_score,
    )

    claimed: list[Span] = []
    results: list[QuoteAlignment] = []

    for quote in quotes:
        alignment = align_quote(quote, text, **kwargs)

        if alignment.char_intervals:
            spans = [(ci.start_pos or 0, ci.end_pos or 0) for ci in alignment.char_intervals]
            overlapping = [c for c in claimed if any(_spans_overlap(s, c) for s in spans)]
            if overlapping:
                retry_start = max(c[1] for c in overlapping)
                retry = align_quote(quote, text, search_start=retry_start, **kwargs)
                if retry.char_intervals and retry.score >= alignment.score:
                    alignment = retry

        claimed.extend(_claim_spans(alignment))
        results.append(alignment)

    return results


def _annotate_gaps_html(s: str, gaps: list[AlignmentGap] | None, css_class: str) -> str:
    """Return HTML for ``s`` with gap positions wrapped in a <mark> of ``css_class``."""
    if not gaps:
        return _html_escape(s)
    parts: list[str] = []
    pos = 0
    for gap in gaps:
        start = gap.char_interval.start_pos or 0
        end = gap.char_interval.end_pos or len(s)
        if pos < start:
            parts.append(_html_escape(s[pos:start]))
        parts.append(f'<mark class="{css_class}" title="quote gap">{_html_escape(s[start:end])}</mark>')
        pos = end
    if pos < len(s):
        parts.append(_html_escape(s[pos:]))
    return "".join(parts)


def _annotate_source_spans_html(
    text: str,
    char_intervals: list[CharInterval],
    source_gaps: list[AlignmentGap] | None,
    context_chars: int,
) -> str:
    """Return HTML for the source region around matched spans.

    Matched spans use ``mark.match``; source gaps between them use ``mark.gap-s``;
    up to *context_chars* characters of plain context appear before and after.
    """
    if not char_intervals:
        return _html_escape(text[: context_chars * 2]) + ("…" if len(text) > context_chars * 2 else "")

    first_start = char_intervals[0].start_pos or 0
    last_end = char_intervals[-1].end_pos or 0
    ctx_start = max(0, first_start - context_chars)
    ctx_end = min(len(text), last_end + context_chars)

    regions: list[tuple[int, int, str]] = [
        (interval.start_pos or 0, interval.end_pos or 0, "match") for interval in char_intervals
    ]
    for gap in source_gaps or []:
        regions.append((gap.char_interval.start_pos or 0, gap.char_interval.end_pos or 0, "gap-s"))
    regions.sort()

    parts: list[str] = []
    if ctx_start > 0:
        parts.append("…")
    pos = ctx_start
    for start, end, cls in regions:
        if end <= ctx_start or start >= ctx_end:
            continue
        clipped_start = max(start, ctx_start)
        clipped_end = min(end, ctx_end)
        if pos < clipped_start:
            parts.append(_html_escape(text[pos:clipped_start]))
        title = "matched" if cls == "match" else "source gap"
        parts.append(f'<mark class="{cls}" title="{title}">{_html_escape(text[clipped_start:clipped_end])}</mark>')
        pos = clipped_end
    if pos < ctx_end:
        parts.append(_html_escape(text[pos:ctx_end]))
    if ctx_end < len(text):
        parts.append("…")
    return "".join(parts)


def _annotate_fuzzy_html(
    quote: str,
    text: str,
    approx_interval: CharInterval,
    context_chars: int,
) -> tuple[str, str]:
    """Return ``(quote_html, source_html)`` for a fuzzy match via SequenceMatcher.

    Matched blocks use ``mark.match``; unmatched quote chars use ``mark.gap-q``;
    unmatched source chars within the approximate window use ``mark.gap-s``.
    Blocks shorter than 2 chars are treated as unmatched to reduce noise.
    """
    approx_start = approx_interval.start_pos or 0
    approx_end = approx_interval.end_pos or len(text)
    source_window = text[approx_start:approx_end]

    matcher = SequenceMatcher(None, quote, source_window, autojunk=False)
    blocks = [(b.a, b.b, b.size) for b in matcher.get_matching_blocks() if b.size >= 2]

    q_parts: list[str] = []
    q_pos = 0
    for qa, _, size in blocks:
        if q_pos < qa:
            q_parts.append(f'<mark class="gap-q" title="quote gap">{_html_escape(quote[q_pos:qa])}</mark>')
        q_parts.append(f'<mark class="match" title="matched">{_html_escape(quote[qa : qa + size])}</mark>')
        q_pos = qa + size
    if q_pos < len(quote):
        q_parts.append(f'<mark class="gap-q" title="quote gap">{_html_escape(quote[q_pos:])}</mark>')

    ctx_start = max(0, approx_start - context_chars)
    ctx_end = min(len(text), approx_end + context_chars)

    s_parts: list[str] = []
    if ctx_start > 0:
        s_parts.append("…")
    s_parts.append(_html_escape(text[ctx_start:approx_start]))
    s_pos = 0
    for _, sb, size in blocks:
        if s_pos < sb:
            s_parts.append(f'<mark class="gap-s" title="source gap">{_html_escape(source_window[s_pos:sb])}</mark>')
        s_parts.append(f'<mark class="match" title="matched">{_html_escape(source_window[sb : sb + size])}</mark>')
        s_pos = sb + size
    if s_pos < len(source_window):
        s_parts.append(f'<mark class="gap-s" title="source gap">{_html_escape(source_window[s_pos:])}</mark>')
    s_parts.append(_html_escape(text[approx_end:ctx_end]))
    if ctx_end < len(text):
        s_parts.append("…")

    return "".join(q_parts), "".join(s_parts)


def render_alignment_view(
    quote: str,
    text: str,
    alignment: QuoteAlignment,
    *,
    context_chars: int = 200,
) -> str:
    """Return an HTML fragment showing quote and source with matched/gap regions annotated.

    Produces a two-row block: the quote row shows unmatched quote chars (``quote_gaps``)
    in a red highlight; the source row shows matched spans in yellow (``match``) and
    source-gap text between them in orange (``gap-s``), with up to *context_chars*
    characters of surrounding plain text.

    For ``MATCH_FUZZY``, both rows are derived from a ``SequenceMatcher`` alignment of
    the quote against the approximate source window; unmatched chars in both strings are
    highlighted accordingly.

    ``quote`` and ``text`` must be the same plain-text strings passed to ``align_quote``
    — not the original HTML. Use ``compute_plain_text`` to convert HTML before calling
    either function.

    The returned fragment references three CSS classes that the caller must define:
    ``mark.match`` (matched span), ``mark.gap-q`` (unmatched quote chars),
    ``mark.gap-s`` (unmatched source chars / source gaps between spans).
    """
    status = alignment.alignment_status

    if status in (AlignmentStatus.MATCH_EXACT, AlignmentStatus.MATCH_GREATER, AlignmentStatus.MATCH_LESSER):
        quote_html = _annotate_gaps_html(quote, alignment.quote_gaps, "gap-q")
        source_html = _annotate_source_spans_html(
            text, alignment.char_intervals or [], alignment.source_gaps, context_chars
        )
    elif status is AlignmentStatus.MATCH_FUZZY:
        if alignment.approximate_char_intervals:
            quote_html, source_html = _annotate_fuzzy_html(
                quote, text, alignment.approximate_char_intervals[0], context_chars
            )
        else:
            quote_html = _html_escape(quote)
            source_html = f'<span class="av-muted">No approximate interval (score {alignment.score:.3f}).</span>'
    else:
        quote_html = _html_escape(quote)
        source_html = '<span class="av-muted">No alignment.</span>'

    return (
        '<div class="alignment-view">'
        '<div class="av-row">'
        '<span class="av-label">Quote</span>'
        f'<pre class="av-text">{quote_html}</pre>'
        "</div>"
        '<div class="av-row">'
        '<span class="av-label">Source</span>'
        f'<pre class="av-text">{source_html}</pre>'
        "</div>"
        "</div>"
    )


_COVERAGE_PALETTE: list[str] = [
    "#bde0fe",
    "#bbf7d0",
    "#fde68a",
    "#fecaca",
    "#ddd6fe",
    "#fed7aa",
    "#a7f3d0",
    "#fce7f3",
]
_COVERAGE_OVERLAP_BG = "repeating-linear-gradient(45deg,#e2e8f0,#e2e8f0 4px,#94a3b8 4px,#94a3b8 8px)"


def render_coverage_view(
    source: str,
    labeled_alignments: list[tuple[str, QuoteAlignment]],
    *,
    palette: list[str] | None = None,
) -> str:
    """Return an HTML fragment showing *source* with each alignment's matched spans highlighted.

    Each ``(label, alignment)`` pair in *labeled_alignments* gets a color from *palette*
    (cycling if there are more alignments than palette entries). Exact spans
    (``char_intervals``) are filled with the color; approximate spans
    (``approximate_char_intervals``) are shown as a colored underline only, signalling
    lower positional confidence. Regions covered by more than one alignment are shown
    with a diagonal-stripe pattern.

    A color legend is rendered above the source text; an "overlap" chip is added if any
    overlapping regions exist.

    *source* and all alignments must share the same plain-text coordinate system —
    i.e., the ``text`` string that was passed to ``align_quote`` when producing each
    ``QuoteAlignment``. Use ``compute_plain_text`` to convert HTML before calling either
    function.

    The returned fragment requires these CSS classes to be defined by the caller:
    ``coverage-view``, ``cov-legend``, ``cov-chip``.
    """
    colors = palette or _COVERAGE_PALETTE

    all_spans: list[tuple[int, int, int, bool]] = []
    for i, (_, alignment) in enumerate(labeled_alignments):
        for interval in alignment.char_intervals or []:
            s, e = interval.start_pos or 0, interval.end_pos or 0
            if s < e <= len(source):
                all_spans.append((s, e, i, False))
        for interval in alignment.approximate_char_intervals or []:
            s, e = interval.start_pos or 0, interval.end_pos or 0
            if s < e <= len(source):
                all_spans.append((s, e, i, True))

    if not all_spans:
        return f'<div class="coverage-view"><pre class="av-text">{_html_escape(source)}</pre></div>'

    boundaries: list[int] = sorted({0, len(source)} | {pos for start, end, _, _ in all_spans for pos in (start, end)})

    n = len(boundaries)
    region_exact: list[set[int]] = [set() for _ in range(n - 1)]
    region_approx: list[set[int]] = [set() for _ in range(n - 1)]
    for start, end, idx, is_approx in all_spans:
        lo = boundaries.index(start)
        hi = boundaries.index(end)
        target = region_approx if is_approx else region_exact
        for r in range(lo, hi):
            target[r].add(idx)

    parts: list[str] = []
    for r in range(n - 1):
        seg = _html_escape(source[boundaries[r] : boundaries[r + 1]])
        all_idxs = region_exact[r] | region_approx[r]
        if not all_idxs:
            parts.append(seg)
        elif len(all_idxs) > 1:
            overlapping = ", ".join(_html_escape(labeled_alignments[i][0]) for i in sorted(all_idxs))
            parts.append(
                f'<mark style="background:{_COVERAGE_OVERLAP_BG};padding:1px 0"'
                f' title="overlapping: {overlapping}">{seg}</mark>'
            )
        else:
            idx = next(iter(all_idxs))
            label = _html_escape(labeled_alignments[idx][0])
            color = colors[idx % len(colors)]
            if idx in region_exact[r]:
                style = f"background:{color};padding:1px 0"
            else:
                style = f"background:none;border-bottom:2px solid {color};padding-bottom:1px"
            parts.append(f'<mark style="{style}" title="{label}">{seg}</mark>')

    has_overlap = any(len(re | ra) > 1 for re, ra in zip(region_exact, region_approx, strict=True))
    legend_chips: list[str] = []
    for i, (label, _) in enumerate(labeled_alignments):
        color = colors[i % len(colors)]
        legend_chips.append(f'<span class="cov-chip" style="background:{color}">{_html_escape(label)}</span>')
    if has_overlap:
        legend_chips.append(f'<span class="cov-chip" style="background:{_COVERAGE_OVERLAP_BG}">overlap</span>')

    return (
        '<div class="coverage-view">'
        f'<div class="cov-legend">{"".join(legend_chips)}</div>'
        f'<pre class="av-text">{"".join(parts)}</pre>'
        "</div>"
    )
