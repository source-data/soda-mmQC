"""Literal quote scoring and span matching helpers."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from rapidfuzz import fuzz

Span = tuple[int, int]
"""Half-open character span."""


class AlignmentStatus(Enum):
    """How a quote aligned to the input text."""

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
    """Structured quote alignment result."""

    quote: str
    score: float
    char_intervals: list[CharInterval] | None = None
    alignment_status: AlignmentStatus | None = None
    matched_text: str | None = None
    source_gaps: list[AlignmentGap] | None = None
    quote_gaps: list[AlignmentGap] | None = None


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


def _find_exact_match(text: str, pattern: str) -> Span | None:
    index = text.find(pattern)
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
) -> _GreedyMatch | None:
    beam_size = 40
    states = [
        _SearchState(
            segments=[],
            quote_pos=0,
            search_start=0,
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
) -> QuoteAlignment:
    """
    Align a quote against text and return spans, score, and alignment status.

    This function performs no normalization or plain-text conversion. Returned
    intervals index into exactly the ``text`` string passed by the caller.
    """
    if not quote or not text:
        return QuoteAlignment(quote=quote, score=0.0)

    exact_match = _find_exact_match(text, quote)
    if exact_match is not None:
        return QuoteAlignment(
            quote=quote,
            score=1.0,
            char_intervals=_char_intervals([exact_match]),
            alignment_status=AlignmentStatus.MATCH_EXACT,
            matched_text=quote,
        )

    greedy_match = _find_greedy_quote_spans(
        text,
        quote,
        min_match_len=min_match_len,
        max_gap_chars=max_gap_chars,
        max_segments=max_segments,
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

    score = float(fuzz.partial_ratio(quote, text)) / 100.0
    return QuoteAlignment(
        quote=quote,
        score=max(0.0, min(1.0, score)),
        alignment_status=AlignmentStatus.MATCH_FUZZY,
    )
