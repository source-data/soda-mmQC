import pytest

from mmqc_utils import (
    AlignmentGap,
    AlignmentStatus,
    CharInterval,
    align_quote,
    compute_plain_text,
)


def test_align_quote_indexes_into_exact_input_text() -> None:
    text = "A Shows protein structure. B Shows binding site."
    quote = "Shows protein structure."

    alignment = align_quote(quote, text)

    assert alignment.char_intervals == [CharInterval(2, 26)]
    assert alignment.approximate_char_intervals is None
    assert alignment.char_intervals is not None
    interval = alignment.char_intervals[0]
    assert text[interval.start_pos : interval.end_pos] == quote


def test_align_quote_reports_exact_match() -> None:
    text = "A Shows protein structure. B Shows binding site."
    quote = "Shows protein structure."

    alignment = align_quote(quote, text)

    assert alignment.alignment_status is AlignmentStatus.MATCH_EXACT
    assert alignment.score == 1.0
    assert alignment.char_intervals == [CharInterval(2, 26)]
    assert alignment.approximate_char_intervals is None
    assert alignment.matched_text == quote
    assert alignment.source_gaps is None
    assert alignment.quote_gaps is None


def test_caller_can_compute_plain_text_before_alignment() -> None:
    html = "<p>The p-value was <em>p</em> &lt; 0.001 in all tests.</p>"
    html_quote = "<span>p &lt; 0.001</span>"
    text = compute_plain_text(html)
    quote = compute_plain_text(html_quote)

    alignment = align_quote(quote, text)

    assert alignment.char_intervals == [CharInterval(16, 25)]
    assert alignment.char_intervals is not None
    assert [text[interval.start_pos : interval.end_pos] for interval in alignment.char_intervals] == ["p < 0.001"]


def test_align_quote_supports_multi_section_quotes() -> None:
    text = (
        "(a-c) Line plots showing runtime across cell counts. "
        "Evaluations were performed on single-cell transcriptomic data (a), "
        "joint profiling of transcriptomic and chromatin accessibility data (b), "
        "and surface protein data (c)."
    )
    quote = (
        "Line plots showing runtime across cell counts. "
        "Evaluations were performed on joint profiling of transcriptomic and chromatin accessibility data"
    )

    alignment = align_quote(quote, text)

    first = "Line plots showing runtime across cell counts. Evaluations were performed on "
    second = "joint profiling of transcriptomic and chromatin accessibility data"
    expected = [
        (text.index(first), text.index(first) + len(first)),
        (text.index(second), text.index(second) + len(second)),
    ]
    source_gap_start = expected[0][1]
    source_gap_end = expected[1][0]
    assert alignment.char_intervals == [CharInterval(start, end) for start, end in expected]
    assert alignment.alignment_status is AlignmentStatus.MATCH_GREATER
    assert alignment.score == 1.0
    assert alignment.matched_text == quote
    assert alignment.source_gaps == [
        AlignmentGap(
            text="single-cell transcriptomic data (a), ",
            char_interval=CharInterval(source_gap_start, source_gap_end),
        )
    ]
    assert alignment.quote_gaps is None


def test_align_quote_reports_lesser_match_when_quote_chars_are_skipped() -> None:
    text = "A Shows protein structure. B Shows binding site."
    quote = "Shows protein structure.!"

    alignment = align_quote(quote, text)

    assert alignment.alignment_status is AlignmentStatus.MATCH_LESSER
    assert alignment.char_intervals == [CharInterval(2, 26)]
    assert alignment.matched_text == "Shows protein structure."
    assert 0.9 < alignment.score < 1.0
    assert alignment.source_gaps is None
    assert alignment.quote_gaps == [
        AlignmentGap(
            text="!",
            char_interval=CharInterval(24, 25),
        )
    ]


def test_align_quote_returns_no_intervals_for_missing_or_empty_quote() -> None:
    text = "A Shows protein structure. B Shows binding site."

    empty_quote = align_quote("", text)
    missing_quote = align_quote("Nonexistent caption segment.", text)
    empty_text = align_quote("Shows protein structure.", "")

    assert empty_quote.char_intervals is None
    assert empty_quote.approximate_char_intervals is None
    assert empty_quote.score == 0.0
    assert empty_quote.alignment_status is None
    assert empty_quote.source_gaps is None
    assert empty_quote.quote_gaps is None
    assert missing_quote.char_intervals is None
    assert missing_quote.approximate_char_intervals is None
    assert missing_quote.alignment_status is AlignmentStatus.MATCH_FUZZY
    assert missing_quote.source_gaps is None
    assert missing_quote.quote_gaps is None
    assert empty_text.char_intervals is None
    assert empty_text.approximate_char_intervals is None
    assert empty_text.score == 0.0
    assert empty_text.alignment_status is None
    assert empty_text.source_gaps is None
    assert empty_text.quote_gaps is None


def test_align_quote_exact_quote_score_is_one() -> None:
    text = "Cells were treated with nevirapine for 24 h."
    quote = "Cells were treated with nevirapine for 24 h."

    assert align_quote(quote, text).score == 1.0


def test_align_quote_close_quote_score_is_below_one() -> None:
    text = "Measurements were taken at 24, 48, and 72 hours post-infection."
    close_quote = "Measurements were done at 24, 48, and 72h after infection."
    absent_quote = "Cells were infected with HSV-1 and treated with interferon."

    close_alignment = align_quote(close_quote, text)
    absent_alignment = align_quote(absent_quote, text)

    assert close_alignment.alignment_status is AlignmentStatus.MATCH_FUZZY
    assert close_alignment.char_intervals is None
    assert close_alignment.approximate_char_intervals == [CharInterval(0, 58)]
    assert close_alignment.score == pytest.approx(0.7758620689655173)
    assert absent_alignment.score < close_alignment.score
    assert absent_alignment.approximate_char_intervals is None
    assert close_alignment.score < 1.0


def test_align_quote_does_not_decode_entities_or_normalize_typography() -> None:
    text = "Runtime of diﬀerent methods was evaluated with p &lt; 0.001 post–infection."
    quote = "different methods was evaluated with p < 0.001 post-infection"

    alignment = align_quote(quote, text)

    assert alignment.char_intervals is None
    assert alignment.score < 1.0


def test_align_quote_empty_inputs_are_zero() -> None:
    assert align_quote("", "source").score == 0.0
    assert align_quote("quote", "").score == 0.0
    assert align_quote("", "").score == 0.0


def test_align_quote_prefers_shorter_source_gap_when_quote_coverage_is_equal() -> None:
    text = (
        "(A) TRITC-dextran flux quantification in HUVECs. "
        "(G, H) Western blot analysis (G) and quantification (H) "
        "of p-VE-cadherin (Tyr658) in NCL-knockdown HUVECs."
    )
    quote = "quantification of p-VE-cadherin (Tyr658) in NCL-knockdown HUVECs."

    alignment = align_quote(quote, text)

    first = "quantification "
    second = "of p-VE-cadherin (Tyr658) in NCL-knockdown HUVECs."
    first_start = text.index("quantification (H)")
    second_start = text.index(second)

    assert alignment.alignment_status is AlignmentStatus.MATCH_GREATER
    assert alignment.char_intervals == [
        CharInterval(first_start, first_start + len(first)),
        CharInterval(second_start, second_start + len(second)),
    ]
    assert alignment.source_gaps == [
        AlignmentGap(
            text="(H) ",
            char_interval=CharInterval(first_start + len(first), second_start),
        )
    ]


def test_align_quote_does_not_return_partial_word_anchor_after_case_mismatch() -> None:
    text = "(G, H) Western blot analysis (G) and quantification (H) of p-VE-cadherin (Tyr658) in NCL-knockdown HUVECs."
    quote = "Quantification of p-VE-cadherin (Tyr658) in NCL-knockdown HUVECs."

    alignment = align_quote(quote, text)

    assert alignment.alignment_status is AlignmentStatus.MATCH_FUZZY
    assert alignment.char_intervals is None
    assert alignment.approximate_char_intervals is not None


def test_align_quote_recovers_spans_when_long_quote_has_single_extra_character() -> None:
    first = "Kidney marker expression in organ panel"
    second = " muscle marker expression in tissue panel."
    quote = f"{first}){second}"
    text = f"{first}{second} unrelated ) mu junk"

    alignment = align_quote(quote, text)

    assert alignment.alignment_status is AlignmentStatus.MATCH_LESSER
    assert alignment.char_intervals == [
        CharInterval(0, len(first)),
        CharInterval(len(first), len(first) + len(second)),
    ]
    assert alignment.quote_gaps == [
        AlignmentGap(
            text=")",
            char_interval=CharInterval(len(first), len(first) + 1),
        )
    ]


def test_align_quote_returns_approximate_interval_for_fuzzy_typo() -> None:
    text = "Prefix Alpha beta gamma delta epsilon suffix"
    quote = "Alpha beta gamma delta epsylon"

    alignment = align_quote(quote, text)

    assert alignment.alignment_status is AlignmentStatus.MATCH_FUZZY
    assert alignment.char_intervals is None
    assert alignment.approximate_char_intervals == [CharInterval(7, 37)]
    assert text[7:37] == "Alpha beta gamma delta epsilon"
    assert alignment.score == pytest.approx(0.9666666666666667)


def test_align_quote_omits_approximate_interval_for_low_confidence_fuzzy_match() -> None:
    text = "A Shows protein structure. B Shows binding site."
    quote = "Nonexistent caption segment."

    alignment = align_quote(quote, text)

    assert alignment.alignment_status is AlignmentStatus.MATCH_FUZZY
    assert alignment.char_intervals is None
    assert alignment.approximate_char_intervals is None
