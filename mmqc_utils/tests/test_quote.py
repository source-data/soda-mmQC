import pytest

from mmqc_utils import (
    AlignmentGap,
    AlignmentStatus,
    CharInterval,
    align_quote,
    align_quotes,
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


def test_align_quote_selects_multi_section_quotes_in_order() -> None:
    text = (
        "a) Some measurements were done and two-way ANOVA was performed. "
        "(b-d) Line plots showing runtime across cell counts. "
        "Evaluations were performed on single-cell transcriptomic data (b), "
        "joint profiling of transcriptomic and chromatin accessibility data (c), "
        "and surface protein data (d) and two-way ANOVA was performed."
    )
    quote = (
        "Line plots showing runtime across cell counts. Evaluations were performed on "
        "joint profiling of transcriptomic and chromatin accessibility data "
        "and two-way ANOVA was performed"
    )

    alignment = align_quote(quote, text)

    first = "Line plots showing runtime across cell counts. Evaluations were performed on "
    second = "joint profiling of transcriptomic and chromatin accessibility data "
    third = "and two-way ANOVA was performed"
    index_third_start = text.index(second)  # correct segment comes after second quote section
    expected = [
        (text.index(first), text.index(first) + len(first)),
        (text.index(second), text.index(second) + len(second)),
        (text.index(third, index_third_start), text.index(third, index_third_start) + len(third)),
    ]
    assert alignment.char_intervals == [CharInterval(start, end) for start, end in expected]
    assert alignment.alignment_status is AlignmentStatus.MATCH_GREATER
    assert alignment.score == 1.0
    assert alignment.matched_text == quote
    assert alignment.source_gaps == [
        AlignmentGap(
            text="single-cell transcriptomic data (b), ",
            char_interval=CharInterval(expected[0][1], expected[1][0]),
        ),
        AlignmentGap(
            text="(c), and surface protein data (d) ",
            char_interval=CharInterval(expected[1][1], expected[2][0]),
        ),
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


# --- align_quotes ---


def test_align_quotes_unique_quotes_are_unchanged() -> None:
    text = "Panel A shows growth. Panel B shows markers. Panel C shows survival."
    quotes = [
        "Panel A shows growth.",
        "Panel B shows markers.",
        "Panel C shows survival.",
    ]

    results = align_quotes(quotes, text)

    assert len(results) == 3
    for quote, result in zip(quotes, results, strict=True):
        assert result.alignment_status is AlignmentStatus.MATCH_EXACT
        assert result.char_intervals is not None
        start, end = result.char_intervals[0].start_pos, result.char_intervals[0].end_pos
        assert text[start:end] == quote


def test_align_quotes_resolves_exact_duplicate() -> None:
    # "Stat test performed." appears twice: once inside panel A's wider quote, once later.
    text = "Alpha data shown. Stat test performed. Beta data shown. Gamma data shown. Stat test performed."
    panel_a = "Alpha data shown. Stat test performed."
    panel_e = "Stat test performed."

    # Confirm the phrase appears twice in the full text.
    assert text.count(panel_e) == 2

    results = align_quotes([panel_a, panel_e], text)
    a, e = results

    assert a.alignment_status is AlignmentStatus.MATCH_EXACT
    assert a.char_intervals is not None
    assert text[a.char_intervals[0].start_pos : a.char_intervals[0].end_pos] == panel_a

    # Panel E should be assigned the second occurrence, not the first (which is inside panel A).
    assert e.alignment_status is AlignmentStatus.MATCH_EXACT
    assert e.occurrence_count == 2
    assert e.char_intervals is not None
    second_occurrence = text.rfind(panel_e)
    assert e.char_intervals[0].start_pos == second_occurrence
    assert text[e.char_intervals[0].start_pos : e.char_intervals[0].end_pos] == panel_e


def test_align_quotes_resolves_greater_overlap() -> None:
    # "alpha omega" is not verbatim in text, so it aligns as MATCH_GREATER.
    # The greedy algorithm prefers the first "alpha" (smaller source gap to "omega"),
    # which falls inside panel A's claimed span. align_quotes must retry from after
    # panel A's claim end and land on the second "alpha".
    text = "alpha claimed omega. filler filler. alpha. filler filler filler. omega end."
    panel_a = "alpha claimed omega."
    panel_e = "alpha omega"

    # Confirm the baseline: without dedup, align_quote picks the overlapping first "alpha".
    baseline = align_quote(panel_e, text)
    assert baseline.alignment_status is AlignmentStatus.MATCH_GREATER
    assert baseline.char_intervals is not None
    assert baseline.char_intervals[0].start_pos == 0  # inside panel A's region

    results = align_quotes([panel_a, panel_e], text)
    a, e = results

    assert a.alignment_status is AlignmentStatus.MATCH_EXACT
    assert a.char_intervals is not None
    a_start = a.char_intervals[0].start_pos
    a_end = a.char_intervals[0].end_pos
    assert a_start is not None and a_end is not None
    assert text[a_start:a_end] == panel_a

    # All of panel E's intervals must lie outside panel A's claimed span.
    assert e.alignment_status is AlignmentStatus.MATCH_GREATER
    assert e.char_intervals is not None
    for interval in e.char_intervals:
        start = interval.start_pos
        end = interval.end_pos
        assert start is not None and end is not None
        assert not (start < a_end and a_start < end), (
            f"interval ({start}, {end}) overlaps panel A claim ({a_start}, {a_end})"
        )


def test_align_quotes_keeps_original_when_no_unclaimed_occurrence() -> None:
    # The phrase appears only once; the second panel cannot be relocated.
    text = "alpha omega single occurrence."
    quotes = ["alpha omega", "alpha omega"]

    results = align_quotes(quotes, text)
    first, second = results

    assert first.char_intervals is not None
    assert first.char_intervals[0].start_pos == 0

    # Second panel keeps the original alignment (same span) rather than degrading to fuzzy.
    assert second.alignment_status is AlignmentStatus.MATCH_EXACT
    assert second.char_intervals is not None
    assert second.char_intervals[0].start_pos == 0
