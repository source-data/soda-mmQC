from mmqc_utils.text import compute_plain_text


def test_compute_plain_text_normalizes_html_and_whitespace() -> None:
    html = "<p>Hello <strong>world</strong></p><div>Line 2<br/>Line 3</div>"
    plain = "Hello world Line 2 Line 3"
    assert compute_plain_text(html) == plain


def test_compute_plain_text_keeps_plain_text_unchanged() -> None:
    html = "A Shows protein structure. B Shows binding site."
    plain = html
    assert compute_plain_text(html) == plain


def test_compute_plain_text_removes_html_tags() -> None:
    html = "A <em>Shows</em> protein structure."
    plain = "A Shows protein structure."
    assert compute_plain_text(html) == plain


def test_compute_plain_text_replaces_block_tag_with_space() -> None:
    html = "Hello<br />World"
    plain = "Hello World"
    assert compute_plain_text(html) == plain


def test_compute_plain_text_removes_inline_tag() -> None:
    html = "Panel <strong>A</strong>) shows something."
    plain = "Panel A) shows something."
    assert compute_plain_text(html) == plain


def test_compute_plain_text_collapses_whitespace() -> None:
    html = "Hello   World\n\nFoo"
    plain = "Hello World Foo"
    assert compute_plain_text(html) == plain


def test_compute_plain_text_strips_leading_trailing_whitespace() -> None:
    html = "  Hello World  "
    plain = "Hello World"
    assert compute_plain_text(html) == plain


def test_compute_plain_text_empty_string() -> None:
    html = ""
    plain = ""
    assert compute_plain_text(html) == plain


def test_compute_plain_text_decodes_html_entities() -> None:
    html = "5 &gt; 3 &amp; 2 &lt; 4"
    plain = "5 > 3 & 2 < 4"
    assert compute_plain_text(html) == plain
