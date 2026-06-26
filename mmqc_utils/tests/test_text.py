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


def test_compute_plain_text_preserves_escaped_comparisons() -> None:
    html = "The p-value was p &lt; 0.001 and x &gt; 2."
    plain = "The p-value was p < 0.001 and x > 2."
    assert compute_plain_text(html) == plain


def test_compute_plain_text_preserves_literal_angle_brackets_in_plain_text() -> None:
    html = "Use x < y > z as plain text."
    plain = "Use x < y > z as plain text."
    assert compute_plain_text(html) == plain


def test_compute_plain_text_strips_escaped_html() -> None:
    html = "&lt;p&gt;Hello &lt;em&gt;world&lt;/em&gt;&lt;/p&gt;"
    plain = "Hello world"
    assert compute_plain_text(html) == plain


def test_compute_plain_text_keeps_superscript_inline_for_display_text() -> None:
    html = "Ca<sup>2+</sup>, 2×10<sup>4</sup> PFU, Dyrk4<sup>+/+</sup>"
    plain = "Ca2+, 2×104 PFU, Dyrk4+/+"
    assert compute_plain_text(html) == plain


def test_compute_plain_text_separates_table_cells() -> None:
    html = "<table><tr><th>Label</th><th>Value</th></tr><tr><td>A</td><td>1</td></tr></table>"
    plain = "Label Value A 1"
    assert compute_plain_text(html) == plain


def test_compute_plain_text_ignores_non_visible_html() -> None:
    html = """
    <html>
      <head><title>Hidden title</title><style>.x { color: red; }</style></head>
      <body>
        <p>Visible<!-- ignored comment --> text.</p>
        <script>alert("hidden");</script>
        <template>Hidden template</template>
      </body>
    </html>
    """
    plain = "Visible text."
    assert compute_plain_text(html) == plain


def test_compute_plain_text_removes_control_characters() -> None:
    html = "\x08YOU ARE DEAD"
    plain = "YOU ARE DEAD"
    assert compute_plain_text(html) == plain


def test_compute_plain_text_preserves_unicode_text() -> None:
    html = "IFN-β was measured at 12.5 μg/mL and 2×10<sup>4</sup> PFU."
    plain = "IFN-β was measured at 12.5 μg/mL and 2×104 PFU."
    assert compute_plain_text(html) == plain


def test_compute_plain_text_handles_malformed_html() -> None:
    html = "<p>Hello <em>world</p>again"
    plain = "Hello world again"
    assert compute_plain_text(html) == plain
