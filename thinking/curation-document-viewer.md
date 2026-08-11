# Curation interface — document-level checks

Plan for extending the Streamlit curation UI so it supports **document-level** checks (whole manuscript, no figure), alongside the existing **figure-level** workflow.

**Status:** Phase 2 complete.

**Related code:**

- Launcher: [`soda_mmqc/scripts/curate.py`](../soda_mmqc/scripts/curate.py)
- UI: [`soda_mmqc/core/curation.py`](../soda_mmqc/core/curation.py)
- Example types: [`soda_mmqc/core/examples.py`](../soda_mmqc/core/examples.py) (`FigureExample`, `WordExample` → `document_to_html`)
- Document conversion: [`mmqc_utils/documents.py`](../mmqc_utils/src/mmqc_utils/documents.py) — `document_to_html`
- Reference check: [`external-data-url-validation-agentic`](../soda_mmqc/data/checklist/doc-checklist/external-data-url-validation-agentic/)

---

## Problem

The curation interface was built for **figure checks**:

1. Select paper (`doc_id`)
2. Select figure (`doc_id/content/{figure_id}`)
3. View **image** + **caption** side by side
4. Edit **expected output** in a schema-driven table (`st.data_editor`)

Document-level checks have a different input shape:

| Aspect | Figure check | Document check |
|--------|--------------|----------------|
| Example granularity | One figure per row | One manuscript per row |
| Primary content | Image + caption | Word file (`.docx`) |
| Example path | `{doc_id}/content/{figure_id}` | `{doc_id}` |
| `example_class` in benchmark | `"figure"` | `"word"` |
| Expected output location | `{doc_id}/content/{fig}/checks/...` **or** `{doc_id}/checks/...`? | `{doc_id}/checks/{check}/expected_output.json` |

For document checks we **do not need an image or caption column**. We need a **document viewer** so curators can read the manuscript while editing structured expected outputs.

The output side is unchanged: JSON with an `outputs` array, edited via the same schema → table mapping and YAML serialization/deserialization already in `curation.py`.

---

## Current state

### Figure-centric assumptions in `curation.py`

1. **`get_example_hierarchy`** walks `examples/*/content/*` and only keeps **numeric** directory names (figure IDs). Document examples (`{doc_id}/content/*.docx`, checks at `{doc_id}/checks/`) are invisible.

2. **`load_example_data` / `save_check_output`** always call `EXAMPLE_FACTORY.create(..., "figure")`.

3. **Layout** is fixed: three columns — Figure (image), Caption, Expected Output.

4. **Navigation** requires both paper and figure selectors.

### What already exists for documents

**`WordExample`** ([`examples.py`](../soda_mmqc/core/examples.py)):

- Source path: `{doc_id}` (relative to `EXAMPLES_DIR`)
- Loads exactly one `.docx` from `{doc_id}/content/` (**raises if zero or multiple**)
- **Target:** convert via `document_to_html` (today: pandoc → markdown — to be replaced in Phase 0)
- `get_expected_output` / `save_expected_output` inherited from `Example` — checks live at `{doc_id}/checks/{check_name}/`

**Benchmark metadata** declares the example type, e.g. [`external-data-url-validation-agentic/benchmark.json`](../soda_mmqc/data/checklist/doc-checklist/external-data-url-validation-agentic/benchmark.json):

```json
{
  "example_class": "word",
  "examples": ["EMBOJ-2024-119734R", ...]
}
```

`run.py` already reads `example_class` from benchmark config; the curation UI does not.

---

## Production vs benchmarking — document conversion

There is currently a **three-way split** in how manuscript text is produced. This matters for curation viewer design and for benchmark/production parity.

### Production ([soda-curation](https://github.com/source-data/soda-curation))

**Main pipeline** — [`manuscript_xml_parser.py` → `extract_docx_content`](https://github.com/source-data/soda-curation/blob/main/src/soda_curation/pipeline/manuscript_structure/manuscript_xml_parser.py#L365):

```python
from mmqc_utils import document_to_html

def extract_docx_content(self, docx_path: str) -> str:
    ...
    return document_to_html(full_path)  # HTML, post-processed
```

Called from [`main.py`](https://github.com/source-data/soda-curation/blob/main/src/soda_curation/main.py):

```python
manuscript_content = extractor.extract_docx_content(zip_structure.docx)
zip_structure.manuscript_text = manuscript_content
```

- Output format: **HTML** (via `pypandoc-binary`, with `mmqc_utils` post-processing)
- No `destination_format` parameter — `document_to_html` always targets HTML
- Stored on `ZipStructure.manuscript_text` and passed to downstream AI steps as `doc_content` / `$manuscript_text`
- Section extraction, caption extraction, hallucination scoring all operate on this **HTML** (caption verification uses `strip_html=False` by default — markup is part of the fidelity check)

Since v3.3.0, soda-curation explicitly migrated to `mmqc_utils` for manuscript conversion ([changelog](https://github.com/source-data/soda-curation)).

**QC pipeline — document checks** — [`ManuscriptQCAnalyzer.extract_word_file_content`](https://github.com/source-data/soda-curation/blob/main/src/soda_curation/qc/base_analyzers.py):

1. **Preferred:** `zip_structure.manuscript_text` → same **HTML** from step above
2. **Fallback:** `python-docx` plain-text extraction (paragraphs + table cells, newline-joined) — **different representation**, only when `manuscript_text` is missing

When QC runs after the main pipeline (typical), document checks see **HTML**.

### Benchmarking ([soda_mmqc](../soda_mmqc/core/examples.py) — `WordExample`)

See next section — today this is **markdown via system pandoc**, not HTML.

### Summary table

| Context | Function | Library | Output | Notes |
|---------|----------|---------|--------|-------|
| Production main pipeline | `extract_docx_content` | `mmqc_utils.document_to_html` | HTML | Canonical production representation |
| Production QC (normal path) | `extract_word_file_content` | reads `zip_structure.manuscript_text` | HTML | Same as main pipeline |
| Production QC (fallback) | `extract_word_file_content` | `python-docx` | plain text | Legacy fallback; should probably use `document_to_html` too |
| mmQC benchmarking | `WordExample.load_from_source` | system `pandoc -t markdown` | markdown | **Diverges from production** |

### Consistency goal *(locked)*

**Production canonical format is HTML** from `mmqc_utils.document_to_html`. Benchmarking will align:

1. **`WordExample.load_from_source`** → `document_to_html` ✅
2. **`prepare_model_input`** → send HTML in the `Content:` block ✅
3. **Curation viewer** → render `WordExample.content` via `st.components.v1.html` ✅
4. **Cache hash** → will change; re-run benchmarks is acceptable ✅
5. **QC fallback** (soda-curation `python-docx` path) — optional follow-up, out of mmQC scope for now

Remove `destination_format` from `WordExample` when switching to `document_to_html` (no longer needed).

---

## Where the model input is produced

Document text sent to the model flows through `WordExample` in [`examples.py`](../soda_mmqc/core/examples.py).

### Target (after Phase 0)

```python
# WordExample.load_from_source()
from mmqc_utils import document_to_html
self.content = document_to_html(word_file_path)
```

- Exactly **one** `.docx` in `{doc_id}/content/`; **raise** if zero or multiple
- Output: **HTML** (same as [soda-curation production](https://github.com/source-data/soda-curation/blob/main/src/soda_curation/pipeline/manuscript_structure/manuscript_xml_parser.py#L365))

### Package — `prepare_model_input`

```python
return {
    "content": [{
        "type": "input_text",
        "text": f"{prompt}\n\nContent:\n{self.content}"   # HTML
    }]
}
```

### Call site — `run.py` → `lib/api.py`

```
run.py → EXAMPLE_FACTORY.create(ex_path, "word")
      → generate_response → prepare_model_input(prompt)
      → OpenAI / Anthropic API
```

**Curation rule:** viewer always reads `WordExample.content` — never calls `document_to_html` independently.

### Current state (pre-Phase 0)

Still uses system `pandoc -t markdown` subprocess. Phase 0 replaces this.

---

## Reference example walkthrough

Check: **external-data-url-validation-agentic**

- Schema: array of objects with `url`, `repository`, `accession_number_or_identifier`, `resolves_to_data_page`, `justification` — same table-editing pattern as figure checks.
- Example `EMBOJ-2024-119734R`:
  - `content/` — contains the `.docx` manuscript
  - `checks/external-data-url-validation-agentic/expected_output.json` — curated URLs and validation fields

Curator workflow we want:

1. Launch: `python soda_mmqc/scripts/curate.py doc-checklist`
2. Select paper `EMBOJ-2024-119734R` (no figure selector)
3. Left (wide): **rendered document** (same representation the model sees), scrollable
4. Right: check selector + prompt expander + editable output table + save

---

## Proposed design

### 1. Detect example mode from checklist

**Decision:** read `example_class` from each check's `benchmark.json` (same source `run.py` uses).

**Assumption (locked):** one checklist = one `example_class`. No mixed figure/word checklists.

#### Technical debt: `example_class` in `benchmark.json`

`example_class` describes input loading, not eval metadata. A future `input_manifest.json` would be cleaner, but **deferred** — the project is moving toward an agentic mechanism instead. Keep reading from `benchmark.json` for now.

### 2. Example discovery

| Mode | Hierarchy function | Selectable units |
|------|-------------------|------------------|
| `figure` | Current `get_example_hierarchy` | `{doc_id}/content/{figure_id}` |
| `word` | New `get_document_example_ids` | `{doc_id}` |

For `word` mode, enumerate top-level directories under `EXAMPLES_DIR` that:

- Appear in the check's `benchmark.json` `examples` list (if present), **or**
- Have `{doc_id}/content/*.docx` and `{doc_id}/checks/{check_name}/` for at least one check in the checklist

Prefer benchmark list when available — matches how `run.py` selects examples.

### 3. Load / save via factory

Replace hardcoded `"figure"` with dynamic `example_class`:

```python
example = EXAMPLE_FACTORY.create(relative_path, example_class)
```

For documents, `relative_path` is just `doc_id` (e.g. `"EMBOJ-2024-119734R"`).

`load_example_data` should return a mode-neutral dict:

```python
{
  "doc_id": ...,
  "example_class": "word" | "figure",
  # figure-only:
  "figure_id": ..., "caption": ..., "image_path": ...,
  # word-only:
  "document_path": Path,           # path to .docx (informational)
  "document_content": str,         # WordExample.content — HTML from document_to_html
  # shared:
  "check_outputs": { check_name: {...} },
}
```

Load via `EXAMPLE_FACTORY.create(doc_id, "word")` so conversion matches inference exactly.

### 4. Document viewer — render what the model sees *(locked)*

**Principle:** curators read the same representation the model received — always from `WordExample.content`.

**Rendering:** HTML via `st.components.v1.html`:

```python
st.components.v1.html(
    f"<div style='height:80vh; overflow:auto;'>{example_data['document_content']}</div>",
    height=800,
    scrolling=True,
)
```

**Prompt context (expander):** full user message from `prepare_model_input(prompt)`.

**Do not** call `document_to_html` in the curation UI — conversion happens once in `WordExample.load_from_source`.

### 5. Layout

| Mode | Columns | Left panel | Middle | Right panel |
|------|---------|------------|--------|-------------|
| `figure` | `[1, 0.7, 1.3]` | Image | Caption | Expected output (unchanged) |
| `word` | `[1.5, 1]` | Document HTML viewer (`st.components.v1.html`) | — | Expected output (unchanged) |

Keep the **expected output column logic identical**: check dropdown, prompt expander, schema-driven `DataFrame` + `st.data_editor`, serialize/deserialize, save with `updated_at` timestamp, session saved-state tracking.

### 6. Navigation UI

| Mode | Selectors |
|------|-----------|
| `figure` | Paper + Figure (current) |
| `word` | Paper only |

Page title could reflect mode: "mmQC Benchmark Curation — Documents" vs "... — Figures".

---

## Implementation phases

### Phase 0 — Align WordExample with production *(do first)* ✅

- [x] `WordExample.load_from_source`: `document_to_html`; remove `destination_format` and pandoc subprocess
- [x] Require exactly one `.docx` in `content/` — raise on zero or multiple
- [x] Accept cache invalidation / re-run benchmarks
- [x] Tests: `tests/test_word_example.py`

### Phase 1 — Minimal document curation (MVP) ✅

- [x] Add `get_document_examples(examples_dir, checklist)` 
- [x] Read `example_class` from checklist benchmark config
- [x] Branch main layout on `example_class`
- [x] `load_example_data` / `save_check_output` use factory with correct type
- [x] Document panel: `WordExample.content` → `st.components.v1.html`
- [x] Optional expander: full `prepare_model_input(prompt)` text for selected check
- [x] Manual smoke test (`doc-checklist` + `external-data-url-validation-agentic`)

### Phase 2 — Polish ✅

- [x] Cache loaded `WordExample` in session state keyed by `(doc_id, docx_mtime)`
- [x] Document search with inline highlight of matches (prev/next navigation)
- [x] Document viewer layout: fixed 40/60 column split, links open in new tab
- [x] Document viewer CSS (typography, link styling, tables)
- [x] Loading/error states (missing docx, conversion failure)
- [x] Filter paper list to benchmark `examples` when defined

### Phase 3 — Optional / out of scope

- [x] ~~URL highlighting in HTML for URL-validation checks~~ **Cancelled** — no schema/link between check output rows and document locations yet
- [x] ~~Refactor shared "expected output editor" into a function~~ **Done in Phase 1** — see `render_expected_output_panel()` in `curation.py`
- [ ] soda-curation QC fallback: replace `python-docx` with `document_to_html` (**separate repo**, optional parity fix)

---

## Decisions (locked)

| Topic | Decision |
|-------|----------|
| Document conversion | `WordExample` uses `document_to_html`; curation reads `WordExample.content` |
| Viewer rendering | `st.html` (width stretch, JS for search) |
| Cache invalidation | Acceptable when conversion changes |
| One checklist, one `example_class` | Assume single class per checklist |
| Multiple `.docx` in `content/` | **Raise exception** — exactly one expected |
| Large documents / performance | TBD — monitor; session cache + mtime key as first mitigation |
| Markdown fidelity | N/A — HTML end-to-end after Phase 0 |
| `input_manifest` migration | **Deferred** — agentic mechanism replaces this direction |
| Tests | Manual QA gates for UI; no Streamlit unit tests required for v1 |

---

## Manual test plan (post-implementation)

1. `source .venv/bin/activate && python soda_mmqc/scripts/curate.py doc-checklist --local-prompts`
2. Select `EMBOJ-2024-119734R` — document renders, no figure selector
3. Select check `external-data-url-validation-agentic` — table loads with URL rows
4. Edit a cell, save — `expected_output.json` updates, `updated_at` set, success message
5. Search the document (e.g. a URL or keyword) — matches highlighted inline; use ↑/↓ or Enter/Shift+Enter to navigate
6. Re-run `curate.py fig-checklist` (or any figure checklist) — existing figure workflow unchanged

---

## Summary

The curation UI needs a **mode switch** from `benchmark.json` → `example_class`, a **document example enumerator**, and an **HTML document panel** reading `WordExample.content` via `st.components.v1.html`. **Phase 0** aligns `WordExample` with production (`document_to_html`). The structured output editor stays as-is. Manual QA gates the UI; `input_manifest` deferred.
