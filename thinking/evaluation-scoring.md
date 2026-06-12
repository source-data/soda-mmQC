# Evaluation scoring (flat leaf model)

Normative design for comparing prediction JSON to gold JSON under **JSON Schema** and an **evaluation manifest** (`eval-manifest.json`).

One comparator invocation takes a single `exp` JSON and a single `pred` JSON with the same schema. The comparator:

1. Identifies **leaf properties** (lowest-level fields).
2. For **predictive** list-of-objects properties (per schema; aligned using manifest `list_alignment`): Hungarian row matching and **layer S** (`by_list`).
3. Compares leaf values (including **extended primitives** — arrays of primitives).
4. Emits a **score** per leaf instance and, when configured, **layer 1** (applicability) and **layer 2** (matching) reporting on that instance.
5. Emits **per leaf property** summaries (`by_property`) and, for each predictive list, **per list** structural summaries (`by_list`).

Worked examples: [evaluation-toy-examples.md](evaluation-toy-examples.md).

---

## Leaf properties

A **leaf property** is a schema path whose value is either:

- a **primitive** (`string`, `number`, `integer`, `boolean`), or
- an **array of primitives** (e.g. `tags: string[]`).

Intermediate objects are not scored as units — only their primitive descendants are leaves. Object nesting in the schema is flattened to dotted paths (`item.meta.author`).

**Path notation:**

| Form | Meaning |
|------|---------|
| `item.label` | One scalar leaf |
| `tags` | One leaf whose value is a primitive array |
| `panels[].status` | Leaf property pattern; one **instance** per aligned panel row (`panels[0].status`, `panels[1].status`, …) |

Nested arrays as leaf values are out of scope.

**Discovery:** walk `schema.json` from the output root; follow `properties` and `items`; stop at primitives or `array` whose `items` is primitive.

**Toy schema leaves (illustration):**

```text
tags
item.id
item.label
item.status
item.meta.author
item.meta.year
panels[].id
panels[].label
panels[].status
```

---

## Leaf instances vs leaf properties

| Term | Meaning |
|------|---------|
| **Leaf property** | Manifest/schema path **pattern** (e.g. `panels[].status`) |
| **Leaf instance** | One concrete path in this comparison (e.g. `panels[1].status`) |

A single `exp` / `pred` pair may contain **many instances** of the same property (multiple panels, multiple document rows in an `outputs[]` list, etc.). Summaries aggregate **within one leaf property only** — never across properties (`tags` is not averaged with `panels[].label`).

---

## Pipeline

```mermaid
flowchart LR
  schema[schema.json] --> flatten[Flatten to leaf paths]
  flatten --> alignObj[Align object lists]
  alignObj --> layerS[Layer S structural reporting]
  layerS --> score[Score each leaf instance]
  flatten --> scorePrim[Compare extended primitives]
  manifest[eval-manifest.json] --> report[Layer 1 and 2 reporting]
  score --> report
  scorePrim --> report
  report --> instances[Per-instance results]
  report --> byProp[by_property summaries]
  layerS --> byList[by_list summaries]
```

1. **Flatten** — enumerate leaf paths from schema and manifest.
2. **Join lists** — positional index for structural ancestors; Hungarian matching for predictive lists using `list_alignment` (see [List of objects](#list-of-objects)); emit **layer S** in `by_list` for predictive lists only.
3. **Score extended primitives** — compare array-of-primitives leaves (e.g. `tags`) as a single leaf instance with aggregate `score` (see [List of primitives](#list-of-primitives)).
4. **Score row leaves** — at each gold row index: use the paired pred row (Hungarian for predictive lists, positional for structural lists); `score ∈ [0, 1]` (see [Scoring rules](#scoring-rules)).
5. **Report** — layer 1 applicability reporting, then layer 2 matching reporting when layer 1 = `correct_applicable`.
6. **Summarize** — required `by_property` per leaf property key; `by_list` only for predictive lists.

Missing required values, disallowed `null`, and `missing_row` gold indices → `score = 0` on the affected leaf instance unless a profile defines otherwise.

---

## Comparator output (required)

### Per leaf instance

Every concrete path:

| Field | Required | Description |
|-------|----------|-------------|
| `path` | yes | e.g. `panels[1].status` |
| `leaf_property` | yes | Pattern, e.g. `panels[].status` |
| `exp_value` / `pred_value` | yes | Compared values after alignment |
| `score` | yes | `[0, 1]` |
| `layer1` | if manifest profile | Applicability reporting outcome |
| `layer2` | if profile and `correct_applicable` | Matching reporting outcome |

For a **list-of-primitives** (extended primitive) leaf such as `tags`, the comparator emits **one** instance on path `tags` with an aggregate `score` only — no layer S structural reporting.

### Per list (`by_list`)

For **each predictive list-of-objects** property (per schema; e.g. `panels` when the model segments panel captions) — the comparator **must** return:

| Field | Description |
|-------|-------------|
| `row_counts` | Layer S structural reporting counts: `correct_row`, `missing_row`, `spurious_row` |
| `rows` | Per-row structural detail (see [Layer S](#layer-s--structural-row-alignment)) |

`by_list` is keyed by the **list property name** (no `[]` suffix; use dotted paths for nested lists, e.g. `figures.panels`). It is separate from `by_property` — structural row events are not mixed into leaf-property rollups.

**Structural lists** (collection wrappers; no `list_alignment`) do **not** appear in `by_list` — see [Structural vs predictive lists](#structural-vs-predictive-lists-schema-driven).

### Per leaf property (`by_property`)

For **each** leaf property key, the comparator **must** return:

| Field | Description |
|-------|-------------|
| `mean_score` | Mean of instance `score`s for this property only |
| `layer1_counts` | Layer 1 applicability reporting counts (manifest-profiled instances) |
| `layer2_counts` | Layer 2 matching reporting counts (instances with `correct_applicable`) |

Properties without a manifest profile still appear with `mean_score`; `layer1_counts` / `layer2_counts` may be empty.

```json
{
  "instances": [ "…" ],
  "by_list": {
    "panels": {
      "row_counts": {
        "correct_row": 2,
        "missing_row": 0,
        "spurious_row": 0
      },
      "rows": [
        { "gold_index": 0, "pred_index": 0, "structural": "correct_row", "similarity": 1.0 },
        { "gold_index": 1, "pred_index": 1, "structural": "correct_row", "similarity": 1.0 }
      ]
    }
  },
  "by_property": {
    "item.label": {
      "mean_score": 1.0,
      "layer1_counts": { "correct_applicable": 1 },
      "layer2_counts": { "match": 1 }
    },
    "item.status": {
      "mean_score": 1.0,
      "layer1_counts": { "correct_NA": 1 },
      "layer2_counts": {}
    },
    "panels[].status": {
      "mean_score": 1.0,
      "layer1_counts": { "correct_applicable": 1, "correct_NA": 1 },
      "layer2_counts": { "TP": 1 }
    },
    "tags": {
      "mean_score": 1.0,
      "layer1_counts": {},
      "layer2_counts": {}
    }
  }
}
```

Do **not** pool counts or means across different leaf properties. Compute precision, recall, and F1 per property from that property’s `layer2_counts` when needed. List-level precision and recall come from `by_list.row_counts` (see [Layer S](#layer-s--structural-row-alignment)).

---

## Scoring rules

### Primitives

| Type | Default |
|------|---------|
| `string` (no schema `enum`) | `score` from manifest `string_compare` when profiled |
| Schema `enum` | Always **exact** literal match → 1.0 or 0.0; **`string_compare` omitted** |
| `number`, `integer`, `boolean` | Exact equality → `score` 1.0 or 0.0 |
| `graded_string` (manifest) | Requires `string_compare`; layer 2 **match** / **mismatch** from `match_threshold` |

**`matching_metric`** (layer 2 reporting) and **`string_compare`** (how `score` is computed) are separate. Use `string_compare` only on **non-enum** strings with `graded_string` — enum fields (`binary_polarity`, `multiclass`) are always exact; do not declare `string_compare` there.

Default `match_threshold` for layer 2: `1.0` when using exact compare; lower values (e.g. `0.8`) for fuzzy or semantic fields.

### Missing, null, and empty string

Treat JSON literally:

- **Absent key** — no value supplied; `score = 0`.
- **`null`** when schema is string-only — no string supplied; `score = 0`.
- **`""`** — a string value exists; compare like any other string. For enums, `""` may mean N/A when listed in manifest `na_values` (layer 1), not “missing”.

Do not use Python truthiness (`if not value:`) — it conflates `""`, `null`, and absent keys.

### List of primitives (extended primitive)

An **extended primitive** is a leaf whose value is an **array of primitives** (e.g. `tags: string[]`). The leaf path is the array property itself (`tags`), not per-element paths. The orchestrator scores it via **`leaves.compare_primitive_list`** (bipartite element matching internally, via `matching.py`); there is **no** layer S.

Example path: `tags` (`string[]`).

1. Build pairwise element similarity `exp[i]` vs `pred[j]` (exact → 0 or 1 by default).
2. Hungarian assignment; pairs with similarity `< τ` count as unmatched slots (score contribution 0).
3. Instance `tags` **score** = mean over `n_all = max(len(exp), len(pred))`, counting unmatched gold slots, unmatched pred slots, and below-threshold pairs as 0.

Layer 1 / 2 apply only if the manifest defines a profile for that path. The aggregate `score` already reflects extra, missing, and mismatched elements; no separate structural layer is emitted for extended primitives.

### List of objects

Example pattern: `panels[]` with leaves `panels[].id`, `panels[].label`, `panels[].status`.

Row leaves under a list-of-objects still emit one instance per **gold index**. How pred rows are joined to gold indices depends on whether the list is **predictive** or **structural** (below).

#### Structural vs predictive lists (schema-driven)

**Terminology (strict):**

| Term | Meaning |
|------|---------|
| **`schema.json`** | The JSON Schema passed to the model for **one** structured-output call. Describes exactly what the model must produce. **Contains no collation wrappers** (`figures[]`, `papers[]`, …). |
| **Evaluation document** | Gold or pred JSON the comparator receives. May **embed** one or more model outputs under collation paths that do not appear in `schema.json`. |
| **Predictive list** | Every `array` of objects in `schema.json` — by definition model-produced, variable-length, needs alignment. |
| **Structural list** | An object array in the **evaluation document only**, nesting repeated model outputs (one slot per figure, paper, run). Never part of `schema.json`. |

**The schema is the reference for predictive lists.** The manifest configures **how** to align them (`list_alignment`) and is validated against the schema. Structural lists are inferred from the evaluation document shape relative to the schema — they are not declared in the schema at all.

| | **Predictive list** | **Structural list** |
|---|---------------------|---------------------|
| **Where it lives** | `schema.json` (model call) | Evaluation gold/pred only (collation) |
| **Definition** | Object array the model was asked to **produce** (`panels[]`, `outputs[]`, …) | Wrapper array **above** embedded model output — how benchmark results are **indexed** (which figure, which paper) |
| **Row join** | **Hungarian** on manifest alignment keys; pairs below `τ` count as unmatched | **Positional:** gold `[i]` ↔ pred `[i]` when that index exists in pred |
| **Layer S / `by_list`** | **Yes** — `correct_row`, `missing_row`, `spurious_row` | **No** |
| **Leaf scoring** | **Yes** — at each gold index after pairing | **Yes** — for eval-only row fields (`figures[1].figure_label`, …) if present in gold/pred and manifest |
| **Extra pred-only rows** | `spurious_row` in `by_list`; no gold-indexed instances | Ignored; not reported as `spurious_row` |
| **Missing pred rows** | `missing_row` in `by_list`; gold leaves get `pred_value = null` | Gold leaves get `pred_value = null`; not reported as `missing_row` |

**Model output vs collation.** One model call returns JSON validating against `schema.json` (e.g. `{ "panels": [ … ] }`). Benchmark gold/pred may **collate** many such outputs: `{ "figures": [ { "panels": [ … ] }, … ] }`. Here `figures[]` is structural (not in `schema.json`); `figures[i].panels[]` is the same predictive list as `panels[]` in the schema, reached through a collation prefix. Evaluation walks structural ancestors by index, then aligns predictive lists per schema inside each slice.

**Example:** panel-caption segmentation — `schema.json` has `panels[]` only. Figure-benchmark eval JSON wraps each run under `figures[]`. `list_alignment` uses `"figures.panels": ["label"]` (path in the **evaluation document**); `fields` may use `figures[].panels[].label`. Row fields on `figures[]` itself (e.g. `figure_label` copied from input) are eval metadata — positional join, no layer S.

**Discovering predictive lists:** walk **`schema.json` only**; every `array` whose `items` is an `object` is predictive. There is no “structural list in schema” case — wrappers exist only in evaluation documents.

**Manifest `list_alignment` — configuration and validation, not classification:**

| Rule | Meaning |
|------|---------|
| **Required** | Every predictive list (per schema) has exactly one `list_alignment` entry |
| **Forbidden** | Keys for structural / collection lists |
| **Key shape** | Dotted path without `[]`: `panels`, `figures.panels`, … matching the list’s position in the eval document |
| **Value shape** | Non-empty array of row field names that exist on that list’s `items.properties` |
| **Profile consistency** | Each alignment key has a matching `fields` entry (e.g. `panels[].label`) when that field is used for `s(i,j)` |

At manifest load (or evaluator init), reject inconsistent manifests: missing alignment for a predictive list, alignment on a structural path, or alignment keys not in the row schema.

#### Where `list_alignment` lives in the manifest

`list_alignment` is a **top-level key** in `eval-manifest.json`, alongside `defaults` and `fields`. It is **not** nested under `fields`.

Keys use the same dotted path as the list in the evaluation document (no `[]` suffix). Values are **row field names** — short names from the row object’s `properties`, not full leaf paths. Keys must correspond to **predictive** lists expected from the schema (see validation table above).

```json
{
  "defaults": { "…": "…" },
  "list_alignment": {
    "panels": ["label"]
  },
  "fields": {
    "panels[].label": {
      "matching_metric": "graded_string",
      "string_compare": "exact",
      "match_threshold": 1.0
    },
    "panels[].status": {
      "na_values": [""],
      "matching_metric": "binary_polarity"
    }
  }
}
```

| Manifest location | Example | Meaning |
|-------------------|---------|---------|
| `list_alignment` key | `"panels"` | Align predictive `panels` at check output root |
| `list_alignment` key | `"figures.panels"` | Align predictive `panels` nested under structural `figures` |
| `list_alignment` value | `["label"]` | Row fields used only for Hungarian `s(i,j)` — not for layer 1/2 rollups |
| `fields` key | `"panels[].label"` | How to **score** that leaf after rows are joined (layer 1/2, `string_compare`, …) |
| *(no key — structural list)* | — | `figures[]` collection wrapper: positional join, no `by_list`; may still have `figures[].figure_label` in `fields` |

For a check whose schema produces `outputs[]`, `list_alignment` must include `"outputs": […]`. A collation-only `figures[]` wrapper must **not** appear in `list_alignment`; row leaves under `figures[].*` are still scored via positional join.

#### Row similarity (alignment only)

For each candidate pair `(exp_i, pred_j)`:

```text
s(i,j) = mean{ score(exp_i[k], pred_j[k]) : k in list_alignment[list_name] }
```

The score for each alignment key `k` uses that field’s **`fields`** profile when present (e.g. `panels[].label` → `string_compare: exact`, `match_threshold: 1.0`). With one key, `s(i,j)` is that field’s leaf score. `s(i,j)` is **only for Hungarian matching** — not reported in `by_property`.

**Concrete example** — toy `panels` with `list_alignment.panels: ["label"]` and exact labels (`τ = 1.0`):

Gold:

```json
[
  { "id": 1, "label": "Fig 1", "status": "yes" },
  { "id": 2, "label": "Fig 2", "status": "" }
]
```

Pred (rows reordered; `Fig 1` has wrong `status`):

```json
[
  { "id": 9, "label": "Fig 2", "status": "" },
  { "id": 8, "label": "Fig 1", "status": "no" }
]
```

Hungarian pairs by `label`: gold `[0]` ↔ pred `[1]`, gold `[1]` ↔ pred `[0]`. Then emit leaves at **gold indices** — `panels[0].status` compares gold `yes` vs pred `no`; `panels[0].id` compares `1` vs `8`. Wrong `id` on a correctly aligned row does not change how rows were matched.

#### Alignment steps (predictive lists only)

For each list named in `list_alignment`:

1. Look up keys, e.g. `list_alignment["panels"]` → `["label"]`.
2. Build `s(i,j)` from those keys only, using each key’s `fields` profile for scoring.
3. Hungarian assignment; emit layer S structural reporting (see below).
4. For each gold row index `k` with `correct_row`, take the paired pred row and emit all row leaves (`panels[k].id`, `panels[k].label`, `panels[k].status`, …).
5. For each gold row index `k` with `missing_row`, emit row leaves at `k` with `pred_value = null` and `score = 0`.
6. Score each leaf instance; apply layer 1 / 2 per `fields` profiles.

**Spurious pred rows** (`spurious_row`) do not receive gold-indexed leaf instances.

Other row fields (e.g. `status`) are scored **after** row pairing. A wrong `status` on a `correct_row` affects only that leaf — not how rows were matched.

#### Positional join (structural lists)

For object lists classified as **structural** (not in the schema’s predictive set — therefore no `list_alignment` entry):

1. For each gold row index `k` from `0` to `len(gold_rows) - 1`, take `pred_rows[k]` when `k < len(pred_rows)`, else treat the pred row as absent.
2. Emit row leaves at gold index `k` with values from the gold row and the positional pred row (or `pred_value = null`).
3. Do **not** run Hungarian alignment; do **not** emit `by_list` for this list.
4. Extra pred rows beyond the gold length are ignored (same effect as `spurious_row` on leaf paths, but not counted in layer S).

When a predictive list is nested under a structural parent (e.g. `panels` under `figures`), repeat the positional join for each parent gold index, then run predictive alignment **within** that parent’s gold/pred row slice.

---

## Evaluation manifest

**Location:** alongside checklist schema, e.g.  
`soda_mmqc/data/checklist/fig-checklist/<checklist-name>/eval-manifest.json`

**Purpose:** map leaf property paths → metric profile. The evaluator loads the model `schema.json` + `eval-manifest.json`. Evaluation semantics live here, **not** in `schema.json` (which is **only** the per-call structured-output contract for the model — no collation wrappers). Manifest `fields` paths may include collation prefixes (e.g. `figures[].panels[].label`) when eval gold/pred embeds model output that way.

**Path keys:** JSON-pointer-style patterns from the output root, e.g. `outputs[].scale_bar_on_image`, `item.status`. `[]` matches any list index.

**Example** (enum polarity slots omit `string_compare`; free-text `graded_string` fields declare it):

```json
{
  "checklist": "toy-eval-examples",
  "defaults": {
    "matching_metric": "binary_polarity",
    "positive_value": "yes",
    "negative_value": "no",
    "na_values": []
  },
  "list_alignment": {
    "panels": ["label"]
  },
  "fields": {
    "item.status": {
      "na_values": [""],
      "matching_metric": "binary_polarity"
    },
    "item.label": {
      "matching_metric": "graded_string",
      "string_compare": "exact",
      "match_threshold": 1.0
    },
    "panels[].status": {
      "na_values": [""],
      "matching_metric": "binary_polarity"
    },
    "panels[].label": {
      "matching_metric": "graded_string",
      "string_compare": "exact",
      "match_threshold": 1.0
    },
    "outputs[].from_the_caption": {
      "matching_metric": "graded_string",
      "string_compare": "semantic",
      "match_threshold": 0.8
    }
  }
}
```

- **`item.status`**, **`panels[].status`** — schema `enum` (`yes` / `no` / `""`); `binary_polarity` only; exact match implicit.
- **`item.label`**, **`panels[].label`** — free `string`; `graded_string` + `string_compare` + `match_threshold` (also used for `list_alignment` on `panels`).
- **`outputs[].from_the_caption`** — free text; semantic compare with a lower threshold.

**Manifest keys:**

| Key | Role |
|-----|------|
| `list_alignment` | Map each **schema predictive list** → row field names for Hungarian `s(i,j)` only (e.g. `"panels": ["label"]`). Must match schema predictive set; validated at load. Structural collection lists omitted. |
| `na_values` | Per field: literals meaning **not applicable** (layer 1). Omit or `[]` = always applicable. |
| `matching_metric` | `binary_polarity` \| `multiclass` \| `graded_string` — **layer 2** reporting shape only |
| `string_compare` | **Required** on non-enum `graded_string` fields. **Omit** on schema `enum` fields (always exact). Not a `defaults` key. |
| `positive_value` / `negative_value` | For `binary_polarity` (layer 2) |
| `match_threshold` | For `graded_string` only: layer 2 **match** if `score >= match_threshold`; also gates list alignment when an alignment key is a graded string |

Put polarity defaults in manifest `defaults`; set `string_compare` per free-text `graded_string` field only.

### `string_compare` — how the leaf score is computed

For free-text and other non-enum strings, the manifest names the **comparison method**. This replaces a single global evaluator setting — each leaf property can differ.


| `string_compare` | `score` computation | Typical use |
|------------------|---------------------|-------------|
| **`exact`** | Character-level equality (after optional normalization) → `1.0` or `0.0` | Titles, labels, citation snippets that must match literally |
| **`fuzzy`** | rapidfuzz `fuzz.ratio` (edit-distance) in `[0, 1]` | Minor typos, whitespace variants |
| **`semantic`** | Embedding cosine similarity in `[0, 1]` (model from evaluator config) | Captions, free-text descriptions where paraphrase is OK |

**Schema `enum` fields** (`binary_polarity`, `multiclass`): always **exact** literal match for `score`. Do not set `string_compare` — it would be redundant and misleading.

**Pipeline for a non-enum `graded_string` field:**

1. `score = compare(exp, pred)` using `string_compare`.
2. Layer 1 from `na_values` (if any).
3. Layer 2: **match** if `score >= match_threshold`, else **mismatch**.

**List alignment:** when an alignment key is a free-text field (e.g. `list_alignment.panels: ["label"]` on a `graded_string` path), use that field’s `string_compare` and `match_threshold` to compute `s(i,j)`. Enum alignment keys use exact match only (no `string_compare`).

**Schema vs manifest:** JSON Schema `enum` lists allowed literals but does **not** define which is N/A or which is “positive”. Configure `na_values` and polarity in the manifest. Repo examples: `enum: ["yes", "no", ""]` needs `na_values: [""]`; `enum: ["yes", "no", "not needed"]` needs `na_values: ["not needed"]`.

Leaves without a manifest entry receive **score** only.

---

## Layer S — Structural row alignment

**Layer S** reports whether each **row slot** in a **predictive** list-of-objects was correctly paired, missing from pred, or spurious in pred. It applies **only** to lists the schema classifies as model-produced — **not** to structural collection lists (`figures`, `papers`, …), and **not** to extended-primitive leaves such as `tags`.

Layer S is **orthogonal** to layer 1 and layer 2:

| Layer | Scope | Question |
|-------|-------|----------|
| **S** | Row slot in a list of objects | Was this gold/pred row correctly present and paired? |
| **1** | Leaf value | Was each side applicable (N/A vs real answer)? |
| **2** | Leaf value | Given both applicable, did values match per `matching_metric`? |

**Structural reporting outcomes:**

| Outcome | Meaning |
|---------|---------|
| **correct_row** | Gold row `i` paired with pred row `j`, `s(i,j) >= τ` |
| **missing_row** | Gold row with no acceptable pred partner (unmatched after Hungarian, or Hungarian pair with `s(i,j) < τ`) |
| **spurious_row** | Pred row with no acceptable gold partner (unmatched after Hungarian, or Hungarian pair with `s(i,j) < τ`) |

When Hungarian assigns a pair below `τ`, emit **both** a `missing_row` for the gold index and a `spurious_row` for the pred index — they are not linked as `correct_row`.

**`by_list` row records** use nullable indices in a single `rows` array:

```json
{
  "gold_index": 0,
  "pred_index": 1,
  "structural": "correct_row",
  "similarity": 1.0
}
```

| Situation | `gold_index` | `pred_index` | `structural` |
|-----------|--------------|--------------|--------------|
| Paired above τ | `k` | `j` | `correct_row` |
| Gold row unmatched | `k` | `null` | `missing_row` |
| Pred row unmatched | `null` | `j` | `spurious_row` |

`similarity` is the alignment score `s(i,j)` on `correct_row` entries; `null` otherwise.

**List-level metrics** from `row_counts`:

```text
recall    = correct_row / (correct_row + missing_row)
precision = correct_row / (correct_row + spurious_row)
```

When `correct_row + missing_row = 0`, recall is undefined (empty gold list). When `correct_row + spurious_row = 0`, precision is undefined (no pred rows accepted).

**Relationship to leaf instances on `missing_row`:**

A `missing_row` still emits leaf instances at the gold index with `pred_value = null` and `score = 0`. Layer 1 applicability reporting may yield `withheld_applicable` or `correct_NA` per field profile. Layer 2 matching reporting is omitted. Those field-level outcomes describe **values within** a structurally missing row; layer S describes the **row event** itself.

**`spurious_row` entries** have no gold-indexed leaf instances. Optional metadata on the row record (e.g. alignment-key values from the pred row) may aid debugging but does not create phantom leaf paths.

---

## Layer 1 — Applicability

For each instance with a manifest profile:


| Gold applicable? | Pred applicable? | Outcome |
|------------------|------------------|---------|
| no (`na_values`) | no | **correct_NA** |
| no | yes | **spurious_applicable** |
| yes | no | **withheld_applicable** |
| yes | yes | **correct_applicable** (eligible for layer 2) |

**Applicable** = value present and ∉ `na_values`. Missing pred when gold expected an applicable value → **withheld_applicable**. Missing pred when gold is N/A (`""` ∈ `na_values`) → **correct_NA** (pred did not spuriously answer).

`layer1_counts` in `by_property` tally these outcomes across instances of that property.

---

## Layer 2 — Matching

Runs only when layer 1 = **correct_applicable**. Denominator: instances where gold was applicable.

**TP**, **FP**, **FN**, and **TN** are **layer 2 matching reporting outcomes only** — value-level polarity on profiled leaves (`binary_polarity`). They do **not** describe missing or spurious **rows**; those are layer S (`missing_row`, `spurious_row`).

| `matching_metric` | Layer 2 outcomes |
|-----------------|------------------|
| **binary_polarity** | **TP**, **FP**, **FN**, **TN** (`positive_value` / `negative_value`, applicable gold only) |
| **multiclass** | **match** / **mismatch**; build confusion matrix per property |
| **graded_string** | **match** / **mismatch** where `score >= match_threshold` |

`no` / `no` when `negative_value` is `"no"` → **TN**. Gold positive + pred negative on a `correct_row` → **FN**; gold negative + pred positive → **FP**. Layer 2 for `correct_NA` and `withheld_applicable` instances is omitted (not counted in `layer2_counts`).

---

## What this design excludes

- Scores at intermediate object or list nodes (objects are not leaf values).
- A single global score mixing unrelated leaf properties.
- Inferring N/A or polarity from enum order or English word shape alone.
- Layer S structural reporting on extended-primitive leaves (`tags`).
- Gold-indexed leaf instances for `spurious_row` pred rows.

---

## Implementation

Target modules: [`leaves.py`](../soda_mmqc/core/leaves.py) (scalar + `compare_primitive_list`), [`matching.py`](../soda_mmqc/core/matching.py) (shared Hungarian), [`object_list_pairing.py`](../soda_mmqc/core/object_list_pairing.py) (row pairing), [`structural_reporting.py`](../soda_mmqc/core/structural_reporting.py) (layer S / `by_list`), [`eval_manifest.py`](../soda_mmqc/core/eval_manifest.py), [`applicability_and_matching.py`](../soda_mmqc/core/applicability_and_matching.py) (layers 1 and 2), [`evaluation.py`](../soda_mmqc/core/evaluation.py) (orchestrator). Contract: flatten → pair **predictive** object lists (Hungarian) / join **structural** lists (positional) → layer S for predictive lists only → score leaf instances → layers 1 and 2 → `instances` + `by_list` + `by_property`.

---

## See also

- [Toy examples](evaluation-toy-examples.md) — gold/pred walkthroughs on a shared schema and manifest.
- [Leaf primitives](evaluation-leaf-primitives.md) — additional detail on string comparison modes and `LeafComparisonResult` target types.
