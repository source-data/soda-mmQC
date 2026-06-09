# Evaluation toy examples

Worked **gold / pred** pairs for [evaluation scoring](evaluation-scoring.md). One shared toy schema and manifest; each example lists **leaf instances** and **`by_property`** summaries.

**Assumptions:**

- **Exact match** on primitives (`τ = 1.0`) unless noted.
- **`tags`** — list-of-primitives leaf: Hungarian alignment, one aggregate `score` on path `tags` (+ alignment TP/FP/FN diagnostics).
- **`panels[]`** — align panel **rows** first; then leaves `panels[k].id`, `panels[k].label`, `panels[k].status` for each gold index `k` using the matched pred row (or missing).
- **Manifest** — layer 1 / 2 only on paths with a profile (`item.status`, `item.label`, `panels[].status`, `panels[].label`).

---

## Leaf inventory (toy schema)

| Path | Value type | Manifest profile |
|------|------------|------------------|
| `tags` | `string[]` | — (score only) |
| `item.id` | integer | — |
| `item.label` | string | `graded_string` |
| `item.status` | enum `""` / `yes` / `no` | `binary_polarity`, `na_values: [""]` |
| `item.meta.author` | string | — |
| `item.meta.year` | integer | — |
| `panels[k].id` | integer | — |
| `panels[k].label` | string | `graded_string` (pattern `panels[].label`) |
| `panels[k].status` | enum | `binary_polarity`, `na_values: [""]` |

**How to read each example:**

1. **What changed** — diff vs baseline gold.
2. **Per leaf instance** — table of concrete paths: `score`, layer 1, layer 2 (— when not applicable).
3. **Per leaf property** — required summary for each property key (`item.label`, `panels[].status`, …): `mean_score`, `layer1_counts`, `layer2_counts` — **never mixed across properties** ([evaluation-scoring.md](evaluation-scoring.md#2--per-leaf-property-required)).
4. **List alignment** — diagnostics for `tags` or `panels` row matching only.

Layer 1 labels: **correct_NA** · **spurious_applicable** · **withheld_applicable** · **correct_applicable**.  
Layer 2 runs only when layer 1 = **correct_applicable**.

---

## Shared toy schema

```json
{
  "type": "object",
  "required": ["tags", "item", "panels"],
  "properties": {
    "tags": { "type": "array", "items": { "type": "string" } },
    "item": {
      "type": "object",
      "required": ["id", "label", "status", "meta"],
      "properties": {
        "id": { "type": "integer" },
        "label": { "type": "string" },
        "status": { "type": "string", "enum": ["", "yes", "no"] },
        "meta": {
          "type": "object",
          "required": ["author", "year"],
          "properties": {
            "author": { "type": "string" },
            "year": { "type": "integer" }
          }
        }
      }
    },
    "panels": {
      "type": "array",
      "items": {
        "type": "object",
        "required": ["id", "label", "status"],
        "properties": {
          "id": { "type": "integer" },
          "label": { "type": "string" },
          "status": { "type": "string", "enum": ["", "yes", "no"] }
        }
      }
    }
  }
}
```

**Baseline gold:**

```json
{
  "tags": ["alpha", "beta"],
  "item": {
    "id": 1,
    "label": "Panel A",
    "status": "",
    "meta": { "author": "Ada", "year": 1843 }
  },
  "panels": [
    { "id": 1, "label": "Fig 1", "status": "yes" },
    { "id": 2, "label": "Fig 2", "status": "" }
  ]
}
```

**Baseline pred** = baseline gold (perfect run) unless an example says otherwise.

---

## Shared toy manifest

```json
{
  "checklist": "toy-eval-examples",
  "defaults": {
    "answer_metric": "binary_polarity",
    "positive_value": "yes",
    "negative_value": "no",
    "na_values": []
  },
  "list_alignment": {
    "panels": ["label"]
  },
  "fields": {
    "item.status": { "na_values": [""], "answer_metric": "binary_polarity" },
    "item.label": { "answer_metric": "graded_string", "match_threshold": 1.0 },
    "panels[].status": { "na_values": [""], "answer_metric": "binary_polarity" },
    "panels[].label": { "answer_metric": "graded_string", "match_threshold": 1.0 }
  }
}
```

---

## A — Baseline (all leaves, perfect)

Pred = baseline gold.

| Path | exp → pred | `score` | Layer 1 | Layer 2 |
|------|------------|---------|---------|---------|
| `tags` | `["alpha","beta"]` → same | 1.0 | — | — |
| `item.id` | `1` → `1` | 1.0 | — | — |
| `item.label` | `"Panel A"` → same | 1.0 | correct_applicable | match |
| `item.status` | `""` → `""` | 1.0 | correct_NA | — |
| `item.meta.author` | `"Ada"` → same | 1.0 | — | — |
| `item.meta.year` | `1843` → same | 1.0 | — | — |
| `panels[0].id` | `1` → `1` | 1.0 | — | — |
| `panels[0].label` | `"Fig 1"` → same | 1.0 | correct_applicable | match |
| `panels[0].status` | `"yes"` → same | 1.0 | correct_applicable | TP |
| `panels[1].id` | `2` → `2` | 1.0 | — | — |
| `panels[1].label` | `"Fig 2"` → same | 1.0 | correct_applicable | match |
| `panels[1].status` | `""` → `""` | 1.0 | correct_NA | — |

**Per leaf property (`by_property`):**

| Leaf property | `mean_score` | `layer1_counts` | `layer2_counts` |
|---------------|--------------|-----------------|-----------------|
| `tags` | 1.0 | — | — |
| `item.id` | 1.0 | — | — |
| `item.label` | 1.0 | correct_applicable: 1 | match: 1 |
| `item.status` | 1.0 | correct_NA: 1 | — |
| `item.meta.author` | 1.0 | — | — |
| `item.meta.year` | 1.0 | — | — |
| `panels[].id` | 1.0 | — | — |
| `panels[].label` | 1.0 | correct_applicable: 2 | match: 2 |
| `panels[].status` | 1.0 | correct_applicable: 1, correct_NA: 1 | TP: 1 |

No row pools `tags` with `panels[].*` or `item.*`.

---

## B — List of primitives (`tags`)

Other leaves = baseline. Only `tags` differs.

### B.1 Extra element

| `tags` pred | `["alpha", "beta", "gamma"]` |
|-------------|------------------------------|

| Path | `score` | List alignment |
|------|---------|----------------|
| `tags` | 2/3 ≈ 0.67 | TP=2, FP=1, FN=0 |

**`by_property`:** `tags` → `mean_score` 0.67 only; other leaf properties unchanged from example A.

### B.2 Substituted element

| `tags` gold | `["alpha", "beta", "gamma"]` |
| `tags` pred | `["alpha", "beta", "delta"]` |

| Path | `score` | List alignment |
|------|---------|----------------|
| `tags` | 2/3 ≈ 0.67 | TP=2, FP=1, FN=1 |

### B.3 Total mismatch (length 1)

| `tags` gold | `["alpha"]` |
| `tags` pred | `["omega"]` |

| Path | `score` | List alignment |
|------|---------|----------------|
| `tags` | 0.0 | TP=0, FP=1, FN=1 |

---

## C — Scalar leaves on `item`

`tags` and `panels` = baseline.

### C.1 Wrong `item.label`

| Field | Gold | Pred |
|-------|------|------|
| `item.label` | `"Panel A"` | `"Panel B"` |

| Path | `score` | Layer 1 | Layer 2 |
|------|---------|---------|---------|
| `item.label` | 0.0 | correct_applicable | mismatch |
| *(other `item.*` leaves)* | 1.0 | *(as in example A)* | *(as in A)* |

**`by_property`:** `item.label` → `mean_score` 0.0, `layer1_counts` { correct_applicable: 1 }, `layer2_counts` { mismatch: 1 }.

### C.2 `item.status` spurious applicable

| Field | Gold | Pred |
|-------|------|------|
| `item.status` | `""` | `"yes"` |

| Path | `score` | Layer 1 | Layer 2 |
|------|---------|---------|---------|
| `item.status` | 0.0 | spurious_applicable | — |

**`by_property`:** `item.status` → `mean_score` 0.0, `layer1_counts` { spurious_applicable: 1 }, `layer2_counts` {}.

### C.3 Missing `item.label`

| Field | Gold | Pred |
|-------|------|------|
| `item.label` | `"Panel A"` | *(absent)* |

| Path | `score` | Layer 1 | Layer 2 |
|------|---------|---------|---------|
| `item.label` | 0.0 | withheld_applicable | — |

### C.4 Wrong `item.meta.year` (flat leaf, not a nested roll-up)

| Field | Gold | Pred |
|-------|------|------|
| `item.meta.year` | `1843` | `1900` |

| Path | `score` | Layer 1 | Layer 2 |
|------|---------|---------|---------|
| `item.meta.year` | 0.0 | — | — |

`item.meta.author` and other leaves unchanged at 1.0. There is no parent `meta` score — only the two primitive leaves matter.

---

## D — `panels[]` leaves (after row alignment)

`tags` and `item` = baseline.

**Row alignment** ([evaluation-scoring.md](evaluation-scoring.md#list-of-objects)): `list_alignment.panels = ["label"]` — Hungarian matching uses **only** `label` scores between candidate rows. Other fields (`id`, `status`) are scored after pairing.

### D.1 Perfect panels

Same as example A panel rows.

### D.2 Wrong `status` on panel 1 (gold index 1)

| | `panels[1].status` | other panel fields |
|---|-------------------|-------------------|
| Gold | `""` | baseline |
| Pred | `"yes"` | baseline |

**Row alignment:** both panels match by `label` (`"Fig 1"`, `"Fig 2"`). The `status` error does not affect pairing.

| Path | `score` | Layer 1 | Layer 2 |
|------|---------|---------|---------|
| `panels[1].status` | 0.0 | spurious_applicable | — |
| `panels[0].*` | 1.0 | *(as in A)* | *(as in A)* |

### D.3 Missing panel (gold row 1 has no pred partner)

| `panels` pred | `[ { "id": 1, "label": "Fig 1", "status": "yes" } ]` |

**Row alignment:** gold panel 0 matches; gold panel 1 unmatched (alignment FN=1).

| Path | exp → pred | `score` | Layer 1 | Layer 2 |
|------|------------|---------|---------|---------|
| `panels[0].*` | baseline → matched | 1.0 | *(as in A)* | *(as in A)* |
| `panels[1].id` | `2` → *(missing)* | 0.0 | — | — |
| `panels[1].label` | `"Fig 2"` → *(missing)* | 0.0 | withheld_applicable | — |
| `panels[1].status` | `""` → *(missing)* | 0.0 | correct_NA | — |

Missing pred on an N/A gold field (`""`) → **correct_NA** (pred did not spuriously answer). Missing pred when gold expected a string label → **withheld_applicable**.

### D.4 Extra panel in pred

| `panels` pred | baseline panels + `{ "id": 99, "label": "Fig 99", "status": "no" }` |

**Row alignment:** baseline rows match; extra pred row → alignment FP=1 (unpaired pred index not given leaf paths in this toy).

| Path | `score` | Layer 1 | Layer 2 |
|------|---------|---------|---------|
| `panels[0].*`, `panels[1].*` | 1.0 | *(as in A)* | *(as in A)* |

---

## Quick reference

| Topic | Section |
|-------|---------|
| All leaves perfect | A |
| `tags` only | B |
| `item.*` / `item.meta.*` scalars | C |
| `panels[k].*` after alignment | D |

---

## See also

- [Evaluation scoring](evaluation-scoring.md) — normative rules for this model.
- [Leaf primitives](evaluation-leaf-primitives.md) — string comparison modes and graded scores.
