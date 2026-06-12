# Evaluation toy examples

Worked **gold / pred** pairs for [evaluation scoring](evaluation-scoring.md). One shared toy schema and manifest; each example lists **leaf instances** and **`by_property`** summaries.

**Assumptions:**

- **Enum leaves** (`item.status`, `panels[].status`) — always exact literal match; no `string_compare` in manifest.
- **Graded strings** (`item.label`, `panels[].label`) — `string_compare: exact`, `match_threshold: 1.0` (`τ = 1.0`).
- **`tags`** — extended-primitive leaf (`string[]`): Hungarian element matching, one aggregate `score` on path `tags`. No layer S. No manifest profile → score only.
- **`panels[]`** — top-level manifest `list_alignment.panels: ["label"]` pairs rows by `label` (using the `panels[].label` field profile); layer S in `by_list.panels`; then score `panels[k].id`, `panels[k].label`, `panels[k].status` at **gold indices** `k`.
- **Manifest** — layer 1 / 2 only on profiled paths (`item.status`, `item.label`, `panels[].status`, `panels[].label`). Unprofiled leaves appear in `by_property` with `mean_score` only (`layer1_counts` / `layer2_counts` = `{}`).

---

## Leaf inventory (toy schema)

| Path | Value type | Manifest profile |
|------|------------|------------------|
| `tags` | `string[]` | — (score only) |
| `item.id` | integer | — |
| `item.label` | string | `graded_string`, `string_compare: exact` |
| `item.status` | enum `""` / `yes` / `no` | `binary_polarity`, `na_values: [""]` |
| `item.meta.author` | string | — |
| `item.meta.year` | integer | — |
| `panels[k].id` | integer | — |
| `panels[k].label` | string | `graded_string`, `string_compare: exact` (pattern `panels[].label`; also `list_alignment` key for `panels`) |
| `panels[k].status` | enum | `binary_polarity`, `na_values: [""]` |

Row alignment: manifest `list_alignment.panels → ["label"]` (top-level key, not under `fields`).

**How to read each example:**

1. **What changed** — diff vs baseline gold.
2. **Per leaf instance** — table of concrete paths: `score`, layer 1, layer 2 (— when not applicable).
3. **Per leaf property** — required summary for each property key (`item.label`, `panels[].status`, …): `mean_score`, `layer1_counts`, `layer2_counts` — **never mixed across properties** ([evaluation-scoring.md](evaluation-scoring.md#per-leaf-property-by_property)). Tables use `{}` for empty count objects.
4. **Layer S (`by_list`)** — structural reporting for list-of-objects properties only: `row_counts` and per-row `structural` outcomes (`correct_row`, `missing_row`, `spurious_row`). Not in `by_property`.

Layer 1 applicability reporting: **correct_NA** · **spurious_applicable** · **withheld_applicable** · **correct_applicable**.  
Layer 2 matching reporting runs only when layer 1 = **correct_applicable**.  
Layer S structural reporting: **correct_row** · **missing_row** · **spurious_row**.

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
    "matching_metric": "binary_polarity",
    "positive_value": "yes",
    "negative_value": "no",
    "na_values": []
  },
  "list_alignment": {
    "panels": ["label"]
  },
  "fields": {
    "item.status": { "na_values": [""], "matching_metric": "binary_polarity" },
    "item.label": { "matching_metric": "graded_string", "string_compare": "exact", "match_threshold": 1.0 },
    "panels[].status": { "na_values": [""], "matching_metric": "binary_polarity" },
    "panels[].label": { "matching_metric": "graded_string", "string_compare": "exact", "match_threshold": 1.0 }
  }
}
```

- **`list_alignment.panels`** — pair rows by `label` only (uses `panels[].label` profile for `s(i,j)`).
- **`item.status`**, **`panels[].status`** — schema `enum`; `binary_polarity` only (no `string_compare`).
- **`item.label`**, **`panels[].label`** — free `string`; `graded_string` + `string_compare` + `match_threshold`.

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
| `tags` | 1.0 | `{}` | `{}` |
| `item.id` | 1.0 | `{}` | `{}` |
| `item.label` | 1.0 | { correct_applicable: 1 } | { match: 1 } |
| `item.status` | 1.0 | { correct_NA: 1 } | `{}` |
| `item.meta.author` | 1.0 | `{}` | `{}` |
| `item.meta.year` | 1.0 | `{}` | `{}` |
| `panels[].id` | 1.0 | `{}` | `{}` |
| `panels[].label` | 1.0 | { correct_applicable: 2 } | { match: 2 } |
| `panels[].status` | 1.0 | { correct_applicable: 1, correct_NA: 1 } | { TP: 1 } |

**`by_list.panels`:**

| `row_counts` | `rows` |
|--------------|--------|
| { correct_row: 2, missing_row: 0, spurious_row: 0 } | `[{ gold_index: 0, pred_index: 0, structural: correct_row, similarity: 1.0 }, { gold_index: 1, pred_index: 1, structural: correct_row, similarity: 1.0 }]` |

No row pools `tags` with `panels[].*` or `item.*`.

---

## B — Extended primitive (`tags`)

Other leaves = baseline. Only `tags` differs. No `by_list` entry for `tags`.

### B.1 Extra element

| `tags` pred | `["alpha", "beta", "gamma"]` |
|-------------|------------------------------|

| Path | `score` |
|------|---------|
| `tags` | 2/3 ≈ 0.67 |

**`by_property`:** `tags` → `mean_score` 0.67 only; other leaf properties unchanged from example A.

### B.2 Substituted element

| `tags` gold | `["alpha", "beta", "gamma"]` |
| `tags` pred | `["alpha", "beta", "delta"]` |

| Path | `score` |
|------|---------|
| `tags` | 2/3 ≈ 0.67 |

### B.3 Total mismatch (length 1)

| `tags` gold | `["alpha"]` |
| `tags` pred | `["omega"]` |

| Path | `score` |
|------|---------|
| `tags` | 0.0 |

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

**`by_property`:** `item.label` → `mean_score` 0.0, `layer1_counts` { withheld_applicable: 1 }, `layer2_counts` `{}`.

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

**Row alignment** ([evaluation-scoring.md](evaluation-scoring.md#list-of-objects)): top-level `list_alignment.panels → ["label"]` — Hungarian matching uses **only** `label` (via the `panels[].label` field profile). `id` and `status` are scored after pairing at gold indices.

### D.1 Perfect panels

Same as example A panel rows.

### D.2 Reordered rows (alignment by `label` only)

Pred rows swapped; `Fig 1` has wrong `status` and `id`:

| `panels` pred |
|---------------|
| `[ { "id": 9, "label": "Fig 2", "status": "" }, { "id": 8, "label": "Fig 1", "status": "no" } ]` |

**`by_list.panels`:** both rows `correct_row` — gold `[0]` ↔ pred `[1]`; gold `[1]` ↔ pred `[0]` (`similarity` 1.0 each). Wrong `status` / `id` on matched rows does not change pairing.

| Path | exp → pred | `score` | Layer 1 | Layer 2 |
|------|------------|---------|---------|---------|
| `panels[0].id` | `1` → `8` | 0.0 | — | — |
| `panels[0].label` | `"Fig 1"` → `"Fig 1"` | 1.0 | correct_applicable | match |
| `panels[0].status` | `"yes"` → `"no"` | 0.0 | correct_applicable | FN |
| `panels[1].id` | `2` → `9` | 0.0 | — | — |
| `panels[1].label` | `"Fig 2"` → `"Fig 2"` | 1.0 | correct_applicable | match |
| `panels[1].status` | `""` → `""` | 1.0 | correct_NA | — |

**`by_property` (panel leaves only):** `panels[].id` → `mean_score` 0.0; `panels[].label` → `mean_score` 1.0, `layer2_counts` { match: 2 }; `panels[].status` → `mean_score` 0.5, `layer1_counts` { correct_applicable: 1, correct_NA: 1 }, `layer2_counts` { FN: 1 }.

### D.3 Wrong `status` on panel 1 (gold index 1)

| | `panels[1].status` | other panel fields |
|---|-------------------|-------------------|
| Gold | `""` | baseline |
| Pred | `"yes"` | baseline |

**Row alignment:** both panels match by `label` (`"Fig 1"`, `"Fig 2"`). The `status` error does not affect pairing.

| Path | `score` | Layer 1 | Layer 2 |
|------|---------|---------|---------|
| `panels[1].status` | 0.0 | spurious_applicable | — |
| `panels[0].*` | 1.0 | *(as in A)* | *(as in A)* |

### D.4 Missing panel (gold row 1 has no pred partner)

| `panels` pred | `[ { "id": 1, "label": "Fig 1", "status": "yes" } ]` |

**`by_list.panels`:**

| `row_counts` | `rows` |
|--------------|--------|
| { correct_row: 1, missing_row: 1, spurious_row: 0 } | `[{ gold_index: 0, pred_index: 0, structural: correct_row, similarity: 1.0 }, { gold_index: 1, pred_index: null, structural: missing_row, similarity: null }]` |

List recall = 1/2.

| Path | exp → pred | `score` | Layer 1 | Layer 2 |
|------|------------|---------|---------|---------|
| `panels[0].*` | baseline → matched | 1.0 | *(as in A)* | *(as in A)* |
| `panels[1].id` | `2` → *(missing)* | 0.0 | — | — |
| `panels[1].label` | `"Fig 2"` → *(missing)* | 0.0 | withheld_applicable | — |
| `panels[1].status` | `""` → *(missing)* | 0.0 | correct_NA | — |

Layer S records the missing row. Field-level layer 1 on that row: missing pred on an N/A gold field (`""`) → **correct_NA**; missing pred when gold expected an applicable label → **withheld_applicable**. No layer 2 on `missing_row` leaves.

### D.5 Extra panel in pred

| `panels` pred | baseline panels + `{ "id": 99, "label": "Fig 99", "status": "no" }` |

**`by_list.panels`:**

| `row_counts` | `rows` |
|--------------|--------|
| { correct_row: 2, missing_row: 0, spurious_row: 1 } | `[{ gold_index: 0, pred_index: 0, structural: correct_row, similarity: 1.0 }, { gold_index: 1, pred_index: 1, structural: correct_row, similarity: 1.0 }, { gold_index: null, pred_index: 2, structural: spurious_row, similarity: null }]` |

List precision = 2/3. The spurious pred row (`Fig 99`) has no gold-indexed leaf instances.

| Path | `score` | Layer 1 | Layer 2 |
|------|---------|---------|---------|
| `panels[0].*`, `panels[1].*` | 1.0 | *(as in A)* | *(as in A)* |

---

## Quick reference

| Topic | Section |
|-------|---------|
| All leaves perfect | A |
| Extended primitive (`tags`) | B |
| Layer S (`by_list.panels`) | D.4, D.5 |
| `item.*` / `item.meta.*` scalars | C |
| `panels[k].*` after alignment | D (D.2 = reordered rows) |

---

## See also

- [Evaluation scoring](evaluation-scoring.md) — normative rules for this model.
- [Leaf primitives](evaluation-leaf-primitives.md) — string comparison modes and graded scores.
