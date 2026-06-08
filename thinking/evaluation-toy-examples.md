# Evaluation toy examples

Worked examples for [hierarchical scoring](evaluation-hierarchical-scoring.md), using **one shared toy schema** and **one toy manifest** throughout. Each section isolates a structural shape; unchanged parts of gold and pred are omitted or shown as `…` for brevity.

**Assumptions (unless noted):**

- **Structural threshold** `τ = 1.0` — exact match on primitives and enums; a pair must score `1.0` to count as a matched list element or property slot.
- **String comparison** — character-exact (no fuzzy matching in these toys).
- **List alignment** — Hungarian assignment on pairwise subtree scores, then threshold gating ([Lists](evaluation-hierarchical-scoring.md#lists)).
- **Manifest** — layers 1–2 are **reporting only**; they do not replace structural `match` ([Bubbling](evaluation-hierarchical-scoring.md#bubbling-three-parallel-aggregates)).

Normative definitions: [evaluation-hierarchical-scoring.md](evaluation-hierarchical-scoring.md). Leaf `score` rules: [evaluation-leaf-primitives.md](evaluation-leaf-primitives.md).

---

## Shared toy schema

Illustrative checklist output root — three top-level properties that cover all four shapes:

| Property | Schema shape | Section |
|----------|--------------|---------|
| `tags` | `array` of `string` | [§1 Lists of primitives](#1-lists-of-primitives) |
| `item` | `object` with primitive fields + nested `meta` | [§2 Objects](#2-objects-with-primitive-properties), [§4 Nested objects](#4-nested-objects) |
| `panels` | `array` of `object` | [§3 Lists of objects](#3-lists-of-objects) |

```json
{
  "type": "object",
  "required": ["tags", "item", "panels"],
  "properties": {
    "tags": {
      "type": "array",
      "items": { "type": "string" }
    },
    "item": {
      "type": "object",
      "required": ["id", "label", "status", "meta"],
      "properties": {
        "id": { "type": "integer" },
        "label": { "type": "string" },
        "status": {
          "type": "string",
          "enum": ["", "yes", "no"]
        },
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
          "status": {
            "type": "string",
            "enum": ["", "yes", "no"]
          }
        }
      }
    }
  }
}
```

**Baseline gold** (reference for all examples — other sections only change the highlighted part):

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
  "fields": {
    "item.status": {
      "na_values": [""],
      "answer_metric": "binary_polarity"
    },
    "item.label": {
      "answer_metric": "graded_string",
      "match_threshold": 1.0
    },
    "panels[].status": {
      "na_values": [""],
      "answer_metric": "binary_polarity"
    },
    "panels[].label": {
      "answer_metric": "graded_string",
      "match_threshold": 1.0
    }
  }
}
```

Enum field `status` uses `na_values: [""]`. Free-text `label` uses exact graded match (`τ = 1.0`). Fields without manifest entries (`id`, `tags`, `meta.*`) use **structural** comparison only.

---

## 1. Lists of primitives

**Shape:** `tags` — `array` of `string`. **Unit of accounting:** [element slots](evaluation-hierarchical-scoring.md#element-slots-and-alignment) after alignment. **Drill-down:** `element_scores` only (no list-level `field_scores`).

### 1.1 Perfect match

| | `tags` |
|---|--------|
| Gold | `["alpha", "beta"]` |
| Pred | `["alpha", "beta"]` |

**Alignment:** `(0,0)`, `(1,1)`; both pairs score `1.0`.

| Metric | Value |
|--------|-------|
| `TP_el` | 2 |
| `FP_el` | 0 |
| `FN_el` | 0 |
| Precision / recall | 1.0 / 1.0 |
| List `score` | 1.0 (mean over `n_all = 2`) |

`element_scores`: `match_0_0`, `match_1_1` (each leaf `score = 1.0`).

### 1.2 Extra predicted element

| | `tags` |
|---|--------|
| Gold | `["alpha", "beta"]` |
| Pred | `["alpha", "beta", "gamma"]` |

**Alignment:** match `alpha` and `beta`; index `2` is unmatched on pred side.

| Metric | Value |
|--------|-------|
| `TP_el` | 2 |
| `FP_el` | 1 |
| `FN_el` | 0 |
| Precision | 2/3 |
| Recall | 1.0 |

`element_scores`: `match_0_0`, `match_1_1`, `unexpected_element_2`.

### 1.3 Same length, one wrong value (`x` vs `c`)

| | `tags` |
|---|--------|
| Gold | `["alpha", "beta", "gamma"]` |
| Pred | `["alpha", "beta", "delta"]` |

**Alignment:** Hungarian assigns `(0,0)`, `(1,1)`, `(2,2)`. Pair `(2,2)` scores `0.0` → below `τ`.

| Metric | Value |
|--------|-------|
| `TP_el` | 2 |
| `FP_el` | 1 (pred index 2) |
| `FN_el` | 1 (gold index 2) |
| Precision / recall | 2/3, 2/3 |

One failed alignment charges **one pred row** and **one gold row** — not the same atom twice. `element_scores`: `match_0_0`, `match_1_1`, `unexpected_element_2`, `missing_element_2`.

### 1.4 Single-element mismatch

| | `tags` |
|---|--------|
| Gold | `["alpha"]` |
| Pred | `["omega"]` |

Hungarian assigns the only pair; similarity `0.0` → sub-threshold → both unmatched element accounting.

| Metric | Value |
|--------|-------|
| `TP_el` | 0 |
| `FP_el` | 1 |
| `FN_el` | 1 |
| Precision / recall | 0 / 0 |

---

## 2. Objects with primitive properties

**Shape:** `item` — `object` with primitive properties `id`, `label`, `status` (and `meta` held equal in these examples). **Unit of accounting:** [property slots](evaluation-hierarchical-scoring.md#required-properties-as-slots). **Drill-down:** `field_scores` keyed by property name; `element_scores` empty.

In §2 examples, `meta` matches baseline on both sides unless noted.

### 2.1 Perfect match

Pred equals baseline `item`.

| Property slot | `match_k` | Structural |
|---------------|-----------|------------|
| `id` | true | TP |
| `label` | true | TP |
| `status` | true | TP |

| Object metric | Value |
|---------------|-------|
| `score` | 1.0 |
| Precision / recall | 1.0 / 1.0 |
| `whole_object_match` | true |

**Manifest (reporting):**

| Path | Layer 1 | Layer 2 |
|------|---------|---------|
| `item.status` `""`/`""` | correct_NA | skipped |
| `item.label` | applicable / applicable | graded match |

### 2.2 Wrong free-text string (`label`)

| | `item.label` | rest |
|---|--------------|------|
| Gold | `"Panel A"` | baseline |
| Pred | `"Panel B"` | baseline |

| Property slot | `match_k` | Structural |
|---------------|-----------|------------|
| `id` | true | TP |
| `label` | false | **FP** (pred asserted a string, wrong) |
| `status` | true | TP |

| Object metric | Value |
|---------------|-------|
| `TP` / `FP` / `FN` | 2 / 1 / 0 |
| Precision | 2/3 |
| `whole_object_match` | false |

**Manifest:** `item.label` — layer 2 graded mismatch (no polarity TP/TN).

### 2.3 Enum polarity error (`status`)

| | `item.status` | rest |
|---|---------------|------|
| Gold | `""` | baseline |
| Pred | `"yes"` | baseline |

| Property slot | Structural |
|---------------|------------|
| `status` | **FP**-like (`match_k` false, `score = 0`) |

**Three tracks on one field** ([composition example](evaluation-hierarchical-scoring.md#composition-example-three-tracks-on-one-enum-field)):

| Track | Outcome |
|-------|---------|
| Structural | `match_k` false → slot not TP |
| Layer 1 | spurious_applicable (gold N/A, pred answered) |
| Layer 2 | skipped (gold not applicable) |

`whole_object_match` false **even though** layer 2 is empty — tracks are independent.

### 2.4 Missing required property

| | `item` |
|---|--------|
| Gold | `{ "id": 1, "label": "Panel A", "status": "", "meta": { … } }` |
| Pred | `{ "id": 1, "status": "", "meta": { … } }` — **no `label`** |

| Property slot | Structural |
|---------------|------------|
| `label` | **FN** (key absent) |

| Object metric | Value |
|---------------|-------|
| `FN` | 1 |
| `whole_object_match` | false |

### 2.5 Extra key in pred

| | `item` |
|---|--------|
| Gold | baseline `item` |
| Pred | baseline + `"noise": true` |

| Structural | |
|------------|--|
| Extra `noise` | **FP** slot |
| `whole_object_match` | false (extra keys forbidden) |

---

## 3. Lists of objects

**Shape:** `panels` — `array` of panel objects. **Two layers of metrics** ([list-of-objects](evaluation-hierarchical-scoring.md#when-x-is-an-object-or-nested-array)):

1. **Element layer** — did each **panel row** align as a whole? (`element_scores`, list P/R/F1).
2. **Field rollup** — how did property `f` behave **across all panels**? (`field_scores` on the **list** node).

Per-panel per-field detail: `element_scores["match_i_j"].field_scores["status"]`, etc.

### 3.1 Perfect match

Pred `panels` equals baseline.

**Element layer:** `TP_el = 2`, precision / recall 1.0. `element_scores`: `match_0_0`, `match_1_1`.

**Field rollups** (on list node `field_scores`):

| Field | Mean score (illustrative) | Notes |
|-------|---------------------------|-------|
| `id` | 1.0 | both panels matched |
| `label` | 1.0 | |
| `status` | 1.0 | panel 2: `""`/`""` |

### 3.2 One wrong field inside a matched panel

| | `panels[1]` | `panels[0]` |
|---|-------------|-------------|
| Gold | `{ "id": 2, "label": "Fig 2", "status": "" }` | baseline |
| Pred | `{ "id": 2, "label": "Fig 2", "status": "yes" }` | baseline |

**Alignment:** diagonal `(0,0)`, `(1,1)`. Panel 1 subtree `score` < 1.0 because `status` failed.

**Element layer** (same pattern as list-of-strings §1.3):

| Metric | Value |
|--------|-------|
| `TP_el` | 1 (panel 0 only) |
| `FP_el` | 1 (pred panel 1) |
| `FN_el` | 1 (gold panel 1 not recovered) |

**Inside** `element_scores["match_1_1"]` (or the unmatched stubs): drill-down shows `field_scores["status"]` with `score = 0`.

**Field rollup `panels[].status`:** counts reflect one spurious `yes` on an N/A field — layer 1 **spurious_applicable** on that path; element layer still charges FP/FN for the **whole row**.

### 3.3 Missing panel

| | `panels` |
|---|----------|
| Gold | baseline (2 panels) |
| Pred | `[ { "id": 1, "label": "Fig 1", "status": "yes" } ]` |

| Element layer | |
|---------------|--|
| `TP_el` | 1 |
| `FN_el` | 1 (gold panel 2 missing) |
| List recall | 1/2 |

**Field rollup `id`:** one matched `id`, one missing row contributes **FN** to the field rollup.

### 3.4 Element vs field rollup — different questions

| | `panels` |
|---|----------|
| Gold | `[ { "id": 1, … }, { "id": 2, … } ]` |
| Pred | `[ { "id": 1, … }, { "id": 2, … }, { "id": 99, … } ]` |

- **Element precision** — penalized for the **extra panel** (`FP_el`).
- **`field_scores["id"]` rollup** — extra panel adds **FP** contributions for `id` (and every required field) on the cross-list aggregate.

Reading `precision` on the list node: check whether you mean **element-layer** or **`field_scores[f]`** ([drill-down](evaluation-hierarchical-scoring.md#list-of-objects-two-questions-two-dicts)).

---

## 4. Nested objects

**Shape:** `item.meta` — object nested under property `meta` on `item`. **Parent slot:** `meta` is **one property slot** on `item`; inner `author` / `year` are slots **inside** the child subtree ([nested object property values](evaluation-hierarchical-scoring.md#nested-object-property-values)).

**Drill-down path:** `field_scores["item"].field_scores["meta"].field_scores["year"]` (from root; exact path depends on compare entry point).

In §4 examples, `item` primitives other than `meta` match baseline.

### 4.1 Perfect nested object

Pred `item.meta` equals `{ "author": "Ada", "year": 1843 }`.

| Level | `whole_object_match` / `match_k` |
|-------|----------------------------------|
| Inner `meta` | true (both fields TP) |
| Parent `item.meta` slot | true (`R.score = 1.0`) |
| Outer `item` | true |

### 4.2 One wrong inner field

| | `item.meta` |
|---|-------------|
| Gold | `{ "author": "Ada", "year": 1843 }` |
| Pred | `{ "author": "Ada", "year": 1900 }` |

**Inner object `meta`:**

| Slot | Structural |
|------|------------|
| `author` | TP |
| `year` | FP (wrong integer) |
| Inner `whole_object_match` | false |

**Parent `item`:**

| Slot | Structural |
|------|------------|
| `meta` | **FP** (pred supplied an object, subtree below `τ` or strict inner flag false) |
| `id`, `label`, `status` | TP |
| `item.whole_object_match` | false |

Drill-down: `field_scores["meta"].field_scores["year"]` shows the failing leaf (`score = 0`).

### 4.3 Missing nested object

| | `item` |
|---|--------|
| Gold | baseline (includes `meta`) |
| Pred | `{ "id": 1, "label": "Panel A", "status": "" }` — **no `meta`** |

| Slot | Structural |
|------|------------|
| `meta` | **FN** (required key absent) |

Parent `item` `whole_object_match` false. Inner metrics do not exist in drill-down — no subtree to attach.

### 4.4 Contrast: same object as list **element** vs nested **property**

The same JSON object shape `{ "id", "label", "status" }` appears under `panels[]` (§3) and under `item.meta` is a different shape — but compare a **panel object as list element** vs **hypothetical** nesting:

| Context | Parent bookkeeping | Key in drill-down |
|---------|-------------------|-------------------|
| `panels[0]` | **Element slot** on list | `element_scores["match_0_0"]` |
| `item.meta` | **Property slot** on object | `field_scores["meta"]` |

Same subtree comparison machinery; different parent TP/FP/FN and key namespace ([nested objects — contrast with lists](evaluation-hierarchical-scoring.md#nested-object-property-values)).

---

## Quick reference

| Section | JSON path | Parent node | Drill-down dict | Slot unit |
|---------|-----------|-------------|-----------------|-----------|
| §1 | `tags[]` | list | `element_scores` | element |
| §2 | `item.*` | object | `field_scores` | property |
| §3 | `panels[]` | list | `element_scores` + `field_scores` rollups | element + cross-list field |
| §4 | `item.meta.*` | object → object | `field_scores` at each object layer | property (nested) |

---

## See also

- [Hierarchical scoring](evaluation-hierarchical-scoring.md) — normative rules referenced by these toys.
- [Leaf primitives](evaluation-leaf-primitives.md) — how terminal `score` is formed.
- [Drill-down: `field_scores` and `element_scores`](evaluation-hierarchical-scoring.md#drill-down-field_scores-and-element_scores) — how to read result trees from these examples in code.
