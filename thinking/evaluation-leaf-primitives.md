# Leaf comparison: terminal primitive values

**Wiki:** catalog `[index.md](index.md)`; conventions `[README.md](README.md)`.

This page is about **terminal values** in schema-guided JSON comparison: the JSON types that do **not** contain nested objects or arrays. We compare them when recursion reaches a node whose schema says `string`, `number`, `integer`, `boolean`, or (if we add it) `null`. **Strings are the richest case** because we support several similarity modes; other primitives are usually **exact** today.

Code reference: `[JSONEvaluator](../soda_mmqc/core/evaluation.py)` in `[soda_mmqc/core/evaluation.py](../soda_mmqc/core/evaluation.py)` (`_compare_values_with_schema`, `_compare_strings`, and the non-string branch for other types).

## Vocabulary: primitive vs scalar

- **JSON primitives** (common wording): `**string`**, `**number`**, `**integer**`, `**boolean**`, `**null**`. They are the values that sit at **leaves** of the JSON value tree (no `{}` or `[]` inside them).
- **“Scalar”** is used loosely in data and ML contexts to mean **one non-container value**. It often includes strings; sometimes people reserve “scalar” for numeric types only. In this wiki we say **primitive** or **leaf value** when we want to be precise.

A **leaf property** is a schema path to a primitive value (or array of primitives). Comparison produces **`score`** only at these paths. Normative model: [evaluation-scoring.md](evaluation-scoring.md).

## Schema-first: the leaf schema fragment decides behavior

Leaf comparison is **not** only “global string metric on the evaluator.” The **sub-schema at the current path** tells us:

- `**type`** — which family of comparators applies.
- `**enum`** (for strings) — prediction must be one of the allowed literals or we score failure.
- Future: `**format**`, numeric bounds (`minimum` / `maximum`), `const`, etc.

The navigator passes **prediction value, expected value, and that fragment** into the leaf layer so behavior stays aligned with the checklist schema.

## Vocabulary at leaves (only what leaves own)


| Term                        | Meaning at a **primitive leaf**                                                                                                                                                                                                                                       |
| --------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `**score`**                 | Numeric quality in `**[0, 1]`** (exact match → `0`/`1`; fuzzy / semantic → graded). Feeds **roll-up averages** and any **parent** decision rule.                                                                                                                      |
| **Validity / kind signals** | Hard failures (`enum` violation, type mismatch) should surface as `**score == 0`** (and/or explicit signals later) for layer 1 / layer 2 reporting ([evaluation-scoring.md](evaluation-scoring.md)). |


**Normative split:** a leaf compares two values and emits **`score`**. **Layer 1 / layer 2** reporting (applicability, TP/FP/FN/TN, match/mismatch) is applied per leaf from the manifest — see [evaluation-scoring.md](evaluation-scoring.md). There is no container roll-up in the flat model.

## Strings

Strings are the main flexible leaf type in our stack.

- **Exact match** — Character-for-character equality (after whatever normalization we apply in code, e.g. none vs strip).
- **Fuzzy match** — Historically LCS-based ratio in-repo; the refactor plan moves this to a **library** (e.g. rapidfuzz) with normalized score in `[0, 1]`.
- **Semantic similarity** — Embedding-based cosine similarity (e.g. SentenceTransformer), mapped to `[0, 1]`.

**Threshold:** `match_threshold` in the manifest turns graded `**score`** into layer 2 **match** / **mismatch** on that leaf ([evaluation-scoring.md](evaluation-scoring.md)). The leaf only produces the score.

**Enum strings:** if the schema lists `enum`, compare **exactly** (unless a profile says otherwise); outside `enum` → **`score = 0`**.

- **N/A sentinel enums** (e.g. `["", "yes", "no"]` where `""` = not applicable): `""` is a **class**, not “empty wrong text.” Which literal is N/A and which is positive/negative is declared in **`eval-manifest.json`**. Reporting: [evaluation-scoring.md](evaluation-scoring.md) (layers 1–2 per leaf).
- **Multi-class enums** (many non-N/A values): leaf **`score`** + **match/mismatch** and mean **`score`**; no binary TP/TN unless you define a profile.
- **Binary polarity enums** (`yes` / `no` + N/A): layer 2 can use TP/FP/TN/FN on **gold-applicable** slots only.

## Numbers and integers

Today: **strict equality** between Python values parsed from JSON (`pred_value == exp_value`) → `**score`** `1.0` or `0.0`. The **object parent** maps that to slot TP/FP/FN (see hierarchical doc).

**Roadmap (design only):** tolerance (epsilon), `multipleOf`, integer-vs-float rules, or schema `const` — still keyed off the same **schema fragment**, not ad-hoc globals.

## Booleans

Same as numbers today: **exact** equality, binary score.

## Null, missing keys, and empty strings

Follow **JSON** and **JSON Schema**: `null`, absent property, and `""` are different. For missing keys vs `null` vs `""` on string leaves, see [evaluation-scoring.md](evaluation-scoring.md) (layer 1 withheld / layer 2 skipped).

If the schema allows `**null`** at a leaf (`type` union includes `"null"`), comparison follows JSON null rules (`null` vs `null` → full `**score`**; `null` vs string → low `**score**`, parent classifies slot).

## Graded scores without declaring container `match`

Leaves often return **partial credit** (e.g. similarity `0.4`). Only the **parent** knows whether that counts as `**match`** for its slot policy (`score >= τ`, strict all-fields, etc.). Keep `**score`** and **slot `match`** **separate in design**: high score with `**match` false** still improves **roll-up averages** above the level where `match` failed.

## Output shape at a leaf (conceptual)

**Normative for now:** every primitive leaf returns one **`score` in `[0, 1]`**, regardless of JSON type:

| Leaf type | How `score` is formed (typical) |
|-----------|----------------------------------|
| **string** | Exact / fuzzy / semantic similarity → `[0, 1]` |
| **number / integer** | Exact equality → `0.0` or `1.0` (tolerance later if needed) |
| **boolean** | Exact equality → `0.0` or `1.0` |
| **enum string** | Exact match on allowed literal → `0.0` or `1.0` |

Optional **leaf-local diagnostics** (e.g. which fuzzy metric fired) are fine; they do not replace **`score`**.

A leaf does **not** define nested **`element_scores` / `field_scores`** — the flat model stores one result per leaf path ([evaluation-scoring.md](evaluation-scoring.md)).

**Separation of concerns:** **leaf** = compare two primitive values → **`score`**. **Parent** = alignment, keys, **`predicate`**, manifest layers, roll-up.

### `LeafComparisonResult` vs `ComparisonResult` (target design)

**Not implemented yet.** Today [`evaluation.py`](../soda_mmqc/core/evaluation.py) still uses one **`ComparisonResult`** type everywhere (with empty `field_scores` at leaves). The refactor should split:

- **`LeafComparisonResult`** — only **`score`** in `[0, 1]` plus optional **`false_positive` / `false_negative`** for leaf-local cases (e.g. `None` vs string). **No** `field_scores` or `element_scores`.
- **`ComparisonResult`** — **objects and lists** only: roll-up **`score`**, drill-down maps, precision/recall on containers.

At the boundary, leaves would call **`leaf.to_comparison_result(match_threshold)`** before a parent stores the child under `field_scores` or `element_scores`:

```text
_compare_strings("yes", "no")  →  LeafComparisonResult(score=0.0)
                                      ↓ to_comparison_result(τ)
field_scores["plot"]           →  ComparisonResult(score=0.0, field_scores={}, …)
```

Example mistake this prevents: attaching `field_scores={"nested": …}` to a bare string compare — the leaf type cannot express that.

**Direction:** move primitive dispatch into **`leaves.py`** returning `LeafComparisonResult`; containers keep returning `ComparisonResult`. See the split-evaluation plan and [evaluation-json-vs-open-source.md](evaluation-json-vs-open-source.md).

## See also

- [Flat leaf scoring](evaluation-scoring.md) — manifest layers and list alignment.
- [Toy examples](evaluation-toy-examples.md) — worked gold/pred cases using a shared schema and manifest.
- [JSON evaluation vs open source](evaluation-json-vs-open-source.md) — why this leaf machinery stays project-specific.

