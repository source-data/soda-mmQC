# Hierarchical scoring in schema-guided JSON comparison

This note defines a **target design** for **roll-up scores**, **drill-down**, and **slot-level FP/FN** when recursively comparing prediction JSON to gold JSON under **JSON Schema**. It is **normative for the project’s direction** (implementations may still diverge until refactors land). It also explains why **`element_scores`** vs **`field_scores`** is easy to misread in a single-result-tree shape.

## Why both roll-up and drill-down?

- **Roll-up at this node** — Two ideas: (1) a **roll-up score** — one quality scalar in `[0, 1]` (often **mean of child `score`s**; lists may use `max(n_pred, n_exp)` in the denominator); (2) **confusion-style summaries** — `precision`, `recall`, `f1_score` (and strict “all matched” booleans) computed from **slot counts** built using each child’s **slot `match`**, declared **at this parent** (see **Where `match` is defined** below). `std_score` summarizes spread of child scores where applicable.
- **Drill-down dictionaries** — Debugging, error messages, and per-path reporting need **where** the mismatch lives: which key, which list slot, which nested branch. Without nested structures, a single number throws away the explanation.

So every level of the recursion ideally carries: **(1) a summary for the node** and **(2) optional child maps** that preserve hierarchy.

**Leaves matter for roll-up:** each **primitive leaf** contributes **`score`** (and hard failures as `score = 0`). **[Slot `match` is not a leaf responsibility](#where-match-is-defined-normative)** for list items or for “did this whole branch succeed?”—parents decide that from subtree results.

## Match vs mismatch, and how parents build TP / FP / FN and P / R / F1

### Where `match` is defined (normative)

| Edge | Who declares **`match`** for that slot? | Rule (default) |
|------|------------------------------------------|------------------|
| **Object → property `k`** | The **object** parent, after comparing `pred[k]` to `exp[k]` | Build child subtree `ComparisonResult` `R`. Then **`match_k = predicate(R)`** — typically **`R.score >= τ`** for a uniform threshold `τ`, plus the **FP/FN kind** for string slots from the [adopted string policy](#adopted-policy-fp-vs-fn-for-required-string-properties-on-objects) (missing / `null` / wrong value). Primitives do **not** pre-bake list semantics; they only supply **`R.score`**. |
| **List → aligned item `(i,j)`** | The **list** parent, after subtree compare for that pair | Same: **`match_item = predicate(R_{i,j})`**, default **`R_{i,j}.score >= τ`**. Works for **any** item type **X** (primitive, object, nested array) because every `X` comparison returns a **rolled-up `score`** in `[0, 1]`. |

**`predicate` in full generality:** `match_slot: (child: ComparisonResult, ctx) -> bool` where `ctx` can carry schema path, slot kind, or strictness. The **threshold-only** form `score >= τ` is the minimal implementation; a richer function can require **`R.true_positive`** at the subtree root, or forbid partial object matches, etc.—but the **declaration site** stays the **parent list or parent object**, not the primitive leaf.

**Primitive leaves** ([evaluation-leaf-primitives.md](evaluation-leaf-primitives.md)) output **`score`** (graded quality and validity). They **do not** decide whether a **list element** “matched” or whether an **object property slot** counts for TP; that is always the **parent’s** job using `predicate`.

**“True negative”** in the classical sense is **not** first-class: metrics are **slot-based** (TP/FP/FN over slots), not a full 2×2 matrix with TN.

At **parents**:

- **Roll-up score** — aggregate child **`score`s** (e.g. mean over required fields or over `n_all` list slots). This answers “how close” even when **`predicate`** marks some slots non-matching.
- **TP / FP / FN** — each **slot** contributes counts from **`match_slot`** and structural cases (extra keys, missing keys, unmatched list rows), using the **adopted FP/FN policy** for object strings (below) where applicable.
- **Precision / recall / F1** — computed at the **parent** from those slot counts: e.g. precision = `TP / (TP + FP)`, recall = `TP / (TP + FN)` over the **same slot universe** defined for that node type.

**Principle (applies beyond strings):** treat **JSON and JSON Schema** literally—do not collapse distinctions with Python idioms (`None`, truthiness on `""`).

### Objects (required keys as slots)

**Unit of accounting:** each **required schema property** is one slot. Extra keys in the prediction (not in the gold schema’s required set for that object) are naturally **FP**-like; a required property for which gold is not satisfied is **FN**-like.

For each required key `k`, compare `pred[k]` to `exp[k]` to get child `R`. The parent computes **`match_k = predicate(R)`** and applies **FP/FN kind** rules (e.g. string table below), then increments TP / FP / FN slot counters.

Then at the **object** node (non-empty required set):

- **Precision** = `TP / (TP + FP)` over property slots (extra keys increase FP).
- **Recall** = `TP / (TP + FN)` over property slots (missing keys increase FN).
- **F1** = harmonic mean of precision and recall.
- A strict **“whole object matched”** flag (if you expose one) is **stricter** than P/R alone: all required properties must satisfy **`predicate`** and there must be **no extra** keys.

### Adopted policy: FP vs FN for required **string** properties on objects

**Definitions (normative for this wiki):**

- **FN (false negative on this slot)** — Gold’s requirement for this property is a **conforming value that pred never supplies** in JSON terms: the key is **absent**, or the value is JSON **`null`** while the schema for this property is **string-only** (type does **not** include `null`). Interpretation: *recall failure* — “we needed a string here and did not get one.”
- **FP (false positive on this slot)** — Pred **does** supply a JSON value for the property, but the **object parent** does **not** mark the slot **`match`** (wrong text, **`predicate`** false on the child subtree—e.g. **`score < τ`**—outside `enum`, **or** empty string `""` when gold expects a non-empty string / when `minLength` and leaf logic yield a failing **`score`**). Interpretation: *precision failure* — “something was asserted for this key, but it is wrong or inadmissible.”

**Why `""` is FP, not FN (JSON-native argument):**

In JSON, `""` is a **string value that exists**. It is neither “key missing” nor `null`. Calling it **FN** would conflate three distinct gold–pred situations that the wire format keeps separate. **FP** keeps the invariant: **FN only when no conforming literal was provided**; **`""` is a literal (length 0)** → if the parent’s **`predicate`** is false (e.g. low **`score`** vs gold), that is pred’s **wrong submission**, same class as `"bad"`. Use **`minLength: 1`** in the schema when the product forbids empty strings; the leaf reflects failure in **`score`**; the parent still classifies the slot as **FP**, consistent with other non-matching strings.

**Why JSON `null` is FN (when `type` is `"string"` only):**

If gold has a string and the schema does not admit `null`, **`"label": null`** means pred did not deliver a string at all—it is **not** the empty string and **not** a missing key, but in evaluation terms it is **“no string answer”** alongside absence. Classifying as **FN** aligns **missing key**, **`null`**, and (if you ever need it) other **non-string** typed values for a string slot under the same *recall* story. If the schema is explicitly `["string", "null"]` and gold is `null`, then **`null` vs `null`** is a normal leaf comparison; the **object parent** marks the slot **`match`** when **`predicate`** holds on the child result (typically high **`score`** for both null).

**Summary table (string slot, gold expects a concrete string like `"ok"`):**

| Pred JSON for `label` | Slot classification | One-line reason |
|------------------------|----------------------|-----------------|
| Key **absent** | **FN** | No property in pred object — nothing to compare to gold’s string. |
| **`null`** and schema is **string-only** | **FN** | No string value delivered (JSON null ≠ string). |
| **`""`** | **FP** | A string was delivered; the parent does not mark **match** vs gold / schema (unless gold is `""` and schema allows it). |
| **`"bad"`** (non-matching) | **FP** | Wrong asserted string. |
| **Extra key** (e.g. `noise`) | **FP** (separate slot or extra-field rule) | Assertion outside gold’s allowed shape for that object. |

**JSON Schema as contract:** `required`, `type` (including union with `"null"`), `minLength`, `enum`, and `pattern` define how the **leaf `score`** is formed; the **object parent** then combines **`score`**, **`predicate`**, and this table to label **FP vs FN** and whether the property **`match`**es.

**Do not use** Python `if not value:` across JSON forms—that erases the **`""` vs `null` vs missing** distinction JSON was designed to preserve.

**Toy — object with two required fields (wrong string on `label`)**

- Gold: `{ "id": 1, "label": "ok" }`  
- Pred: `{ "id": 1, "label": "bad" }`  
- `id` **matches** → **one TP** slot. `label` **does not match** (wrong string) → **one FP** slot, **not** FN.

Slot totals: **`TP = 1`, `FP = 1`, `FN = 0`** → precision `1/2`, recall `1`. A strict “whole object matched” flag would be **false** (not every property matched; no extra keys).

**Toy — extra key**

- Gold: `{ "id": 1 }`, Pred: `{ "id": 1, "noise": true }`  
- One **FP** slot from `noise`, so object precision drops; object `true_positive` is false.

### List of X (including primitives and nested non-object items)

**Unit of accounting:** **list elements after alignment**, not raw indices. Build pairwise **subtree `score`s**, **align** (e.g. Hungarian), then for each aligned pair let the **list parent** set **`match_item = predicate(R)`** (default **`R.score >= τ`**). Rows that are pred-only or gold-only after that step are **FP elements** / **FN elements** in the list story.

Let `TP_el` = number of matched pairs above threshold, `FP_el` = extra predicted elements, `FN_el` = missing expected elements. The implementation uses `n_all = max(len(pred), len(exp))` when averaging **scores** across slots (including zero-score stubs for missing/extra).

- **List precision** ≈ `TP_el / (TP_el + FP_el)` — “of what the model predicted as aligned mass, how much was good?”
- **List recall** ≈ `TP_el / (TP_el + FN_el)` — “of what gold required, how much was recovered above threshold?”
- **List F1** from precision and recall as usual.

**Toy — list of strings, exact match threshold 1.0**

- Gold: `["a", "b"]`, Pred: `["a", "b", "x"]`  
- Alignment: match two pairs (`a`,`b`), one prediction row `x` is **extra** → `TP_el = 2`, `FP_el = 1`, `FN_el = 0` → precision `2/3`, recall `1`.

**Toy — same length, one substituted value (`x` vs `c`)**

- Gold: `["a", "b", "c"]`, Pred: `["a", "b", "x"]`  
- Hungarian matching still assigns **position-wise** pairs that maximize total similarity (here `(0,0)`, `(1,1)`, `(2,2)`). The first two pairs score `1.0` at exact match; **`"x"` vs `"c"`** scores `0.0` (below threshold `1.0`).

Under **threshold-gated** alignment, a pair **below threshold** does **not** count toward `TP_el`. The prediction index `2` is then treated as an **unmatched prediction slot** → **one FP element** (pred side paid a “bad / unmatched” row). The expected index `2` is an **unmatched expected slot** → **one FN element** (gold side paid a “not recovered” row).

So **both** labels apply at the **element-slot** layer: **`x` → FP**, **`c` → FN**. That is not a logical contradiction: one **failed alignment** charges **one pred row** and **one gold row** in the bookkeeping, not the same JSON atom wearing two hats.

With `TP_el = 2`, `FP_el = 1`, `FN_el = 1`: **precision** `= 2/(2+1) = 2/3`, **recall** `= 2/(2+1) = 2/3`. Intuitively: one wrong pred slot and one missed gold slot relative to a perfect 3-for-3 alignment.

**Toy — list of strings, one wrong value (sub-threshold pair)**

- Gold: `["a"]`, Pred: `["z"]`  
- Hungarian still assigns the pair, but if similarity is **below** `match_threshold`, the pair does **not** count as `TP_el`. The prediction and expected rows then fall through to **unmatched** element accounting, so element-level FP/FN drive list precision/recall.

### List of X when **X** is not a leaf (objects or nested arrays)

**Normative rule:** at any list node, **TP / FP / FN always describe list *elements*** (whole **items** after alignment), never individual fields *inside* those items. Fields and inner lists have **their own** TP/FP/FN **inside** the nested `ComparisonResult` for that item.

**How an item `match` is decided when X is an object or array:**

1. For every pair `(pred_i, exp_j)`, run the **full subtree comparison** for type **X** and read that pair’s **rolled-up subtree `score`** in `[0, 1]` (already aggregating whatever is inside: object fields, inner lists, leaves).
2. Build the **assignment** on those scores (e.g. Hungarian to maximize sum of scores).
3. **`match_item` at the list parent** — `predicate(R_{i,j})`; default **`R_{i,j}.score >= τ`** (one decision per **whole** item subtree, independent of whether **X** is primitive, object, or array).
4. **`TP_el` / `FP_el` / `FN_el`** at this list are **exactly** the same slot story as for list-of-strings: matched slots above threshold, unmatched pred rows → FP, unmatched exp rows → FN.

**Why not propagate inner TP/FP/FN to the list row?** The list layer answers **“which checklist items aligned?”** (set-style), not **“how many fields inside an item?”** Inner metrics live in **`element_scores["match_i_j"].field_scores`** (objects) or deeper **`element_scores`** (arrays).

**List of objects (`X` = `object`):**

- **Element layer** — as above; each matched item carries a full object subtree (`field_scores` per property, object-level P/R/F1 inside that subtree).
- **Cross-list field rollups** — **additionally**, for each required **object field name** across items, the **second** `field_scores` block on the **list** node aggregates how that field behaved over all alignments (see below). Element TP/FP/FN and field rollups answer different questions.

**List of arrays (`X` = `array`):**

- **No** list-level `field_scores` rollup (arrays have no named fields). Drill-down is **`element_scores`** only: each slot holds a nested list’s own `ComparisonResult` (with its own element-level TP/FP/FN for the inner list).
- **Outer** TP/FP/FN = quality of **which inner lists** matched as wholes; **inner** TP/FP/FN = quality **inside** each inner list.

**Toy — list of two small objects (same structure, one wrong field inside one item)**

- Gold: `[ { "id": 1, "t": "a" }, { "id": 2, "t": "b" } ]`  
- Pred: `[ { "id": 1, "t": "a" }, { "id": 2, "t": "wrong" } ]`  

After alignment, both pairs can sit on the diagonal; the **second item’s subtree score** is below threshold if `t` drives the object roll-up down. Then that row behaves like the **`"x"` vs `"c"`** string case: **one FP element** (bad second pred object) and **one FN element** (second gold object not recovered as a `match`) at the **list** element layer—**even though** the first object matched. Inside the second item’s drill-down, you still see **which** field failed (`t`) and object-internal FP/FN there.

**Toy — list of lists**

- Gold: `[ [1,2], [3] ]`, Pred: `[ [1,2], [9] ]`  
- Outer list: two items; compare inner lists pairwise; if inner list `[3]` vs `[9]` fails threshold, same **FP/FN element** pattern at the **outer** slot level. Inner list `[1,2]` vs `[1,2]` has its own `TP_el` etc. inside the nested result.

When **X** is a **primitive** only, there is **no** second `field_scores` rollup under the list—only **`element_scores`** with synthetic alignment keys.

**List-of-objects — second layer only (`field_scores` on the list node):** beyond element-layer `TP_el` / `FP_el` / `FN_el`, for each **required object field name** `f` aggregate over all list slots: matched items contribute that field’s subtree; each **extra** pred list row adds **FP** for every `f`; each **missing** gold row adds **FN** for every `f`. That yields **another** precision/recall/F1 per `f` in **`field_scores[f]`**—different question from element-layer P/R (“how did field `title` behave across the whole checklist list?”).

The words **precision / recall** on a **list-of-objects** node can therefore mean **element-layer** metrics **or** **`field_scores[f]` rollups**—always check which subtree you are reading.

**Toy — list of `{ "id": integer }`, threshold such that only `id` must match**

- Gold: `[ { "id": 1 }, { "id": 2 } ]`  
- Pred: `[ { "id": 1 } ]`  

Hungarian matches one object; one gold object is missing → element-level **FN**; list recall < 1. For field `id`, the rollup counts reflect one matched `id` and one missing row’s **FN** contribution to that field’s rollup.

### Accuracy

**Accuracy** in the sense “(TP + TN) / all” is **not** currently a first-class exported metric on `ComparisonResult`; parents emphasize **precision / recall / F1** and mean **score**. If you add accuracy, define the **denominator** explicitly (e.g. property slots, or `n_all` list slots) because **TN** is not otherwise modeled.

---

## Three structural shapes (why we distinguish them)

The dispatcher looks at the JSON Schema **`type`** (and `required`, `items`, etc.) at the current path. Three shapes matter for how we attach drill-down:

| Shape | Meaning | Drill-down role |
|--------|---------|-------------------|
| **Object** | JSON object with named properties | One nested result **per property name** (schema `required` keys). |
| **List of X** | JSON array whose **`items`** schema describes element type **X** | Drill-down is **per list alignment slot** (see below). **X is not “scalar only.”** |
| **List of objects** | **X** is `object` — each element is an object with its own `properties` | Same per-slot drill-down, **plus** a second need: roll-ups **per object field name across the whole list** (e.g. average score for field `title` across all matched list elements). |

### What is X in “list of X”?

**X** is whatever the schema says **`items`** is: the single element sub-schema. In our evaluator that can be:

- **Primitives** in the JSON Schema sense: `string`, `number`, `integer`, `boolean` (sometimes loosely called “scalars” together with `null` when you add it).
- **`object`** — list of objects (the case that triggers extra cross-element field roll-ups).
- **`array`** — **nested list**; recursion continues with the same list machinery on inner arrays. So **X can absolutely be a list** (list of lists, list of lists of objects, …).

So “list of X” is **not** restricted to primitive leaves; **X** is any valid `items` schema the navigator recurses into.

## Hungarian alignment and synthetic keys

For lists whose elements are **not** 1:1 with positional identity (we use **optimal bipartite matching** on pairwise similarity), a matched pair is identified by **indices** `(pred_idx, exp_idx)` after assignment. Keys such as `match_0_2` mean: “prediction index 0 was paired with expected index 2.” Unmatched prediction rows become `unexpected_element_{pred_idx}`; unmatched expected rows become `missing_element_{exp_idx}`.

Those strings are **synthetic** because the **schema does not name list slots**—only order and value. The key encodes the **alignment event**, not a field name from JSON.

## `element_scores` vs `field_scores` (and why `field_scores` feels wrong on lists)

The implementation uses **one** dataclass for every node type, with two dict fields. That keeps serialization uniform but **overloads names**:

### On an **object** node

- **`field_scores`** — Keys are **real JSON property names**. Values are full nested `ComparisonResult` trees for that property. This matches the word “field.”
- **`element_scores`** — Unused (empty dict).

### On a **list** node

- **`element_scores`** — Keys are the **synthetic alignment keys** above. Values are nested `ComparisonResult` trees for **each matched pair** or **stub rows** for extra/missing elements. This is the natural “list drill-down.”
- **`field_scores`** — Filled **only when items are objects**. Keys are again **object field names**, but values are **aggregated summaries across the entire list** (mean score and P/R/F1-style counts for that field name over all alignments)—**not** another layer of per-slot object keys. Per-slot **per-field** detail still lives **inside** each match’s nested `ComparisonResult.field_scores`.

So **`field_scores` means two different things** depending on the parent:

1. **Under an object:** “per-property subtree.”
2. **Under a list-of-objects:** “per-property **list-level rollup**,” not the same semantic as (1).

Your doubt is reasonable: reusing **`field_scores`** for the list-of-objects rollup is **compact** (one JSON shape everywhere) but **confusing** because “field” suggests object keys, not “summary across many list elements.” Alternatives discussed for code (not all done yet): rename to roles like `list_item_field_rollups`, or use a **discriminated union** (`ObjectBreakdown` vs `ListBreakdown`) instead of two parallel dicts. See the refactor plan’s section on hierarchical results.

## How this should guide implementation

For **schema-guided** recursive comparison:

1. **Every recursive node** should be able to expose a **roll-up** so parents can aggregate (mean, counts, Hungarian cost, etc.).
2. **Every non-leaf node** should be able to expose **drill-down** keyed in a way that matches **object vs list** semantics—not only generic “children.”
3. **List-of-objects** is the only shape that needs **two** drill-down concepts: **per alignment slot** (whole element comparison) and **per schema field name across the list** (cross-element rollup). That is why the current type carries two dicts even though names collide mentally.
4. **Object string slots** should implement the **adopted FP/FN policy** above (JSON `""` vs `null` vs missing key)—refactors should converge code to this contract.

When we refactor toward a package layout (e.g. schema-driven **leaves**, **compare_lists** / **compare_objects**, and a clearer **ComparisonResult**), this page is the **normative contract** for behavior and naming. Legacy code may still differ until aligned.

## See also

- [Leaf primitives (strings, numbers, booleans)](evaluation-leaf-primitives.md) — how terminal / non-structured values are compared schema-first.
- [evaluation-json-vs-open-source.md](evaluation-json-vs-open-source.md) — why this comparator is custom vs OSS frameworks.
- [README.md](README.md) — wiki conventions.
