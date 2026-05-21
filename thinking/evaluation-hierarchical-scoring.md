# Hierarchical scoring in schema-guided JSON comparison

This note defines a **target design** for **roll-up scores**, **drill-down**, and **slot-level FP/FN** when recursively comparing prediction JSON to gold JSON under **JSON Schema**. It is **normative for the project’s direction** (implementations may still diverge until refactors land). It also explains why `**element_scores`** vs `**field_scores**` is easy to misread in a single-result-tree shape.

## Vocabulary: slot, element, value

This wiki uses three terms consistently:


| Term         | Meaning                                                                                                                                                                                                                                                                                                                   |
| ------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Value**    | JSON content being compared: `pred[...]` vs `exp[...]`. A **leaf value** is a primitive (`string`, `number`, …) with no nested `{}` or `[]`.                                                                                                                                                                              |
| **Property** | A named key on an **object**; comparison uses **property values** at `pred[k]` and `exp[k]`.                                                                                                                                                                                                                              |
| **Element**  | One entry in a JSON **array**; comparison uses **element values** (or full subtrees when schema `items` type **X** is object or array).                                                                                                                                                                                   |
| **Slot**     | One **accounting unit** at a parent for slot `**match`** and TP/FP/FN: a **property slot** (required key `k` on an object) or an **element slot** (aligned pair, or an unmatched extra/missing row on a list). The **slot** is the parent’s bookkeeping label; the **value** (or rolled-up subtree) is what was compared. |


See also [evaluation-leaf-primitives.md](evaluation-leaf-primitives.md) for **leaf values** only.

## Why both roll-up and drill-down?

- **Roll-up at this node** — Two ideas: (1) a **roll-up score** — one quality scalar in `[0, 1]` (often **mean of child `score`s**; lists may use `max(n_pred, n_exp)` in the denominator); (2) **confusion-style summaries** — `precision`, `recall`, `f1_score` (and strict “all matched” booleans) computed from **slot counts** built using each child’s **slot `match`**, declared **at this parent** (see **Where `match` is defined** below). `std_score` summarizes spread of child scores where applicable.
- **Drill-down dictionaries** — Debugging, error messages, and per-path reporting need **where** the mismatch lives: which key, which list slot, which nested branch. Without nested structures, a single number throws away the explanation.

So every level of the recursion ideally carries: **(1) a summary for the node** and **(2) optional child maps** that preserve hierarchy.

**Leaves matter for roll-up:** each **primitive leaf** contributes `**score`** (and hard failures as `score = 0`). A leaf compares **property values** on objects (e.g. `pred["label"]` vs `exp["label"]`) or **element values** on lists when **X** is primitive—but in every case it only supplies `**score`**. **[Slot `match` is not a leaf responsibility](#where-match-is-defined-normative)** for **property slots** or **element slots** (nor for “did this whole nested branch succeed?”); the **object** or **list** parent decides that from the **child subtree** result (primitive or composite).

## Match vs mismatch, and how parents build TP / FP / FN and P / R / F1

### Where `match` is defined (normative)


| Slot (parent)                            | Who declares `**match`** for that slot?                                       | Rule (default)                                                                                                                                                                                                                                                                                                                           |
| ---------------------------------------- | ----------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Object property slot** `k`             | The **object** parent, after comparing property values `pred[k]` and `exp[k]` | Build child subtree `ComparisonResult` `R`. Then `**match_k = predicate(R)`** — typically `**R.score >= τ**`, plus **FP/FN kind** for string property slots from the [adopted string policy](#adopted-policy-fp-vs-fn-for-required-string-properties-on-objects) (missing / `null` / wrong value). Primitives only supply `**R.score`**. |
| **List element slot** (aligned `i`, `j`) | The **list** parent, after subtree compare for that pair                      | `**match_el = predicate(R_{i,j})`**, default `**R_{i,j}.score >= τ**`. Works for **any** element type **X** (primitive, object, nested array) because every **X** comparison returns a rolled-up `**score`** in `[0, 1]`.                                                                                                                |


`**predicate` in full generality:** `match_slot: (child: ComparisonResult, ctx) -> bool` where `ctx` can carry schema path, slot kind, or strictness. The **threshold-only** form `score >= τ` is the minimal implementation; a richer function can require `**R.true_positive`** at the subtree root, or forbid partial object matches, etc.—but the **declaration site** stays the **parent list or parent object**, not the primitive leaf.

**Primitive leaves** ([evaluation-leaf-primitives.md](evaluation-leaf-primitives.md)) output `**score`** (graded quality and validity). They **do not** decide slot `**match`** or reporting labels; parents and a **metric profile** for the field do (see below).

At **parents** (structural roll-up, all field types):

- **Roll-up score** — aggregate child `**score`s** (e.g. mean over required fields or over `n_all` list slots). Answers “how close” even when slot `**match`** is false.
- **Structural slot TP / FP / FN** — extra keys, missing required properties, unmatched list rows (see **Objects** / **Lists**). Distinct from **applicability** / **answer** reporting below.

**Principle:** treat **JSON and JSON Schema** literally—do not collapse distinctions with Python idioms (`None`, truthiness on `""`).

## Applicability and answer metrics (reporting)

Checklist fields are not one-size-fits-all. **Reporting** should usually split into **two layers** so summary F1 is not driven by trivial **correct N/A** (`""` / `""`) pairs.

Configure per field in a separate **evaluation manifest** (see below)—**not** in the checklist `schema.json` used for OpenAI structured output.

- `**na_values`** — literals meaning **not applicable** (often `""` only).
- `**answer_metric`** — how to score when gold is applicable: `binary_polarity` | `multiclass` | `graded_string` | …
- For `binary_polarity`: `**positive_value**` (usually `"yes"`) and `**negative_value**` (usually `"no"`).

Leaves still emit `**score**` (exact enum → 0/1; fuzzy/semantic strings → graded). Parents still set slot `**match**` via `**predicate**`. The layers below are **how we aggregate** for dashboards and experiments.

### Schema contract: is N/A or yes/no unambiguous?

**No — not from JSON Schema alone.** Standard `type` + `enum` only lists allowed **literals**. It does **not** say which literal is N/A, which is “positive”, or that `"yes"` means boolean true.

**What the repo does today (examples):**


| Pattern                             | Example in repo                                                                                                                        | Ambiguity                                                                                                             |
| ----------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------- |
| `enum: ["yes", "no"]`               | `micrograph`, `involves_replicates`                                                                                                    | No N/A in-schema; every panel must answer. Applicability is implicit (always applicable).                             |
| `enum: ["yes", "no", ""]`           | `scale_bar_on_image` in [micrograph-scale-bar/schema.json](../soda_mmqc/data/checklist/fig-checklist/micrograph-scale-bar/schema.json) | `""` **means** N/A by convention, but JSON Schema does not **declare** it — eval must configure `na_values: [""]`.    |
| `enum: ["yes", "no", "not needed"]` | `average_values` in [individual-data-points/schema.json](../soda_mmqc/data/checklist/fig-checklist/individual-data-points/schema.json) | Third value is N/A under a **different string** than `""`. Eval must list `na_values: ["not needed"]` for that field. |


**Strings vs booleans:**


| Approach                                     | Pros                                                                                      | Cons                                                                                                         |
| -------------------------------------------- | ----------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------ |
| `**type: "string"`, `enum: ["yes","no",…]`** | Matches LLM strict outputs; easy 3-state in one field (`""` or `"not needed"`).           | Polarity and N/A are **not** machine-obvious without extra metadata.                                         |
| `**type: "boolean"`**                        | `true`/`false` are unambiguous.                                                           | **No honest N/A** in a required boolean field; you need `null`, optional property, or a separate gate field. |
| **Gate + boolean**                           | e.g. required `micrograph: yes/no`, then `scale_bar_present: boolean` only when relevant. | More fields; conditional schema is harder for strict JSON mode.                                              |


**Normative recommendation:**

- Keep `**schema.json`** as the **OpenAI structured-output** contract only (`type`, `enum`, `required`, descriptions). **Do not** add custom `x-`* evaluation keys there—they are not part of that API surface and would fork schema maintenance.
- Keep **string enums** in schema for LLM outputs (`yes` / `no` / `""` / `not needed`, etc.).
- Put all evaluation semantics in a **sidecar manifest** next to each checklist (or one manifest per checklist family), keyed by **JSON paths** into the payload.

**Do not** rely on: enum order, English word shape, Python truthiness of `""`, or guessing N/A from the schema file alone.

### Evaluation manifest (sidecar)

**Location (convention):** alongside the checklist schema, e.g.  
`soda_mmqc/data/checklist/fig-checklist/<checklist-name>/eval-manifest.json`  
next to `schema.json`.

**Purpose:** map **property paths** → **metric profile** for layers 1–2. The evaluator loads `schema.json` + `eval-manifest.json` when scoring a run.

**Path keys:** JSON-pointer-style paths from the checklist output root, e.g. `outputs[].scale_bar_on_image` (one profile per leaf field; list `[]` means “each element”).

**Example (illustrative):**

```json
{
  "checklist": "micrograph-scale-bar",
  "defaults": {
    "answer_metric": "binary_polarity",
    "positive_value": "yes",
    "negative_value": "no",
    "na_values": []
  },
  "fields": {
    "outputs[].micrograph": {
      "answer_metric": "binary_polarity"
    },
    "outputs[].scale_bar_on_image": {
      "na_values": [""],
      "answer_metric": "binary_polarity"
    },
    "outputs[].scale_bar_defined_in_caption": {
      "na_values": [""],
      "answer_metric": "binary_polarity"
    },
    "outputs[].from_the_caption": {
      "answer_metric": "graded_string"
    }
  }
}
```

**Field profile keys:**


| Key                                 | Role                                                                                |
| ----------------------------------- | ----------------------------------------------------------------------------------- |
| `na_values`                         | Literals treated as **not applicable** (layer 1). Omit or `[]` = always applicable. |
| `answer_metric`                     | `binary_polarity` | `multiclass` | `graded_string`                                  |
| `positive_value` / `negative_value` | For `binary_polarity` on applicable slots (layer 2).                                |
| `match_threshold` | Optional **`τ`** for structural / answer slot `match` (graded strings). |
| `applicability_threshold` | Optional **`τ_applic`** for layer 1 (default `1` = exact N/A vs applicable agreement). |


**Defaults:** `defaults` supplies project-wide assumptions (e.g. yes/no polarity); per-field entries override only what differs (`na_values` for `""` vs `not needed` checklists).

**Validation:** manifest paths should refer to properties that exist in `schema.json`; `na_values` must be subsets of that property’s `enum` when `enum` is present. CI or a small linter can check manifest ↔ schema consistency without changing the OpenAI schema.

**Implementation:** not in repo yet; this wiki defines the contract before `JSONEvaluator` reads it.

### Layer 1 — Applicability (is this slot in scope?)

For each property or list element slot, after comparing values:


| Gold applicable? | Pred applicable? | Label                   | Meaning                                                              |
| ---------------- | ---------------- | ----------------------- | -------------------------------------------------------------------- |
| no (`na_values`) | no               | **correct_NA**          | Right to abstain (e.g. plot panel → `""`)                            |
| no               | yes              | **spurious_applicable** | Asserted yes/no (or other answer) when check does not apply          |
| yes              | no               | **withheld_applicable** | Gold required an answer; pred sent N/A (`""` or configured sentinel) |
| yes              | yes              | *(pass to layer 2)*     | Both sides in scope for content/polarity                             |


Define **applicability precision / recall / F1** on the binary event “should this slot be scored?” (gold applicable vs pred applicable). **Do not** mix **correct_NA** into the same denominator as polarity errors unless you explicitly want a “full panel” metric.

**Applicability as its own threshold (not baked into structural `match`):**

For enums, layer 1 is usually already discrete. Define an **applicability score** in `[0, 1]` per slot, e.g. `1.0` when gold/pred agree on applicable vs N/A, `0.0` on **spurious_applicable** or **withheld_applicable**. Then:

```text
applicability_match = (applicability_score >= τ_applic)
```

Default **`τ_applic = 1`** for checklist enums (hard agree on in-scope vs N/A). Lower **`τ_applic`** only if you later add soft applicability (e.g. model confidence).

**Do not** require “no **spurious_applicable**” as an extra conjunct on **`whole_object_match`** or on structural **`match_k`**: that is equivalent to folding layer 1 into structural match with **`τ_applic = 1`**, which is often **too tight** (one spurious `yes` on an N/A field would fail the whole object flag even when answer-layer errors are what you care about). Keep **structural `match_k`** on value match (`predicate(R)`, usually **`R.score >= τ`** for the full string/enum). Report applicability via **layer 1 counts / applicability F1**, not by AND-ing into structural match unless you deliberately want one strict gate.

**Lists:** run Hungarian (or positional) alignment first; classify **each aligned element slot** (and unmatched rows as structural FP/FN). Applicability is per element, not per document.

### Layer 2 — Answer (only when gold is applicable)

**Denominator:** slots where gold ∉ `na_values` (applicable-only). Trivial N/A matches never appear here.


| Field shape                              | Example                              | Primary reporting                                                                                                                                                                                 |
| ---------------------------------------- | ------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Binary polarity enum**                 | `yes` / `no` (+ `""` N/A in layer 1) | Map `yes` → positive, `no` → negative; then **TP / FP / FN / TN** on applicable slots only. `no`/`no` is **TN** (correct negative finding), not “match” only.                                     |
| **Multi-class enum (≥3 non-N/A values)** | `low` / `medium` / `high`            | No natural FP/TN — use **match / mismatch** per slot, **confusion matrix**, **macro-F1**, and **mean `score`**.                                                                                   |
| **Free-text string**                     | title, description                   | `**score`** + threshold → slot `**match**`; mean `**score**`; optional slot TP/FP/FN via [free-text policy](#free-text-string-properties-fp-vs-fn-on-objects) (missing / `null` / wrong literal). |


**Scale bar (concrete):** `enum: ["", "yes", "no"]` with `na_values = [""]`.

- Layer 1: `""`/`""` → **correct_NA**; `""`/`yes` or `""`/`no` → **spurious_applicable**; `yes`/`""` or `no`/`""` → **withheld_applicable**.
- Layer 2 (gold `yes` or `no` only): `yes`/`yes` → TP; `no`/`no` → TN; `yes`/`no` or `no`/`yes` → polarity error (FP or FN depending on which class is “positive” in config).

Full 3×3 for debugging (all nine gold×pred pairs); **headline F1** should use **applicable-only** or **weighted** counts so **correct_NA** does not dominate.

### What we are *not* doing

- One global slot TP/FP/FN for every string field without a **metric profile**.
- Treating checklist `""` as “empty wrong string” when it means **N/A** (see enum note in [evaluation-leaf-primitives.md](evaluation-leaf-primitives.md)).
- Classical document-level TN without defining the event universe.

Structural comparison (objects, lists, Hungarian) is unchanged; only **reporting** gains this split.

### Bubbling: two layers → property slots → object `match`

Do **not** fold applicability F1 and answer F1 into one opaque flag. At each node, think of **three parallel aggregates** (nested children roll up into parents the same way):

| Aggregate | What it measures | Typical roll-up |
|-----------|------------------|-----------------|
| **Roll-up `score`** | How close (graded) | Mean (or list `n_all` rule) of child **`score`s** |
| **Structural slot counts** | Required keys / aligned elements / extras | Sum child slot TP, FP, FN; object/list P, R, F1 |
| **Manifest reporting** | Applicability + answer (layers 1–2) | Sum layer labels from manifest profile per path |

```mermaid
flowchart TB
  subgraph leaf [Leaf property value]
    L["score from compare"]
  end
  subgraph prop [Property slot k on object]
    P["predicate(R) → slot_match_k"]
    M["manifest → layer1 + layer2 labels"]
  end
  subgraph obj [Object node]
    Oscore["score = mean child scores"]
    Oslot["TP/FP/FN over property slots"]
    Oreport["sum layer1/layer2 from children"]
    Ostrict["whole_object_match = AND slot_match_k, no extra keys"]
  end
  L --> P
  L --> M
  P --> Oslot
  P --> Ostrict
  M --> Oreport
  L --> Oscore
```

**Per required property `k` (pipeline):**

1. Recurse: `R = compare(pred[k], exp[k])` (leaf, nested object, or list).
2. **Structural slot `match_k`** = `predicate(R)` — default **`R.score >= τ`**. Today’s code often mirrors this as `R.true_positive` on the child; the parent increments structural TP/FP/FN from that (and missing/extra keys).
3. **Manifest (sidecar):** load profile for path `…/k`. Classify **layer 1** (applicability) and **layer 2** (answer, only if gold applicable) from `(exp[k], pred[k])`. Optional **`applicability_score`** + **`τ_applic`**. These feed **reporting aggregates only** — they **do not** replace structural **`match_k`** (do not AND “no spurious_applicable” into `predicate`; that duplicates **`τ_applic = 1`** on applicability alone).
4. Store `R` in `field_scores[k]` (drill-down).

**Object-level “whole object matched” (strict flag):**

```text
whole_object_match =
  (∀ k in required: slot_match_k)
  ∧ (no extra keys in pred)
```

- Same idea as today’s `ComparisonResult.true_positive` on objects in [`evaluation.py`](../soda_mmqc/core/evaluation.py): all required fields “passed” **and** no spurious properties.
- **Stricter than object P/R:** you can have precision `1.0` with one FN if the denominator counts slots differently; the strict flag is **all-or-nothing** on required keys.
- **Independent of headline applicability/answer F1:** e.g. every `""`/`""` panel field can be **correct_NA** (layer 1) while one `scale_bar_on_image` `yes`/`no` flip makes `whole_object_match` false via `slot_match` false.

**When gold is `""` and pred is `""` on an enum field:**

- **Structural:** `score = 1` → `slot_match_k` true → that property counts as structural **TP**.
- **Layer 1:** **correct_NA** (good, but do not let many of these dominate **applicability F1** if that metric is meant to be hard — use a separate denominator or report **correct_NA** rate alone).
- **Layer 2:** **skipped** (gold not applicable).

**When gold is `yes` and pred is `no`:**

- **Structural:** `slot_match_k` false → property **FP**-like (and may also increment FN-style slot counts depending on child flags).
- **Layer 1:** both applicable → pass to layer 2.
- **Layer 2:** polarity **FP** or **FN** (from manifest `positive_value`).

**Lists and nested objects (e.g. `outputs[]` panel row):**

- Each **element slot** gets a child `R` that may be a **whole object** subtree.
- **Element `match_el`** = `predicate(R)` on that subtree (default: subtree roll-up **`score >= τ`**, or strict: subtree **`whole_object_match`** if you want element = “perfect panel”).
- List P/R/F1 uses **element slots** (alignment, extras, misses).
- Inside the panel object, property-level layer 1/2 counts **bubble up** by summing over `field_scores`; the list’s own manifest profile on `outputs[]` is optional if each element is an object with per-field profiles under `outputs[].<field>`.

**Checklist root:** top-level `outputs` array → list metrics; each panel object → object metrics + summed manifest counts. Export **both** strict flags and layer summaries so dashboards are not forced to infer one from the other.

**Implementation note:** code today implements **structural** roll-up + `true_positive` only. **Manifest layer sums** and **`τ_applic`** are not wired yet. **`whole_object_match`** stays structural (all required **`match_k`**, no extra keys). Applicability strictness belongs in **layer 1 metrics**, not as a hidden second conjunct on structural match.

### Objects (required keys as slots)

**Unit of accounting:** each **required schema property** is one slot. Extra keys in the prediction (not in the gold schema’s required set for that object) are naturally **FP**-like; a required property for which gold is not satisfied is **FN**-like.

For each required key `k`, compare `pred[k]` to `exp[k]` to get child `R`. The parent computes **`match_k = predicate(R)`** and applies structural FP/FN rules (free-text table below when no manifest), then increments **structural** TP / FP / FN slot counters. With a manifest, also attach layer 1/2 labels for reporting on that path.

Then at the **object** node (non-empty required set):

- **`score`** — mean of child scores (missing keys → 0).
- **Precision / recall / F1** — from **structural** property-slot TP, FP, FN (extra keys increase FP).
- **`whole_object_match`** — all required `match_k` true, no extra keys (see [bubbling](#bubbling-two-layers--property-slots--object-match) above).

### Free-text string properties: FP vs FN on objects

**Scope:** ordinary **content** strings (titles, labels, free text). **Not** checklist enums where `""` ∈ `na_values` — use [applicability and answer metrics](#applicability-and-answer-metrics-reporting) instead.

**Definitions (normative for this wiki):**

- **FN (false negative on this slot)** — Gold’s requirement for this property is a **conforming value that pred never supplies** in JSON terms: the key is **absent**, or the value is JSON `**null`** while the schema for this property is **string-only** (type does **not** include `null`). Interpretation: *recall failure* — “we needed a string here and did not get one.”
- **FP (false positive on this slot)** — Pred **does** supply a JSON value for the property, but the **object parent** does **not** mark the slot `**match`** (wrong text, `**predicate**` false on the child subtree—e.g. `**score < τ**`—outside `enum`, **or** empty string `""` when gold expects a non-empty string / when `minLength` and leaf logic yield a failing `**score`**). Interpretation: *precision failure* — “something was asserted for this key, but it is wrong or inadmissible.”

**Why `""` is FP, not FN (JSON-native argument):**

In JSON, `""` is a **string value that exists**. It is neither “key missing” nor `null`. Calling it **FN** would conflate three distinct gold–pred situations that the wire format keeps separate. **FP** keeps the invariant: **FN only when no conforming literal was provided**; `**""` is a literal (length 0)** → if the parent’s `**predicate*`* is false (e.g. low `**score**` vs gold), that is pred’s **wrong submission**, same class as `"bad"`. Use `**minLength: 1`** in the schema when the product forbids empty strings; the leaf reflects failure in `**score**`; the parent still classifies the slot as **FP**, consistent with other non-matching strings.

**Why JSON `null` is FN (when `type` is `"string"` only):**

If gold has a string and the schema does not admit `null`, `**"label": null`** means pred did not deliver a string at all—it is **not** the empty string and **not** a missing key, but in evaluation terms it is **“no string answer”** alongside absence. Classifying as **FN** aligns **missing key**, `**null`**, and (if you ever need it) other **non-string** typed values for a string slot under the same *recall* story. If the schema is explicitly `["string", "null"]` and gold is `null`, then `**null` vs `null`** is a normal leaf comparison; the **object parent** marks the slot `**match`** when `**predicate**` holds on the child result (typically high `**score**` for both null).

**Summary table (string slot, gold expects a concrete string like `"ok"`):**


| Pred JSON for `label`                    | Slot classification                        | One-line reason                                                                                                         |
| ---------------------------------------- | ------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------- |
| Key **absent**                           | **FN**                                     | No property in pred object — nothing to compare to gold’s string.                                                       |
| `**null`** and schema is **string-only** | **FN**                                     | No string value delivered (JSON null ≠ string).                                                                         |
| `**""`**                                 | **FP**                                     | A string was delivered; the parent does not mark **match** vs gold / schema (unless gold is `""` and schema allows it). |
| `**"bad"`** (non-matching)               | **FP**                                     | Wrong asserted string.                                                                                                  |
| **Extra key** (e.g. `noise`)             | **FP** (separate slot or extra-field rule) | Assertion outside gold’s allowed shape for that object.                                                                 |


**JSON Schema as contract:** `required`, `type` (including union with `"null"`), `minLength`, `enum`, and `pattern` define how the **leaf `score`** is formed; the **object parent** then combines `**score`**, `**predicate**`, and this table to label **FP vs FN** and whether the property `**match`**es.

**Do not use** Python `if not value:` across JSON forms—that erases the `**""` vs `null` vs missing** distinction JSON was designed to preserve.

**Toy — object with two required fields (wrong string on `label`)**

- Gold: `{ "id": 1, "label": "ok" }`  
- Pred: `{ "id": 1, "label": "bad" }`  
- `id` **matches** → **one TP** slot. `label` **does not match** (wrong string) → **one FP** slot, **not** FN.

Slot totals: `**TP = 1`, `FP = 1`, `FN = 0`** → precision `1/2`, recall `1`. A strict “whole object matched” flag would be **false** (not every property matched; no extra keys).

**Toy — extra key**

- Gold: `{ "id": 1 }`, Pred: `{ "id": 1, "noise": true }`  
- One **FP** slot from `noise`, so object precision drops; object `true_positive` is false.

### List of X (including primitives and nested non-object items)

**Unit of accounting:** **list elements after alignment**, not raw indices. Build pairwise **subtree `score`s**, **align** (e.g. Hungarian), then for each aligned pair let the **list parent** set **element-slot `match`** via `**match_el = predicate(R)**` (default `**R.score >= τ**`). Rows that are pred-only or gold-only after that step are **FP elements** / **FN elements** in the list story.

Let `TP_el` = number of matched pairs above threshold, `FP_el` = extra predicted elements, `FN_el` = missing expected elements. The implementation uses `n_all = max(len(pred), len(exp))` when averaging **scores** across slots (including zero-score stubs for missing/extra).

- **List precision** ≈ `TP_el / (TP_el + FP_el)` — “of what the model predicted as aligned mass, how much was good?”
- **List recall** ≈ `TP_el / (TP_el + FN_el)` — “of what gold required, how much was recovered above threshold?”
- **List F1** from precision and recall as usual.

**Toy — list of strings, exact match threshold 1.0**

- Gold: `["a", "b"]`, Pred: `["a", "b", "x"]`  
- Alignment: match two pairs (`a`,`b`), one prediction row `x` is **extra** → `TP_el = 2`, `FP_el = 1`, `FN_el = 0` → precision `2/3`, recall `1`.

**Toy — same length, one substituted value (`x` vs `c`)**

- Gold: `["a", "b", "c"]`, Pred: `["a", "b", "x"]`  
- Hungarian matching still assigns **position-wise** pairs that maximize total similarity (here `(0,0)`, `(1,1)`, `(2,2)`). The first two pairs score `1.0` at exact match; `**"x"` vs `"c"`** scores `0.0` (below threshold `1.0`).

Under **threshold-gated** alignment, a pair **below threshold** does **not** count toward `TP_el`. The prediction index `2` is then treated as an **unmatched prediction slot** → **one FP element** (pred side paid a “bad / unmatched” row). The expected index `2` is an **unmatched expected slot** → **one FN element** (gold side paid a “not recovered” row).

So **both** labels apply at the **element-slot** layer: `**x` → FP**, `**c` → FN**. That is not a logical contradiction: one **failed alignment** charges **one pred row** and **one gold row** in the bookkeeping, not the same JSON atom wearing two hats.

With `TP_el = 2`, `FP_el = 1`, `FN_el = 1`: **precision** `= 2/(2+1) = 2/3`, **recall** `= 2/(2+1) = 2/3`. Intuitively: one wrong pred slot and one missed gold slot relative to a perfect 3-for-3 alignment.

**Toy — list of strings, one wrong value (sub-threshold pair)**

- Gold: `["a"]`, Pred: `["z"]`  
- Hungarian still assigns the pair, but if similarity is **below** `match_threshold`, the pair does **not** count as `TP_el`. The prediction and expected rows then fall through to **unmatched** element accounting, so element-level FP/FN drive list precision/recall.

### List of X when **X** is not a leaf (objects or nested arrays)

**Normative rule:** at any list node, **TP / FP / FN always describe list *elements*** (whole **items** after alignment), never individual fields *inside* those items. Fields and inner lists have **their own** TP/FP/FN **inside** the nested `ComparisonResult` for that item.

**How element-slot `match` is decided when X is an object or array:**

1. For every pair `(pred_i, exp_j)`, run the **full subtree comparison** for type **X** and read that pair’s **rolled-up subtree `score`** in `[0, 1]` (already aggregating whatever is inside: object fields, inner lists, leaves).
2. Build the **assignment** on those scores (e.g. Hungarian to maximize sum of scores).
3. **Element-slot `match` at the list parent** — `predicate(R_{i,j})`; default `**R_{i,j}.score >= τ`** (one decision per **whole element** subtree, independent of whether **X** is primitive, object, or array).
4. `**TP_el` / `FP_el` / `FN_el`** at this list are **exactly** the same slot story as for list-of-strings: matched slots above threshold, unmatched pred rows → FP, unmatched exp rows → FN.

**Why not propagate inner TP/FP/FN to the list row?** The list layer answers **“which checklist items aligned?”** (set-style), not **“how many fields inside an item?”** Inner metrics live in `**element_scores["match_i_j"].field_scores`** (objects) or deeper `**element_scores`** (arrays).

**List of objects (`X` = `object`):**

- **Element layer** — as above; each matched item carries a full object subtree (`field_scores` per property, object-level P/R/F1 inside that subtree).
- **Cross-list field rollups** — **additionally**, for each required **object field name** across items, the **second** `field_scores` block on the **list** node aggregates how that field behaved over all alignments (see below). Element TP/FP/FN and field rollups answer different questions.

**List of arrays (`X` = `array`):**

- **No** list-level `field_scores` rollup (arrays have no named fields). Drill-down is `**element_scores`** only: each slot holds a nested list’s own `ComparisonResult` (with its own element-level TP/FP/FN for the inner list).
- **Outer** TP/FP/FN = quality of **which inner lists** matched as wholes; **inner** TP/FP/FN = quality **inside** each inner list.

**Toy — list of two small objects (same structure, one wrong field inside one item)**

- Gold: `[ { "id": 1, "t": "a" }, { "id": 2, "t": "b" } ]`  
- Pred: `[ { "id": 1, "t": "a" }, { "id": 2, "t": "wrong" } ]`

After alignment, both pairs can sit on the diagonal; the **second item’s subtree score** is below threshold if `t` drives the object roll-up down. Then that row behaves like the `**"x"` vs `"c"`** string case: **one FP element** (bad second pred object) and **one FN element** (second gold object not recovered as a `match`) at the **list** element layer—**even though** the first object matched. Inside the second item’s drill-down, you still see **which** field failed (`t`) and object-internal FP/FN there.

**Toy — list of lists**

- Gold: `[ [1,2], [3] ]`, Pred: `[ [1,2], [9] ]`  
- Outer list: two items; compare inner lists pairwise; if inner list `[3]` vs `[9]` fails threshold, same **FP/FN element** pattern at the **outer** slot level. Inner list `[1,2]` vs `[1,2]` has its own `TP_el` etc. inside the nested result.

When **X** is a **primitive** only, there is **no** second `field_scores` rollup under the list—only `**element_scores`** with synthetic alignment keys.

**List-of-objects — second layer only (`field_scores` on the list node):** beyond element-layer `TP_el` / `FP_el` / `FN_el`, for each **required object field name** `f` aggregate over all list slots: matched items contribute that field’s subtree; each **extra** pred list row adds **FP** for every `f`; each **missing** gold row adds **FN** for every `f`. That yields **another** precision/recall/F1 per `f` in `**field_scores[f]`**—different question from element-layer P/R (“how did field `title` behave across the whole checklist list?”).

The words **precision / recall** on a **list-of-objects** node can therefore mean **element-layer** metrics **or** `**field_scores[f]` rollups**—always check which subtree you are reading.

**Toy — list of `{ "id": integer }`, threshold such that only `id` must match**

- Gold: `[ { "id": 1 }, { "id": 2 } ]`  
- Pred: `[ { "id": 1 } ]`

Hungarian matches one object; one gold object is missing → element-level **FN**; list recall < 1. For field `id`, the rollup counts reflect one matched `id` and one missing row’s **FN** contribution to that field’s rollup.

### Accuracy

**Accuracy** depends on which layer and field profile you mean. **Applicable-only** accuracy for a `yes`/`no` field can use (TP + TN) / (applicable slots). **Full-panel** accuracy that counts **correct_NA** needs its own denominator and label — do not merge with polarity F1 without an explicit definition.

---

## Three structural shapes (why we distinguish them)

The dispatcher looks at the JSON Schema `**type`** (and `required`, `items`, etc.) at the current path. Three shapes matter for how we attach drill-down:


| Shape               | Meaning                                                                 | Drill-down role                                                                                                                                                                     |
| ------------------- | ----------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Object**          | JSON object with named properties                                       | One nested result **per property name** (schema `required` keys).                                                                                                                   |
| **List of X**       | JSON array whose `**items`** schema describes element type **X**        | Drill-down is **per list alignment slot** (see below). **X is not “scalar only.”**                                                                                                  |
| **List of objects** | **X** is `object` — each element is an object with its own `properties` | Same per-slot drill-down, **plus** a second need: roll-ups **per object field name across the whole list** (e.g. average score for field `title` across all matched list elements). |


### What is X in “list of X”?

**X** is whatever the schema says `**items`** is: the single element sub-schema. In our evaluator that can be:

- **Primitives** in the JSON Schema sense: `string`, `number`, `integer`, `boolean` (sometimes loosely called “scalars” together with `null` when you add it).
- `**object`** — list of objects (the case that triggers extra cross-element field roll-ups).
- `**array`** — **nested list**; recursion continues with the same list machinery on inner arrays. So **X can absolutely be a list** (list of lists, list of lists of objects, …).

So “list of X” is **not** restricted to primitive leaves; **X** is any valid `items` schema the navigator recurses into.

## Hungarian alignment and synthetic keys

For lists whose elements are **not** 1:1 with positional identity (we use **optimal bipartite matching** on pairwise similarity), a matched pair is identified by **indices** `(pred_idx, exp_idx)` after assignment. Keys such as `match_0_2` mean: “prediction index 0 was paired with expected index 2.” Unmatched prediction rows become `unexpected_element_{pred_idx}`; unmatched expected rows become `missing_element_{exp_idx}`.

Those strings are **synthetic** because the **schema does not name element slots**—only order and value. The key encodes the **alignment event**, not a property name from JSON.

## `element_scores` vs `field_scores` (and why `field_scores` feels wrong on lists)

The implementation uses **one** dataclass for every node type, with two dict fields. That keeps serialization uniform but **overloads names**:

### On an **object** node

- `**field_scores`** — Keys are **real JSON property names**. Values are full nested `ComparisonResult` trees for that property. This matches the word “field.”
- `**element_scores`** — Unused (empty dict).

### On a **list** node

- `**element_scores`** — Keys are the **synthetic alignment keys** above. Values are nested `ComparisonResult` trees for **each matched pair** or **stub rows** for extra/missing elements. This is the natural “list drill-down.”
- `**field_scores`** — Filled **only when items are objects**. Keys are again **object field names**, but values are **aggregated summaries across the entire list** (mean score and P/R/F1-style counts for that field name over all alignments)—**not** another layer of per-slot object keys. Per-slot **per-field** detail still lives **inside** each match’s nested `ComparisonResult.field_scores`.

So `**field_scores` means two different things** depending on the parent:

1. **Under an object:** “per-property subtree.”
2. **Under a list-of-objects:** “per-property **list-level rollup**,” not the same semantic as (1).

Your doubt is reasonable: reusing `**field_scores`** for the list-of-objects rollup is **compact** (one JSON shape everywhere) but **confusing** because “field” suggests object keys, not “summary across many list elements.” Alternatives discussed for code (not all done yet): rename to roles like `list_item_field_rollups`, or use a **discriminated union** (`ObjectBreakdown` vs `ListBreakdown`) instead of two parallel dicts. See the refactor plan’s section on hierarchical results.

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

