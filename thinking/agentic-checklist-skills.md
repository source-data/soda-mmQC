---
title: Agentic checklist skills (design brief)
date: 2026-07-22
tags: [agentic, skills, checklist, anthropic, evaluation, mcp]
status: draft
---

# Agentic checklist skills — design brief

Planning only. No implementation in this note.

## Goal

Replace the current one-prompt-per-check model call (`run_model` → `generate_response` with prompt + schema) with a **hierarchy of nested skills** per checklist (dependencies form a directed acyclic graph, or DAG):

- Skills can be **chained and reused** (e.g. panel identification → applicability → leaf analysis).
- Skills may invoke **Python tools** (and later MCP).
- **CLI surface of `run.py` / `evaluate` stays the same** for running checks on examples.
- **Evaluation is split out** into a separate CLI (cleaner than today’s combined run+score path).
- First runtime: **Anthropic Claude Agent SDK**; later **OpenAI Agents**, behind a small code layer.

## Current state (baseline)

Today a check is a directory under `soda_mmqc/data/checklist/{checklist}/{check}/` with:

| Asset | Role |
|-------|------|
| `prompts/prompt.N.txt` | Single monolithic instruction |
| `schema.json` | Structured output contract (OpenAI wrapper or inner schema) |
| `eval-manifest.json` | Scoring policy for flat leaf eval |
| `model_config.json` | Optional provider tools / reasoning (mostly OpenAI-shaped) |
| `benchmark.json` | Which examples under `data/examples/` to run |

`run.py` discovers checks, loads examples, caches by key, calls the model once per example+prompt, then scores. See [README](../README.md) and [run.py](../soda_mmqc/scripts/run.py).

There is already a check named `external-data-url-validation-agentic`, but it is still a **single prompt**, not a skill graph.

## Decisions (locked for this brief)

| Topic | Decision |
|-------|----------|
| What owns the skill graph? | The **checklist** (scope of related reusable nested skills) |
| What is a check? | A **leaf skill** in that hierarchy (a skill with no checklist-local dependents; it owns `schema.json`) |
| Leaf contents | Folder with `SKILL.md` + `schema.json` (+ `eval-manifest.json`, `benchmark.json` as today) |
| Intermediate skills | Folder with `SKILL.md` only (no eval schema); may declare tools/needs |
| Graph source of truth | **Skills** — frontmatter `requires` / `produces` (option C earlier) |
| Hierarchy for humans/tools | **Generated** `dag.yaml` + README from skill metadata (not hand-edited) |
| Example / cache loop | **Python orchestrator (A)** — deterministic batching; agent runs **per example** |
| Evaluation | **Separate CLI** from check execution |
| Model/agent config | Skill declares **capability needs**; **checklist-level defaults** are the practical primary config; runner maps needs → Anthropic/OpenAI agent settings |
| Provider priority | Anthropic Agent SDK first; OpenAI Agents later via same abstraction |
| Skill versioning | **Checklist versioning manifest** pins every skill. Runner default = all pinned. CLI may **unpin exactly one** skill; runner sweeps that skill’s versions against the pins. Skill store + runtime assembly; agent sees one version per name. Cache per expanded SkillSet. Langfuse exposes the pinned (production) manifest. |
| Production exposure | **Langfuse remains the surface** that applications use to pull the **production** prompt/skill (as today). Repo/git holds all versions; Langfuse labels/points at what is production for runtime apps. |

## Skill versioning and Langfuse

### The problem

We need all of:

| Consumer | Needs |
|----------|--------|
| Apps (production) | One labeled “production” resolution (today: Langfuse) |
| Benchmarking | Compare skill revisions (`prompt.1` vs `prompt.4` today) — **including** changing one skill in a chain while holding others fixed |
| Agent runtime | **Exactly one** body per skill name (no `SKILL.1` + `SKILL.2` in view) |
| Cache | Correct, non-mysterious keys when a leaf depends on three versioned upstream skills |

**Pure “edit one `SKILL.md`; git history = versions” is not enough** for comparative evaluation: you cannot conveniently pin “leaf@4 + identify-panels@2” vs “leaf@4 + identify-panels@3” and report them side by side the way we do with prompt versions today.

**Side-by-side files in the agent-visible tree** confuse the agent.

So we need an explicit **skill store** (many versions) plus a **runtime assembly** step (one coherent view).

### What a “skill version” is

An immutable, addressable object: `{skill_name, version_id}` → folder contents (`SKILL.md`, and for leaves usually `schema.json` / related contracts).

`version_id` can be an integer, semver, or content hash — the important part is **addressability**, not that it equals a git commit (though the store may live *in* git).

### Proposed model: skill store + runtime assembly

```text
┌──────────────────────────── Skill store (many versions) ────────────────────────────┐
│  skills-store/identify-panels/v1/SKILL.md                                            │
│  skills-store/identify-panels/v2/SKILL.md                                            │
│  skills-store/micrograph-scale-bar/v3/SKILL.md + schema.json                         │
│  skills-store/micrograph-scale-bar/v4/...                                            │
└─────────────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      │  resolve SkillSet (lock / CLI)
                                      ▼
┌──────────────────────── Runtime view (one version per name) ────────────────────────┐
│  .runtime/<run_id>/skills/identify-panels      → symlink → store/.../v2             │
│  .runtime/<run_id>/skills/micrograph-scale-bar → symlink → store/.../v4             │
│  Agent + CLAUDE.md only see this flat skills/ tree                                  │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

- **Store:** all versions coexist; humans and eval tooling can list/compare them; git still versions the store.
- **SkillSet (lockfile):** explicit map of `skill_name → version_id` for a run, e.g.

  ```yaml
  # skillset.yaml (per evaluate run or checked-in “profiles”)
  leaf: micrograph-scale-bar
  skills:
    identify-panels: 2
    micrograph-scale-bar: 4
  ```

- **Assembly:** orchestrator builds a temp (or cacheable) runtime directory with **symlinks** (or copies) so the agent sees `skills/<name>/SKILL.md` only.
- **Comparative eval:** run the same examples under SkillSet A vs SkillSet B (two assemblies, two prediction dirs). Reporting keys off SkillSet id, not a single “prompt version” integer.

Langfuse `production` becomes a **named SkillSet** (or a pointer to one): apps fetch that resolution; they do not browse the store.

### Combinatorial explosion (3 × 4 ≠ “run 12”)

If `skill_1` has 3 versions and `skill_2` has 4, the store contains **12 possible** pairings. That does **not** mean evaluation runs the cartesian product.

### Checklist versioning manifest (proposed workflow)

One **versioning manifest** per checklist pins **every** skill in that checklist’s DAG to exactly one version.

```yaml
# fig-checklist/version-manifest.yaml  (name TBD)
skills:
  identify-panels: 2
  applicability-micrograph: 1
  micrograph-scale-bar: 4
  # ... every skill in the checklist must appear with a pin
```

**Rules:**

1. **Complete pins.** The manifest must list every skill in the checklist; each has a pinned version. Creating a new skill requires adding it to the manifest (CI/check fails otherwise).
2. **Pre-run validation (required).** Before assembling the runtime or calling the agent, the orchestrator **validates** the manifest against the checklist skill inventory (from `requires`/`produces` / generated `dag.yaml`):
   - every skill in the DAG appears in the manifest with a pin;
   - every pin resolves to an existing version in the skill store;
   - no unknown skill names in the manifest (orphan pins);
   - if `--unpin` is set: that skill exists, and each version to sweep exists in the store.
   Fail fast with a clear error if any pin is missing or invalid — do not run partially.
3. **Baseline run.** `evaluate` with no unpin → assemble runtime from the manifest as-is (one SkillSet). This is the default / production-shaped run.
4. **Unpin via CLI, one skill only.** e.g. `--unpin micrograph-scale-bar` (or `--unpin micrograph-scale-bar --versions 3,4` if we want a subset).
5. **What the runner expands.** With one skill unpinned, “combinations” means: **for each version of that skill** (or each version you list), run once with **all other skills still at manifest pins**.  
   - Unpin leaf with 4 versions → **4** runs, not a grid.  
   - Unpinning two skills at once is **rejected** (keeps eval 1-dimensional).
6. **Agent view.** Each of those runs still gets one assembled tree (symlinks); the agent never sees other versions.

So: inventory can grow per skill; eval cost is `1` (baseline) or `N` (versions of the single unpinned skill).

**Policy summary:** versions are inventory; the **manifest** is the pinned baseline; **unpin one skill** is how you compare; never grid-search the DAG.

### Hard rules

1. **Agent never sees the store** — only the assembled runtime tree.
2. **One name → one version** per assembly.
3. **Manifest is complete** — every checklist skill pinned; new skills must be registered.
4. **Validate before run** — missing/invalid pins abort the run; no partial execution.
5. **At most one unpinned skill per run** — runner expands that skill’s versions against the manifest pins.
6. **No cartesian eval** across multiple unpinned skills.

### Caching: the chain-version mess (and how to make it boring)

Worry: leaf check + three chained skills at independent versions → combinatorial cache nightmare **if** we treated every pairing as a first-class eval target or tried clever partial reuse across the matrix.

**Mitigation — treat the SkillSet as an atomic unit (and keep the set of SkillSets small):**

| Do | Don’t |
|----|--------|
| Cache key includes `skillset_hash = hash(sorted(skill→version→content_hash))` | Cache “leaf only” and hope upstream matches |
| One prediction artifact per `(example, skillset_hash, model, config)` | Reuse a leaf result when only an upstream skill version changed |
| Store `skillset.yaml` + content hashes next to predictions for provenance | Infer versions from filenames after the fact |
| Optional later: cache **intermediate** `produces` artifacts keyed by `(skill, version, input_hash)` if worth the complexity | Build a clever partial-reuse graph in v1 |
| Run baseline + 1D variants | Nested loops over all skill versions |

v1 recommendation: **session-level cache only** (whole check run for one example under one expanded SkillSet from manifest ± one unpin). Intermediate skill caches deferred.

Comparative reporting: baseline manifest vs each version of the unpinned skill (same idea as today’s prompt.1 vs prompt.4, with the rest of the chain held fixed).

### Langfuse’s role

- **Not** the multi-version store (the repo skill store is).
- **Is** the app-facing **production** view of the checklist **versioning manifest** (all pins).
- Benchmark against production = run with no unpin (or explicitly use the published manifest).
- Experiments = `--unpin <skill>` locally without publishing.

### Still open

- Store layout on disk (`skills-store/<name>/vN/` vs content-addressed `sha256/…` with names as tags).
- Whether `version_id` is author-chosen (`v4`) or content hash (tags for human names).
- CLI shape: `--unpin SKILL` and optional `--versions 1,2,4` vs always sweep all versions of the unpinned skill.
- Promote: publish versioning manifest → Langfuse (multi-skill payload or bundled closure for legacy apps).
- Symlink vs copy on Windows / sandbox.
- Manifest filename / whether unused skills outside a leaf’s closure must still be pinned (lean yes: whole-checklist manifest).

## Recommended architecture

```text
CLI (evaluate CHECKLIST [--check LEAF] [--model ...] ...)
        │
        ▼
┌─────────────────────── Python orchestrator ───────────────────────┐
│  resolve checklist / leaf(s)                                      │
│  load benchmark examples                                          │
│  cache get/set                                                    │
│  for each example:                                                │
│      run_agent_session(leaf, example, checklist_config)  ─────────┼──► Agent
│  write raw outputs (predictions)                                  │
└───────────────────────────────────────────────────────────────────┘

Separate CLI (e.g. score / evaluate-outputs):
        │
        ▼
  load predictions + expected_output + leaf schema + eval-manifest
  FlatEvaluator → analysis.json / reports
```

### Per-example agent session

1. Load checklist context (`CLAUDE.md` / project instructions describing `soda_mmqc/data`).
2. Resolve the **closure** of skills needed for the target leaf (walk `requires` backward from the leaf; optionally use generated `dag.yaml` for the same graph).
3. Run skills in dependency order (or let the agent schedule within the closure), passing artifacts via `produces` contracts.
4. Enforce / request structured output against the **leaf** `schema.json`.
5. Return JSON to the orchestrator for caching and later scoring.

### On-disk sketch (illustrative)

```text
soda_mmqc/data/checklist/fig-checklist/
  CLAUDE.md                 # or repo-root CLAUDE.md pointing here
  model_defaults.json       # checklist-level agent defaults (needs → settings)
  dag.yaml                  # GENERATED — do not hand-edit
  README.md                 # GENERATED hierarchy view
  skills/
    identify-panels/
      SKILL.md              # requires: [] ; produces: [panels]
    micrograph-scale-bar/   # LEAF = check
      SKILL.md              # requires: [identify-panels] ; produces: [scale_bar_report]
      schema.json
      eval-manifest.json
      benchmark.json
```

`.claude/skills/...` may **symlink** into `skills/` so Claude Code / Agent SDK discovers the same files without duplicating content.

### Skill frontmatter (illustrative)

```yaml
---
name: micrograph-scale-bar
description: Check scale bars on micrograph panels.
kind: leaf   # or intermediate
requires: [identify-panels]
produces: [scale_bar_report]
needs: []    # e.g. [web, code] — usually empty; checklist defaults apply
---
```

`schema.json` remains the **eval/curation contract** for leaves only. Intermediate skills use informal or typed `produces` names; they do not need full JSON Schema unless we later want typed intermediate artifacts.

## Approaches considered

### 1. Orchestration: Python loop vs top-level agent

| | Python loop (chosen) | Top-level agent |
|--|----------------------|-----------------|
| Batching / cache / resume | Deterministic, cheap | Fragile, expensive |
| Agentic value | Skill chaining **inside** each example | Also decides which examples to run |
| Fit for benchmarking | Strong | Weak for CI |

**Rejected for v1:** top-level agent owning the example loop. Optional later as a thin wrapper over the same tools.

### 2. Graph representation: `dag.yaml` vs skills-only vs FS tree

| | Explicit only | Skills SoT + generated dag (chosen) | FS tree |
|--|---------------|--------------------------------------|---------|
| Reuse (multi-parent) | Easy | Easy | Awkward |
| Drift | High if hand-edited | Low if generated + CI check | Medium |
| Visualize | Good | Good (generated README) | Local only |

**Rejected:** filesystem tree as graph; hand-maintained `dag.yaml` as SoT.

### 3. Schema placement

| | Leaf schema (chosen) | Checklist-wide schema | Schema only in prompt text |
|--|----------------------|------------------------|----------------------------|
| Eval/curation | Natural path to leaf dir | Ambiguous multi-check | Brittle |
| Reuse of upstream skills | Unaffected | N/A | N/A |

**Chosen:** schema on leaf folders; evaluation and curation resolve `…/skills/{leaf}/schema.json` (or keep today’s path shape via compatibility).

### 4. Code layer: in-process Python tools vs MCP

Needed so skills can call deterministic helpers (load figure, parse caption, validate URL, etc.).

| | In-process tools (SDK tool defs) | MCP server wrapping same functions |
|--|----------------------------------|-------------------------------------|
| Anthropic Agent SDK | First-class | Also supported |
| OpenAI Agents later | Map tool registry | Same MCP or dual adapter |
| Local/CI | Simple | Extra process |
| External clients (Cursor, etc.) | Harder | Natural |

**Recommendation:** implement a **small Python tool registry** first (functions the agent can call). Expose the **same registry via MCP** as a thin adapter so Cursor/other hosts can use checklist skills without forking logic. Do not invent two implementations of “load example” / “read schema”.

## Config model (needs + checklist defaults)

1. **Checklist** `model_defaults.json` (name TBD): provider-agnostic defaults used in practice — effort, max tokens, allowed tool families, etc.
2. **Skill** `needs: […]`: rare overrides when a skill truly requires web/code/etc.
3. **Runner** maps `(defaults ∪ needs ∪ CLI --model)` → Anthropic Agent SDK session config (v1) or OpenAI Agents config (later).
4. **Cache key** includes: leaf id, skill closure hash (or content hashes), example id, model id, effective config hash, schema hash.

Avoid shipping OpenAI-shaped `model_config.json` as the long-term skill contract; keep migration shims if needed.

## CLAUDE.md role

Repo or checklist-level `CLAUDE.md` should set the stage for the agent:

- Layout of `soda_mmqc/data/` (checklist vs examples vs evaluation outputs).
- Meaning of a **skill** vs **leaf/check** vs **generated dag**.
- Where schemas, benchmarks, expected outputs live.
- How to invoke Python/MCP tools (names, contracts).
- Output discipline: final answer must validate against the leaf schema.

This is agent orientation, not a substitute for per-skill `SKILL.md` procedures.

## CLI split

| Command (names TBD) | Responsibility |
|---------------------|----------------|
| Existing `evaluate` / `run.py` (compat) | Discover examples, run agent sessions, write predictions, cache |
| New `score` (or similar) | Load predictions + gold + leaf schema + eval-manifest → analysis |
| Existing `curate` / `init` | Continue to read leaf `schema.json` for fields to curate |

Keep argument names of the run CLI stable where possible (`CHECKLIST_NAME`, `--check`, `--model`, `--no-cache`, …).

## Migration sketch (not scheduled here)

1. Introduce `skills/` layout alongside existing check dirs (or gradually rename).
2. Convert one pilot checklist (e.g. subset of `fig-checklist`) to skills + one shared upstream skill.
3. Generator: skills → `dag.yaml` + README; CI fails on drift.
4. Swap `run_model` path for agent session behind a flag / provider.
5. Split scoring into new CLI; `evaluate` stops writing analysis or shells out.
6. Retire monolithic `prompts/prompt.N.txt` once leaves are skills; versions live in the skill store; production SkillSet published to Langfuse.

## Open questions

1. **Manifest + CLI:** filename; `--unpin` / optional `--versions`; replace `--prompt-version`.
2. **Git → Langfuse promote:** publish versioning manifest / closure.
3. **Intermediate artifact schema:** informal `produces` names vs optional JSON Schema for upstream skills (better debugging, more authoring cost).
4. **Shared skills across checklists:** global store vs per-checklist store (manifest still per checklist).
5. **Parallelism:** Python may run multiple example agent sessions concurrently — rate limits and cache atomicity.
6. **Mock mode:** `--mock` should short-circuit agent and return expected leaf JSON (keep for CI).
7. **Langfuse tracing:** whole agent session vs per-skill spans (separate from Langfuse-as-production-surface).
8. **When (if ever) to cache intermediate skill outputs** — deferred; v1 session-level only.

## Non-goals (this brief)

- Implementing the Agent SDK integration.
- Rewriting all existing prompts to `SKILL.md`.
- Replacing `FlatEvaluator` scoring semantics.
- Committing to MCP-only (MCP is an adapter, not the only tool surface).

## Related

- Data layout today: [README — Checklists and checks](../README.md)
- Evaluation scoring: [evaluation-scoring.md](evaluation-scoring.md)
- Orchestration code: [soda_mmqc/scripts/run.py](../soda_mmqc/scripts/run.py)
