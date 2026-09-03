---
title: Agentic checklist conversion plan
date: 2026-08-06
tags: [agentic, skills, checklist, migration, implementation-plan]
status: draft
---

# Agentic Checklist Conversion Implementation Plan

> **For Copilot:** Execute this plan phase by phase. Do not pass a human decision gate without explicit approval.

**Goal:** Convert mmQC from one model call per monolithic check prompt into an agentic DAG pipeline of versioned, reusable checklist skills while preserving leaf JSON evaluation and the existing run CLI surface.

**Architecture:** Keep the entire implementation inside mmQC. `soda_mmqc/agentic/` owns DAG resolution, versioning, runtime assembly, orchestration, provider adapters, operation permissions, persistence, and provenance. Each checklist owns its agentic DAG under `soda_mmqc/data/checklist/<checklist>/agentic/`, including versioned skills, shared artifact contracts, complete pins, and generated graph documentation. For each example and target leaf, the orchestrator executes the resolved closure in topological order with one fresh, skill-scoped sub-agent per skill.

**Tech stack:** Python 3.12, Anthropic Claude Agent SDK (exact package and version to be verified), MCP, approved pre-generated script wrappers, JSON Schema, YAML, pytest, and the existing `FlatEvaluator`; OpenAI Agents adapter deferred until the Anthropic path is stable.

**Design relationship:** [agentic-checklist-skills.md](agentic-checklist-skills.md) remains the design brief. This plan is authoritative for the concrete mmQC layout, orchestrator-controlled sub-agents, typed artifact contracts, operation restrictions, and phased implementation.

---

## 1. Boundaries and success criteria

This is an implementation plan, not an implementation. File paths marked **proposed** do not exist yet.

The conversion is complete only when all of these are true:

- `evaluate CHECKLIST`, `--check`, `--checks`, `--model`, `--mock`, and `--no-cache` still run checklist examples with their current user-facing meanings.
- A checklist version manifest pins exactly one version of every skill in that checklist's agentic DAG.
- `--unpin SKILL` expands only that skill's selected versions against all other manifest pins; attempts to unpin more than one skill fail before model calls.
- The agent sees exactly one version of each skill and cannot access the version store.
- For each scientific input and target leaf, the orchestrator resolves the closure, orders it topologically, and launches one fresh sub-agent per skill.
- Each sub-agent can access only its selected skill, required input artifacts, output contracts, and mmQC-approved MCP or pre-generated script operations.
- Sub-agents cannot open a shell, write or modify scripts, execute arbitrary code, install packages, register tools, launch agents, or write directly to artifact storage.
- `requires` and `produces` frontmatter entries name versioned artifact contracts so producer/consumer compatibility is machine-checkable.
- Skill leaves own their scientific output `schema.json`; mmQC owns `eval-manifest.json`, `benchmark.json`, and curated gold because those define application-specific evaluation policy.
- Intermediate skill artifacts may be schema-validated and persisted for debugging, but are not folded into leaf gold or scored by `FlatEvaluator`.
- Prediction cache identity includes the checklist, complete resolved SkillSet, model, effective session policy, artifact contracts, leaf, and example.
- The mmQC agentic pipeline persists artifacts, predictions, cache entries, and provenance under configured mmQC data roots.
- Scoring can run independently from model execution against persisted predictions.
- Legacy prompt execution can be removed without changing curated `expected_output.json` files or flat scoring semantics.

Explicit non-goals for v1:

- OpenAI Agents runtime implementation.
- Cartesian sweeps across multiple skill versions.
- Intermediate artifact scoring.
- Intermediate artifact caching.
- A global panel-kind taxonomy.
- Converting every checklist before the pilot is accepted.
- mmQC deployment, production promotion, and rollback policy.
- Langfuse integration or publication.
- Publishing an independent Python library or standalone skill-collection package.
- Cross-checklist skill sharing or a global skill registry.

## 2. Recommended target layout

The branch adds one agentic pipeline subtree to the existing mmQC package. Agentic code remains under `soda_mmqc/agentic/`; each checklist's skills and contracts remain beside that checklist's existing benchmark and evaluation assets.

```text
mmQC/
├── soda_mmqc/
│   ├── agentic/                                   # mmQC's agentic DAG pipeline.
│   │   ├── __init__.py                            # Public pipeline exports.
│   │   ├── contracts/
│   │   │   ├── __init__.py                        # Contract exports.
│   │   │   ├── skills.py                          # Skill metadata and version models.
│   │   │   ├── artifacts.py                       # Typed artifact requirement/output models.
│   │   │   ├── sessions.py                        # Skill sub-agent request/result models.
│   │   │   ├── permissions.py                     # MCP/script request and permission models.
│   │   │   └── errors.py                          # Typed pipeline errors.
│   │   ├── graph/
│   │   │   ├── __init__.py                        # Graph API exports.
│   │   │   ├── inventory.py                       # Discovers checklist skills and immutable versions.
│   │   │   ├── validation.py                      # Rejects invalid edges, contracts, and cycles.
│   │   │   ├── resolution.py                      # Resolves leaf closure and topological order.
│   │   │   └── generation.py                      # Generates dag.yaml and SKILLS.md.
│   │   ├── versioning/
│   │   │   ├── __init__.py                        # Versioning API exports.
│   │   │   ├── manifests.py                       # Loads and validates checklist pins.
│   │   │   ├── skillsets.py                       # Builds baseline and one-skill sweeps.
│   │   │   └── hashing.py                         # Computes skill and SkillSet hashes.
│   │   ├── runtime/
│   │   │   ├── __init__.py                        # Runtime API exports.
│   │   │   ├── assembly.py                        # Builds one-version-per-skill runtime trees.
│   │   │   ├── orchestrator.py                    # Runs skills topologically for one example.
│   │   │   ├── validation.py                      # Validates produced artifacts.
│   │   │   ├── permissions.py                     # Resolves skill requests against mmQC policy.
│   │   │   └── cleanup.py                         # Removes or retains temporary runtime state.
│   │   ├── providers/
│   │   │   ├── __init__.py                        # Provider adapter exports.
│   │   │   ├── base.py                            # Skill sub-agent provider protocol.
│   │   │   ├── fake.py                            # Deterministic test provider.
│   │   │   └── anthropic.py                       # Claude Agent SDK adapter.
│   │   ├── tools/
│   │   │   ├── __init__.py                        # Approved operation exports.
│   │   │   ├── registry.py                        # Registry of existing MCP/script operations.
│   │   │   ├── mcp.py                             # Connects approved MCP servers.
│   │   │   └── scripts.py                         # Wraps approved pre-generated scripts.
│   │   └── persistence/
│   │       ├── __init__.py                        # Persistence API exports.
│   │       ├── artifacts.py                       # Writes validated skill artifacts.
│   │       ├── cache.py                           # Caches complete example/SkillSet runs.
│   │       └── provenance.py                      # Records versions, hashes, model, and usage.
│   ├── data/
│   │   ├── checklist/
│   │   │   └── fig-checklist/
│   │   │       ├── agentic/                       # Agentic DAG owned by fig-checklist.
│   │   │       │   ├── CLAUDE.md                  # Figure-checklist agent orientation.
│   │   │       │   ├── version-manifest.yaml      # Complete pin for every checklist skill.
│   │   │       │   ├── dag.yaml                   # Generated graph; never hand-edited.
│   │   │       │   ├── SKILLS.md                  # Generated hierarchy and pinned versions.
│   │   │       │   ├── contracts/
│   │   │       │   │   ├── scientific-panels/
│   │   │       │   │   │   └── v1.schema.json    # Contract for identified panels.
│   │   │       │   │   └── scale-bar-report/
│   │   │       │   │       └── v1.schema.json    # Contract for the scale-bar leaf.
│   │   │       │   └── skills-store/
│   │   │       │       ├── identify-panels/
│   │   │       │       │   ├── v1/
│   │   │       │       │   │   └── SKILL.md       # First immutable panel procedure.
│   │   │       │       │   └── v2/
│   │   │       │       │       └── SKILL.md       # Revised panel procedure.
│   │   │       │       └── micrograph-scale-bar/
│   │   │       │           └── v1/
│   │   │       │               └── SKILL.md       # Scale-bar leaf procedure.
│   │   │       └── micrograph-scale-bar/
│   │   │           ├── benchmark.json             # Selects benchmark examples.
│   │   │           └── eval-manifest.json         # Defines FlatEvaluator scoring policy.
│   │   ├── examples/                              # Existing inputs and expected outputs.
│   │   ├── cache/                                 # Existing or migrated run cache.
│   │   ├── predictions/                           # Raw agentic leaf predictions.
│   │   └── evaluation/                            # Existing analysis.json outputs.
│   └── scripts/
│       ├── run.py                                 # Existing evaluate compatibility CLI.
│       ├── score.py                               # Separate prediction-scoring CLI.
│       └── generate_skill_dag.py                  # Regenerates dag.yaml and SKILLS.md.
└── tests/
    ├── agentic/
    │   ├── test_inventory.py                      # Skill/version discovery tests.
    │   ├── test_graph.py                          # DAG and contract validation tests.
    │   ├── test_manifests.py                      # Pin and sweep tests.
    │   ├── test_assembly.py                       # Runtime isolation tests.
    │   ├── test_orchestrator.py                   # Topological sub-agent execution tests.
    │   ├── test_permissions.py                    # MCP/script isolation tests.
    │   ├── test_providers.py                      # Fake and Anthropic adapter tests.
    │   └── test_persistence.py                    # Cache and provenance tests.
    ├── test_agentic_cli.py                        # Existing CLI compatibility tests.
    └── test_score_cli.py                          # Independent scoring tests.
```

Ownership is direct:

```text
evaluate CLI
  -> soda_mmqc.agentic pipeline
  -> checklist-owned agentic DAG
  -> existing mmQC examples and prediction roots
  -> separate FlatEvaluator scoring
```

`fig-checklist/agentic/` is the source of truth for the figure checklist DAG. Existing leaf directories continue to own benchmark and evaluation policy. The agentic pipeline is reusable across mmQC checklists, but it is not an independently distributed library.

Runtime directories should live outside the source tree, for example under a temporary directory or a gitignored cache root:

```text
.runtime/<run-id>/
  skillset.yaml                         # Exact resolved skill versions for this session.
  content-hashes.json                   # Immutable content identities used for provenance and cache keys.
  skills/
    identify-panels/                    # Agent-visible copy/link of exactly one shared-skill version.
    micrograph-scale-bar/               # Agent-visible copy/link of exactly one leaf-skill version.
```

Do not expose `skills-store/` as a parent or sibling reachable from the agent's working directory. Isolation must be tested, not assumed.

### Locked execution model

The mmQC orchestrator, not an agent, controls DAG execution:

1. Resolve and validate the checklist's complete version manifest.
2. Resolve the target leaf's dependency closure and topological order.
3. For each skill in that order, launch a fresh sub-agent scoped to that skill.
4. Give the sub-agent only its `SKILL.md`, validated upstream artifacts, declared output contracts, and the MCP/script operations requested by that skill and approved by the host.
5. Receive structured artifacts as the sub-agent response; the sub-agent does not write them directly.
6. Validate and persist artifacts through the orchestrator-owned artifact store before launching dependent skills.
7. Write and return the final leaf prediction through mmQC's run pipeline.

This is one orchestrated run per scientific input and target leaf, containing multiple isolated sub-agent executions. Sub-agents cannot schedule other skills or launch other agents.

### Locked permission model

Skills request capabilities; they do not grant capabilities. For each skill, the runtime computes:

```text
skill-requested MCP/script operations
  intersect mmQC-approved operations
  = operations exposed to that skill's sub-agent
```

Missing, unknown, or denied required operations fail validation before the sub-agent starts. The pipeline exposes named semantic operations with typed inputs, not a shell or a generic `run_script(path, args)` capability. Pre-generated scripts execute behind mmQC-registered wrappers; any temporary writes belong to those wrappers, not to the sub-agent.

### Locked artifact frontmatter direction

Dependencies and outputs identify both the artifact and its versioned scientific contract:

```yaml
---
name: micrograph-scale-bar
kind: leaf
requires:
  - skill: identify-panels
    artifact: panels
    contract: scientific-panels/v1
produces:
  - artifact: scale-bar-report
    contract: micrograph-scale-bar-report/v1
needs:
  mcps: [figure-data]
  scripts: [inspect-scale-bars]
---
```

Shared contract schemas live under `soda_mmqc/data/checklist/<checklist>/agentic/contracts/`. The checklist validator must resolve every contract identifier there and reject incompatible or missing producer/consumer contracts before execution.

### Locked persistence boundary

The pipeline writes through `soda_mmqc/agentic/persistence/` to configured mmQC cache, prediction, and evaluation roots. Cache identity includes checklist ID, leaf ID, example content hash, SkillSet hash, artifact-contract hashes, provider/model identity, and effective permission/session policy. Sub-agents never write directly; only the orchestrator and persistence layer may mutate run state.

## 3. Cross-phase decision policy

Each phase ends at a human decision gate. At a gate, produce the named evidence, record the decision in this document or an ADR, and wait for approval before proceeding. A gate has three valid outcomes:

- **Go:** accept the proposed contract and continue.
- **Revise:** update the design or phase output, then repeat the gate.
- **Stop:** preserve the current legacy path and defer the migration.

No benchmark score alone authorizes a migration change. Human review must also cover sample traces, leaf JSON quality, cost, latency, and failure behavior.

## Phase 0: Freeze and characterize the current baseline

**Purpose:** Establish behavior that the migration must preserve and separate existing failures from migration regressions.

**Current anchors:**

- [run.py](../soda_mmqc/scripts/run.py) combines discovery, prompt/config resolution, model calls, cache use, evaluation, and analysis persistence.
- [api.py](../soda_mmqc/lib/api.py) selects OpenAI or Anthropic and calls the model once with a prompt and schema.
- [cache.py](../soda_mmqc/lib/cache.py) hashes current model input, check, model, and model configuration.
- [evaluation.py](../soda_mmqc/core/evaluation.py) contains the leaf-oriented `FlatEvaluator` that remains authoritative.

**Steps:**

1. Run the existing unit suite and record passing, skipped, and failing tests without modifying code.
2. Run one `--mock` check and one `--mock` checklist invocation; record generated paths and JSON shapes.
3. When credentials and budget permit, run one small real check through each supported provider and retain redacted metadata, cost, and latency.
4. Capture representative cache keys, legacy Langfuse prompt resolution behavior, and `analysis.json` shape as baseline facts only; Langfuse is not part of the target architecture.
5. Select a fixed pilot benchmark subset containing easy, ambiguous, multi-panel, and inapplicable examples.
6. Store the baseline run identifiers and environment details in the implementation PR or experiment record, not in gold data.

**Validation commands:**

```bash
pytest tests/test_run_analyze_results.py tests/test_model_api.py \
  tests/test_evaluation.py tests/test_eval_manifest.py -v
evaluate fig-checklist --check micrograph-scale-bar --mock --no-cache
```

**Human decision gate 0 - Baseline accepted**

Review: existing failures, pilot example set, current output contract, baseline cost/latency, and whether `micrograph-scale-bar` is still the preferred pilot.

Approve only if the team can state which behaviors are compatibility requirements and which current behaviors may intentionally change. A failure here stops all implementation work because later parity would be unmeasurable.

**Exit artifact:** A short baseline record attached to the implementation issue/PR, with no runtime changes.

## Phase 1: Resolve architecture contracts and open names

**Purpose:** Turn the brief's open choices into explicit contracts before code or data depends on them.

**Files:**

- Modify: this plan and/or add project ADRs in the team's chosen documentation location.
- Do not modify runtime code yet.

**Steps:**

1. Confirm `soda_mmqc/data/checklist/<checklist>/agentic/skills-store/<name>/vN/` as the skill-store layout.
2. Choose immutable version identifiers. Recommendation: author-visible monotonically increasing `vN` directories plus a required content hash in resolved provenance.
3. Confirm `version-manifest.yaml` as the checklist pin filename and require one pin for every skill in the checklist DAG.
4. Define frontmatter semantics precisely:
  - `name` is unique inside a checklist DAG.
   - `kind` is `intermediate` or `leaf`.
  - `requires` contains producer skill, artifact, and versioned contract identifiers.
  - `produces` contains unique artifact and versioned contract identifiers.
  - `needs.mcps` and `needs.scripts` contain named operations, never executable code or arbitrary paths.
5. Choose CLI spelling. Recommendation: `--unpin SKILL --versions v1,v2,v4`; reject repeated `--unpin` and multiple names.
6. Choose assembly strategy. Recommendation: symlink on supported Unix CI/local hosts, with copy fallback; record whether Windows is required for v1.
7. Define the mmQC run, prediction, intermediate-artifact, cache, and provenance formats under configured mmQC data roots.
8. Define the temporary compatibility policy for `--prompt-version`, `--config-from-version`, and `--config-source-check`.
9. Confirm contract identifiers resolve under `soda_mmqc/data/checklist/<checklist>/agentic/contracts/<contract>/vN.schema.json`.
10. Record deferred topics separately: mock semantics, concurrency limits, repair count, runtime retention, MCP transport lifecycle, and OpenAI support.

**Human decision gate 1 - Contracts approved**

Review and sign off on: checklist DAG scope, version ID policy, manifest name, artifact frontmatter, CLI flags, runtime assembly portability, mmQC run-artifact format, and legacy flag behavior.

Do not proceed with unresolved placeholders such as "name TBD." These choices become persisted interfaces and are expensive to reverse after skill conversion.

**Exit artifact:** Approved ADR(s) or a completed decision table in this plan.

## Phase 2: Add agentic contracts, graph, and manifest validation

**Purpose:** Build mmQC-internal contracts that reject malformed checklist skill DAGs before any sub-agent or model call.

**Proposed files:**

- Create: `soda_mmqc/agentic/__init__.py`
- Create: `soda_mmqc/agentic/contracts/skills.py`
- Create: `soda_mmqc/agentic/contracts/artifacts.py`
- Create: `soda_mmqc/agentic/contracts/permissions.py`
- Create: `soda_mmqc/agentic/contracts/errors.py`
- Create: `soda_mmqc/agentic/graph/inventory.py`
- Create: `soda_mmqc/agentic/graph/validation.py`
- Create: `soda_mmqc/agentic/graph/resolution.py`
- Create: `soda_mmqc/agentic/versioning/manifests.py`
- Create: `soda_mmqc/agentic/versioning/skillsets.py`
- Create: `soda_mmqc/agentic/versioning/hashing.py`
- Create: `tests/agentic/test_inventory.py`
- Create: `tests/agentic/test_graph.py`
- Create: `tests/agentic/test_manifests.py`
- Add small checklist-DAG fixtures under `tests/agentic/fixtures/`

**Steps:**

1. Write failing tests for duplicate skill names, missing frontmatter, invalid kinds, duplicate artifacts, unresolved contracts, unknown dependencies, cycles, and leaves without output contracts.
2. Add immutable data models for skill references, stored skill versions, typed artifact requirements/outputs, permission requests, resolved SkillSets, dependency closures, and validation errors.
3. Parse `SKILL.md` frontmatter with the existing YAML dependency; never parse YAML with string splitting.
4. Discover and validate the complete checklist agentic inventory independently from generated `dag.yaml`; require exactly one valid manifest pin for every skill and reject orphan pins.
5. Resolve contract identifiers from the checklist's `agentic/contracts/` directory and reject missing or incompatible producer/consumer contracts.
6. Validate graph edges and topologically sort each target leaf closure deterministically.
7. Write failing tests for missing pins, unknown pins, nonexistent versions, malformed version IDs, invalid unpin targets, empty version selections, and attempts to unpin multiple skills.
8. Resolve the pinned baseline and one-dimensional sweep SkillSets.
9. Compute a canonical SkillSet hash from sorted skill name, version ID, and content hash tuples.
10. Ensure checklist, graph, manifest, artifact-contract, and permission-request validation completes before provider/session objects can be constructed.

**Validation commands:**

```bash
pytest tests/agentic/test_inventory.py tests/agentic/test_graph.py \
  tests/agentic/test_manifests.py -v
```

**Human decision gate 2 - Resolution semantics accepted**

Review malformed-fixture error messages, one pinned resolution, and one unpinned expansion. Confirm that the resolved set is deterministic, complete, contract-compatible, and limited to one changed dimension.

Approve only if a reviewer can identify the exact selected version and content hash of every skill without opening runtime directories.

**Exit criterion:** Invalid inventories and manifests fail fast with actionable errors; valid inputs yield stable SkillSet and closure hashes.

## Phase 3: Generate and verify the DAG documentation

**Purpose:** Make the skill-authored graph inspectable without creating a second source of truth.

**Proposed files:**

- Create: `soda_mmqc/agentic/graph/generation.py`
- Create: `soda_mmqc/scripts/generate_skill_dag.py`
- Extend: `tests/agentic/test_graph.py`
- Generate per checklist: `agentic/dag.yaml` and `agentic/SKILLS.md`

**Steps:**

1. Define a deterministic `dag.yaml` schema containing inventory, edges, produced artifacts, leaf status, versions, and generation format version.
2. Generate `dag.yaml` only from parsed `SKILL.md` metadata.
3. Generate `SKILLS.md` showing shared skills, leaves, dependency closures, and pinned versions; keep `README.md` human-authored.
4. Add `--check` mode that exits nonzero when checked-in generated files differ.
5. Add the check to CI only after generated output is reviewed.
6. Keep the runtime capable of rebuilding the graph from skills; generated files are documentation/cache, never authority.

**Validation commands:**

```bash
pytest tests/agentic/test_graph.py -v
python -m soda_mmqc.scripts.generate_skill_dag fig-checklist --check
```

**Human decision gate 3 - Generated view is useful and non-authoritative**

Review a sample graph and generated `SKILLS.md`. Confirm that they expose enough information for review and debugging but contain no hand-maintained edge data.

Approve the CI drift check only after the generation command is stable and deterministic on a clean checkout.

**Exit criterion:** Regeneration is idempotent and drift is machine-detectable.

## Phase 4: Assemble isolated runtime views

**Purpose:** Guarantee the agent sees one coherent version per skill and never sees the version store.

**Proposed files:**

- Create: `soda_mmqc/agentic/runtime/assembly.py`
- Create: `soda_mmqc/agentic/runtime/cleanup.py`
- Create: `tests/agentic/test_assembly.py`

**Steps:**

1. Write failing tests proving one version per skill, deterministic contents, and cleanup after success or failure.
2. Assemble only the target leaf closure, checklist-level orientation, resolved artifact contracts, and the selected skill's approved operation descriptors.
3. Materialize symlinks or copies according to the approved portability policy.
4. Write `skillset.yaml` and `content-hashes.json` into the runtime root for provenance.
5. Reject links that resolve outside approved version directories.
6. Set the agent working directory so `skills-store/`, repository credentials, unrelated examples, and prior runtime trees are not reachable through normal traversal.
7. Add an optional retain-on-failure/debug mode with an explicit retention policy; default to cleanup.
8. Test concurrent assemblies to ensure run IDs and cleanup cannot collide.

**Validation commands:**

```bash
pytest tests/agentic/test_assembly.py -v
```

**Human decision gate 4 - Isolation accepted**

Review an assembled tree, a traversal/security test, Windows behavior if required, cleanup behavior, and disk usage under copy fallback.

Do not connect an agent until a reviewer confirms the store is absent from the agent-visible root and each name resolves to exactly one approved version.

**Exit criterion:** Runtime assembly is deterministic, isolated, concurrent-safe, and self-describing.

## Phase 5: Define artifacts, predictions, provenance, and cache identity

**Purpose:** Separate raw generation from scoring and make every output reproducible.

**Proposed files:**

- Create: `soda_mmqc/agentic/persistence/artifacts.py`
- Create: `soda_mmqc/agentic/persistence/cache.py`
- Create: `soda_mmqc/agentic/persistence/provenance.py`
- Create: `tests/agentic/test_persistence.py`

**Recommended output shape:**

```text
soda_mmqc/data/predictions/<checklist>/<leaf>/<run-id>/
  run.json                              # Checklist, leaf, example, provider, policy, and format identities.
  skillset.yaml                         # Exact skill pins and content hashes.
  input.json                            # Example descriptor and content hash.
  artifacts/<skill>/<artifact>.json     # Orchestrator-written, contract-validated skill artifacts.
  metadata/<skill>.json                 # Usage, local trace IDs, and validation attempts per sub-agent.
  result.json                           # Final leaf prediction consumed by scoring.
```

**Steps:**

1. Define the versioned mmQC run manifest, skill-artifact records, final prediction, and provenance schemas.
2. Permit only the orchestrator and persistence implementation to write run artifacts; sub-agents return values and receive no storage-write capability.
3. Make writes atomic so interrupted/concurrent runs cannot leave valid-looking partial JSON.
4. Map example identifiers to paths deterministically while retaining the original identifier and content hash in metadata.
5. Include checklist ID, leaf ID, example content hash, SkillSet hash, artifact-contract hashes, model ID, effective permission/session policy hash, and runner format version in the whole-run cache key.
6. Treat the resolved SkillSet atomically; do not reuse a leaf result when any upstream skill content changes.
7. Persist intermediate artifacts as orchestrator-owned sidecars when requested, but never fold them into the final leaf prediction consumed by scoring.
8. Define resume behavior from completed skill artifacts without introducing cross-run intermediate caching in v1.
9. Add format-version checks so future readers fail clearly on incompatible artifacts.

**Validation commands:**

```bash
pytest tests/agentic/test_persistence.py -v
```

**Human decision gate 5 - Reproducibility contract approved**

Review one artifact tree, cache-key inputs, retention needs, redaction rules, and expected disk growth. Confirm whether intermediate sidecars are default-off or default-on for pilot runs.

Approve only if a prediction can be traced to exact skill contents, artifact contracts, provider/model identity, and effective session policy.

**Exit criterion:** Predictions are independently scoreable and cache collisions across SkillSets are covered by tests.

## Phase 6: Add session policy and approved operation registry

**Purpose:** Resolve each skill's requested MCP/script operations against host policy and expose only existing, approved operations to its sub-agent.

**Proposed files:**

- Create: `soda_mmqc/agentic/runtime/permissions.py`
- Create: `soda_mmqc/agentic/tools/registry.py`
- Create: `soda_mmqc/agentic/tools/mcp.py`
- Create: `soda_mmqc/agentic/tools/scripts.py`
- Create: `tests/agentic/test_permissions.py`

**Steps:**

1. Inventory existing MCP and pre-generated script operations needed by the pilot, beginning with permitted figure/caption access and scale-bar inspection.
2. Represent each operation as a named semantic capability with typed inputs/outputs; never expose shell, generic Python, arbitrary code execution, or `run_script(path, args)`.
3. Let each skill request operation names through `needs.mcps` and `needs.scripts`; mmQC supplies the approved registry and deny policy.
4. Resolve permissions separately for each skill as `requested ∩ mmQC-approved`; do not expose all operations needed by the full closure to every sub-agent.
5. Fail before sub-agent startup when a required operation is missing, unknown, or denied.
6. Enforce path confinement, input/output validation, payload limits, timeouts, and read-only access at the wrapper or MCP boundary.
7. Keep script source immutable to the sub-agent. Any temporary writes are performed inside the approved wrapper's sandbox and returned as structured results or host-controlled references.
8. Prohibit sub-agent shell access, filesystem writes, package installation, tool registration, agent launching, and direct artifact-store writes.
9. Add tests for path traversal, oversized payloads, invalid arguments, undeclared operations, host denial, capability escalation, and attempts to access generic execution.

**Validation commands:**

```bash
pytest tests/agentic/test_permissions.py -v
```

**Human decision gate 6 - Capabilities and trust boundary approved**

Review every pilot MCP/script operation, its filesystem/network reach, timeout/payload limits, data sensitivity, and provider mapping. Arbitrary code execution and sub-agent-authored scripts are prohibited in v1.

No agent integration may proceed until the allowed tool surface is approved. Default recommendation: example-read tools only for the first pilot.

**Exit criterion:** Each skill-scoped sub-agent receives only that skill's minimum mmQC-approved operations and has no generic execution or write capability.

## Phase 7: Add the skill-scoped sub-agent protocol and Anthropic adapter

**Purpose:** Execute a leaf closure deterministically with one isolated, observable sub-agent per skill without coupling orchestration to one SDK.

**Proposed files:**

- Create: `soda_mmqc/agentic/contracts/sessions.py`
- Create: `soda_mmqc/agentic/runtime/orchestrator.py`
- Create: `soda_mmqc/agentic/runtime/validation.py`
- Create: `soda_mmqc/agentic/providers/base.py`
- Create: `soda_mmqc/agentic/providers/fake.py`
- Create: `soda_mmqc/agentic/providers/anthropic.py`
- Create: `tests/agentic/test_orchestrator.py`
- Create: `tests/agentic/test_providers.py`
- Modify: root `pyproject.toml` and `requirements.txt` only after SDK selection is approved

**Steps:**

1. Verify the current official Anthropic Claude Agent SDK Python package, supported Python version, license, structured-output behavior, tool API, cancellation, and tracing hooks.
2. Pin a tested SDK range; do not assume the existing `anthropic` Messages API package is the Agent SDK.
3. Define a provider-neutral sub-agent request containing the selected skill, its assembled files, validated upstream artifacts, output contract schemas, approved MCP/script operations, model/session settings, and local trace metadata.
4. Define a sub-agent result containing structured artifact values, usage, local trace identifiers, validation history, and terminal status; it contains no direct storage mutations.
5. Implement a fake provider first and test topological orchestration, fresh isolation per skill, retries, timeout/cancellation, malformed outputs, denied operations, tool failures, and cleanup.
6. Implement the Anthropic adapter behind the same protocol.
7. Make the orchestrator launch one fresh sub-agent per skill in topological order; dependent skills start only after all required artifacts validate and persist successfully.
8. Validate every produced artifact against its resolved contract before persistence or downstream handoff; require the final leaf artifact to conform to its scientific output contract.
9. Define a bounded repair process for invalid structured output and retain all validation errors and attempts in metadata; the exact attempt limit remains a Phase 1 deferred decision.
10. Record local per-skill spans and one whole-run trace without introducing a Langfuse dependency.
11. Never fall back silently to the legacy one-prompt call when an agentic run fails.

**Validation commands:**

```bash
pytest tests/agentic/test_orchestrator.py tests/agentic/test_providers.py -v
pytest -m integration tests/agentic/test_providers.py -v
```

**Human decision gate 7 - SDK and session behavior accepted**

Review the dependency/security assessment, fake-provider failure matrix, one redacted real trace, structured-output reliability, token cost, latency, and retry policy.

Approve only if the adapter can be replaced without changing orchestrator or skill contracts and if terminal failures are explicit and resumable.

**Exit criterion:** One scientific input can run through fake and real Anthropic skill-scoped sub-agents in deterministic DAG order, producing contract-valid leaf JSON and complete provenance.

## Phase 8: Convert the first `fig-checklist` pilot

**Purpose:** Prove reuse and chaining with the smallest useful DAG before migrating more checks.

**Proposed data changes:**

- Create: `soda_mmqc/data/checklist/fig-checklist/agentic/CLAUDE.md`
- Create: `soda_mmqc/data/checklist/fig-checklist/agentic/version-manifest.yaml`
- Create: `soda_mmqc/data/checklist/fig-checklist/agentic/contracts/scientific-panels/v1.schema.json`
- Create: `soda_mmqc/data/checklist/fig-checklist/agentic/contracts/scale-bar-report/v1.schema.json`
- Create: `soda_mmqc/data/checklist/fig-checklist/agentic/skills-store/identify-panels/v1/SKILL.md`
- Create: `soda_mmqc/data/checklist/fig-checklist/agentic/skills-store/micrograph-scale-bar/v1/SKILL.md`
- Copy/adapt the existing leaf `schema.json` as the scientific basis for `micrograph-scale-bar-report/v1`.
- Keep the existing mmQC `benchmark.json` and `eval-manifest.json` under `soda_mmqc/data/checklist/fig-checklist/micrograph-scale-bar/`
- Generate: `soda_mmqc/data/checklist/fig-checklist/agentic/dag.yaml` and `SKILLS.md`
- Create: `tests/agentic/test_fig_checklist_pilot.py`

**Steps:**

1. Extract panel identification instructions from current figure prompts into `identify-panels`; define a stable `panels` artifact schema.
2. Keep ambiguous panel-kind policy out of the shared skill unless required to label the physical panels consistently.
3. Rewrite `micrograph-scale-bar` as a leaf that requires `identify-panels`, applies its own micrograph gate, performs scale-bar analysis, and conforms to the unchanged leaf schema.
4. Do not paste the old full JSON example or panel-identification section into the leaf skill.
5. Pin both skills in the complete checklist version manifest. Keep the pilot DAG inventory limited to real converted skills rather than adding placeholders for unconverted checks.
6. Run the graph generator and inspect the closure.
7. Run mock/fake-provider tests first, then the fixed real pilot benchmark.
8. Compare leaf JSON and `FlatEvaluator` results against the Phase 0 baseline; inspect intermediate panel artifacts manually.
9. Create at least one alternate `identify-panels` or leaf version and prove a one-dimensional sweep changes only the requested pin.

**Validation commands:**

```bash
pytest tests/agentic/test_fig_checklist_pilot.py -v
evaluate fig-checklist --check micrograph-scale-bar --mock --no-cache
evaluate fig-checklist --check micrograph-scale-bar \
  --unpin micrograph-scale-bar --versions v1,v2 --no-cache
```

The last two commands are target commands; use a temporary feature/backend flag until Phase 9 wires the compatible default surface.

**Human decision gate 8 - Pilot skill quality accepted**

Review side-by-side: old prompt, shared skill, leaf skill, panel artifacts, final leaf predictions, scores, failures, cost, latency, and traces. Have a domain expert inspect cases where panel labels or applicability changed even when aggregate score did not.

Decide whether to remain at two levels (`identify-panels` to leaf) or introduce `classify-panel-kind`. Recommendation: add the middle skill only after a second converted image leaf demonstrates duplicated classification logic.

Approve only if shared panel identification improves consistency without forcing changes to leaf gold or scoring semantics.

**Exit criterion:** The two-skill pilot is reproducible, scoreable by the existing evaluator, and acceptable to a scientific/domain reviewer.

## Phase 9: Integrate agentic execution with the existing run CLI

**Purpose:** Replace the internal execution path while preserving the command users employ to run checks.

**Proposed files:**

- Refactor: `soda_mmqc/scripts/run.py`
- Extend: `tests/test_run_analyze_results.py`
- Create: `tests/test_agentic_cli.py`

**Steps:**

1. Extract current discovery/example-loading behavior behind functions that can serve either backend without changing selected examples.
2. Make the orchestrator resolve and validate the checklist DAG, artifact contracts, permissions, and all requested SkillSets before running the first example.
3. For each SkillSet, leaf, and example, check the whole-run cache, assemble a runtime, execute skill-scoped sub-agents in topological order, and persist artifacts/provenance atomically through mmQC persistence.
4. Preserve deterministic ordering and `--mock` behavior.
5. Add `--unpin` and `--versions` using the Phase 1 contract.
6. Reject agentic-incompatible legacy flags clearly; during transition, allow them only on the explicitly selected legacy backend.
7. Add a temporary backend selector for rollout, for example `--runner legacy|agentic`, but do not make it permanent unless users need both.
8. Keep scoring callable for compatibility during this phase, but route it through the same persisted predictions that the separate CLI will consume.
9. Test interruption, resume, one failed example among many, cache hits, cache bypass, and multiple checks.

**Validation commands:**

```bash
pytest tests/test_agentic_cli.py tests/test_run_analyze_results.py -v
evaluate fig-checklist --check micrograph-scale-bar --mock --no-cache
```

**Human decision gate 9 - CLI compatibility accepted**

Review `--help` before/after, exit codes, selected example IDs, prediction paths, mock output, failure summaries, and a legacy-versus-agentic pilot comparison.

Approve CLI compatibility only after scripted callers and documentation have been checked. Choosing a deployment default or rollout window is outside this plan.

**Exit criterion:** Existing run commands still select the same checklist/check/examples while internally producing independently scoreable agentic predictions.

## Phase 10: Split scoring into an independent CLI

**Purpose:** Remove model/provider concerns from evaluation and allow repeated scoring without repeated generation.

**Proposed files:**

- Create: `soda_mmqc/scripts/score.py`
- Modify: `pyproject.toml` to register the approved command name
- Reuse: `soda_mmqc/core/evaluation.py`
- Reuse: `soda_mmqc/core/eval_manifest.py`
- Create: `tests/test_score_cli.py`
- Update: `soda_mmqc/docs/benchmarking.md`

**Steps:**

1. Define score CLI inputs: prediction run/directory, optional example filter, and output directory.
2. Load leaf schema, eval manifest, benchmark/gold, and prediction records without importing provider SDKs.
3. Validate run format, leaf identity, schema hash, missing/extra examples, and malformed predictions before scoring.
4. Call the existing `FlatEvaluator` unchanged for each leaf prediction.
5. Preserve the current `analysis.json` structure unless a versioned extension is approved.
6. Add clear policies for partial runs and failed generation records.
7. During transition, let `evaluate` optionally invoke the scorer after generation; then remove implicit scoring at the approved compatibility boundary.
8. Verify existing reporting and notebook consumers against the independently produced analysis.

**Validation commands:**

```bash
pytest tests/test_score_cli.py tests/test_evaluation.py \
  tests/test_eval_manifest.py tests/test_reporting_compare.py -v
```

**Human decision gate 10 - Execution/scoring boundary accepted**

Review command naming (`score` versus `evaluate-outputs`), partial-run policy, output compatibility, and downstream reporting/notebook behavior.

Approve removal of implicit scoring only after users and CI jobs have migrated. Leaf gold and `FlatEvaluator` semantics are not part of this gate and must remain unchanged.

**Exit criterion:** A stored prediction run can be scored repeatedly on a machine with no model credentials and without an agent SDK session.

## Phase 11: Convert a second leaf, decide on the middle skill, then scale out

**Purpose:** Prove reuse is real before converting the rest of the checklist.

**Recommended second leaf:** `image-annotation-defined` or `single-channel-for-overlay` because it reuses panel identification and tests whether panel-kind classification is genuinely shared.

**Steps:**

1. Convert one second image/micrograph leaf using the same extraction recipe as the pilot.
2. Compare duplicated applicability/type-gate text across the two converted leaves.
3. If duplication is substantial and semantics align, design `classify-panel-kind` with a coarse taxonomy and leaf-level override rules.
4. If semantics do not align, keep two levels and preserve leaf-local applicability.
5. Run both leaves against a shared set of figures and compare panel lists/intermediate artifacts.
6. Convert remaining image leaves, then plot leaves, then cross-cutting leaves in small batches.
7. At each batch, preserve existing leaf schemas, manifests, benchmark selection, and expected outputs unless a separately reviewed curation change is required.
8. Repeat generation, unit, mock, benchmark, domain review, cost, and latency checks for every batch.
9. Convert other checklists only after `fig-checklist` operation and authoring are stable.

**Human decision gate 11 - Reuse model accepted for scale-out**

Decide whether `classify-panel-kind` exists, approve its taxonomy if added, and confirm that shared skills reduce disagreement rather than hide leaf-specific judgment.

Approve each migration batch separately. A regression in panel identity, applicability, cost ceiling, or operational reliability pauses further conversion without forcing rollback of accepted batches.

**Exit criterion:** At least two leaves share a reviewed upstream artifact, and the team has an evidence-based pattern for the rest of the checklist.

## Phase 12: Retire the legacy prompt path

**Purpose:** Remove dual behavior only after all consumers and accepted checklists use agentic skills.

**Likely files:**

- Simplify: `soda_mmqc/scripts/run.py`
- Simplify or retain as low-level compatibility: `soda_mmqc/lib/api.py`
- Remove obsolete prompt-loading branches and legacy-only CLI flags after deprecation
- Remove migrated `prompts/prompt.N.txt` files only after legacy compatibility and archival requirements are met
- Update: root `README.md`, checklist docs, provider docs, and CI

**Steps:**

1. Inventory application, CI, notebook, and external users of prompt files and legacy flags.
2. Announce and document deprecation with an exact removal release/date.
3. Verify every active migrated leaf has complete checklist pins and a successful compatibility benchmark.
4. Remove legacy execution branches and tests that assert obsolete behavior; retain historical prediction artifacts.
5. Remove obsolete dependencies only after import and packaging checks.
6. Run the full unit suite, integration suite, all-checklist mock run, and selected provider-backed smoke tests.
7. Tag/archive the last legacy-compatible release or commit for reference.

**Validation commands:**

```bash
pytest -v
evaluate fig-checklist --mock --no-cache
score <prediction-run>
```

**Human decision gate 12 - Legacy removal authorized**

Review consumer inventory, deprecation completion, full test results, provider-backed traces, benchmark comparisons, fallback documentation, and the last known legacy release.

This gate requires explicit project-owner approval. If approval is withheld, keep the legacy backend isolated and documented rather than partially deleting it.

**Exit criterion:** One supported execution model remains, with no active consumer depending on monolithic prompt files or prompt-version CLI semantics.

## 4. Required review matrix

Use this matrix at every pilot or migration-batch gate.

| Area | Evidence | Human owner | Reject when |
|------|----------|-------------|-------------|
| Scientific behavior | Predictions, intermediates, disagreement cases | Domain reviewer | Panel/applicability meaning drifts without justification |
| Evaluation | Leaf schema/hash, gold, `FlatEvaluator` output | Evaluation owner | Gold or scoring changes implicitly |
| Agent behavior | Trace, tools used, validation/repair attempts | Agent/runtime engineer | Agent escapes closure or failures are silent |
| Reproducibility | SkillSet manifest, content hashes, policy hash | Reviewer | Exact inputs cannot be reconstructed |
| Operations | Cost, latency, retries, cache hit rate | Project owner | Budget/SLO ceiling exceeded |
| Security/privacy | Tool allowlist, paths, payloads, logs | Security/data owner | Store/secrets/unapproved data are reachable |
| Rollback | Prior manifest/backend and tested procedure | Release owner | Rollback depends on editing immutable versions |

## 5. Test strategy by layer

**Pure unit tests:** frontmatter parsing, graph validation, topological ordering, manifest validation, unpin expansion, canonical hashes, session-policy resolution, path confinement, schema validation, and artifact formats.

**Contract tests:** fake provider against the provider-neutral session protocol; tool schemas against implementations; generated DAG against skill metadata; score CLI against persisted prediction fixtures.

**Integration tests:** real Anthropic sub-agents on a tiny synthetic checklist DAG, concurrent assemblies, interruption/resume, mmQC-approved MCP/script wrappers, and cache behavior.

**Behavior benchmarks:** fixed pilot examples scored with the existing leaf schema, eval manifest, expected output, and `FlatEvaluator`. Compare quality distribution, not only averages; inspect regressions by example and property.

**Operational tests:** cost/token budget, latency percentiles, retry rate, invalid-output rate, trace completeness, disk growth, and cleanup.

**Full regression:** existing tests in `tests/`, all-checklist `--mock`, report generation, curation reads of leaf schemas, and notebooks/scripts that consume `analysis.json`.

Real provider tests must be marked and skipped cleanly without credentials. Unit and mock tests must remain credential-free.

## 6. Suggested implementation sequence and commits

Keep each commit independently testable and avoid mixing skill-content rewrites with runtime infrastructure.

1. `test: characterize checklist execution baseline`
2. `feat: add agentic contracts and graph validation`
3. `feat: add checklist skillset resolution`
4. `feat: generate checklist DAG documentation`
5. `feat: assemble isolated skill runtime views`
6. `feat: persist agentic run provenance`
7. `feat: enforce sub-agent operation policy`
8. `feat: add provider-neutral skill sub-agents`
9. `feat: add Anthropic sub-agent adapter`
10. `feat: add fig checklist agentic pilot`
11. `feat: run agentic checks from evaluate CLI`
12. `feat: add independent score CLI`
13. `docs: document checklist skill authoring and operations`

Do not commit generated runtime directories, caches, credentials, raw sensitive traces, or unredacted provider payloads.

## 7. Risks and planned controls

| Risk | Control | Gate |
|------|---------|------|
| Agent sees multiple skill versions | Isolated closure-only assembly and traversal tests | 4 |
| Manifest/store drift | Complete pre-run validation and generated-doc CI check | 2, 3 |
| Cache reuses stale upstream behavior | Atomic SkillSet/content hash in cache key | 5 |
| Agent output breaks current scoring | Leaf-schema validation; leaf-only prediction artifact | 7, 8 |
| Shared panel skill changes all leaves | Fixed pilot benchmark, intermediate review, pinned versions | 8, 12 |
| SDK locks orchestration to Anthropic | Provider-neutral request/result protocol and fake provider | 7 |
| Sub-agents gain code or write access | Per-skill host policy, no shell, semantic wrappers, orchestrator-owned stores | 6, 7 |
| CLI split breaks users | Transitional backend/scoring compatibility and help snapshots | 9, 10 |
| Artifact contracts drift across skills | Versioned contract IDs and pre-run compatibility validation | 2, 8 |
| Dual path persists indefinitely | Dated deprecation and owner-controlled removal gate | 12 |

## 8. First implementation milestone

The first milestone should stop after Phase 8. Its deliverable is a reviewable experiment, not an mmQC deployment decision, with:

- complete `soda_mmqc/agentic/` pipeline modules inside the existing mmQC package;
- validated `fig-checklist/agentic/` skill inventory, artifact contracts, and complete pins;
- generated `dag.yaml`/`SKILLS.md`;
- isolated runtime assembly;
- orchestrator-controlled fake and Anthropic skill-scoped sub-agents;
- minimum mmQC-approved MCP and pre-generated script operations;
- no sub-agent shell, script-writing, arbitrary-code, package-install, or artifact-write capability;
- `identify-panels@v1` and `micrograph-scale-bar@v1`;
- leaf-only predictions with full provenance;
- unchanged leaf scoring against the fixed pilot benchmark;
- human-reviewed quality, cost, latency, and traces;
- a recorded go/revise/stop decision for CLI integration.

This milestone gives the team a useful stopping point. If agentic execution does not justify its cost or complexity, the experiment can be removed without altering the current `evaluate` path, curated gold, or `FlatEvaluator`.