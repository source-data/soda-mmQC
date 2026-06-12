# Thinking catalog

Skim this file first when answering from the wiki; then open linked pages. **Update on every ingest** (new page, major edit, or merge of external notes) so links and blurbs stay current.

Optional columns: **date** (last meaningful update), **sources** (rough count or “—”).

| Page | Blurb | date | sources |
|------|--------|------|---------|
| [README.md](README.md) | Markdown-only wiki: usage, maintenance, conventions. | 2026-04-30 | — |
| [evaluation-json-vs-open-source.md](evaluation-json-vs-open-source.md) | Custom `JSONEvaluator` vs Inspect.ai / OSS; Hungarian lists; no full framework replacement. | 2026-04-30 | ext |
| [evaluation-scoring.md](evaluation-scoring.md) | Normative flat leaf scoring: instances, `by_property`, `by_list` (layer S), layers 1–2, extended primitives, object-list alignment. | 2026-06-08 | — |
| [evaluation-implementation-plan.md](evaluation-implementation-plan.md) | Progressive build plan: `leaves` → manifest → layers 1–2 → `matching` / object pairing / layer S → evaluator. | 2026-06-08 | — |
| [evaluation-toy-examples.md](evaluation-toy-examples.md) | Worked gold/pred toys: leaf instances, `by_property`, `by_list` (layer S on `panels`). | 2026-06-08 | — |
| [evaluation-hierarchical-scoring-DEPRECATED.md](evaluation-hierarchical-scoring-DEPRECATED.md) | **DEPRECATED:** recursive roll-up/drill-down; use [evaluation-scoring.md](evaluation-scoring.md). | 2026-05-21 | — |
| [evaluation-leaf-primitives.md](evaluation-leaf-primitives.md) | Primitives, extended primitives (`string[]`), and **`score`**; layer S is out of scope here. | 2026-06-08 | — |
