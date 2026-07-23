# Production prompts

Index of checklist assets selected for production use (across fig-, doc-, and
other checklists).

`production.json` stores **checklist + check + prompt version**. Full paths are
derived from `path_templates` (consumers fill `{checklist}`, `{check}`, and
`{version}`).

## Layout

```text
production_prompts/
├── production.json
└── README.md
```

## Editing winners

Change only `prompt_version` (and `checklist` / `check` when adding entries):

```json
{ "checklist": "fig-checklist", "check": "error-bars-defined", "prompt_version": 2 }
{ "checklist": "doc-checklist", "check": "author-contribution-in-ms", "prompt_version": 1 }
```

Use `"prompt_version": null` when no production prompt is selected yet.

## Path resolution

| Key | Template |
|-----|----------|
| `prompt` | `soda_mmqc/data/checklist/{checklist}/{check}/prompts/prompt.{version}.txt` |
| `schema` | `soda_mmqc/data/checklist/{checklist}/{check}/schema.json` |
| `eval_manifest` | `soda_mmqc/data/checklist/{checklist}/{check}/eval-manifest.json` |
| `model_config` | `soda_mmqc/data/model_config.json` (shared default) |

Example: `fig-checklist` / `error-bars-defined` / version `2` →
`soda_mmqc/data/checklist/fig-checklist/error-bars-defined/prompts/prompt.2.txt`

Not every check has every file (e.g. some lack `eval-manifest.json`); consumers
should treat missing optional assets accordingly.

## Current selections

| Checklist | Check | Version |
|-----------|-------|--------:|
| `fig-checklist` | `error-bars-defined` | 2 |
| `fig-checklist` | `image-annotation-defined` | 2 |
| `fig-checklist` | `individual-data-points` | 2 |
| `fig-checklist` | `micrograph-scale-bar` | 1 |
| `fig-checklist` | `panel-image-matches-caption` | 1 |
| `fig-checklist` | `plot-axis-units` | 1 |
| `fig-checklist` | `plot-gap-labeling` | 1 |
| `fig-checklist` | `replication-reporting` | 3 |
| `fig-checklist` | `single-channel-for-overlay` | 2 |
| `fig-checklist` | `stat-significance-level` | 2 |
| `fig-checklist` | `stat-test` | 3 |
| `doc-checklist` | `AB-target-reagent-consistency` | 1 |
| `doc-checklist` | `DAS-present-and-correct` | 2 |
| `doc-checklist` | `author-contribution-in-ms` | 1 |
| `doc-checklist` | `biorender-protocol.io-mentions` | 1 |
| `doc-checklist` | `data-not-shown` | 2 |
| `doc-checklist` | `external-data-url-validation` | 1 |
| `doc-checklist` | `external-data-url-validation-agentic` | — |
| `doc-checklist` | `no-overclaim-in-abstract` | 1 |
| `doc-checklist` | `reference-format-alphabetical` | 2 |
| `doc-checklist` | `section-order` | 2 |
| `doc-checklist` | `section-order-alt` | 1 |

Doc-checklist multi-prompt checks currently use the latest available version
(except `author-contribution-in-ms`, kept at 1). `external-data-url-validation-agentic`
is `null` because its file is named `prompt1.txt`, which does not match the
`prompt.{version}.txt` template.
