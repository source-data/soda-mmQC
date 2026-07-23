# Production prompts

Index of fig-checklist assets selected for production use.

`production.json` stores **prompt version numbers only**. Full paths are derived
from `path_templates` (JSON cannot expand placeholders itself; consumers fill
`{check}` and `{version}`).

## Layout

```text
production_prompts/
├── production.json
└── README.md
```

## Editing winners

Change only `prompt_version` in the `checks` list:

```json
{ "check": "error-bars-defined", "prompt_version": 2 }
```

Use `"prompt_version": null` when no production prompt is selected yet.

## Path resolution

| Key | Template |
|-----|----------|
| `prompt` | `soda_mmqc/data/checklist/fig-checklist/{check}/prompts/prompt.{version}.txt` |
| `schema` | `soda_mmqc/data/checklist/fig-checklist/{check}/schema.json` |
| `eval_manifest` | `soda_mmqc/data/checklist/fig-checklist/{check}/eval-manifest.json` |
| `model_config` | `soda_mmqc/data/model_config.json` (shared default; no `{check}`) |

Example: `error-bars-defined` + version `2` →
`soda_mmqc/data/checklist/fig-checklist/error-bars-defined/prompts/prompt.2.txt`

## Current selections

| Check | Version |
|-------|--------:|
| `error-bars-defined` | 2 |
| `image-annotation-defined` | 2 |
| `individual-data-points` | 2 |
| `micrograph-scale-bar` | 1 |
| `panel-image-matches-caption` | 1 |
| `plot-axis-units` | 1 |
| `plot-gap-labeling` | 1 |
| `replication-reporting` | 3 |
| `single-channel-for-overlay` | 2 |
| `stat-significance-level` | 2 |
| `stat-test` | 3 |
