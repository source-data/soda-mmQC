# Production prompts

Snapshot of fig-checklist assets selected for production use.

Each subdirectory is one check. Layout:

```text
production_prompts/
├── <check-name>/
│   ├── prompt.<N>.txt          # production prompt (version N), if selected
│   ├── schema.json             # structured-output schema
│   ├── eval-manifest.json      # evaluation field config
│   └── model_config.json       # model/API options (default if check has none)
├── …
└── README.md
```

## Checks

| Check | Prompt version | schema | eval-manifest | model_config |
|-------|----------------|:------:|:-------------:|:------------:|
| `error-bars-defined` | `prompt.2.txt` | ✓ | ✓ | default |
| `image-annotation-defined` | `prompt.2.txt` | ✓ | ✓ | default |
| `individual-data-points` | `prompt.2.txt` | ✓ | ✓ | default |
| `micrograph-scale-bar` | `prompt.1.txt` | ✓ | ✓ | default |
| `panel-image-matches-caption` | `prompt.1.txt` | ✓ | ✓ | default |
| `plot-axis-units` | `prompt.1.txt` | ✓ | ✓ | default |
| `plot-gap-labeling` | — (none selected) | ✓ | ✓ | default |
| `replication-reporting` | `prompt.3.txt` | ✓ | ✓ | default |
| `single-channel-for-overlay` | `prompt.2.txt` | ✓ | ✓ | default |
| `stat-significance-level` | — (none selected) | ✓ | ✓ | default |
| `stat-test` | `prompt.3.txt` | ✓ | ✓ | default |

## Notes

- Prompt versions are copies from `soda_mmqc/data/checklist/fig-checklist/<check>/prompts/`.
- `schema.json` and `eval-manifest.json` are copies from the same check folder.
- `model_config.json` is the shared default from `soda_mmqc/data/model_config.json` (no check-specific configs were present).
- Folders without a prompt are placeholders (schema / eval-manifest / model_config only).
