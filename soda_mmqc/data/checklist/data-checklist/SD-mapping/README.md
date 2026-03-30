# SD-mapping

This check maps the **source data files** provided for a figure to the **panels** of that figure. The model receives the figure, caption, and all source data images, and returns for each panel which source data file(s) correspond to it.

## Providing source data

Same convention as western-blot-matching-SD: place source data image files under `content/source_data/`, either in **panel subfolders** (e.g. `source_data/A/`, `source_data/B/`) or **flat** in `source_data/`. See the [western-blot-matching-SD README](../western-blot-matching-SD/README.md) for the exact layout.

Supported image formats: `.png`, `.jpg`, `.jpeg`, `.tiff`, `.tif`, '.xlsx', '.csv'.
If `source_data/` is missing or empty, the check still runs; the model will return empty `source_data_filenames` for panels.

## No agentic behaviour

Source data is included in the same request as the figure (no tools or multi-turn). The pipeline sends figure + caption + source data images with their filenames; the model responds with the mapping in one structured output.
