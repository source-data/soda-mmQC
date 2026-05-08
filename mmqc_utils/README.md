# mmqc-utils

Reusable conversion utilities for MMQC projects.

## Included utilities

- Convert and downscale common image formats (TIFF, JPEG, PNG, GIF, WebP, PDF) to bounded JPEG previews, with optional byte-size budget enforcement
- Convert DOCX, RTF, ODT, TeX, and PDF documents to HTML
- Normalize HTML to plain text

## System requirements

- **ImageMagick** — required for image conversion (`convert_to_bounded_jpeg`). Install via your system package manager:
  ```bash
  # macOS
  brew install imagemagick

  # Debian / Ubuntu
  apt-get install imagemagick
  ```
- **Pandoc** — bundled automatically via `pypandoc-binary`; no separate installation needed.

## Installation

```bash
pip install mmqc-utils
# or
uv add mmqc-utils
```

## Usage

All functions accept a file path (`str` or `Path`), raw `bytes`/`bytearray`, or a `BinaryIO` object.

### Document conversion

```python
from mmqc_utils import document_to_html

# From a file path
html = document_to_html("paper.docx")

# From bytes — input_format is required when there is no file extension to infer from
html = document_to_html(raw_bytes, input_format="rtf")
```

Supported formats: `docx`, `rtf`, `odt`, `tex`, `pdf`.

For PDFs, pandoc is tried first; if it cannot convert the file, text is extracted page-by-page via `pypdf` and wrapped in `<div class='page'>` elements.

### Image conversion

```python
from mmqc_utils import convert_to_bounded_jpeg, compress_to_bounded_jpeg

# Downscale to pixel dimensions
jpeg_bytes = convert_to_bounded_jpeg(
    "figure.tiff",
    rasterization_dpi=150,   # DPI for vector/PDF rasterization
    max_dimension=2000,       # downscale if width or height exceeds this
    compression_quality=80,   # JPEG quality 1–100
    background="white",       # background when removing transparency
)

# Compress until the result fits within a byte-size budget
jpeg_bytes = compress_to_bounded_jpeg(
    "figure.tiff",
    max_bytes=5 * 1024 * 1024,  # 5 MB
    max_dimension=2000,
    compression_quality=80,
)
```

Both functions accept `Path`, `str`, `bytes`, `bytearray`, or `BinaryIO` as input and only render the first page/layer of multi-page TIFFs.

`compress_to_bounded_jpeg` steps down JPEG quality first (`80 → 70 → … → 30`), then halves `max_dimension` and repeats, until the result fits within `max_bytes`. If the budget cannot be met even at minimum quality and dimension, it returns the smallest result achieved.

### Text normalization

```python
from mmqc_utils import html_to_text, compute_plain_text

text = html_to_text("<p>Hello <b>world</b></p>")
# → "Hello world"
```

`compute_plain_text` is an alias for `html_to_text`. Block-level tags (`<p>`, `<div>`, `<br>`, headings, list items, …) are replaced by a space; inline tags are stripped; whitespace is collapsed.

## Development

### Releasing a New Version

To release a new version of `mmqc-utils` to PyPI:

1. **Update the version number** in `pyproject.toml`.
2. **Update the uv.lock file**:
   ```bash
   uv lock
   ```
3. **Update the changelog** in `CHANGELOG.md`.
4. **Build the distribution**:
   ```bash
   just build
   ```
5. **Check the distribution**:
   ```bash
   uv run --group publish twine check dist/*
   ```
6. **Upload to TestPyPI** (optional but recommended):
   ```bash
   uv run --group publish twine upload --repository testpypi dist/*
   ```
7. **Upload to PyPI**:
   ```bash
   uv run --group publish twine upload dist/*
   ```
