# Changelog

All notable changes to this project will be documented here.

## [0.1.0] - 2026-05-07

### Added
- `document_to_html`: convert DOCX, RTF, ODT, TeX, and PDF files to HTML via pandoc (bundled through `pypandoc-binary`); PDF falls back to `pypdf` page-by-page text extraction
- `convert_to_bounded_jpeg`: rasterize and downscale images (TIFF, JPEG, PNG, GIF, WebP, PDF) to a bounded JPEG preview via ImageMagick/Wand
- `compress_to_bounded_jpeg`: wrapper around `convert_to_bounded_jpeg` that steps down JPEG quality and then halves the pixel dimension until the output fits within a byte-size budget
- `html_to_text` / `compute_plain_text`: strip HTML tags and collapse whitespace to plain text
- All public functions accept `Path`, `str`, `bytes`, `bytearray`, or `BinaryIO` as input
