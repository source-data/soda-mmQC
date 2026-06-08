# Changelog

All notable changes to this project will be documented here.

## [0.3.1] - 2026-06-08

### Added
- match multiple quotes from the same source text: order-aware quote alignment to prevent multiple quotes from matching to the same span in the source text, if that span is repeated later in the source.

## [0.3.0] - 2026-05-11

### Added
- quote matching in & similarity to source text: find the best matching span(s) in the source text for a quote and provide a score for how literal the quote is.
- visualization of quote matching.

### Changed
- use stdlib HTMLParser in `compute_plain_text` instead of regex solution.

## [0.2.1] - 2026-05-11

### Added
- make wand imports lazy to avoid unnecessary ImageMagick dependency for users who don't need image processing features.

## [0.2.0] - 2026-05-08

### Added
- type hinting support: added `py.typed` marker for PEP 561 compliance

## [0.1.0] - 2026-05-07

### Added
- `document_to_html`: convert DOCX, RTF, ODT, TeX, and PDF files to HTML via pandoc (bundled through `pypandoc-binary`); PDF falls back to `pypdf` page-by-page text extraction
- `convert_to_bounded_jpeg`: rasterize and downscale images (TIFF, JPEG, PNG, GIF, WebP, PDF) to a bounded JPEG preview via ImageMagick/Wand
- `compress_to_bounded_jpeg`: wrapper around `convert_to_bounded_jpeg` that steps down JPEG quality and then halves the pixel dimension until the output fits within a byte-size budget
- `html_to_text` / `compute_plain_text`: strip HTML tags and collapse whitespace to plain text
- All public functions accept `Path`, `str`, `bytes`, `bytearray`, or `BinaryIO` as input
