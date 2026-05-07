"""Image conversion helpers."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import BinaryIO

from wand.color import Color
from wand.image import Image

PathLikeOrFile = str | Path | bytes | bytearray | BinaryIO

_QUALITY_LADDER = (80, 70, 60, 50, 40, 30)


def _read_bytes(source: PathLikeOrFile) -> bytes:
    if isinstance(source, Path):
        return source.read_bytes()
    if isinstance(source, str):
        return Path(source).read_bytes()
    if isinstance(source, (bytes, bytearray)):
        return bytes(source)
    return source.read()


def convert_to_bounded_jpeg(
    source: PathLikeOrFile,
    rasterization_dpi: int = 150,
    max_dimension: int = 2000,
    compression_quality: int = 80,
    background: str = "white",
) -> bytes:
    """
    Create a JPG preview for a file.

    Uses wand/ImageMagick to handle a variety of formats including vector PDFs and multi-page TIFFs. For TIFFs, only
    the first page/layer is used for the preview to avoid issues with some TIFFs where later layers are tiny
    insets/labels that end up being returned by `merge_layers("flatten")`.

    Parameters
    ----------
    source : PathLikeOrFile
        The image source, either a file path or a file-like object
    rasterization_dpi : int, optional
        The DPI to use when rasterizing vector formats (PDF, PS). Higher values produce sharper previews but increase
        processing time and memory usage. Default is 150, which is a good balance for typical use cases.
    max_dimension : int, optional
        The maximum allowed dimension (width or height) for the output JPEG. The image will be downscaled if either
        dimension exceeds this value, while preserving aspect ratio. Default is 2000, which provides decent fullscreen
        quality on typical monitors without producing excessively large files.
    compression_quality : int, optional
        The JPEG compression quality (1-100). Higher values produce better quality but larger file sizes. Default is 80,
        which is a good balance for most use cases.
    background : str, optional
        The background color to use when flattening images with transparency (e.g., PNGs). Default is "white".
    """
    image_bytes = _read_bytes(source)
    output_format = "jpg"

    with Image(file=BytesIO(image_bytes), resolution=rasterization_dpi) as img:
        # Important:
        # Some TIFFs are stored as a *stack* of pages/layers where the first page is the full-size composite and later
        # pages are tiny sub-images at offsets (labels/insets/etc.).
        # In wand, `merge_layers("flatten")` can end up effectively returning only the last layer/page (a tiny patch),
        # producing a seemingly "blank" preview. The ImageMagick CLI `convert -layers flatten` does not exhibit that
        # behavior.
        # For previews we therefore render only the first page/layer.
        with Image(img.sequence[0]) as page:
            # JPG doesn't support transparency; composite onto white.
            page.background_color = Color(background)
            page.alpha_channel = "remove"

            # Only downscale (never upscale). The trailing '>' keeps aspect ratio.
            if max(page.width, page.height) > max_dimension:
                page.transform(resize=f"{max_dimension}x{max_dimension}>")

            page.format = output_format
            page.compression_quality = compression_quality
            return page.make_blob()


def compress_to_bounded_jpeg(
    source: PathLikeOrFile,
    max_bytes: int,
    *,
    rasterization_dpi: int = 150,
    max_dimension: int = 2000,
    compression_quality: int = 80,
    background: str = "white",
) -> bytes:
    """Convert an image to JPEG, reducing quality then dimensions until under max_bytes.

    Tries progressively lower JPEG quality settings first; if quality alone is
    insufficient, halves max_dimension and repeats. Returns the smallest result
    achieved even if max_bytes cannot be met.

    Parameters mirror convert_to_bounded_jpeg; see that function for details.
    """
    # Materialise streams so we can call convert_to_bounded_jpeg multiple times.
    if not isinstance(source, (str, Path, bytes, bytearray)):
        source = source.read()

    quality_steps = [compression_quality] + [q for q in _QUALITY_LADDER if q < compression_quality]

    dim = max_dimension
    result = b""
    while dim >= 250:
        for q in quality_steps:
            result = convert_to_bounded_jpeg(
                source,
                rasterization_dpi=rasterization_dpi,
                max_dimension=dim,
                compression_quality=q,
                background=background,
            )
            if len(result) <= max_bytes:
                return result
        dim //= 2

    return result
