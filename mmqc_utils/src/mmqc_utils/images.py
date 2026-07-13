"""Image conversion helpers."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import BinaryIO

from .exceptions import TiffWandReadError

PathLikeOrFile = str | Path | bytes | bytearray | BinaryIO

_QUALITY_LADDER = (80, 70, 60, 50, 40, 30)
_TIFF_MAGIC = (b"II*\x00", b"MM\x00*")
_TIFF_SUFFIXES = {".tif", ".tiff"}


def _read_bytes(source: PathLikeOrFile) -> bytes:
    if isinstance(source, Path):
        return source.read_bytes()
    if isinstance(source, str):
        return Path(source).read_bytes()
    if isinstance(source, (bytes, bytearray)):
        return bytes(source)
    return source.read()


def _is_tiff_bytes(data: bytes) -> bool:
    return len(data) >= 4 and data[:4] in _TIFF_MAGIC


def _is_tiff_source(source: PathLikeOrFile) -> bool:
    if isinstance(source, (str, Path)):
        return Path(source).suffix.lower() in _TIFF_SUFFIXES
    if isinstance(source, (bytes, bytearray)):
        return _is_tiff_bytes(bytes(source))
    return False


def _is_tiff_wand_sampleformat_error(exc: BaseException) -> bool:
    """Return True when ImageMagick cannot read a TIFF SampleFormat tag."""
    from wand.exceptions import CoderError

    msg = str(exc)
    return isinstance(exc, CoderError) and "SampleFormat" in msg and "TIFFReadDirectory" in msg


def _normalize_to_uint8(arr):
    import numpy as np

    if arr.dtype == np.uint8:
        return arr
    if np.issubdtype(arr.dtype, np.floating):
        values = arr.astype(np.float64)
        if values.size and values.max() <= 1.0:
            values *= 255.0
        return np.clip(values, 0, 255).astype(np.uint8)
    if arr.dtype == np.uint16:
        return (arr.astype(np.float64) / 257.0).astype(np.uint8)
    if np.issubdtype(arr.dtype, np.integer):
        info = np.iinfo(arr.dtype)
        if info.max > 255:
            return (arr.astype(np.float64) / info.max * 255.0).astype(np.uint8)
    return arr.astype(np.uint8)


def _tiff_array_to_rgb_uint8(arr):
    """Convert a tifffile array to an ``H x W x 3`` uint8 RGB array."""
    import numpy as np

    data = np.asarray(arr)
    if data.ndim == 4:
        data = data[0]

    if data.ndim == 2:
        gray = _normalize_to_uint8(data)
        return np.stack([gray, gray, gray], axis=-1)

    if data.ndim == 3 and data.shape[-1] == 1:
        gray = _normalize_to_uint8(data[..., 0])
        return np.stack([gray, gray, gray], axis=-1)

    if data.ndim == 3 and data.shape[-1] == 4:
        rgb = data[..., :3].astype(np.float64)
        alpha = data[..., 3:4].astype(np.float64) / 255.0
        rgb = rgb * alpha + 255.0 * (1.0 - alpha)
        return np.clip(rgb, 0, 255).astype(np.uint8)

    if data.ndim == 3 and data.shape[-1] >= 3:
        return _normalize_to_uint8(data[..., :3])

    raise ValueError(f"Unsupported TIFF array shape: {data.shape}")


def _encode_wand_page_to_jpeg(page, *, max_dimension: int, compression_quality: int, background: str) -> bytes:
    from wand.color import Color

    page.background_color = Color(background)
    page.alpha_channel = "remove"

    if max(page.width, page.height) > max_dimension:
        page.transform(resize=f"{max_dimension}x{max_dimension}>")

    page.format = "jpg"
    page.compression_quality = compression_quality
    return page.make_blob()


def _convert_with_wand(
    image_bytes: bytes,
    *,
    rasterization_dpi: int,
    max_dimension: int,
    compression_quality: int,
    background: str,
) -> bytes:
    from wand.exceptions import WandException
    from wand.image import Image

    try:
        with Image(file=BytesIO(image_bytes), resolution=rasterization_dpi) as img:
            # Important:
            # Some TIFFs are stored as a stack of pages/layers where the first page is the
            # full-size composite and later pages are tiny sub-images at offsets.
            # In wand, `merge_layers("flatten")` can return only the last layer/page,
            # producing a seemingly "blank" preview. The ImageMagick CLI `convert -layers flatten` does not exhibit that
            # behavior.
            # For previews we therefore render only the first page/layer.
            with Image(img.sequence[0]) as page:
                return _encode_wand_page_to_jpeg(
                    page,
                    max_dimension=max_dimension,
                    compression_quality=compression_quality,
                    background=background,
                )
    except WandException as exc:
        if _is_tiff_bytes(image_bytes) and _is_tiff_wand_sampleformat_error(exc):
            raise TiffWandReadError(f"ImageMagick failed to read TIFF: {exc}") from exc
        raise


def _convert_tiff_with_tifffile(
    image_bytes: bytes,
    *,
    max_dimension: int,
    compression_quality: int,
    background: str,
) -> bytes:
    import tifffile
    from PIL import Image as PILImage

    arr = tifffile.imread(BytesIO(image_bytes))
    rgb = _tiff_array_to_rgb_uint8(arr)
    image = PILImage.fromarray(rgb, mode="RGB")

    width, height = image.size
    if max(width, height) > max_dimension:
        scale = max_dimension / max(width, height)
        image = image.resize(
            (max(1, int(width * scale)), max(1, int(height * scale))),
            PILImage.Resampling.LANCZOS,
        )

    output = BytesIO()
    image.save(output, format="JPEG", quality=compression_quality)
    return output.getvalue()


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

    When ImageMagick fails on a TIFF with a malformed ``SampleFormat`` tag, falls back to ``tifffile`` for decoding.

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
    try:
        return _convert_with_wand(
            image_bytes,
            rasterization_dpi=rasterization_dpi,
            max_dimension=max_dimension,
            compression_quality=compression_quality,
            background=background,
        )
    except TiffWandReadError:
        if not (_is_tiff_source(source) or _is_tiff_bytes(image_bytes)):
            raise
        return _convert_tiff_with_tifffile(
            image_bytes,
            max_dimension=max_dimension,
            compression_quality=compression_quality,
            background=background,
        )


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
