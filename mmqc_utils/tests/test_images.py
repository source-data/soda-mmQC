from io import BytesIO
from unittest.mock import patch

import pytest
from PIL import Image
from wand.exceptions import CoderError

from mmqc_utils.exceptions import TiffWandReadError
from mmqc_utils.images import (
    _is_tiff_bytes,
    _is_tiff_source,
    _is_tiff_wand_sampleformat_error,
    _tiff_array_to_rgb_uint8,
    compress_to_bounded_jpeg,
    convert_to_bounded_jpeg,
)


def test_convert_to_bounded_jpeg_downscales_and_converts_rgba() -> None:
    source = Image.new("RGBA", (4000, 2000), (255, 0, 0, 128))
    buffer = BytesIO()
    source.save(buffer, format="PNG")

    result = convert_to_bounded_jpeg(buffer.getvalue(), max_dimension=1000)

    converted = Image.open(BytesIO(result))
    assert converted.format == "JPEG"
    assert max(converted.size) <= 1000
    assert converted.mode == "RGB"


def test_convert_to_bounded_jpeg_uses_first_frame_for_multiframe_tiff() -> None:
    first = Image.new("RGB", (1200, 1200), "red")
    second = Image.new("RGB", (200, 200), "blue")
    buffer = BytesIO()
    first.save(buffer, format="TIFF", save_all=True, append_images=[second])

    result = convert_to_bounded_jpeg(buffer.getvalue(), max_dimension=600)

    converted = Image.open(BytesIO(result))
    assert converted.size == (600, 600)


# --- compress_to_bounded_jpeg tests ---


def _large_png_bytes(size: int = 3000) -> bytes:
    img = Image.new("RGB", (size, size), (100, 149, 237))
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_compress_to_bounded_jpeg_meets_size_limit() -> None:
    png = _large_png_bytes()
    limit = 100_000  # 100 KB

    result = compress_to_bounded_jpeg(png, max_bytes=limit)

    assert isinstance(result, bytes)
    assert len(result) <= limit
    assert Image.open(BytesIO(result)).format == "JPEG"


def test_compress_to_bounded_jpeg_from_binary_io() -> None:
    limit = 100_000
    result = compress_to_bounded_jpeg(BytesIO(_large_png_bytes()), max_bytes=limit)

    assert len(result) <= limit


def test_compress_to_bounded_jpeg_returns_bytes_when_limit_unreachable() -> None:
    # An impossibly small limit — function should still return valid JPEG bytes.
    result = compress_to_bounded_jpeg(_large_png_bytes(), max_bytes=1)

    assert isinstance(result, bytes)
    assert len(result) > 0
    assert Image.open(BytesIO(result)).format == "JPEG"


def test_compress_to_bounded_jpeg_no_compression_needed() -> None:
    # A tiny image that already fits — should come back on the first attempt.
    img = Image.new("RGB", (50, 50), "green")
    buf = BytesIO()
    img.save(buf, format="PNG")
    tiny_png = buf.getvalue()

    result = compress_to_bounded_jpeg(tiny_png, max_bytes=10_000_000)

    assert len(result) <= 10_000_000


# --- TIFF helpers and fallback ---


def test_is_tiff_bytes() -> None:
    buf = BytesIO()
    Image.new("RGB", (10, 10), "red").save(buf, format="TIFF")
    assert _is_tiff_bytes(buf.getvalue()) is True
    assert _is_tiff_bytes(b"not-a-tiff") is False


def test_is_tiff_source() -> None:
    assert _is_tiff_source("figure.tiff") is True
    assert _is_tiff_source("figure.png") is False


def test_is_tiff_wand_sampleformat_error() -> None:
    exc = CoderError(
        'Incorrect count for "SampleFormat". `TIFFReadDirectory\' @ error/tiff.c/TIFFErrors/574'
    )
    assert _is_tiff_wand_sampleformat_error(exc) is True
    assert _is_tiff_wand_sampleformat_error(CoderError("other tiff error")) is False


def test_tiff_array_to_rgb_uint8_handles_rgba() -> None:
    import numpy as np

    rgba = np.zeros((2, 2, 4), dtype=np.uint8)
    rgba[..., 0] = 255
    rgba[..., 3] = 128

    rgb = _tiff_array_to_rgb_uint8(rgba)

    assert rgb.shape == (2, 2, 3)
    assert rgb.dtype == np.uint8


def test_convert_to_bounded_jpeg_tifffile_fallback_on_sampleformat_error() -> None:
    buf = BytesIO()
    Image.new("RGBA", (400, 300), (255, 0, 0, 128)).save(buf, format="TIFF")
    tiff_bytes = buf.getvalue()

    with patch("mmqc_utils.images._convert_with_wand", side_effect=TiffWandReadError("wand failed")):
        result = convert_to_bounded_jpeg(tiff_bytes, max_dimension=200)

    assert result.startswith(b"\xff\xd8")
    converted = Image.open(BytesIO(result))
    assert converted.format == "JPEG"
    assert max(converted.size) <= 200


@pytest.mark.skipif(
    not __import__("pathlib").Path("/tmp/emboj121381/graphic/EMBOJ2025121381_Fig1.tiff").exists(),
    reason="EMBOJ TIFF fixtures not available",
)
def test_convert_to_bounded_jpeg_emboj_sampleformat_tiffs() -> None:
    from pathlib import Path

    base = Path("/tmp/emboj121381/graphic")

    for fig in sorted(base.glob("EMBOJ2025121381_Fig*.tiff")):
        with patch("mmqc_utils.images._convert_with_wand", side_effect=TiffWandReadError("wand failed")):
            result = convert_to_bounded_jpeg(fig, max_dimension=1000)
        assert result.startswith(b"\xff\xd8"), fig.name
        assert len(result) > 0, fig.name
