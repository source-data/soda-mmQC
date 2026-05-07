from io import BytesIO

from PIL import Image

from mmqc_utils.images import compress_to_bounded_jpeg, convert_to_bounded_jpeg


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
