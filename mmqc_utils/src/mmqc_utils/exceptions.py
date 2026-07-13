"""Package exceptions."""


class MMQCUtilsError(Exception):
    """Base exception for mmqc_utils."""


class DocumentConversionError(MMQCUtilsError):
    """Raised when a document cannot be converted to HTML."""


class UnsupportedDocumentFormatError(DocumentConversionError):
    """Raised when a document format is unsupported."""


class TiffWandReadError(MMQCUtilsError):
    """Wand/ImageMagick failed to read a TIFF; tifffile fallback may work."""
