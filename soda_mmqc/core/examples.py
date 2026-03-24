from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, Optional, Type, List, Tuple
import hashlib
import io
import json
import logging
import base64
import mimetypes
import subprocess
from soda_mmqc.config import EXAMPLES_DIR

logger = logging.getLogger(__name__)

# Image MIME types supported by OpenAI Responses API (TIFF is not supported)
_API_SUPPORTED_IMAGE_TYPES = frozenset(
    {"image/jpeg", "image/png", "image/gif", "image/webp"}
)


# Class mapping for factory pattern - will be populated after class definitions
EXAMPLE_TYPES: Dict[str, Type['Example']] = {}


class Example(ABC):
    """Base class for all examples.
    
    This abstract class defines the interface for all example types.
    Each example type should know how to:
    1. Load its content from a source directory
    2. Generate a cache key
    3. Prepare its content for model input
    """
    
    _example_type: Optional[str] = None
    
    def __init__(self, relative_source_path: str):
        """Initialize an example from a dictionary.
        
        Args:
            source_path: Path to the example source directory
            
        Raises:
            ValueError: If required data is missing
        """
        self.relative_source_path = relative_source_path
        self.source_path = EXAMPLES_DIR / Path(self.relative_source_path)
        self.doc_id: Optional[str] = None
        self._content_hash: Optional[str] = None
        self.expected_output = None

    @abstractmethod
    def load_from_source(self) -> None:
        """Load the example's content from the provided path.
        
        Raises:
            FileNotFoundError: If required files are missing
            ValueError: If required data is missing
        """
        pass

    @property
    def example_class_name(self) -> str:
        """Get the example type identifier (read-only)."""
        if self._example_type is None:
            raise ValueError(
                "Example type not set. This should be set by the factory."
            )
        return self._example_type
    
    @example_class_name.setter
    def example_class_name(self, value: str) -> None:
        """Set the example type identifier (factory use only)."""
        if not isinstance(value, str):
            raise ValueError("Example type must be a string")
        if not value.strip():
            raise ValueError("Example type cannot be empty")
        self._example_type = value.strip()
    
    def get_expected_output(self, check_name: str) -> Dict[str, Any]:
        """Set the expected output for this example.
        
        Args:
            check_name: Name of the check
            
        Returns:
            Path to the expected output file 
        """

        expected_output_path = (
            self.source_path / "checks" / check_name /
            "expected_output.json"
        )
        if expected_output_path.exists():     
            with open(expected_output_path, "r", encoding="utf-8") as f:
                expected_output_json = json.load(f)
        else:
            logger.warning(
                f"Expected output file not found: {expected_output_path}"
            )
            expected_output_json = {}
        return expected_output_json

    @abstractmethod
    def get_content_hash(self) -> str:
        """Get a hash of the example's content for caching.
        
        Returns:
            A string hash of the content
        """
        pass

    def _ensure_loaded(self) -> None:
        """Ensure content is loaded. Call this before accessing 
        content-dependent methods."""
        if self.doc_id is None:
            self.load_from_source()
    
    @abstractmethod
    def prepare_model_input(
        self, prompt: str, model_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Prepare the example's content for model input.
        
        Args:
            prompt: The prompt to use for the model
            model_config: Optional check-level config (e.g. source_data_files.enabled)
            
        Returns:
            Dictionary containing:
            - content: List of content items for the model
            - metadata: Dictionary with tracing information
        """
        pass

    def to_dict(self) -> Dict[str, Any]:
        """Convert the example to a dictionary.

        Returns:
            Dictionary representation of the example
        """
        return {
            "source_path": str(self.source_path),
            "content_hash": self.get_content_hash(),
            "model_input": self.prepare_model_input("")
        }

    def save_expected_output(
        self,
        output: Dict[str, Any],
        check_name: str,
        overwrite: bool = False
    ) -> Path:
        """Save an expected output for this example.

        Args:
            output: The output to save
            check_name: Name of the check
            overwrite: Whether to overwrite existing files (default: False)
            
        Returns:
            Path to the saved expected output file
        """
        # Create expected output directory
        expected_output_dir = self.source_path / "checks" / check_name
        expected_output_dir.mkdir(parents=True, exist_ok=True)
        expected_output_path = expected_output_dir / "expected_output.json"

        # Check if expected output already exists
        if expected_output_path.exists() and not overwrite:
            logger.info(
                f"Expected output already exists and overwrite=False: "
                f"{expected_output_path}"
            )
            return expected_output_path

        # Write expected output
        with open(expected_output_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=4, ensure_ascii=False)

        # Write HTML version for easier viewing
        try:
            from soda_mmqc.lib.expected_output_html import output_to_html
            html_path = expected_output_dir / "expected_output.html"
            with open(html_path, "w", encoding="utf-8") as f:
                f.write(output_to_html(output, title=f"Expected output — {check_name}"))
        except Exception as e:
            logger.debug("Could not write expected_output.html: %s", e)

        logger.info(
            f"Saved expected output: {expected_output_path}"
        )
        return expected_output_path


class FigureExample(Example):
    """Example containing a figure with caption."""
    
    def __init__(self, relative_source_path: str):
        super().__init__(relative_source_path)
        self.caption: Optional[str] = None
        self.image_path: Optional[Path] = None
        self.figure_id: Optional[str] = None

    def load_from_source(self) -> None:
        """Load the example's content from the provided path.

        Raises:
            FileNotFoundError: If required files are missing
            ValueError: If required data is missing
        """
        # Convert string to Path immediately
        path_obj = self.source_path
        # Pattern: doc_id/content/figure_id/content/caption.txt
        # So source_path points to figure_id, need to go up 2 levels to get 
        # doc_id
        self.doc_id = path_obj.parent.parent.name
        self.figure_id = path_obj.name

        if not self.source_path.exists():
            raise FileNotFoundError(
                f"Content directory not found: {self.source_path}"
            )

        # Load caption
        caption_path = self.source_path / "content" / "caption.txt"
        if not caption_path.exists():
            raise FileNotFoundError(
                f"Caption file not found: {caption_path}"
            )
        with open(caption_path, "r", encoding="utf-8") as f:
            self.caption = f.read().strip()

        # Find image (extensions must match what we can send to the API or convert)
        self.image_path = None
        for ext in [".png", ".jpg", ".jpeg", ".webp", ".tiff", ".tif"]:
            for image_path in self.source_path.glob(f"content/*{ext}"):
                self.image_path = image_path
                break
            if self.image_path:
                break

        if not self.image_path:
            raise ValueError(
                f"No image found in {self.source_path}"
            )

    def get_content_hash(self) -> str:
        """Get a hash of the example's content for caching.
        
        Includes caption, main figure image, and any images in content/source_data/
        so cache invalidates when source data is added or changed.
        """
        self._ensure_loaded()
        if self._content_hash is None:
            hasher = hashlib.sha256()
            if self.caption is not None:
                hasher.update(self.caption.encode('utf-8'))
            if self.image_path and self.image_path.exists():
                with open(self.image_path, "rb") as f:
                    hasher.update(f.read())
            for path in self._get_all_source_data_paths():
                with open(path, "rb") as f:
                    hasher.update(f.read())
            self._content_hash = hasher.hexdigest()
        return self._content_hash

    def _get_image_mime_type(self) -> str:
        """Get the MIME type of the image file based on its extension."""
        if not self.image_path:
            raise ValueError("No image path available")
        mime_type, _ = mimetypes.guess_type(str(self.image_path))
        if mime_type:
            if mime_type.startswith('image/'):
                return mime_type
            else:
                raise ValueError(f"Not an image: {self.image_path}")
        else:
            raise ValueError(f"Could not guess mime type: {self.image_path}")

    def _encode_image(self) -> str:
        """Encode image to base64 string."""
        if not self.image_path or not self.image_path.exists():
            raise ValueError("Image file not found")
        with open(self.image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def _encode_image_from_path(self, image_path: Path) -> tuple[str, str]:
        """Encode an image file to base64 and return (mime_type, base64_string)."""
        if not image_path.exists():
            raise ValueError(f"Image file not found: {image_path}")
        mime_type, _ = mimetypes.guess_type(str(image_path))
        if not mime_type or not mime_type.startswith("image/"):
            raise ValueError(f"Not an image: {image_path}")
        with open(image_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        return self._ensure_api_supported_image(mime_type, b64)

    def _ensure_api_supported_image(self, mime_type: str, b64: str) -> tuple[str, str]:
        """Return (mime_type, b64) with a format the API accepts (jpeg, png, gif, webp).
        Converts TIFF and other unsupported formats to PNG via PIL if available.
        """
        if mime_type in _API_SUPPORTED_IMAGE_TYPES:
            return mime_type, b64
        try:
            from PIL import Image
        except ImportError:
            logger.warning(
                "PIL not available; cannot convert %s to a supported format. "
                "Install Pillow or use JPEG/PNG/GIF/WebP images.",
                mime_type,
            )
            raise ValueError(
                f"Image format {mime_type} is not supported by the API. "
                "Use image/jpeg, image/png, image/gif, or image/webp, or install Pillow to convert TIFF."
            )
        raw = base64.b64decode(b64)
        img = Image.open(io.BytesIO(raw))
        if img.mode in ("RGBA", "P"):
            img = img.convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return "image/png", base64.b64encode(buf.getvalue()).decode("utf-8")

    _SOURCE_DATA_IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".tiff", ".tif")

    def _get_source_data_items(
        self,
    ) -> List[Tuple[Optional[str], List[Path]]]:
        """Return source data as (panel_label, image_paths) items.
        
        If content/source_data/ has subdirectories, each subfolder name is
        treated as a panel label (e.g. A, B, 1G) and images inside are
        grouped under that panel. If there are no subdirectories, all images
        in source_data/ are returned as one item with panel_label None.
        """
        source_data_dir = self.source_path / "content" / "source_data"
        if not source_data_dir.is_dir():
            return []
        exts = list(self._SOURCE_DATA_IMAGE_EXTS)
        subdirs = [d for d in source_data_dir.iterdir() if d.is_dir()]
        if subdirs:
            items: List[Tuple[Optional[str], List[Path]]] = []
            for d in sorted(subdirs):
                paths = []
                for ext in exts:
                    paths.extend(d.glob(f"*{ext}"))
                paths = sorted(paths)
                if paths:
                    items.append((d.name, paths))
            return items
        paths = []
        for ext in exts:
            paths.extend(source_data_dir.glob(f"*{ext}"))
        paths = sorted(paths)
        return [(None, paths)] if paths else []

    def _get_all_source_data_paths(self) -> List[Path]:
        """Flat list of all source data image paths (for hashing)."""
        out: List[Path] = []
        for _panel, paths in self._get_source_data_items():
            out.extend(paths)
        return sorted(out)

    def _get_source_data_file_list(self) -> List[str]:
        """List of all file names (and relative paths) in source_data folder.
        Recursively includes every file so the model can see e.g. .xlsx, .csv
        without opening them. Returns paths relative to source_data dir.
        """
        source_data_dir = self.source_path / "content" / "source_data"
        if not source_data_dir.is_dir():
            return []
        out: List[str] = []
        for f in sorted(source_data_dir.rglob("*")):
            if f.is_file():
                try:
                    rel = f.relative_to(source_data_dir)
                    out.append(rel.as_posix())
                except ValueError:
                    out.append(f.name)
        return sorted(out)

    def prepare_model_input(
        self, prompt: str, model_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Prepare the example's content for model input.
        
        When model_config has source_data_files.enabled, also includes any
        images from content/source_data/ so the model can compare them to the figure.
        """
        self._ensure_loaded()
        fig_mime, fig_b64 = self._get_image_mime_type(), self._encode_image()
        fig_mime, fig_b64 = self._ensure_api_supported_image(fig_mime, fig_b64)
        content: List[Dict[str, Any]] = [
            {
                "type": "input_text",
                "text": f"{prompt}\n\nFigure Caption:\n{self.caption}"
            },
            {
                "type": "input_image",
                "image_url": f"data:{fig_mime};base64,{fig_b64}"
            }
        ]
        # Include source data files if this check requests them
        if model_config:
            source_config = model_config.get("source_data_files") or {}
            if source_config.get("enabled"):
                source_data_dir = self.source_path / "content" / "source_data"
                if source_data_dir.is_dir():
                    file_list = self._get_source_data_file_list()
                    if file_list:
                        content.append({
                            "type": "input_text",
                            "text": (
                                "---\n"
                                "File names in the **source_data** folder (same path as the figure and caption):\n"
                                + "\n".join(f"  - {name}" for name in file_list)
                            )
                        })
                source_items = self._get_source_data_items()
                num_source_images = sum(len(paths) for _, paths in source_items)
                if source_items:
                    logger.info(
                        "Including %d source data image(s) for %s",
                        num_source_images,
                        getattr(self, "doc_id", self.source_path),
                    )
                    content.append({
                        "type": "input_text",
                        "text": (
                            "Images from the **source_data** folder (same path as the figure and caption). "
                            "Each image is labelled with its filename and optionally the panel (e.g. Source data for panel A: filename). "
                            "Use these images to compare with the western blot panels in the figure above."
                        )
                    })
                for panel_label, paths in source_items:
                    group_header = (
                        f"Source data for panel {panel_label}:"
                        if panel_label is not None
                        else "Source data files:"
                    )
                    for path in paths:
                        try:
                            mime_type, b64 = self._encode_image_from_path(path)
                            content.append({
                                "type": "input_text",
                                "text": f"{group_header} {path.name}".strip()
                            })
                            content.append({
                                "type": "input_image",
                                "image_url": f"data:{mime_type};base64,{b64}"
                            })
                            group_header = ""  # subsequent images: filename only in next text
                        except Exception as e:
                            logger.warning(
                                "Skipping source data file %s: %s", path, e
                            )
        return {"content": content}

    def to_dict(self) -> Dict[str, Any]:
        """Convert the example to a dictionary.
        
        Returns:
            Dictionary representation of the example
        """
        base_dict = super().to_dict()
        base_dict["figure_id"] = self.figure_id
        return base_dict


class WordExample(Example):
    """Example containing only text content."""
    
    def __init__(self, relative_source_path: str, destination_format: str = "markdown"):
        super().__init__(relative_source_path)
        self.content: Optional[str] = None
        self.destination_format = destination_format

    def load_from_source(self) -> None:
        """Load the example's content from the provided dictionary.
        
        Args:
            example: Dictionary containing example data with at least 'doc_id'
                
        Raises:
            FileNotFoundError: If content file is missing
            ValueError: If required data is missing
        """

        if not self.source_path.exists():
            raise FileNotFoundError(
                f"Content directory not found: {self.source_path}"
            )

        # Load Word file
        content_path = self.source_path / "content"
        if not content_path.exists():
            raise FileNotFoundError(
                f"Content directory not found: {content_path}"
            )

        # Find Word file (.docx only)
        word_files = list(content_path.glob("*.docx"))
        if not word_files:
            raise FileNotFoundError(
                f"No .docx file found in {content_path}. "
                "Only .docx files are supported."
            )

        word_file_path = word_files[0]  # Use the first .docx file found

        # Extract text from Word file using Pandoc
        try:
            result = subprocess.run(
                ["pandoc", str(word_file_path), "-t", self.destination_format],
                capture_output=True,
                text=True,
                check=True
            )
            self.content = result.stdout.strip()

        except subprocess.CalledProcessError as e:
            raise ValueError(
                f"Pandoc error reading Word file {word_file_path}: {e.stderr}"
            )
        except FileNotFoundError:
            raise ValueError(
                "Pandoc not found. Install from: "
                "https://pandoc.org/installing.html"
            )
        except Exception as e:
            raise ValueError(
                f"Error reading Word file {word_file_path}: {str(e)}"
            )

        # Store doc_id
        self.doc_id = self.source_path.name

    def get_content_hash(self) -> str:
        """Get a hash of the example's content for caching.
        
        Returns:
            A string hash of the content
        """
        self._ensure_loaded()
        if self._content_hash is None:
            hasher = hashlib.sha256()
            if self.content is not None:
                hasher.update(self.content.encode('utf-8'))
            self._content_hash = hasher.hexdigest()
        return self._content_hash

    def prepare_model_input(
        self, prompt: str, model_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Prepare the example's content for model input."""
        self._ensure_loaded()
        return {
            "content": [
                {
                    "type": "input_text",
                    "text": f"{prompt}\n\nContent:\n{self.content}"
                }
            ]
        }


# Register example types
EXAMPLE_TYPES["figure"] = FigureExample
EXAMPLE_TYPES["word"] = WordExample


class ExampleFactory:
    """Factory for creating Example instances.
    
    Simple factory that maps example types to their classes.
    """

    def __init__(self):
        self._example_types = EXAMPLE_TYPES.copy()

    def create(self, relative_source_path: str, example_type: str, **kwargs) -> Example:
        """Create an example from a source path.
        
        Args:
            source_path: Path to the example source directory
            example_type: Type of example to create ("figure" or "word")
            
        Returns:
            Example instance
            
        Raises:
            ValueError: If example_type is not supported
        """
        if example_type not in self._example_types:
            raise ValueError(
                f"Unsupported example type: {example_type}. "
                f"Supported types: {list(self._example_types.keys())}"
            )

        example_class = self._example_types[example_type]
        example = example_class(relative_source_path, **kwargs)
        
        # Set the example type identifier
        example.example_class_name = example_type
        
        # Explicitly load the content after initialization
        example.load_from_source()
        
        if example.doc_id is None:
            raise ValueError(
                f"Example at {relative_source_path} has no doc_id!"
            )
        return example


# Create a default factory instance
EXAMPLE_FACTORY = ExampleFactory()
