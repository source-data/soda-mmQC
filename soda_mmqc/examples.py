from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, Optional, Type
import hashlib
import json
import logging
import base64
import mimetypes
import subprocess
from soda_mmqc.config import EXAMPLES_DIR

logger = logging.getLogger(__name__)


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
    
    def __init__(self, source_path: str):
        """Initialize an example from a dictionary.
        
        Args:
            source_path: Path to the example source directory
            
        Raises:
            ValueError: If required data is missing
        """
        self.source_path = EXAMPLES_DIR / Path(source_path)
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
    def prepare_model_input(self, prompt: str) -> Dict[str, Any]:
        """Prepare the example's content for model input.
        
        Args:
            prompt: The prompt to use for the model
            
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
        check_name: str
    ) -> Path:
        """Save an expected output for this example.

        Args:
            output: The output to save
            check_name: Name of the check
            
        Returns:
            Path to the saved expected output file
        """
        # Create expected output directory
        expected_output_dir = self.source_path / "checks" / check_name
        expected_output_dir.mkdir(parents=True, exist_ok=True)
        expected_output_path = expected_output_dir / "expected_output.json"

        # Check if expected output already exists
        if expected_output_path.exists():
            logger.info(
                f"Expected output already exists: {expected_output_path}"
            )
            return expected_output_path

        # Write expected output
        with open(expected_output_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=4, ensure_ascii=False)

        logger.info(
            f"Created expected output: {expected_output_path}"
        )
        return expected_output_path


class FigureExample(Example):
    """Example containing a figure with caption."""

    def __init__(self, source_path: str):
        super().__init__(source_path)
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

        # Find image
        self.image_path = None
        for ext in [".png", ".jpg", ".jpeg", ".tiff"]:
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
        
        Returns:
            A string hash of the content
        """
        self._ensure_loaded()
        if self._content_hash is None:
            # Hash both caption and image
            hasher = hashlib.sha256()
            if self.caption is not None:
                hasher.update(self.caption.encode('utf-8'))
            if self.image_path and self.image_path.exists():
                with open(self.image_path, "rb") as f:
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

    def prepare_model_input(self, prompt: str) -> Dict[str, Any]:
        """Prepare the example's content for model input.
        
        Args:
            prompt: The prompt to use for the model
            
        Returns:
            Dictionary containing:
            - content: List of content items for the model
            - metadata: Dictionary with tracing information
        """
        self._ensure_loaded()
        return {
            "content": [
                {
                    "type": "input_text",
                    "text": f"{prompt}\n\nFigure Caption:\n{self.caption}"
                },
                {
                    "type": "input_image",
                    "image_url": (
                        f"data:{self._get_image_mime_type()};base64,"
                        f"{self._encode_image()}"
                    )
                }
            ],
            "metadata": {
                "doc_id": self.doc_id,
                "figure_id": self.figure_id
            }
        }

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

    def __init__(self, source_path: str, destination_format: str = "markdown"):
        super().__init__(source_path)
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

    def prepare_model_input(self, prompt: str) -> Dict[str, Any]:
        """Prepare the example's content for model input.
        
        Args:
            prompt: The prompt to use for the model
            
        Returns:
            Dictionary containing:
            - content: List of content items for the model
            - metadata: Dictionary with tracing information
        """
        self._ensure_loaded()
        return {
            "content": [
                {
                    "type": "input_text",
                    "text": f"{prompt}\n\nContent:\n{self.content}"
                }
            ],
            "metadata": {
                "doc_id": self.doc_id
            }
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

    def create(self, source_path: str, example_type: str, **kwargs) -> Example:
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
        example = example_class(source_path, **kwargs)
        
        # Explicitly load the content after initialization
        example.load_from_source()
        
        if example.doc_id is None:
            raise ValueError(
                f"Example at {source_path} has no doc_id!"
            )
        return example


# Create a default factory instance
EXAMPLE_FACTORY = ExampleFactory()
