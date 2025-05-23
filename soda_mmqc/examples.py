from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, Optional, TypeVar, Generic
import hashlib
import json
import logging
import base64
import mimetypes
from dataclasses import dataclass

from soda_mmqc.config import get_content_path, get_expected_output_path

logger = logging.getLogger(__name__)

@dataclass
class ExampleMetadata:
    """Metadata about an example."""
    doi: str
    source_path: Path

T = TypeVar('T', bound=ExampleMetadata)

class Example(ABC, Generic[T]):
    """Base class for all examples.
    
    This abstract class defines the interface for all example types.
    Each example type should know how to:
    1. Load its content from a source directory
    2. Generate a cache key
    3. Prepare its content for model input
    """
    
    def __init__(self, example: Dict[str, Any]):
        """Initialize an example from a dictionary.
        
        Args:
            example: Dictionary containing example data with at least 'doi'
            
        Raises:
            ValueError: If required data is missing
        """
        if "doi" not in example:
            raise ValueError("Example dictionary must contain 'doi'")
        
        self._load_from_source(example)
        self._content_hash: Optional[str] = None
    
    @abstractmethod
    def _load_from_source(self, example: Dict[str, Any]) -> None:
        """Load the example's content from the provided dictionary.
        
        Args:
            example: Dictionary containing example data
            
        Raises:
            FileNotFoundError: If required files are missing
            ValueError: If required data is missing
        """
        pass
    
    @abstractmethod
    def get_content_hash(self) -> str:
        """Get a hash of the example's content for caching.
        
        Returns:
            A string hash of the content
        """
        pass
    
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
            "doi": self.metadata.doi,
            "source_path": str(self.metadata.source_path),
            "content_hash": self.get_content_hash(),
            "model_input": self.prepare_model_input("")
        }

    def get_expected_output(self, check_name: str) -> Dict[str, Any]:
        """Get the expected output for this example.
        
        Args:
            check_name: Name of the check
            
        Returns:
            Dictionary containing the expected output
            
        Raises:
            FileNotFoundError: If expected output file is not found
        """
        expected_output_path = self.metadata.source_path / "checks" / check_name / "expected_output.json"
        if not expected_output_path.exists():
            logger.error(
                f"Expected output file not found: {expected_output_path}"
            )
            raise FileNotFoundError(
                f"Expected output file not found: {expected_output_path}"
            )
        with open(expected_output_path, "r", encoding="utf-8") as f:
            return json.load(f)

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
        expected_output_dir = self.metadata.source_path / "checks" / check_name
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

@dataclass
class FigureMetadata(ExampleMetadata):
    """Metadata about a figure example."""
    figure_id: str

class FigureExample(Example[FigureMetadata]):
    """Example containing a figure with caption."""
    
    def _load_from_source(self, example: Dict[str, Any]) -> None:
        """Load the example's content from the provided dictionary.
        
        Args:
            example: Dictionary containing example data with at least:
                - doi: The DOI of the example
                - figure_id: The ID of the figure
                
        Raises:
            FileNotFoundError: If required files are missing
            ValueError: If required data is missing
        """
        if "figure_id" not in example:
            raise ValueError("Figure example must contain 'figure_id'")
        
        # Get content path
        self.source_path = get_content_path(example)
        if not self.source_path.exists():
            raise FileNotFoundError(
                f"Content directory not found: {self.source_path}"
            )
        
        # Load caption
        caption_path = self.source_path / "caption.txt"
        if not caption_path.exists():
            raise FileNotFoundError(
                f"Caption file not found: {caption_path}"
            )
        with open(caption_path, "r", encoding="utf-8") as f:
            self.caption = f.read().strip()
        
        # Find image
        self.image_path = None
        for ext in [".png", ".jpg", ".jpeg", ".tiff"]:
            for img_file in self.source_path.glob(f"*{ext}"):
                self.image_path = img_file
                break
            if self.image_path:
                break
        
        if not self.image_path:
            raise ValueError(
                f"No image found in {self.source_path}"
            )
        
        # Create metadata
        self.metadata = FigureMetadata(
            doi=example["doi"],
            figure_id=example["figure_id"],
            source_path=self.source_path
        )
    
    def get_content_hash(self) -> str:
        """Get a hash of the example's content for caching.
        
        Returns:
            A string hash of the content
        """
        if self._content_hash is None:
            # Hash both caption and image
            hasher = hashlib.sha256()
            hasher.update(self.caption.encode('utf-8'))
            with open(self.image_path, "rb") as f:
                hasher.update(f.read())
            self._content_hash = hasher.hexdigest()
        return self._content_hash
    
    def _get_image_mime_type(self) -> str:
        """Get the MIME type of the image file based on its extension."""
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
                "doi": self.metadata.doi,
                "figure_id": self.metadata.figure_id
            }
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the example to a dictionary.
        
        Returns:
            Dictionary representation of the example
        """
        base_dict = super().to_dict()
        base_dict["figure_id"] = self.metadata.figure_id
        return base_dict

class WordExample(Example[ExampleMetadata]):
    """Example containing only text content."""
    
    def _load_from_source(self, example: Dict[str, Any]) -> None:
        """Load the example's content from the provided dictionary.
        
        Args:
            example: Dictionary containing example data with at least:
                - doi: The DOI of the example
                
        Raises:
            FileNotFoundError: If content file is missing
        """
        # Get content path
        self.source_path = get_content_path(example)
        if not self.source_path.exists():
            raise FileNotFoundError(
                f"Content directory not found: {self.source_path}"
            )
        
        # Load content
        content_path = self.source_path / "content.txt"
        if not content_path.exists():
            raise FileNotFoundError(
                f"Content file not found: {content_path}"
            )
        with open(content_path, "r", encoding="utf-8") as f:
            self.content = f.read().strip()
        
        # Create metadata
        self.metadata = ExampleMetadata(
            doi=example["doi"],
            source_path=self.source_path
        )
    
    def get_content_hash(self) -> str:
        """Get a hash of the example's content for caching.
        
        Returns:
            A string hash of the content
        """
        if self._content_hash is None:
            hasher = hashlib.sha256()
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
        return {
            "content": [
                {
                    "type": "input_text",
                    "text": f"{prompt}\n\nContent:\n{self.content}"
                }
            ],
            "metadata": {
                "doi": self.metadata.doi
            }
        }

class DataFigureWordExample(FigureExample):
    """Example containing a figure with caption and additional data."""
    
    def _load_from_source(self, example: Dict[str, Any]) -> None:
        """Load the example's content from the provided dictionary.
        
        Args:
            example: Dictionary containing example data with at least:
                - doi: The DOI of the example
                - figure_id: The ID of the figure
                
        Raises:
            FileNotFoundError: If required files are missing
            ValueError: If required data is missing
        """
        # First load the base figure content
        super()._load_from_source(example)
        
        # Load additional data
        data_path = self.source_path / "data.json"
        if not data_path.exists():
            raise FileNotFoundError(
                f"Data file not found: {data_path}"
            )
        with open(data_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)
    
    def get_content_hash(self) -> str:
        """Get a hash of the example's content for caching.
        
        Returns:
            A string hash of the content
        """
        if self._content_hash is None:
            # Hash caption, image, and data
            hasher = hashlib.sha256()
            hasher.update(self.caption.encode('utf-8'))
            with open(self.image_path, "rb") as f:
                hasher.update(f.read())
            hasher.update(
                json.dumps(self.data, sort_keys=True).encode('utf-8')
            )
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
        base_input = super().prepare_model_input(prompt)
        # Add data as a new text input
        base_input["content"].append({
            "type": "input_text",
            "text": f"\nAdditional Data:\n{json.dumps(self.data, indent=2)}"
        })
        return base_input

def create_example(example: Dict[str, Any]) -> Example:
    """Create an example from a dictionary.
    
    This factory function determines the appropriate Example type based on
    the files present in the content directory.
    
    Args:
        example: Dictionary containing example data with at least 'doi'
        
    Returns:
        An instance of the appropriate Example class
        
    Raises:
        FileNotFoundError: If required files are missing
        ValueError: If required data is missing
    """
    # Try to create a FigureExample first
    try:
        # Check if it has a figure_id
        if "figure_id" in example:
            # Check if it has additional data
            source_path = get_content_path(example)
            if (source_path / "data.json").exists():
                return DataFigureWordExample(example)
            else:
                return FigureExample(example)
    except (FileNotFoundError, ValueError):
        # If figure loading fails, try word example
        pass
    
    # Try to create a WordExample
    return WordExample(example) 