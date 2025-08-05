import json
import hashlib
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime


class ModelCache:
    """Manages caching of model outputs with metadata."""
    
    def __init__(self, cache_dir: Path):
        """Initialize the cache manager.
        
        Args:
            cache_dir: Directory to store cached outputs
        """
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _generate_cache_key(self, data: Dict[str, Any]) -> str:
        """Generate a unique cache key for the given inputs.
        
        Args:
            inputs: Dictionary containing model inputs
            
        Returns:
            String hash of the inputs
        """
        # Sort the inputs to ensure consistent hashing
        sorted_data = json.dumps(data, sort_keys=True)
        return hashlib.sha256(sorted_data.encode()).hexdigest()

    def generate_cache_key(
        self, model_input, check_name: str, model: str
    ) -> str:
        """Generate a cache key for model input.
        
        Args:
            model_input: The ModelInput object containing example, prompt, and 
                schema
            check_name: Name of the check being processed
            model: The model being used for generation
            
        Returns:
            String hash of the cache key data
        """
        cache_key_data = {
            "content_hash": model_input.example.get_content_hash(),
            "prompt": model_input.prompt,
            "schema": model_input.schema,
            "check_name": check_name,
            "model": model,
        }
        return self._generate_cache_key(cache_key_data)

    def get_cached_output(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get cached output for given cache key if it exists.
        
        Args:
            cache_key: String hash of the cache key data
            
        Returns:
            Cached output if found, None otherwise
        """
        cache_file = self.cache_dir / f"{cache_key}.json"

        if cache_file.exists():
            with open(cache_file, "r", encoding="utf-8") as f:
                return json.load(f)
        return None

    def cache_output(
        self,
        cache_key: str,
        data: Dict[str, Any],
        metadata: Dict[str, Any] = {}
    ) -> None:
        """Cache model output with metadata.

        Args:
            cache_key: String hash of the cache key data
            data: Model output to cache
            metadata: Additional metadata about the model call
        """
        cache_file = self.cache_dir / f"{cache_key}.json"

        cache_entry = {
            "data": data,
            "metadata": {
                **metadata,
                "cached_at": datetime.now().isoformat()
            }
        }

        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(cache_entry, f, indent=2, ensure_ascii=False)