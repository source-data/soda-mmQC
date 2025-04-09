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

    def get_cached_output(
        self, 
        data_for_cache_key: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Get cached output for given inputs if it exists.
        
        Args:
            data_for_cache_key: Dictionary containing data for the cache key
            
        Returns:
            Cached output if found, None otherwise
        """
        cache_key = self._generate_cache_key(data_for_cache_key)
        cache_file = self.cache_dir / f"{cache_key}.json"

        if cache_file.exists():
            with open(cache_file, "r", encoding="utf-8") as f:
                return json.load(f)
        return None

    def cache_output(
        self,
        data_for_cache_key: Dict[str, Any],
        data: Dict[str, Any],
        metadata: Dict[str, Any]
    ) -> None:
        """Cache model output with metadata.

        Args:
            inputs: Dictionary containing model inputs
            output: Model output to cache
            metadata: Additional metadata about the model call
        """
        cache_key = self._generate_cache_key(data_for_cache_key)
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