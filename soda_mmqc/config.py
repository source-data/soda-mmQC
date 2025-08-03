import os
from pathlib import Path
import torch
import logging

from dotenv import load_dotenv

load_dotenv()


# Device validation and setup
def _validate_and_setup_device() -> str:
    """Validate the requested device and return the best available device.
    
    Returns:
        str: The device string to use ('cuda', 'mps', or 'cpu')
    """
    requested_device = os.getenv("DEVICE", "cpu").lower()
    
    # Setup logging for device validation
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    if requested_device == "mps":
        if torch.backends.mps.is_available():
            logger.info(
                "✅ MPS (Metal Performance Shaders) is available "
                "and will be used"
            )
            return "mps"
        else:
            logger.warning(
                "⚠️  MPS requested but not available, falling back to CPU"
            )
            return "cpu"
    elif requested_device in ["cuda", "gpu"]:
        if torch.cuda.is_available():
            logger.info(
                f"✅ CUDA is available with {torch.cuda.device_count()} GPU(s)"
            )
            return "cuda"
        else:
            logger.warning(
                "⚠️  CUDA requested but not available, falling back to CPU"
            )
            return "cpu"
    elif requested_device == "cpu":
        logger.info("📱 CPU device selected")
        return "cpu"
    else:
        logger.warning(
            f"⚠️  Unknown device '{requested_device}' requested, "
            "falling back to CPU"
        )
        return "cpu"


DEVICE = _validate_and_setup_device()

# Get the package root directory
PACKAGE_ROOT = Path(__file__).parent

# Base data directory - can be overridden by environment variable
DATA_DIR = PACKAGE_ROOT / "data"
CACHE_DIR = Path(os.getenv("SODA_MMQC_CACHE_DIR", DATA_DIR / "cache"))

# Subdirectories
CHECKLIST_DIR = DATA_DIR / "checklist"
EXAMPLES_DIR = DATA_DIR / "examples"
EVALUATION_DIR = DATA_DIR / "evaluation"
PLOTS_DIR = DATA_DIR / "plots"

# String comparison metrics configuration
STRING_METRICS = [
    "perfect_match",
    "semantic_similarity", 
    "longest_common_subsequence"
]

# Default match threshold for string comparisons
DEFAULT_MATCH_THRESHOLD = 0.3

# SentenceTransformer model for semantic similarity
DEFAULT_SENTENCE_TRANSFORMER_MODEL = "all-MiniLM-L6-v2"
