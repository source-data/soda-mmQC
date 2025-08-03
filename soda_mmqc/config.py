import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

DEVICE = os.getenv("DEVICE", "cpu")

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
