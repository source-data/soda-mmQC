import os
from pathlib import Path
from typing import Dict, Any, Optional

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
CONTENT_SUBDIR = "content"
EXPECTED_OUTPUT_SUBDIR = "checks"
EXPECTED_OUTPUT_FILE = "expected_output.json"
CHECK_DATA_FILE = "benchmark.json"
SCHEMA_FILE = "schema.json"
CAPTION_FILE = "caption.txt"
# Ensure all directories exist
for directory in [CHECKLIST_DIR, EXAMPLES_DIR, EVALUATION_DIR, CACHE_DIR, PLOTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


def get_checklist(checklist_name: str) -> Path:
    """Get the full path to a checklist file.
    
    Args:
        checklist_name: Name of the checklist subfolder (e.g., 'minimal-requirement-for-figure-caption')
    """
    return CHECKLIST_DIR / checklist_name


def get_check(checklist_name: str, check_name: str) -> Path:
    """Get the full path to a checklist file.
    
    Args:
        checklist_name: Name of the checklist subfolder (e.g., 'minimal-requirement-for-figure-caption')
        check_name: Name of the specific check JSON file (without .json extension)
    """
    return get_checklist(checklist_name) / check_name / CHECK_DATA_FILE


def get_check_data_file(check_dir: Path) -> Path:
    """Get the full path to a checklist file.
    
    Args:
        check_dir: Path to the check directory
    """
    return check_dir / CHECK_DATA_FILE


def get_schema_path(checklist_name: str, check_name: str) -> Path:
    """Get the full path to a schema file."""
    return get_checklist(checklist_name) / check_name / SCHEMA_FILE


def list_checks(checklist_dir: Path) -> Dict[str, Path]:
    """List all checks in a checklist."""
    # enumerate the subdirectories of the checklist directory
    checks = {}
    for check_dir in checklist_dir.iterdir():
        if check_dir.is_dir():
            checks[check_dir.name] = check_dir
    return checks


def get_figure_path(example: Dict[str, Any]) -> Path:
    """Get the full path to a figure directory."""
    return EXAMPLES_DIR / example["doi"] / example["figure_id"]


def get_content_path(example: Dict[str, Any]) -> Path:
    """Get the full path to a figure directory."""
    return get_figure_path(example) / CONTENT_SUBDIR


def get_caption_path(example: Dict[str, Any]) -> Path:
    """Get the full path to a caption file."""
    return get_content_path(example) / CAPTION_FILE


def get_image_path(example: Dict[str, Any]) -> Optional[Path]:
    """Get the full path to an image file."""
    content_path = get_content_path(example)
    image_path = None
    for ext in [".png", ".jpg", ".jpeg", ".tiff"]:
        image_files = list(content_path.glob(f"*{ext}"))
        if image_files:
            image_path = image_files[0]
            break
    return image_path


def get_expected_output_path(example: Dict[str, Any], check_name: str) -> Path:
    """Get the relative path to expected results for a check."""
    return get_figure_path(example) / EXPECTED_OUTPUT_SUBDIR / check_name / EXPECTED_OUTPUT_FILE


def get_evaluation_path(checklist_name: str) -> Path:
    """Get the full path to evaluation results for a checklist."""
    return EVALUATION_DIR / checklist_name


def get_cache_path() -> Path:
    """Get the full path to cache results."""
    return CACHE_DIR


def get_plots_path() -> Path:
    """Get the full path to plots directory."""
    return PLOTS_DIR
