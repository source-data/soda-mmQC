import os
from pathlib import Path
from typing import List, Dict

# Get the package root directory
PACKAGE_ROOT = Path(__file__).parent

# Base data directory - can be overridden by environment variable
DATA_DIR = Path(os.getenv("SODA_MMQC_DATA_DIR", PACKAGE_ROOT / "data"))

# Subdirectories
CHECKLIST_DIR = DATA_DIR / "checklist"
FIGURE_DIR = DATA_DIR / "figure"
PROMPT_DIR = DATA_DIR / "prompt"
EVALUATION_DIR = DATA_DIR / "evaluation"
EXPECTED_OUTPUT_SUBDIR = "checks"


# Ensure all directories exist
for directory in [CHECKLIST_DIR, FIGURE_DIR, PROMPT_DIR, EVALUATION_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


def get_checklist_path(checklist_name: str, check_name: str) -> Path:
    """Get the full path to a checklist file.
    
    Args:
        checklist_name: Name of the checklist subfolder (e.g., 'minimal-requirement-for-figure-caption')
        check_name: Name of the specific check JSON file (without .json extension)
    """
    return CHECKLIST_DIR / checklist_name / f"{check_name}.json"


def list_checklist_files(checklist_name: str = None) -> Dict[str, List[Path]]:
    """List all checklist JSON files.
    
    Args:
        checklist_name: Optional name of a specific checklist subfolder to list.
                       If None, lists all checklist subfolders and their JSON files.
    
    Returns:
        Dictionary mapping checklist names to lists of their JSON files
    """
    if checklist_name:
        checklist_path = CHECKLIST_DIR / checklist_name
        if not checklist_path.exists():
            return {}
        return {checklist_name: list(checklist_path.glob("*.json"))}
    
    # List all checklist subfolders and their JSON files
    result = {}
    for checklist_dir in CHECKLIST_DIR.iterdir():
        if checklist_dir.is_dir():
            result[checklist_dir.name] = list(checklist_dir.glob("*.json"))
    return result


def get_prompt_path(prompt_name: str) -> Path:
    """Get the full path to a prompt file."""
    return PROMPT_DIR / f"{prompt_name}.txt"


def get_figure_path(doi: str, figure_id: str) -> Path:
    """Get the full path to a figure directory."""
    return FIGURE_DIR / doi / figure_id


def get_expected_output_path(doi: str, figure_id: str, check_name: str) -> Path:
    """Get the relative path to expected results for a check."""
    fig_path = get_figure_path(doi, figure_id)
    return fig_path / EXPECTED_OUTPUT_SUBDIR / check_name / "expected_output.json"


def get_evaluation_path(checklist_name: str) -> Path:
    """Get the full path to evaluation results for a checklist."""
    return EVALUATION_DIR / checklist_name