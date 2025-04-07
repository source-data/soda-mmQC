import os
import json
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Any
from tqdm import tqdm
from soda_mmqc.evaluation import evaluate_response
from soda_mmqc.config import list_checklist_files


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def load_json(file_path):
    """Load JSON data from a file."""
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def analyze_results(
    results: List[Dict[str, Any]], 
    metrics: List[str]
) -> List[Dict[str, Any]]:
    """Analyze model outputs against expected outputs using specified metrics.
    
    Args:
        results: List of dictionaries containing model outputs
        metrics: List of metrics to compute
        
    Returns:
        List of dictionaries containing analysis results
    """
    logger.info("Analyzing all results")
    
    analyzed_results = []
    for result in tqdm(results, desc="Analyzing results", unit="example"):
        # Analyze response
        analysis_results = evaluate_response(
            result["model_output"],
            result["expected_output"],
            metrics
        )
        
        # Store analyzed result
        analyzed_results.append({
            "doi": result["doi"],
            "figure_id": result["figure_id"],
            "expected_output": result["expected_output"],
            "model_output": result["model_output"],
            "analysis": analysis_results
        })
        
    return analyzed_results


def process_check_analysis(
    check_data: Dict[str, Any], 
    results_dir: Path,
    cache_dir: Path
) -> None:
    """Process analysis for a single check:
    1. Load cached model outputs
    2. Analyze all results
    
    Args:
        check_data: The checklist data
        results_dir: Directory to save analysis results
        cache_dir: Directory containing cached model outputs
    """
    check_name = check_data["name"]
    metrics = check_data["metrics"]
    
    logger.info(f"Processing check: {check_name}")
    logger.debug(f"Check data: {json.dumps(check_data, indent=2)}")
    
    try:
        # Load cached outputs
        # First try the expected directory structure
        cache_path = cache_dir / check_name
        logger.info(f"Looking for cached outputs in: {cache_path}")
        
        if cache_path.exists():
            # Find all cache files for this check in the subdirectory
            cache_files = list(cache_path.glob("*.json"))
            logger.info(f"Found {len(cache_files)} cache files in {cache_path}")
        else:
            # If the subdirectory doesn't exist, look for cache files in the main cache directory
            # that match the check name in their metadata
            logger.info(f"Cache subdirectory not found, searching in main cache directory")
            cache_files = []
            
            # Load all cache files and filter by check name
            for cache_file in cache_dir.glob("*.json"):
                try:
                    with open(cache_file, "r", encoding="utf-8") as f:
                        cache_data = json.load(f)
                        if cache_data.get("metadata", {}).get("check_name") == check_name:
                            cache_files.append(cache_file)
                except Exception as e:
                    logger.warning(f"Error reading cache file {cache_file}: {str(e)}")
                    continue
            
            logger.info(f"Found {len(cache_files)} cache files for {check_name} in main cache directory")
        
        if not cache_files:
            logger.error(f"No cache files found for {check_name}")
            return
            
        # Load all cached outputs
        results = []
        for cache_file in cache_files:
            try:
                logger.debug(f"Loading cache file: {cache_file}")
                cache_data = load_json(cache_file)
                results.append({
                    "doi": cache_data["metadata"]["doi"],
                    "figure_id": cache_data["metadata"]["figure_id"],
                    "expected_output": cache_data["inputs"]["expected_output"],
                    "model_output": cache_data["output"]
                })
            except Exception as e:
                logger.error(
                    f"Error loading cache file {cache_file}: {str(e)}"
                )
                logger.debug("Exception details:", exc_info=True)
                continue
            
        logger.info(f"Successfully loaded {len(results)} results for {check_name}")
        
        # Analyze results
        try:
            analyzed_results = analyze_results(results, metrics)
            logger.info(
                f"Successfully analyzed {len(analyzed_results)} results for "
                f"{check_name}"
            )
        except Exception as e:
            logger.error(
                f"Error during analysis for {check_name}: {str(e)}"
            )
            logger.debug("Analysis exception details:", exc_info=True)
            raise
        
        # Save analysis results
        try:
            analysis_path = Path(results_dir) / check_name
            os.makedirs(analysis_path, exist_ok=True)
            analysis_file = analysis_path / "analysis.json"
            
            with open(analysis_file, "w", encoding="utf-8") as f:
                json.dump(analyzed_results, f, indent=4)
                
            logger.info(f"Saved analysis for {check_name} to {analysis_file}")
        except Exception as e:
            logger.error(
                f"Error saving analysis results for {check_name}: {str(e)}"
            )
            logger.debug("Save exception details:", exc_info=True)
            raise
        
    except Exception as e:
        logger.error(f"Error processing check {check_name}: {str(e)}")
        logger.debug("Full exception details:", exc_info=True)
        raise


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Analyze model outputs for figure caption quality checks"
    )
    parser.add_argument(
        "--checklist", 
        "-c",
        help=(
            "Name of the checklist subfolder to analyze (e.g., "
            "'minimal-requirement-for-figure-caption'). "
            "If not provided, all checklists will be analyzed."
        ),
        default=None
    )
    parser.add_argument(
        "--check",
        help=(
            "Name of the specific check to analyze (without .json extension). "
            "If not provided, all checks in the checklist will be analyzed."
        ),
        default=None
    )
    parser.add_argument(
        "--results-dir",
        "-r",
        help="Directory to save analysis results (default: evaluation/analysis)",
        default="evaluation/analysis"
    )
    parser.add_argument(
        "--cache-dir",
        help="Directory containing cached model outputs (default: evaluation/cache)",
        default="evaluation/cache"
    )
    parser.add_argument(
        "--log-level",
        "-l",
        help="Set the logging level (default: INFO)",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO"
    )
    args = parser.parse_args()

    # Set logging level from command line argument
    logging.getLogger().setLevel(getattr(logging, args.log_level))

    # Create results directory
    results_dir = Path(args.results_dir)
    os.makedirs(results_dir, exist_ok=True)
    logger.info(f"Analysis results will be saved to: {results_dir}")
    
    # Get cache directory
    cache_dir = Path(args.cache_dir)
    if not cache_dir.exists():
        logger.error(f"Cache directory not found: {cache_dir}")
        return
        
    # Get all checklist files
    checklist_files_dict = list_checklist_files(args.checklist)
    
    if not checklist_files_dict:
        logger.error(f"No checklist files found for checklist: {args.checklist}")
        return
        
    # Process each checklist file
    for checklist_name, checklist_files in checklist_files_dict.items():
        logger.info(f"Processing checklist: {checklist_name}")
        for checklist_file in checklist_files:
            try:
                # Load checklist data - each file represents a single check
                check_data = load_json(checklist_file)
                
                # Skip if check name doesn't match filter
                if args.check and check_data["name"] != args.check:
                    continue
                    
                # Process the check
                process_check_analysis(
                    check_data,
                    results_dir,
                    cache_dir
                )
                    
            except Exception as e:
                logger.error(
                    f"Error processing checklist {checklist_file}: {str(e)}"
                )
                logger.debug("Exception details:", exc_info=True)
                continue


if __name__ == "__main__":
    main() 