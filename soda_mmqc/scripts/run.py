import os
import json
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Any
from tqdm import tqdm
from soda_mmqc.model_api import generate_response
from soda_mmqc.model_cache import ModelCache
from soda_mmqc.config import (
    get_prompt_path,
    get_figure_path,
    get_expected_output_path,
    list_checklist_files
)


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


def gather_inputs(check_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Step 1: Gather all necessary inputs and validate file paths.
    
    Args:
        check_data: The checklist data containing examples and prompt info
        
    Returns:
        List of dictionaries containing all necessary inputs for model processing
    """
    check_name = check_data["name"]
    prompt_name = Path(check_data["prompt_path"]).stem
    examples = check_data["examples"]

    logger.info(f"Gathering inputs for check: {check_name}")
    logger.info(f"Found {len(examples)} examples to process")

    # Get and validate prompt file
    prompt_path = get_prompt_path(prompt_name)
    if not prompt_path.exists():
        logger.error(f"Prompt file not found: {prompt_path}")
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")

    with open(prompt_path, "r", encoding="utf-8") as f:
        prompt_template = f.read()

    # Gather and validate all inputs
    inputs = []
    for example in tqdm(examples, desc="Validating inputs", unit="example"):
        doi = example["doi"]
        figure_id = example["figure_id"]
        
        # Get and validate paths
        figure_path = get_figure_path(doi, figure_id)
        expected_output_path = get_expected_output_path(
            doi, figure_id, check_name
        )
        
        logger.debug(f"Loading figure: {doi}/{figure_id}")
        
        # Validate caption file
        caption_path = figure_path / "caption.txt"
        if not caption_path.exists():
            logger.error(f"Caption file not found: {caption_path}")
            raise FileNotFoundError(f"Caption file not found: {caption_path}")
            
        # Validate expected output file
        if not expected_output_path.exists():
            logger.error(
                f"Expected output file not found: {expected_output_path}"
            )
            raise FileNotFoundError(
                f"Expected output file not found: {expected_output_path}"
            )
            
        # Find and validate image file
        image_path = None
        for ext in [".png", ".jpg", ".jpeg", ".tiff"]:
            for img_file in figure_path.glob(f"*{ext}"):
                image_path = img_file
                break
            if image_path:
                break
                
        if not image_path:
            logger.error(f"No image found for {doi}/{figure_id}")
            raise ValueError(f"No image found for {doi}/{figure_id}")
            
        # Read caption and expected output
        with open(caption_path, "r", encoding="utf-8") as f:
            caption = f.read().strip()
            
        with open(expected_output_path, "r", encoding="utf-8") as f:
            expected_output = f.read().strip()
            
        # Store all inputs
        inputs.append({
            "doi": doi,
            "figure_id": figure_id,
            "image_path": str(image_path),
            "caption": caption,
            "expected_output": expected_output,
            "prompt_template": prompt_template,
            "check_name": check_name
        })
        
    return inputs


def run_model(
    inputs: List[Dict[str, Any]], 
    mock: bool = False,
    use_cache: bool = True
) -> List[Dict[str, Any]]:
    """Step 2: Run the model on all inputs.
    
    Args:
        inputs: List of dictionaries containing model inputs
        mock: If True, use expected outputs as model outputs (no API calls)
        use_cache: If True, use cached outputs when available
        
    Returns:
        List of dictionaries containing model outputs
    """
    logger.info("Running model on all inputs")
    
    # Initialize model cache
    cache_dir = Path("evaluation/cache")
    model_cache = ModelCache(cache_dir)
    
    results = []
    for input_data in tqdm(inputs, desc="Running model", unit="example"):
        try:
            if mock:
                # In mock mode, use expected output as model output
                model_output = input_data["expected_output"]
            else:
                # Check cache first if enabled
                if use_cache:
                    cached_result = model_cache.get_cached_output(input_data)
                    if cached_result:
                        logger.debug(
                            "Using cached output for "
                            f"{input_data['doi']}"
                        )
                        model_output = cached_result["output"]
                    else:
                        # Generate new output if not cached
                        model_input = {
                            "prompt": input_data["prompt_template"],
                            "image": input_data["image_path"],
                            "caption": input_data["caption"]
                        }
                        try:
                            model_output = generate_response(model_input)
                        except KeyError as e:
                            logger.error(
                                f"Missing required key in model_input: {e}. "
                                f"Available keys: {list(model_input.keys())}"
                            )
                            raise
                        except Exception as e:
                            logger.error(
                                f"Error generating response for "
                                f"{input_data['doi']}: {str(e)}"
                            )
                            raise
                        # Cache the new output
                        model_cache.cache_output(
                            input_data,
                            {"output": model_output},
                            {
                                "doi": input_data["doi"],
                                "figure_id": input_data["figure_id"],
                                "check_name": input_data["check_name"]
                            }
                        )
                else:
                    # Generate new output without caching
                    model_input = {
                        "prompt": input_data["prompt_template"],
                        "image": input_data["image_path"],
                        "caption": input_data["caption"]
                    }
                    try:
                        model_output = generate_response(model_input)
                    except KeyError as e:
                        logger.error(
                            f"Missing required key in model_input: {e}. "
                            f"Available keys: {list(model_input.keys())}"
                        )
                        raise
                    except Exception as e:
                        logger.error(
                            f"Error generating response for "
                            f"{input_data['doi']}: {str(e)}"
                        )
                        raise
            
            # Store result
            results.append({
                "doi": input_data["doi"],
                "figure_id": input_data["figure_id"],
                "expected_output": input_data["expected_output"],
                "model_output": model_output
            })
        except Exception as e:
            logger.error(
                f"Error processing example "
                f"{input_data['doi']}/{input_data['figure_id']}: {str(e)}"
            )
            # Continue with next example instead of failing completely
            continue
        
    return results


def process_check(
    check_data: Dict[str, Any], 
    cache_dir: Path,
    mock: bool = False,
    use_cache: bool = True
) -> None:
    """Process a single check through two steps:
    1. Gather and validate all inputs
    2. Run the model on all inputs and cache outputs
    
    Args:
        check_data: The checklist data
        cache_dir: Directory to save cached outputs
        mock: If True, use expected outputs as model outputs (no API calls)
        use_cache: If True, use cached outputs when available
    """
    check_name = check_data["name"]
    
    try:
        # Step 1: Gather inputs
        inputs = gather_inputs(check_data)
        
        # Step 2: Run model and cache outputs
        run_model(inputs, mock=mock, use_cache=use_cache)
        
        logger.info(f"Completed processing for {check_name}")
        
    except Exception as e:
        logger.error(f"Error processing check {check_name}: {str(e)}")
        raise


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Run model for figure caption quality checks"
    )
    parser.add_argument(
        "--checklist", 
        "-c",
        help=(
            "Name of the checklist subfolder to process (e.g., "
            "'mini'). "
            "If not provided, all checklists will be processed."
        ),
        default=None
    )
    parser.add_argument(
        "--check",
        help=(
            "Name of the specific check to process (without .json extension). "
            "If not provided, all checks in the checklist will be processed."
        ),
        default=None
    )
    parser.add_argument(
        "--cache-dir",
        help="Directory to save cached outputs (default: evaluation/cache)",
        default="evaluation/cache"
    )
    parser.add_argument(
        "--log-level",
        "-l",
        help="Set the logging level (default: INFO)",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO"
    )
    parser.add_argument(
        "--mock",
        "-m",
        action="store_true",
        help=(
            "Run in mock mode - use expected outputs as model outputs "
            "(no API calls)"
        )
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable caching of model outputs"
    )
    args = parser.parse_args()

    # Set logging level from command line argument
    logging.getLogger().setLevel(getattr(logging, args.log_level))

    # Create cache directory
    cache_dir = Path(args.cache_dir)
    os.makedirs(cache_dir, exist_ok=True)
    logger.info(f"Cached outputs will be saved to: {cache_dir}")
    
    # Get all checklist files
    checklist_files = list_checklist_files(args.checklist)
    
    if not checklist_files:
        logger.error(
            f"No checklist files found for checklist: {args.checklist}"
        )
        return
        
    # Process each checklist file
    for checklist_name, files in checklist_files.items():
        for checklist_file in files:
            try:
                # Load checklist data
                check_data = load_json(checklist_file)
                
                # Skip if check name doesn't match filter
                if args.check and check_data["name"] != args.check:
                    continue
                    
                process_check(
                    check_data,
                    cache_dir,
                    mock=args.mock,
                    use_cache=not args.no_cache
                )
                    
            except Exception as e:
                logger.error(
                    f"Error processing checklist {checklist_file}: {str(e)}"
                )
                continue


if __name__ == "__main__":
    main()
