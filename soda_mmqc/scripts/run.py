import os
import json
import hashlib
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Any
from tqdm import tqdm
from soda_mmqc.model_api import generate_response
from soda_mmqc.model_cache import ModelCache
from soda_mmqc.config import (
    get_content_path,
    get_expected_output_path,
    get_cache_path,
    get_checklist,
    list_checks,
    get_evaluation_path
)
from soda_mmqc.evaluation import evaluate_response


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


def gather_examples(check_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Step 1: Gather all necessary inputs and validate file paths.
    
    Args:
        check_data: The checklist data containing examples and prompt info
        
    Returns:
        List of dictionaries containing all necessary inputs for model processing
    """
    check_name = check_data["name"]
    examples = check_data["examples"]

    logger.info(f"Gathering inputs for check: {check_name}")
    logger.info(f"Found {len(examples)} examples to process")

    # Gather and validate all inputs
    inputs = []
    for example in tqdm(examples, desc="Validating inputs", unit="example"):
        
        doi = example["doi"]
        figure_id = example["figure_id"]

        # Get and validate paths
        content_path = get_content_path(example)

        # Validate caption file
        caption_path = content_path / "caption.txt"
        if not caption_path.exists():
            logger.error(f"Caption file not found: {caption_path}")
            raise FileNotFoundError(f"Caption file not found: {caption_path}")
        # Read caption and expected output
        with open(caption_path, "r", encoding="utf-8") as f:
            caption = f.read().strip()

        # Find and validate image file
        image_path = None
        for ext in [".png", ".jpg", ".jpeg", ".tiff"]:
            for img_file in content_path.glob(f"*{ext}"):
                image_path = img_file
                break
        # we delay the loading and encoding of the image until the last moment when call the model
        # but we keep a hash so that it is part of the cache key
        if not image_path:
            logger.error(f"No image found for {example['doi']}/{example['figure_id']}")
            raise ValueError(f"No image found for {example['doi']}/{example['figure_id']}")
        with open(image_path, "rb") as f:
            img_hash = hashlib.sha256(f.read()).hexdigest()

        expected_output_path = get_expected_output_path(example, check_name)
        # Validate expected output file
        if not expected_output_path.exists():
            logger.error(
                f"Expected output file not found: {expected_output_path}"
            )
            raise FileNotFoundError(
                f"Expected output file not found: {expected_output_path}"
            )
        expected_output = load_json(expected_output_path)

        # Store all inputs
        inputs.append({
            "doi": doi,
            "figure_id": figure_id,
            "caption": caption,
            "image_path": str(image_path),
            "img_hash": img_hash,
            "expected_output": expected_output,
        })

    return inputs


def run_model(
    inputs: Dict[str, Any], 
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
    cache_dir = get_cache_path()
    model_cache = ModelCache(cache_dir)

    results = []

    check_data = inputs["check_data"]
    check_name = check_data["name"]
    examples = inputs["examples"]
    prompt = inputs["prompt"]
    schema = inputs["schema"]

    for example in tqdm(examples, desc="Running model", unit="example"):
        try:
            if mock:
                # In mock mode, use expected output as model output
                model_output = example["expected_output"]
            else:
                # Check cache first if enabled
                if use_cache:
                    # the cache key needs to be unique with respect to 
                    # the example, the prompt, the schema, and the check name
                    data_for_cache_key = {
                        "example": example,
                        "prompt": prompt,
                        "schema": schema,
                        "check_name": check_name,
                    }
                    cached_result = model_cache.get_cached_output(data_for_cache_key)
                    if cached_result:
                        logger.debug(
                            "Using cached output for "
                            f"{example['doi']}/{example['figure_id']}"
                        )
                        # cache entries have a data and metadata field
                        # we only need the data field
                        model_output = cached_result["data"]
                    else:
                        # Generate new output if not cached
                        model_input = {
                            "prompt": prompt,
                            "schema": schema,
                            "image_path": example["image_path"],
                            "caption": example["caption"],
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
                                f"{example['doi']}: {str(e)}"
                            )
                            raise
                        # Cache the new output
                        model_cache.cache_output(
                            data_for_cache_key,
                            data=model_output,
                            metadata={
                                "doi": example["doi"],
                                "figure_id": example["figure_id"],
                                "check_name": check_name
                            }
                        )
                else:
                    # Generate new output without caching
                    model_input = {
                        "prompt": prompt,
                        "schema": schema,
                        "image_path": example["image_path"],
                        "caption": example["caption"],
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
                            f"{example['doi']}: {str(e)}"
                        )
                        raise

            # Accumulate result
            results.append({
                "doi": example["doi"],
                "figure_id": example["figure_id"],
                "expected_output": example["expected_output"],
                "model_output": model_output
            })

        except Exception as e:
            logger.error(
                f"Error processing example "
                f"{example['doi']}/{example['figure_id']}: {str(e)}"
            )
            # Continue with next example instead of failing completely
            continue

    return results


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


def save_analysis(
    analyzed_results: List[Dict[str, Any]],
    checklist_name: str,
    check_name: str
):
    """Save the analysis results to a file.
    
    Args:
        analyzed_results: List of dictionaries containing analysis results
        checklist_name: Name of the checklist
        check_name: Name of the check
    """
    
    # Save analysis results
    try:
        analysis_path = get_evaluation_path(checklist_name) / check_name
        os.makedirs(analysis_path, exist_ok=True)
        analysis_file = analysis_path / "analysis.json"

        with open(analysis_file, "w", encoding="utf-8") as f:
            json.dump(analyzed_results, f, indent=4, ensure_ascii=False)

        logger.info(f"Saved analysis for {check_name} to {analysis_file}")
    except Exception as e:
        logger.error(
            f"Error saving analysis results for {check_name}: {str(e)}"
        )
        logger.debug("Save exception details:", exc_info=True)
        raise


def process_check(
    check_dir: Path,
    mock: bool = False,
    use_cache: bool = True
) -> List[Dict[str, Any]]:
    """Process a single check through two steps:
    1. Gather and validate all inputs
    2. Run the model on all inputs and cache outputs
    
    Args:
        check_data: The checklist data
        cache_dir: Directory to save cached outputs
        mock: If True, use expected outputs as model outputs (no API calls)
        use_cache: If True, use cached outputs when available
    """
    check_data = load_json(check_dir/'benchmark.json')
    check_name = check_data["name"]
    with open(check_dir/'prompt.txt', 'r') as f:
        prompt = f.read()
    schema = load_json(check_dir/'schema.json')
    try:
        # Step 1: Gather examples
        examples = gather_examples(check_data)

        inputs = {
            "check_data": check_data,  # will be part of the hash key
            "examples": examples,
            "prompt": prompt,
            "schema": schema
        }

        # Step 2: Run model and cache outputs
        results = run_model(
            inputs,
            mock=mock,
            use_cache=use_cache
        )

        # Step 3: Evaluate results
        metrics = check_data["metrics"]
        analyzed_results = analyze_results(results, metrics)
        return analyzed_results

    except Exception as e:
        logger.error(f"Error processing check {check_name}: {str(e)}")
        raise


def process_checklist(
    checklist_dir: Path,
    checklist_name: str,
    mock: bool = False,
    use_cache: bool = True
):
    """Process all checks in a checklist."""

    # Get all checks frmo the checklist
    checks = list_checks(checklist_dir)

    if not checks:
        logger.error(
            f"Checklist not found: {checklist_name}"
        )
        return

    for check_dir_name, check_dir in checks.items():
        try:
            analyzed_results = process_check(
                check_dir,
                mock=mock,
                use_cache=use_cache
            )
            save_analysis(analyzed_results, checklist_name, check_dir_name)

        except Exception as e:
            logger.error(
                f"Error processing check in {check_dir_name}: {str(e)}"
            )
            continue


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Run model for figure caption quality checks"
    )
    # positional argument
    parser.add_argument(
        "checklist",
        help=(
            "Name of the checklist subfolder to process (e.g., "
            "'mini'). "
            "If not provided, all checklists will be processed."
        ),
        default=None
    )
    # optional arguments
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
    if not args.no_cache:
        cache_dir = get_cache_path()
        logger.info(f"Cached outputs will be saved to: {cache_dir}")

    checklist_dir = get_checklist(args.checklist)

    process_checklist(
        checklist_dir=checklist_dir,
        checklist_name=args.checklist,
        mock=args.mock,
        use_cache=not args.no_cache
    )


if __name__ == "__main__":
    main()
