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
    get_evaluation_path,
    get_check_data_file
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

        # Store all inputs
        inputs.append({
            "doi": doi,
            "figure_id": figure_id,
            "caption": caption,
            "image_path": str(image_path),
            "img_hash": img_hash,
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
    logger.info(f"Running model on {len(inputs.get('examples', []))} inputs")

    # Initialize model cache
    cache_dir = get_cache_path()
    model_cache = ModelCache(cache_dir)

    results = []

    check_data = inputs["check_data"]
    check_name = check_data["name"]
    examples = inputs["examples"]
    prompt = inputs["prompt"]
    prompt_name = inputs["prompt_name"]
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
                        "prompt_name": prompt_name,
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
                                "check_name": check_name,
                                "prompt_name": prompt_name
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
                "example": example,
                "model_output": model_output,
                "prompt_name": prompt_name
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
    metrics: List[str],
    schema: Dict[str, Any]
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
            metrics,
            schema
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
    analyzed_results: Dict[str, List[Dict[str, Any]]],
    checklist_name: str,
    check_name: str
):
    """Save the analysis results to a file.
    
    Args:
        analyzed_results: Dictionary mapping prompt names to their analysis 
            results
        checklist_name: Name of the checklist
        check_name: Name of the check
    """
    
    # Save analysis results
    try:
        analysis_path = get_evaluation_path(checklist_name) / check_name
        os.makedirs(analysis_path, exist_ok=True)
        
        # Save results for each prompt
        for prompt_name, results in analyzed_results.items():
            # Create a subdirectory for each prompt
            prompt_dir = analysis_path / prompt_name
            os.makedirs(prompt_dir, exist_ok=True)
            
            # Save the analysis results for this prompt
            analysis_file = prompt_dir / "analysis.json"
            with open(analysis_file, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=4, ensure_ascii=False)
            
            logger.info(
                f"Saved analysis for {check_name} with prompt {prompt_name} "
                f"to {analysis_file}"
            )
            
        # Also save a summary file with all prompts
        summary_file = analysis_path / "summary.json"
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(analyzed_results, f, indent=4, ensure_ascii=False)
            
        logger.info(
            f"Saved summary for {check_name} to {summary_file}"
        )
        
    except Exception as e:
        logger.error(
            f"Error saving analysis results for {check_name}: {str(e)}"
        )
        logger.debug("Save exception details:", exc_info=True)
        raise


def prepare_check_data(check_dir: Path):
    """Prepare data for processing a check.
    
    This function consolidates common functionality for loading check data,
    schema, prompts, and gathering examples. It can be used by both
    initialize() and process_check() functions.
    
    Args:
        check_dir: Path to the check directory
        use_all_prompts: If True, return all prompts; if False, return only the first prompt
        
    Returns:
        Dictionary containing prepared data for the check, or None if preparation fails
    """
    check_dir_name = check_dir.name
    logger.info(f"Preparing data for check: {check_dir_name}")
    
    # Load check data
    check_data_file = get_check_data_file(check_dir)
    if not check_data_file.exists():
        logger.error(f"Check data file not found: {check_data_file}")
        return None
    
    try:
        check_data = load_json(check_data_file)
    except Exception as e:
        logger.error(f"Error loading check data: {str(e)}")
        return None
    
    # Load schema
    schema_file = check_dir / "schema.json"
    if not schema_file.exists():
        logger.error(f"Schema file not found: {schema_file}")
        return None
    
    try:
        schema = load_json(schema_file)
    except Exception as e:
        logger.error(f"Error loading schema: {str(e)}")
        return None
    
    # Get all prompts from the prompts directory
    prompts_dir = check_dir / "prompts"
    if not prompts_dir.exists():
        logger.error(f"Prompts directory not found: {prompts_dir}")
        return None
    
    # Get all prompt files
    prompt_files = list(prompts_dir.glob("*.txt"))
    if not prompt_files:
        logger.error(f"No prompt files found in {prompts_dir}")
        return None
    
    # Sort the prompt files by name in ascending order
    prompt_files.sort()
    
    # Load prompts
    prompts = {}
    for prompt_file in prompt_files:
        prompt_name = prompt_file.stem
        with open(prompt_file, "r", encoding="utf-8") as f:
            prompts[prompt_name] = f.read()
    
    # Get examples from benchmark.json
    examples = check_data.get("examples", [])
    if not examples:
        logger.warning(f"No examples found in check: {check_dir_name}")
        return None
    
    # Use the existing gather_examples function
    try:
        gathered_examples = gather_examples(check_data)
    except Exception as e:
        logger.error(f"Error gathering examples: {str(e)}")
        return None
    
    if not gathered_examples:
        logger.warning(
            f"No valid examples gathered for check: {check_dir_name}"
        )
        return None
    
    # Return prepared data
    return {
        "check_dir_name": check_dir_name,
        "check_data": check_data,
        "schema": schema,
        "prompts": prompts,
        "examples": gathered_examples
    }


def process_check(
    check_dir: Path,
    mock: bool = False,
    use_cache: bool = True
) -> Dict[str, List[Dict[str, Any]]]:
    """Process a single check through two steps:
    1. Gather and validate all inputs
    2. Run the model on all inputs and cache outputs
    
    Args:
        check_dir: Path to the check directory
        mock: If True, use expected outputs as model outputs (no API calls)
        use_cache: If True, use cached outputs when available
        
    Returns:
        Dictionary mapping prompt names to their analysis results
    """
    
    def get_expected_output(example: Dict[str, Any], check_name: str) -> Dict[str, Any]:
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
        return expected_output
    
    # Prepare check data
    prepared_data = prepare_check_data(check_dir)
    if not prepared_data:
        return {}

    # check_dir_name = prepared_data["check_dir_name"]
    check_data = prepared_data["check_data"]
    check_name = check_data["name"]
    metrics = check_data["metrics"]
    schema = prepared_data["schema"]
    prompts = prepared_data["prompts"]
    examples = prepared_data["examples"]

    # Dictionary to store results for each prompt
    all_results = {}
    
    # Process each prompt
    for prompt_name, prompt in prompts.items():
        logger.info(f"Processing prompt: {prompt_name}")
        
        # Prepare inputs for run_model
        inputs = {
            "check_data": check_data,
            "examples": examples,
            "prompt": prompt,
            "schema": schema,
            "prompt_name": prompt_name
        }

        # Run model and cache outputs
        results = run_model(
            inputs,
            mock=mock,
            use_cache=use_cache
        )
        
        # Evaluate results

        # add expected outputs to results
        for result in results:
            example = result["example"]
            expected_output = get_expected_output(example, check_name)
            result["expected_output"] = expected_output

        analyzed_results = analyze_results(results, metrics, schema)

        # Store results for this prompt
        all_results[prompt_name] = analyzed_results

    return all_results


def initialize(checklist_dir: Path, use_cache: bool = True):
    """Initialize expected_output.json files for all examples in a checklist.
    
    This function iterates through the checks of a checklist. For each check,
    it uses the prompts in prompts/ and the schema.json to execute the check
    using the image and caption of each example. The output becomes the
    preliminary expected_output.json which is written in the
    <doi>/checks/<check_name>/ folder.
    
    Args:
        checklist_dir: Path to the checklist directory
        use_cache: If True, use cached outputs when available
    """
    
    def create_expected_output(example: Dict[str, Any], check_name: str) -> Path:
        doi = example["doi"]
        figure_id = example["figure_id"]
        model_output = example["model_output"]
        # Create expected output directory
        example = {"doi": doi, "figure_id": figure_id}
        expected_output_path = get_expected_output_path(
            example, check_name
        )
        expected_output_dir = expected_output_path.parent
        expected_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Check if expected output already exists
        if expected_output_path.exists():
            logger.info(
                f"Expected output already exists: {expected_output_path}"
            )
            return expected_output_path
        
        # Write expected output
        with open(expected_output_path, "w", encoding="utf-8") as f:
            json.dump(model_output, f, indent=4, ensure_ascii=False)
        
        logger.info(
            f"Created expected output: {expected_output_path}"
        )
        return expected_output_path

    logger.info(f"Initializing expected outputs for checklist: {checklist_dir.name}")
    
    # Get all checks from the checklist
    checks = list_checks(checklist_dir)
    if not checks:
        logger.error(f"No checks found in checklist: {checklist_dir.name}")
        return
    
    # Process each check
    for check_dir_name, check_dir in checks.items():
        logger.info(f"Processing check: {check_dir_name}")
        
        # Prepare check data (use only the first prompt)
        prepared_data = prepare_check_data(check_dir)
        if not prepared_data:
            continue
        
        check_dir_name = prepared_data["check_dir_name"]
        check_data = prepared_data["check_data"]
        check_name = check_data["name"]
        schema = prepared_data["schema"]
        prompts = prepared_data["prompts"]
        examples = prepared_data["examples"]
        
        # Get the first prompt for initialization
        # they are sorted alphabetically
        prompt_name = list(prompts.keys())[0]
        prompt = prompts[prompt_name]
        
        # Prepare inputs for run_model
        inputs = {
            "check_data": check_data,
            "examples": examples,
            "prompt": prompt,
            "schema": schema,
            "prompt_name": prompt_name
        }
        
        # Run the model
        try:
            results = run_model(
                inputs,
                mock=False,
                use_cache=use_cache
            )
            
            # Write expected outputs
            for result in results:
                try:
                    create_expected_output(result, check_name)
                except Exception as e:
                    logger.error(
                        f"Error writing expected output for "
                        f"{result['doi']}/{result['figure_id']}: {str(e)}"
                    )
                    continue

        except Exception as e:
            logger.error(
                f"Error running model for check {check_dir_name}: {str(e)}"
            )
            continue


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
    parser = argparse.ArgumentParser(description="Run MMQC checks on examples")
    parser.add_argument("checklist", type=str, help="Name of the checklist to process")
    parser.add_argument("--initialize", action="store_true", help="Initialize expected output files")
    parser.add_argument("--mock", action="store_true", help="Use mock responses instead of calling the model")
    parser.add_argument("--no-cache", action="store_true", help="Disable caching of model responses")
    parser.add_argument("--check", type=str, help="Name of the check to process")
    args = parser.parse_args()

    # Get the checklist directory using the config function
    checklist_dir = get_checklist(args.checklist)
    if not checklist_dir.exists():
        logger.error(f"Checklist directory not found: {checklist_dir}")
        return

    # Initialize if requested
    if args.initialize:
        initialize(checklist_dir, not args.no_cache)
        return
        
    if args.check:
        # Find the check in the checklist
        check_dir = checklist_dir / args.check
        if check_dir.exists():
            process_check(check_dir, args.mock, not args.no_cache)
        else:
            logger.error(f"Check not found: {args.check}")
    else:
        # Process the entire checklist
        process_checklist(checklist_dir, args.checklist, args.mock, not args.no_cache)


def initialize_main():
    """Entry point for the initialize command that passes --initialize to main."""
    import sys
    sys.argv.append("--initialize")
    main()


def launch_curation():
    """Launch the curation interface using streamlit."""
    import sys
    import os
    from pathlib import Path
    
    # Set environment variables before importing streamlit
    os.environ["STREAMLIT_SERVER_RUN_ON_SAVE"] = "false"
    os.environ["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"
    os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"  # Disable file watching completely
    
    # Import streamlit after environment is configured
    import streamlit.web.cli as stcli
    
    # Get the path to the benchmark_curation.py file
    workspace_root = Path(__file__).resolve().parent.parent
    curation_script = workspace_root / "tools" / "benchmark_curation.py"
    
    # Prepare streamlit arguments
    sys.argv = [
        "streamlit",
        "run",
        str(curation_script),
        "--server.headless=true",
        "--server.address=localhost",
        "--server.port=8501",
        "--server.enableCORS=false",
        "--server.enableXsrfProtection=false",
        "--global.developmentMode=false",
        "--server.fileWatcherType=none"  # Disable file watching via CLI as well
    ]
    
    # Run streamlit
    stcli.main()


if __name__ == "__main__":
    main()
