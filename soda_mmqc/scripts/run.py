import os
import json
import argparse
import logging
from pathlib import Path
from tqdm import tqdm
from soda_mmqc.model_api import generate_response
from soda_mmqc.evaluation import evaluate_response
from soda_mmqc.config import (
    get_checklist_path,
    get_prompt_path,
    get_figure_path,
    get_evaluation_path,
    get_expected_output_relative_path,
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
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def process_check(check_data, results_dir):
    check_name = check_data["name"]
    prompt_name = Path(check_data["prompt_path"]).stem  # Get name without extension
    metrics = check_data["metrics"]
    examples = check_data["examples"]

    logger.info(f"Processing check: {check_name}")
    logger.info(f"Found {len(examples)} examples to process")

    # Get prompt path from config
    prompt_path = get_prompt_path(prompt_name)
    if not prompt_path.exists():
        logger.error(f"Prompt file not found: {prompt_path}")
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")

    with open(prompt_path, "r", encoding="utf-8") as f:
        prompt_template = f.read()

    results = []
    for example in tqdm(
        examples, 
        desc=f"Processing {check_name}", 
        unit="example"
    ):
        # Get figure path from config
        figure_path = get_figure_path(example["doi"], example["figure_id"])
        expected_output_path = figure_path / get_expected_output_relative_path(check_name)

        logger.debug(f"Processing figure: {example['doi']}/{example['figure_id']}")

        # Read caption
        with open(figure_path / "caption.txt", "r", encoding="utf-8") as f:
            caption = f.read().strip()

        # Read expected output
        with open(expected_output_path, "r", encoding="utf-8") as f:
            expected_output = f.read().strip()

        # Find image file
        image_path = None
        for ext in [".png", ".jpg", ".jpeg", ".tiff"]:
            for img_file in figure_path.glob(f"*{ext}"):
                image_path = img_file
                break
            if image_path:
                break

        if not image_path:
            logger.error(f"No image found for {example['doi']}/{example['figure_id']}")
            raise ValueError(f"No image found for {example['doi']}/{example['figure_id']}")

        # Generate model response
        model_input = {
            "image": str(image_path),
            "caption": caption,
            "prompt": prompt_template
        }
        model_output = generate_response(model_input)

        # Evaluate response
        evaluation_results = evaluate_response(
            model_output, expected_output, metrics
        )

        # Prepare result
        result = {
            "doi": example["doi"],
            "figure_id": example["figure_id"],
            "expected_output": expected_output,
            "model_output": model_output,
            "evaluation": evaluation_results
        }

        results.append(result)

    # Save results
    results_path = Path(results_dir) / check_name
    os.makedirs(results_path, exist_ok=True)
    results_file = results_path / "metrics.json"

    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)

    logger.info(f"Saved results for {check_name} to {results_file}")


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Run figure caption quality checks"
    )
    parser.add_argument(
        "--checklist", 
        "-c",
        help="Name of the checklist subfolder to process (e.g., "
             "'minimal-requirement-for-figure-caption'). "
             "If not provided, all checklists will be processed.",
        default=None
    )
    parser.add_argument(
        "--check",
        help="Name of the specific check to process (without .json extension). "
             "If not provided, all checks in the checklist will be processed.",
        default=None
    )
    parser.add_argument(
        "--results-dir",
        "-r",
        help="Directory to save results (default: evaluation/results)",
        default="evaluation/results"
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
    logger.info(f"Results will be saved to: {results_dir}")
    # Get all checklist files
    checklist_files = list_checklist_files(args.checklist)
    
    if not checklist_files:
        logger.error("No checklist files found")
        return

    # Process checklist files
    for checklist_name, check_files in checklist_files.items():
        logger.info(f"Processing checklist: {checklist_name}")
        
        # Filter for specific check if requested
        if args.check:
            check_files = [f for f in check_files if f.stem == args.check]
            if not check_files:
                logger.error(f"Check '{args.check}' not found in checklist '{checklist_name}'")
                continue
        
        # Process each check file
        for check_file in tqdm(
            check_files, 
            desc=f"Processing checks in {checklist_name}", 
            unit="check"
        ):
            check_data = load_json(check_file)
            process_check(check_data, results_dir)


if __name__ == "__main__":
    main()
