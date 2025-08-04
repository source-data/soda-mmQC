import os
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from tqdm import tqdm
from soda_mmqc.lib.api import generate_response
from soda_mmqc.lib.cache import ModelCache
from soda_mmqc.config import (
    CHECKLIST_DIR,
    CACHE_DIR,
    EVALUATION_DIR,
    STRING_METRICS,
    DEFAULT_MATCH_THRESHOLD,
    DEFAULT_SENTENCE_TRANSFORMER_MODEL,
    DEFAULT_MODEL,
    API_PROVIDER,
)
from soda_mmqc.lib.api import validate_model_for_provider, get_compatible_models
from soda_mmqc.core.evaluation import JSONEvaluator
from soda_mmqc import logger
from soda_mmqc.core.examples import EXAMPLE_FACTORY, Example


@dataclass
class CheckData:
    """Container for all data needed to process a single check.
    
    This dataclass holds the essential information required to run a check
    against a set of examples, including the schema, examples, and metadata.
    
    Attributes:
        check_dir_name: Directory name of the check (e.g., 
            "error-bars-defined")
        check_name: Human-readable name of the check (e.g., 
            "Error Bars Defined")
        schema: JSON schema defining the expected output structure
        examples: List of Example instances
        expected_outputs: List of expected outputs for each example
    """
    check_dir_name: str
    check_name: str
    schema: Dict[str, Any]
    examples: List[Example]
    expected_outputs: List[Dict[str, Any]]


@dataclass
class ModelInput:
    """Container for model input data.
    
    Holds all the data needed to generate a response from the model,
    including the example, prompt, and expected output schema.
    
    Attributes:
        example: The Example instance containing image and caption data
        prompt: The prompt template to send to the model
        schema: JSON schema defining the expected structured output format
    """
    example: Example
    prompt: str
    schema: Dict[str, Any]


@dataclass
class ModelResult:
    """Container for a single model evaluation result.
    
    Attributes:
        doc_id: The document identifier for the example (e.g., 
            "10.1038/emboj.2009.312")
        model_output: The raw structured output from the model API
    """
    doc_id: str | None
    model_output: Dict[str, Any]
    metadata: Dict[str, Any]


def load_json(file_path):
    """Load JSON data from a file."""
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def run_model(
    check_data: CheckData,
    prompt: str,
    use_cache: bool = True,
    model: str = DEFAULT_MODEL
) -> List[ModelResult]:
    """Step 2: Run the model on all inputs.
    
    Args:
        inputs: Dictionary containing:
            - check_data: The checklist data
            - examples: List of Example instances
            - prompt: The prompt to use
            - schema: The schema for structured output
            - prompt_name: Name of the prompt being used
        use_cache: If True, use cached outputs when available
        mock: If True, use expected outputs as model outputs (no API calls)
        model: The model to use for generation
        
    Returns:
        List of dictionaries containing model outputs
    """
    logger.info(f"Running model on {len(check_data.examples)} inputs")

    # Initialize model cache
    cache_dir = CACHE_DIR
    model_cache = ModelCache(cache_dir)

    results = []

    check_name = check_data.check_name
    examples = check_data.examples
    schema = check_data.schema

    for example in tqdm(examples, desc="Running model", unit="example"):
        try:
            # Generate new output without caching
            model_input = ModelInput(
                example=example,
                prompt=prompt,
                schema=schema
            )
            input_metadata = {
                "doc_id": example.doc_id,
                "source": example.relative_source_path,
                "example_type": example.example_class_name
            }
            # Check cache first if enabled
            if use_cache:
                cache_key = model_cache.generate_cache_key(
                    model_input, check_name, model
                )
                cached_result = model_cache.get_cached_output(cache_key)
                if cached_result:
                    logger.debug(
                        "Using cached output for "
                        f"{example.doc_id}"
                    )
                    model_output = cached_result["data"]
                    response_metadata = cached_result["metadata"]
                else:
                    try:
                        model_output, response_metadata = generate_response(
                            model_input,
                            model=model,
                            metadata=input_metadata
                        )
                    except Exception as e:
                        logger.error(
                            f"Error generating response for "
                            f"{example.doc_id}: {str(e)}"
                        )
                        raise
                    # Cache the new output
                    model_cache.cache_output(
                        cache_key,
                        data=model_output,
                        metadata=response_metadata
                    )
            else:
                try:
                    model_output, response_metadata = generate_response(
                        model_input,
                        model=model,
                        metadata=input_metadata
                    )
                except Exception as e:
                    logger.error(
                        f"Error generating response for "
                        f"{example.doc_id}: {str(e)}"
                    )
                    raise

            # Accumulate result
            results.append(ModelResult(
                doc_id=example.doc_id,
                model_output=model_output,
                metadata=response_metadata
            ))

        except Exception as e:
            logger.error(
                f"Error processing example "
                f"{example.doc_id}: {str(e)}"
            )
            # Continue with next example instead of failing completely
            continue
    return results


def analyze_results(
    results: List[ModelResult],
    schema: Dict[str, Any],
    expected_outputs: List[Dict[str, Any]],
    match_threshold: float = DEFAULT_MATCH_THRESHOLD,
    sentence_transformer_model: str = (
        DEFAULT_SENTENCE_TRANSFORMER_MODEL
    )
) -> Dict[str, List[Dict[str, Any]]]:
    """Analyze model outputs against expected outputs using all string metrics.
    
    Args:
        results: List of ModelResult objects containing model outputs
        schema: Dict[str, Any] for the check
        expected_outputs: List of expected outputs
        match_threshold: Threshold for considering a match (0-1)
    Returns:
        Dictionary mapping string metric names to analysis results
    """
    logger.info("Analyzing all results with all string metrics")

    # Dictionary to store results for each string metric
    all_metric_results = {}
    
    # Run evaluation for each string metric
    for string_metric in STRING_METRICS:
        logger.info(f"Running analysis with metric: {string_metric}")
        
        analyzed_results = []
        evaluator = JSONEvaluator(
            schema, 
            string_metric=string_metric, 
            match_threshold=match_threshold,
            sentence_transformer_model=sentence_transformer_model
        )
        
        for result, expected_output in tqdm(
            zip(results, expected_outputs), 
            desc=f"Analyzing with {string_metric}", 
            unit=" example"
        ):
            logger.debug(
                f"\n\n\n========= Analyzing: {result.doc_id} "
                f"with {string_metric}\n\n\n"
            )
            # Analyze response (metrics are set inside the evaluator)
            analysis = evaluator.evaluate(
                result.model_output,
                expected_output,
            )
            analyzed_results.append({
                "doc_id": result.doc_id,
                "expected_output": expected_output,
                "model_output": result.model_output,
                "metadata": result.metadata,
                "analysis": analysis
            })
        
        # Store results for this metric
        all_metric_results[string_metric] = analyzed_results

    return all_metric_results


def save_analysis(
    analyzed_results: Dict[str, Dict[str, List[Dict[str, Any]]]],
    checklist_name: str,
    check_name: str,
    model: str
):
    """Save the analysis results to a file.
    
    Args:
        analyzed_results: Dictionary mapping prompt names to their results,
            where each result contains string metric results
        checklist_name: Name of the checklist
        check_name: Name of the check
        model: Model name
    """
    
    # Save analysis results
    try:
        analysis_path = EVALUATION_DIR / checklist_name / check_name / model
        os.makedirs(analysis_path, exist_ok=True)
        
        # Save a comprehensive file with all prompts and all string metrics
        analysis_file = analysis_path / "analysis.json"
        with open(analysis_file, "w", encoding="utf-8") as f:
            json.dump(analyzed_results, f, indent=4, ensure_ascii=False)

        logger.info(
            f"Saved analysis for {check_name} to {analysis_file}"
        )
        
    except Exception as e:
        logger.error(
            f"Error saving analysis results for {check_name}: {str(e)}"
        )
        logger.debug("Save exception details:", exc_info=True)
        raise


def prepare_check_data(
    check_dir: Path
) -> Tuple[CheckData | None, Dict[str, str]]:
    """Prepare data for processing a check.
    
    This function consolidates common functionality for loading check data,
    schema, prompts, and gathering examples. It can be used by both
    initialize() and process_check() functions.
    
    Args:
        check_dir: Path to the check directory
    Returns:
        Tuple of (CheckData object, prompts dict) or (None, {}) if 
        preparation fails
    """
    logger.info(f"Preparing data for check: {check_dir.name}")

    # Load check data
    check_benchmark_file = check_dir / "benchmark.json"
    if not check_benchmark_file.exists():
        logger.warning(
            f"Check data file not found: {check_benchmark_file}"
        )
        return (None, {})

    try:
        benchmark_data = load_json(check_benchmark_file)
    except Exception as e:
        logger.error(f"Error loading check data: {str(e)}")
        return (None, {})

    check_name = benchmark_data["name"]

    # Load schema
    schema_file = check_dir / "schema.json"
    if not schema_file.exists():
        logger.error(f"Schema file not found: {schema_file}")
        return (None, {})

    try:
        schema = load_json(schema_file)
    except Exception as e:
        logger.error(f"Error loading schema: {str(e)}")
        return (None, {})

    # Get all prompts from the prompts directory
    prompts_dir = check_dir / "prompts"
    if not prompts_dir.exists():
        logger.error(
            f"Prompts directory not found: {prompts_dir}"
        )
        return (None, {})

    # Get all prompt files
    prompt_files = list(prompts_dir.glob("prompt*.txt"))
    if not prompt_files:
        logger.error(f"No prompt files found in {prompts_dir}")
        return (None, {})

    # Sort the prompt files by name in ascending order
    prompt_files.sort()

    # Load prompts
    prompts = {}
    for prompt_file in prompt_files:
        prompt_name = prompt_file.stem
        with open(prompt_file, "r", encoding="utf-8") as f:
            prompts[prompt_name] = f.read()

    # Get list of paths to examples from benchmark.json
    example_paths = benchmark_data.get("examples", [])
    if not example_paths:
        logger.warning(f"No examples found in check: {check_dir.name}")
        return (None, {})

    # Get example class from benchmark.json
    try:
        example_class_name = benchmark_data["example_class"]
    except KeyError:
        logger.error(f"No example class found in {check_benchmark_file}")
        return (None, {})

    try:
        examples = [
            EXAMPLE_FACTORY.create(ex_path, example_class_name)
            for ex_path in example_paths
        ]
    except Exception as e:
        logger.error(f"Error gathering examples: {str(e)}")
        return (None, {})
    if not examples:
        logger.warning(
            f"No valid examples gathered for check: {check_dir.name}"
        )
        return (None, {})

    try:
        expected_outputs = []
        for example in examples:
            try:
                expected_output = example.get_expected_output(check_name)
            except Exception as e:
                logger.error(f"Error getting expected outputs ({example.doc_id}): {str(e)}")
                continue
            if expected_output is not None:
                expected_outputs.append(expected_output)
            else:
                logger.warning(
                    f"No expected output found for example: {example.doc_id}. "
                    f"This may indicate that the example is not properly "
                    f"initialized."
                )
                continue
    except Exception as e:
        logger.error(f"Error getting expected outputs: {str(e)}")
        return (None, {})
    if not expected_outputs:
        # this can happen when initializing
        logger.warning(
            f"No valid expected outputs gathered for check: {check_dir.name}"
        )
        return (None, {})
    else:
        # this should not happen and indicates some expected outputs are 
        # missing
        assert len(expected_outputs) == len(examples), (
            f"Expected outputs not found for all examples in "
            f"check: {check_dir.name}"
        )

    # Validate that we have the same number of examples with doc_ids as 
    # expected outputs
    # This ensures consistency between examples and expected_outputs
    if len(examples) != len(expected_outputs):
        logger.error(
            f"Mismatch between examples with doc_ids "
            f"({len(examples)}) and expected outputs "
            f"({len(expected_outputs)}) for check: {check_dir.name}. "
            f"This may indicate examples with None doc_ids."
        )
        return (None, {})

    check_data = CheckData(
        check_dir_name=check_dir.name,
        check_name=check_name,
        schema=schema,
        examples=examples,
        expected_outputs=expected_outputs
    )

    return check_data, prompts


def process_check(
    check_dir: Path,
    checklist_name: str,
    mock: bool = False,
    use_cache: bool = True,
    model: str = DEFAULT_MODEL,
    match_threshold: float = DEFAULT_MATCH_THRESHOLD,
    sentence_transformer_model: str = (
        DEFAULT_SENTENCE_TRANSFORMER_MODEL
    )
) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
    """Process a single check.
    
    Args:
        check_dir: Path to the check directory
        checklist_name: Name of the checklist
        mock: If True, use expected outputs as model outputs (no API calls)
        use_cache: If True, use cached outputs when available
        model: The model to use for generation
        match_threshold: Threshold for considering a match (0-1)
        sentence_transformer_model: SentenceTransformer model for semantic similarity
        
    Returns:
        Dictionary mapping string metric names to analysis results
    """
    # Validate model compatibility
    if not validate_model_for_provider(model):
        compatible_models = get_compatible_models()
        raise ValueError(
            f"Model '{model}' is not compatible with provider '{API_PROVIDER}'. "
            f"Compatible models: {', '.join(compatible_models)}"
        )

    
    # Prepare check data
    check_data, prompts = prepare_check_data(check_dir)
    if not check_data:
        return {}

    # Dictionary to store results for each prompt
    all_results = {}

    # Process each prompt
    for prompt_name, prompt in prompts.items():
        logger.info(f"Processing prompt: {prompt_name}")
        if mock:
            results = [
                ModelResult(
                    doc_id=example.doc_id,
                    model_output=expected_output,
                    metadata={
                        "doc_id": example.doc_id,
                        "source": example.relative_source_path,
                        "example_type": example.example_class_name
                    }
                )
                for example, expected_output in zip(
                    check_data.examples, check_data.expected_outputs
                )
            ]
        else:
            results = run_model(
                check_data,
                prompt,
                use_cache=use_cache,
                model=model
            )

        # Analyze results with all string metrics
        analyzed_results = analyze_results(
            results,
            check_data.schema,
            check_data.expected_outputs,
            match_threshold=match_threshold,
            sentence_transformer_model=sentence_transformer_model
        )

        # Store results for this prompt
        all_results[prompt_name] = analyzed_results

    save_analysis(
        all_results, checklist_name, check_data.check_name, model
    )

    return all_results


def list_checks(checklist_dir: Path) -> Dict[str, Path]:
    """List all checks in a checklist."""
    # enumerate the subdirectories of the checklist directory
    checks = {}
    for check_dir in checklist_dir.iterdir():
        if check_dir.is_dir():
            checks[check_dir.name] = check_dir
    return checks


def initialize(checklist_dir: Path, use_cache: bool = True, model: str = DEFAULT_MODEL):
    """Initialize expected_output.json files for all examples in a checklist.
    
    This function iterates through the checks of a checklist. For each check,
    it uses the prompts in prompts/ and the schema.json to execute the check
    using the image and caption of each example. The output becomes the
    preliminary expected_output.json which is written in the
    <doc_id>/checks/<check_name>/ folder.
    
    Args:
        checklist_dir: Path to the checklist directory
        use_cache: If True, use cached outputs when available
    """
    logger.info(
        f"Initializing expected outputs for checklist: {checklist_dir.name}"
    )
    
    # Get all checks from the checklist
    checks = list_checks(checklist_dir)
    if not checks:
        logger.error(f"No checks found in checklist: {checklist_dir.name}")
        return
    
    # Process each check
    for check_dir_name, check_dir in checks.items():
        logger.info(f"Processing check: {check_dir_name}")
        
        # Prepare check data (use only the first prompt)
        prepared_data, prompts = prepare_check_data(check_dir)

        if not prepared_data:
            continue

        # Get the first prompt for initialization
        # sort prompts alphabetically
        prompts_sorted = sorted(prompts.keys())
        first_prompt_name = prompts_sorted[0]
        first_prompt = prompts[first_prompt_name]

        try:
            results = run_model(
                prepared_data,
                first_prompt,
                use_cache=use_cache,
                model=model
            )

            # Write expected outputs
            for result, example in zip(results, prepared_data.examples):
                try:
                    example.save_expected_output(
                        result.model_output,
                        prepared_data.check_name
                    )
                except Exception as e:
                    logger.error(
                        f"Error writing expected output for "
                        f"{result.doc_id}: {str(e)}"
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
    use_cache: bool = True,
    model: str = DEFAULT_MODEL,
    match_threshold: float = DEFAULT_MATCH_THRESHOLD,
    sentence_transformer_model: str = (
        DEFAULT_SENTENCE_TRANSFORMER_MODEL
    )
):
    """Process an entire checklist.
    
    Args:
        checklist_dir: Path to the checklist directory
        checklist_name: Name of the checklist
        mock: If True, use expected outputs as model outputs (no API calls)
        use_cache: If True, use cached outputs when available
        model: The model to use for generation
        match_threshold: Threshold for considering a match (0-1)
        sentence_transformer_model: SentenceTransformer model for semantic similarity
    """
    # Validate model compatibility
    if not validate_model_for_provider(model):
        compatible_models = get_compatible_models()
        raise ValueError(
            f"Model '{model}' is not compatible with provider '{API_PROVIDER}'. "
            f"Compatible models: {', '.join(compatible_models)}"
        )


    # Get all checks frmo the checklist
    checks = list_checks(checklist_dir)

    if not checks:
        logger.error(
            f"Checklist not found: {checklist_name}"
        )
        return

    for check_dir_name, check_dir in checks.items():
        try:
            process_check(
                check_dir,
                checklist_name,
                mock=mock,
                use_cache=use_cache,
                model=model,
                match_threshold=match_threshold,
                sentence_transformer_model=sentence_transformer_model
            )
        except Exception as e:
            logger.error(
                f"Error processing check in {check_dir_name}: {str(e)}"
            )
            continue


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Run MMQC checks on examples"
    )
    parser.add_argument(
        "checklist", type=str, help="Name of the checklist to process"
    )
    parser.add_argument(
        "--initialize", action="store_true", 
        help="Initialize expected output files"
    )
    parser.add_argument(
        "--mock", action="store_true", 
        help="Use mock responses instead of calling the model"
    )
    parser.add_argument(
        "--no-cache", action="store_true", 
        help="Disable caching of model responses"
    )
    parser.add_argument(
        "--check", type=str, help="Name of the check to process"
    )
    parser.add_argument(
        "--model", type=str, default=DEFAULT_MODEL, 
        help="The model to use for generation"
    )
    parser.add_argument(
        "--match-threshold", type=float, default=DEFAULT_MATCH_THRESHOLD,
        help="Threshold for considering a match (0-1)"
    )
    parser.add_argument(
        "--sentence-transformer-model", type=str, 
        default=DEFAULT_SENTENCE_TRANSFORMER_MODEL,
        help="SentenceTransformer model for semantic similarity"
    )
    args = parser.parse_args()

    # Get the checklist directory using the config function
    checklist_name = args.checklist
    checklist_dir = CHECKLIST_DIR / checklist_name
    if not checklist_dir.exists():
        logger.error(f"Checklist directory not found: {checklist_dir}")
        return

    # Validate model compatibility with current provider
    if not validate_model_for_provider(args.model):
        compatible_models = get_compatible_models()
        logger.error(
            f"Model '{args.model}' is not compatible with provider '{API_PROVIDER}'. "
            f"Compatible models: {', '.join(compatible_models)}"
        )
        return
    
    # Initialize if requested
    if args.initialize:
        initialize(checklist_dir, not args.no_cache, args.model)
        return

    if args.check:
        # Find the check in the checklist
        check_dir = checklist_dir / args.check
        if check_dir.exists():
            process_check(
                check_dir, args.checklist, args.mock, not args.no_cache, 
                model=args.model,
                match_threshold=args.match_threshold,
                sentence_transformer_model=args.sentence_transformer_model
            )
        else:
            logger.error(f"Check not found: {args.check}")
    else:
        # Process the entire checklist
        process_checklist(
            checklist_dir, args.checklist, args.mock, not args.no_cache, 
            model=args.model,
            match_threshold=args.match_threshold,
            sentence_transformer_model=args.sentence_transformer_model
        )


def initialize_main():
    """Entry point for the initialize command that passes --initialize to 
    main."""
    import sys
    sys.argv.append("--initialize")
    main()


if __name__ == "__main__":
    main()
