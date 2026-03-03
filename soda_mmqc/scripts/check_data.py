from pathlib import Path
from soda_mmqc import logger
from langfuse import get_client
import json

# Initialize Langfuse client
langfuse = get_client()


def load_json(file_path):
    """Load JSON data from a file."""
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_check_data(check_dir: Path, remote: bool = True):
    if remote:
        # Use the canonical prompt key used elsewhere in the codebase:
        # "checklists/{checklist_name}/{check_name}" so Langfuse prompt
        # management returns the intended prompt rather than a file/path
        # based name which can point to unrelated prompt entities.
        try:
            checklist_name = check_dir.parent.name
        except Exception:
            checklist_name = str(check_dir.parent)
        prompt_key = f"checklists/{checklist_name}/{check_dir.name}"
        logger.info(f"Langfuse: fetching prompt for key={prompt_key}")
        prompt_data = langfuse.get_prompt(prompt_key)
        prompt_text = prompt_data.prompt
        schema = prompt_data.config.get("output_schema", {})
        benchmark = prompt_data.config.get("benchmark", {})
        return prompt_text, schema, benchmark
    else:
        prompts = get_local_prompts(check_dir)
        schema = get_local_schema(check_dir)
        benchmark = get_local_benchmark(check_dir)
        return prompts, schema, benchmark


def get_remote_prompts(check_dir: Path, versions = [1, 2, 3, 4]):
    prompts = {}
    schemas = {}
    benchmarks = {}
    
    try:
        checklist_name = check_dir.parent.name
    except Exception:
        checklist_name = str(check_dir.parent)
    prompt_key = f"checklists/{checklist_name}/{check_dir.name}"
    prompt_text = None
    schema = {}
    benchmark = {}
    for version in versions:
        logger.info(f"Langfuse: fetching prompt for key={prompt_key} version={version}")
        prompt_data = langfuse.get_prompt(prompt_key, version=version)
        if prompt_data:
            prompt_text = prompt_data.prompt
            schema = prompt_data.config.get("output_schema", {})
            benchmark = prompt_data.config.get("benchmark", {})
            break
    return prompt_text, schema, benchmark


def get_local_prompts(check_dir: Path):
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
    return prompts


def get_local_schema(check_dir: Path):
    # Get the schema from the schema.json file
    schema_file = check_dir / "schema.json"
    if not schema_file.exists():
        logger.error(f"Schema file not found: {schema_file}")
        return None
    schema = load_json(schema_file)
    return schema


def get_local_benchmark(check_dir: Path):
    # Get the benchmark from the benchmark.json file
    benchmark_file = check_dir / "benchmark.json"
    if not benchmark_file.exists():
        logger.error(f"Benchmark file not found: {benchmark_file}")
        return None
    benchmark = load_json(benchmark_file)
    # verify that the bacnhmark name is identical to the name of the check directory
    if benchmark.get("name") != check_dir.name:
        logger.error(f"Benchmark name '{benchmark.get('name', '')}' does not match check directory name '{check_dir.name}'")
        return None

    return benchmark

