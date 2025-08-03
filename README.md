# SODA MMQC

## System Requirements

- Python 3.8+
- [Pandoc](https://pandoc.org/installing.html) (must be installed and available in your PATH)

### Install Pandoc

- **macOS:** `brew install pandoc`
- **Ubuntu/Debian:** `sudo apt-get install pandoc`
- **Windows:** [Download installer](https://pandoc.org/installing.html)

---

A resource to develop, evaluate and distributeAI-generated quality checks of scientific manuscritps.

## Package Structure

The `soda_mmqc` package is organized as follows:

```
soda_mmqc/
├── __init__.py          # Package initialization and logging
├── config.py            # Configuration constants and paths
├── core/                # Core business logic
│   ├── examples.py      # Example data classes and factory
│   ├── evaluation.py    # Model evaluation tools
│   └── curation.py      # Streamlit curation interface
├── lib/                 # Model-related functionality
│   ├── api.py           # API client for model calls
│   └── cache.py         # Caching functionality
├── scripts/             # CLI scripts
├── utils/               # Utility functions
└── data/                # Static data files (checklists, examples, evaluations)
```

## Installation

You can install the package in development mode using pip:

```bash
# Clone the repository
git clone https://github.com/yourusername/soda-mmqc.git
cd soda-mmqc

# Create virtual environment and install dependencies
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

## Configuration

### API Provider Setup

SODA MMQC supports both OpenAI and Anthropic APIs with structured output capabilities. You need to configure your API provider before using the system.

#### Method 1: Environment File (Recommended)

Create a `.env` file in the project root:

```bash
# API Provider Configuration
API_PROVIDER=openai  # Choose: 'openai' or 'anthropic'

# OpenAI Configuration (required if API_PROVIDER=openai)
OPENAI_API_KEY=your_openai_api_key_here

# Anthropic Configuration (required if API_PROVIDER=anthropic)  
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# Optional: Device configuration for ML operations
DEVICE=cpu  # Options: 'cpu', 'cuda', 'mps' (Apple Silicon)
```

#### Method 2: Environment Variables

Alternatively, export environment variables directly:

```bash
# For OpenAI (default)
export API_PROVIDER=openai
export OPENAI_API_KEY=your_openai_api_key

# For Anthropic
export API_PROVIDER=anthropic
export ANTHROPIC_API_KEY=your_anthropic_api_key
```

#### Supported Models

The system automatically selects appropriate default models for each provider:

**OpenAI Models:**
- `gpt-4o-2024-08-06` (default)
- `gpt-4o-mini`
- Other GPT-4 variants with structured output support

**Anthropic Models:**
- `claude-3-5-sonnet-20241022` (default)
- `claude-3-7-sonnet-20250219`
- `claude-4-sonnet` and `claude-4-opus` (when available)

#### Configuration Validation

The system validates your configuration on startup and provides helpful feedback:

- ✅ **Valid setup**: "OpenAI API provider configured"
- ⚠️ **Missing API key**: "OpenAI selected but OPENAI_API_KEY not found"
- ⚠️ **Invalid provider**: "Unknown API provider 'xyz', falling back to OpenAI"

For detailed configuration options, see the [API Provider Documentation](soda_mmqc/docs/api_providers.md).

## Testing

The project uses pytest for testing. You can run tests using:

```bash
# Run all tests
pytest

# Run tests with verbose output
pytest -v

# Run specific test files
pytest tests/test_specific_file.py

# Run tests matching a pattern
pytest -k "test_name_pattern"
```

The test configuration is defined in `pyproject.toml` with test paths set to the `tests/` directory.

## Usage

After installation, you can use the following commands:

```bash
# Run all checks in a checklist
evaluate CHECKLIST_NAME [--model MODEL_NAME] [--mock] [--no-cache] [--match-threshold THRESHOLD]

# Run a specific check in a checklist
evaluate CHECKLIST_NAME --check CHECK_NAME [--model MODEL_NAME] [--mock] [--no-cache] [--match-threshold THRESHOLD]

# Initialize expected output files for a checklist
init CHECKLIST_NAME [--no-cache]

# Curate and manage checklists
curate CHECKLIST_NAME
```

Command line options:
- `--model`: Specify the model to use. Supports both OpenAI and Anthropic models (auto-selected based on API_PROVIDER configuration)
- `--mock`: Use expected outputs as model outputs (no API calls)
- `--no-cache`: Disable caching of model outputs
- `--check`: Specify a particular check to run within a checklist
- `--initialize`: Initialize expected output files (alternative to `init` command)
- `--match-threshold`: Set threshold for string matching across all metrics (default: 0.3)

**Note**: If no model is specified, the system automatically uses the default model for your configured API provider (see [Configuration](#configuration)).

## String Comparison Metrics

When evaluating model outputs against expected results, SODA MMQC automatically uses three different string comparison methods to provide comprehensive scoring:

### 1. Perfect Match
- **Method**: Exact string matching (case-sensitive)
- **Use Case**: Identifies when model outputs match expectations exactly
- **Score**: 1.0 for exact matches, 0.0 otherwise
- **Example**: "yes" vs "yes" → 1.0, "yes" vs "Yes" → 0.0

### 2. Semantic Similarity
- **Method**: SentenceTransformer embeddings with cosine similarity
- **Use Case**: Captures semantic relationships between different phrasings
- **Score**: 0.0 to 1.0 based on semantic closeness
- **Example**: "cat" vs "feline" → ~0.8, "cat" vs "dog" → ~0.6

### 3. Longest Common Subsequence (LCS)
- **Method**: Character-level similarity using longest common subsequence
- **Use Case**: Handles partial text overlaps and extracted segments
- **Score**: 0.0 to 1.0 based on shared character sequences
- **Example**: "statistical significance" vs "significance was tested" → ~0.6

### Evaluation Output Structure

Results are organized by prompt and metric for comprehensive analysis:

```json
{
  "prompt.1": {
    "perfect_match": [{"doc_id": "...", "analysis": {"score": 1.0, ...}}],
    "semantic_similarity": [{"doc_id": "...", "analysis": {"score": 0.85, ...}}],
    "longest_common_subsequence": [{"doc_id": "...", "analysis": {"score": 0.72, ...}}]
  },
  "prompt.2": { ... }
}
```

This multi-metric approach allows researchers to:
- **Compare methodologies**: See how different similarity measures perform on their specific tasks
- **Understand failure modes**: Identify whether issues are exact matching vs semantic understanding
- **Optimize thresholds**: Choose appropriate match thresholds for each metric type
- **Validate robustness**: Ensure results are consistent across different evaluation approaches

## Dependencies

The project requires Python 3.8 or higher and includes the following main dependencies:

- **python-dotenv**: Environment variable management
- **openai & anthropic**: LLM API integration with structured output support (see [Configuration](#configuration))
- **Pillow**: Image processing
- **nltk & sentence-transformers**: Text processing and embeddings
- **scikit-learn & numpy**: Data processing and analysis
- **plotly & matplotlib**: Data visualization
- **streamlit**: Web interface
- **jupyter & jupyterlab**: Interactive development
- **pytest**: Testing framework

**Note**: You must configure at least one API provider (OpenAI or Anthropic) before using the system. See the [Configuration](#configuration) section for setup instructions.

The Open Library of Multimodal Data Checklists (mmQC)
===============================================================

This is an open library of multimodal prompts dedicated to the verification of the quality, rigor and compliance to editorial policies of scientific figures, including the image depicting the results, the respective figure caption and the linked research data. 

Example of checks:

- Check whether error bars are defined in the figure caption.
- Check whether statistical tests used to assess significance are mentioned in the caption.
- Check whether scale bars are defined in micrograph figure captions.
- Check whether individual data points are shown in bar charts.


## Checks:

A check is characterized by the following components:
- **Prompts**: Multiple prompt templates that can be compared and optimized for the specific check
- **Output Schema**: A `schema.json` file that defines the structure of the model output given the prompts
- **Benchmarking Dataset**: A `benchmark.json` file containing test examples and expected outputs to measure prompt performance

## Checklists:

A checklist is a collection of loosely related checks, organized for conveniencein a directory structure. The project currently supports two main checklists:

### 1. Document Checklists (`doc-checklist/`)
Checks that analyze entire documents or manuscript structure:
- **extract-figures**: Extracts and identifies figures from documents [not implemented]
- **section-order**: Validates the order and structure of document sections
- **species-identified**: Checks if species are properly identified in the document [not implemented]

### 2. Figure Checklists (`fig-checklist/`)
Checks that analyze individual figures and their captions:
- **error-bars-defined**: Verifies if error bars are explained in captions
- **individual-data-points**: Checks if individual data points are shown in bar charts
- **micrograph-scale-bar**: Validates presence of scale bars in micrographs
- **micrograph-symbols-defined**: Ensures symbols in micrographs are explained
- **plot-axis-units**: Checks if axis units are properly labeled
- **plot-gap-labeling**: Validates gap labeling in plots
- **replicates-defined**: Verifies if replicates are properly defined
- **stat-significance-level**: Checks statistical significance reporting
- **stat-test**: Validates statistical test specifications
- **structure-identified**: Ensures structures are properly identified [not implemented]

Each check directory contains:
```
checklist/
├── doc-checklist/                    # Document-level checks
│   ├── section-order/
│   │   ├── prompts/                  # Directory containing prompt templates
│   │   ├── benchmark.json           # Test examples and expected outputs
│   │   └── schema.json              # JSON schema for check output
│   └── ...
├── fig-checklist/                    # Figure-level checks
│   ├── error-bars-defined/
│   │   ├── prompts/
│   │   ├── benchmark.json
│   │   └── schema.json
│   └── ...


```

## Benchmarking data:

The structure of the repository keeps each example as human readable directories, grouping the content files as well as the expected output for each of the checks:

### Structure of the examples used for benchmarking:

    data/examples/
      └── 10.1038_emboj.2009.312/           # Example document
          ├── content/
          │   ├── manuscirpt.docx
          │   └── 1/                        # Figure 
          │       ├── content/
          │       │   ├── embj2009312-fig-0001-m.jpg
          │       │   ├── 10.1038-emboj.2009.312Figure20093120001.pptx
          │       │   └── caption.txt
          │       └── checks/                # Figure-level checks
          │           ├── error-bars-defined/
          │           │   └── expected_output.json
          │           ├── individual-data-points/
          │           │   └── expected_output.json
          │           ├── micrograph-scale-bar/
          │           │   └── expected_output.json
          │           ├── micrograph-symbols-defined/
          │           │   └── expected_output.json
          │           ├── plot-axis-units/
          │           │   └── expected_output.json
          │           ├── plot-gap-labeling/
          │           │   └── expected_output.json
          │           ├── replicates-defined/
          │           │   └── expected_output.json
          │           ├── stat-significance-level/
          │           │   └── expected_output.json
          │           └── stat-test/
          │               └── expected_output.json
          └── checks/                        # Document-level checks
              ├── section-order/
              │   └── expected_output.json
              └── section-order-alt/
                  └── expected_output.json

### Structure of the checklists:

    data/checklist/
      ├── doc-checklist/                    # A series of checks
      │   ├── extract-figures/
      │   │   ├── prompts/
      │   │   │   ├── prompt.1.txt
      │   │   │   ├── prompt.2.txt
      │   │   │   └── ...
      │   │   ├── schema.json
      │   │   └── benchmark.json
      │   ├── section-order/
      │   │   ├── prompts/
      │   │   ├── schema.json
      │   │   └── benchmark.json
      │   └── species-identified/
      │       ├── prompts/
      │       ├── schema.json
      │       └── benchmark.json
      │
      ├── fig-checklist/                    # Another series of checks
      │   ├── error-bars-defined/
      │   │   ├── prompts/
      │   │   ├── schema.json
      │   │   └── benchmark.json
      │   ├── individual-data-points/
      │   ├── micrograph-scale-bar/
      │   ├── micrograph-symbols-defined/
      │   ├── plot-axis-units/
      │   ├── plot-gap-labeling/
      │   ├── replicates-defined/
      │   ├── stat-significance-level/
      │   ├── stat-test/
      │   └── structure-identified/
      │
      ├── ...

### Structure of the evaluation:

    data/evaluation/
      ├── doc-checklist/
      │   └── section-order/
      │       └── gpt-4o-2024-08-06/
      │           └── analysis.json
      └── fig-checklist/
          ├── error-bars-defined/
          │   ├── gpt-4o-2024-08-06/
          │   ├── gpt-4o-mini/
          │   └── o4-mini-2025-04-16/
          │       └── analysis.json
          ├── individual-data-points/
          ├── micrograph-scale-bar/
          ├── micrograph-symbols-defined/
          ├── plot-axis-units/
          ├── plot-gap-labeling/
          ├── replicates-defined/
          ├── stat-significance-level/
          └── stat-test/


## Example

### Prompt:

```text
You are a scientific figure quality control expert. Your task is to analyze a scientific figure and its caption to check for the presence of error bars and whether they are properly defined.

For each panel in the figure:

1. Look for error bars (lines extending from data points indicating variability).
2. Check if the caption explains what these error bars represent.
3. Report your findings in a structured format.

Provide your analysis in the following JSON format for EACH panel:

{
    "outputs": [
        {
            "panel_label": "[panel letter]",
            "error_bar_on_figure": "[yes/no]",
            "error_bar_defined_in_caption": "[yes/no/not needed]",
            "error_bar_definition": "[exact text describing what the error bars represent, or an empty string]"
        }
    ]
}


Be concise and focus only on the presence and definition of error bars.
```

#### Output Schema:

The output schema defines the structured format for model responses. SODA MMQC automatically enforces this schema for both OpenAI (using structured output) and Anthropic (using tool calling) APIs.


```json
{
    "format": {
        "type": "json_schema",
        "name": "error-bars-defined",
        "schema": {
            "type": "object",
            "properties": {
                "outputs": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "panel_label": {
                                "type": "string",
                                "description": "Label of the panel (e.g., A, B, C)"
                            },
                            "error_bar_on_figure": {
                                "type": "string",
                                "enum": ["yes", "no"],
                                "description": "Whether error bars are present"
                            },
                            "error_bar_defined_in_caption": {
                                "type": "string",
                                "enum": ["yes", "no", "not needed"],
                                "description": "Whether error bars are defined in caption"
                            },
                            "from_the_caption": {
                                "type": "string",
                                "description": "Text from caption describing error bars, when error bars are present"
                            }
                        },
                        "required": [
                            "panel_label",
                            "error_bar_on_figure",
                            "error_bar_defined_in_caption",
                            "from_the_caption"
                        ],
                        "additionalProperties": false
                    }
                }
            },
            "required": ["outputs"],
            "additionalProperties": false
        },
        "strict": true
    }
}
```

#### Benchmarking Dataset:

A benchmark dataset is a JSON file that contains the relative path to the examples to be used for benchmarking.
The example directory is specified as `EXAMPLE_DIR` in `soda_mmqc/config.py` and is set to `data/examples/` by default.
The benchmark JSON file includes the field "example_class" that specifies the type of example to be used for benchmarking.
The field "examples" is an array of relative paths to the examples to be used for benchmarking.

The benchmark dataset is used to evaluate the performance of the prompts and the output schema.


```json
{
  "name": "error-bars-defined",
  "description": "Checks whether the error bars are defined in the figure caption.",
  "example_class": "figure",
  "examples": [
    "10.1038_emboj.2009.312/content/1",
    "10.1038_s44319-025-00432-6/content/7",
    "10.1038_emboj.2009.340/content/3",
    "10.1038_s44319-025-00438-0/content/1",
    "10.1038_embor.2009.217/content/4",
    "10.1038_s44320-025-00092-7/content/1",
    "10.1038_s44320-025-00092-7/content/2",
    "10.1038_embor.2009.233/content/2",
    "10.1038_s44320-025-00094-5/content/2",
    "10.1038_s44318-025-00409-0/content/1",
    "10.1038_s44320-025-00096-3/content/9",
    "10.1038_s44318-025-00412-5/content/1",
    "10.1038_s44321-025-00219-1/content/1",
    "10.1038_s44318-025-00416-1/content/8",
    "10.1038_s44319-025-00415-7/content/2",
    "10.1038_s44321-025-00224-4/content/1"
  ]
}
```

### Expected output:

The expected output is a JSON file that contains the expected output for each of the checks. It has to follow the output schema.

```json
{
    "name": "error-bars-defined",
    "results": [
        {
            "panel_label": "A",
            "error_bar_on_figure": "yes",
            "error_bar_defined_in_caption": "yes",
            "quote_from_caption": "Mean±s.d. of three experiments is reported.",
        },
        {
            "panel_label": "B",
            "error_bar_on_figure": "yes",
            "error_bar_defined_in_caption": "yes",
            "quote_from_caption": "Mean±s.d. of three experiments is reported.",
        },
        {
            "panel_label": "C",
            "error_bar_on_figure": "yes",
            "error_bar_defined_in_caption": "yes",
            "quote_from_caption": "Mean±s.d. of three experiments is reported.",
        },
        {
            "panel_label": "D",
            "error_bar_on_figure": "yes",
            "error_bar_defined_in_caption": "yes",
            "from_the_caption": "Mean±s.d. of three experiments is reported.",
        }
    ]
}
```




