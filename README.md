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

## Installation

You can install the package in development mode using pip:

```bash
# Clone the repository
git clone https://github.com/yourusername/soda-mmqc.git
cd soda-mmqc

# Install in development mode
pip install -e .
```

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
evaluate CHECKLIST_NAME [--model MODEL_NAME] [--mock] [--no-cache]

# Run a specific check in a checklist
evaluate CHECKLIST_NAME --check CHECK_NAME [--model MODEL_NAME] [--mock] [--no-cache]

# Initialize expected output files for a checklist
init CHECKLIST_NAME [--no-cache]

# Curate and manage checklists
curate CHECKLIST_NAME
```

Command line options:
- `--model`: Specify the model to use (default: "gpt-4o-2024-08-06")
- `--mock`: Use expected outputs as model outputs (no API calls)
- `--no-cache`: Disable caching of model outputs
- `--check`: Specify a particular check to run within a checklist
- `--initialize`: Initialize expected output files (alternative to `init` command)


## Dependencies

The project requires Python 3.8 or higher and includes the following main dependencies:

- python-dotenv: Environment variable management
- openai & anthropic: LLM API integration
- Pillow: Image processing
- nltk & sentence-transformers: Text processing and embeddings
- scikit-learn & numpy: Data processing and analysis
- plotly & matplotlib: Data visualization
- streamlit: Web interface
- jupyter & jupyterlab: Interactive development
- pytest: Testing framework

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
T
#### Output Schema:

The output schema is based on the OpenAI JSON Schema format for the `client.responses.create` method.


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




