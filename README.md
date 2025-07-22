# SODA MMQC

## System Requirements

- Python 3.8+
- [Pandoc](https://pandoc.org/installing.html) (must be installed and available in your PATH)

### Install Pandoc

- **macOS:** `brew install pandoc`
- **Ubuntu/Debian:** `sudo apt-get install pandoc`
- **Windows:** [Download installer](https://pandoc.org/installing.html)

---

A tool for evaluating figure caption quality.

## Installation

You can install the package in development mode using pip:

```bash
# Clone the repository
git clone https://github.com/yourusername/soda-mmqc.git
cd soda-mmqc

# Install in development mode
pip install -e .
```

## Usage

After installation, you can use the following commands:

```bash
# Run all checklists
soda-mmqc run [--model MODEL_NAME] [--mock] [--no-cache]

# Run a specific checklist
soda-mmqc run --checklist CHECKLIST_NAME [--model MODEL_NAME] [--mock] [--no-cache]

# Initialize the project (first time setup)
soda-mmqc init

# Curate and manage checklists
soda-mmqc curate CHECKLIST_NAME

# Visualize results
soda-mmqc viz

# Set logging level
soda-mmqc run --log-level DEBUG

# Specify custom results directory
soda-mmqc run --results-dir path/to/results
```

Command line options:
- `--model`: Specify the model to use (default: "gpt-4o-2024-08-06")
- `--mock`: Use expected outputs as model outputs (no API calls)
- `--no-cache`: Disable caching of model outputs
- `--checklist`: Specify a particular checklist to run
- `--log-level`: Set logging level (DEBUG, INFO, WARNING, ERROR)
- `--results-dir`: Specify custom directory for results

## Project Structure

```
soda-mmqc/
├── soda_mmqc/
│   ├── __init__.py
│   ├── config.py
│   ├── model_api.py
│   ├── model_cache.py
│   ├── evaluation.py
│   ├── scripts/
│   │   ├── __init__.py
│   │   ├── run.py
│   │   ├── curate.py
│   │   └── visualize.py
│   ├── tools/
│   │   └── ...
│   ├── data/
│   │   ├── __init__.py
│   │   ├── checklist/
│   │   ├── examples/
│   │   ├── evaluation/
│   │   │   └── fig-checklist/
│   │   │       ├── error-bars-defined/
│   │   │       ├── individual-data-points/
│   │   │       ├── micrograph-scale-bar/
│   │   │       ├── micrograph-symbols-defined/
│   │   │       ├── plot-axis-units/
│   │   │       ├── plot-gap-labeling/
│   │   │       ├── replicates-defined/
│   │   │       ├── stat-significance-level/
│   │   │       └── stat-test/
│   │   └── plots/
│   └── logs/
├── notebooks/
│   ├── plots.ipynb
│   ├── images/
│   └── cache/
├── pyproject.toml
└── README.md
```

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

- Check if the experimental method, platform or assay used in the depicted experiment is indicated in the caption.
- Check if the object of the observations or measurements are indicated in the caption.
- Check if the source data files associated with this figure panel corresponda to the results displayed in the image.
- Should the number of replicates be mentioned in the caption given the results depicted in the figure?
- If the number of replicates is mentioned, is it clear what is the nature of these replicates (experimental, technical) and what source of variability they are supposed to account for?
- In fluorescence microscopy, are the channels identified are the clearly linked to specific biological objects or features or interest?
- For ligh microscopy, are the staining or labeling methods indicated?
- Are there annotations (arrows, arrow heads, marking, symbols, stars, etc) on the images all explained and referred to in the caption?
- For overlay images that superpose multiple channels, are the respective grey scale images provided for each separate channel?
- For images showing a phenotype, microscopic or macroscopic, are there multiple images illustrating the range of variability of the phenotype?
- In a figure displaying plots that include error bars, are the error bars explained in the figure caption?
- In bar charts, are the individual points also displayed?
- If statistical significance or testing is mentioned in the caption or displayed on the image, is the statistical test specified?


## Checks:

For each of these checks we design a prompt that can be optmized based on a test set assembling examples covering the expected outcomes.

Each check is defined by a JSON file in the `data/checks/` directory.

```json
{
  "name": "error-bars-defined",
  "description": "Checks whether the error bars are defined in the figure caption.",
  "prompt_path": "prompts/error-bars-defined.txt",
  "metrics": ["exact_match", "semantic_similarity", "BLEU"],
  "examples": [
    {
      "figure_path": "data/figure/10.1038_emboj.2009.312/1/",
      "expected_output_path": "data/figure/10.1038_emboj.2009.312/1/error-bars-defined/expected_output.txt"
    },
    {
      "figure_path": "data/figure/10.1038_emboj.2009.340/3/",
      "expected_output_path": "data/figure/10.1038_emboj.2009.340/3/error-bars-defined/expected_output.txt"
    }
  ]
}
```

## Checklists:

A checklist is a collection of related checks, organized in a directory structure. Each check in a checklist has its own directory containing:

```
checklist/
├── fig-checklist/                    # Main checklist directory
│   ├── error-bars-defined/           # Individual check directory
│   │   ├── prompts/                  # Directory containing prompt templates
│   │   ├── benchmark.json           # Test examples and expected outputs
│   │   └── schema.json              # JSON schema for check output
│   ├── individual-data-points/
│   │   ├── prompts/
│   │   ├── benchmark.json
│   │   └── schema.json
│   └── ...
└── doc-checklist/                    # Another checklist
    └── ...
```

Each check directory contains:
- `prompts/`: Directory containing the prompt templates used for the check
- `benchmark.json`: Contains the test examples and their expected outputs
- `schema.json`: Defines the structure of the expected output for the check

## Benchmarking data:

The structure of the repository keeps each example as human readable directories, grouping the image, the caption as well as the expected output for each of the checks:

    data/
      ├── examples/
      │   ├── 10.1038_embor.2009.233/
      │   │   ├── content/
      │   │   │   ├── figure.png
      │   │   │   └── caption.txt
      │   │   └── checks/
      │   │       ├── check-experimental-method-mentioned/
      │   │       │   └── expected_output.txt
      │   │       └── check-error-bars-defined/
      │   │           └── expected_output.txt
      │   ├── 10.1038_embor.2009.217/
      │   │   ├── content/
      │   │   │   ├── figure.png
      │   │   │   └── caption.txt
      │   │   └── checks/
      │   │       └── ...
      │   └── ...
      │
      ├── checklist/
      │   └── mini/
      │       └── error-bars-defined/
      │           ├── prompt.txt
      │           ├── schema.json
      │           └── benchmark.json
      │
      └── evaluation/
          └── results/
              └── minimal-requirement-for-figure-caption/
                  ├── check-experimental-method-mentioned_metrics.json
                  └── check-error-bars-defined_metrics.json


## Exepected output:

The expected output is a JSON file that contains the expected output for each of the checks.

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




