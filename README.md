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
│   ├── curation.py
│   ├── model_api.py
│   ├── model_cache.py
│   ├── evaluation.py
│   ├── examples.py
│   ├── scripts/
│   │   ├── __init__.py
│   │   ├── run.py
│   │   ├── curate.py
│   │   └── visualize.py
│   ├── utils/
│   │   └── hash_utils.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── checklist/
│   │   │   ├── doc-checklist/
│   │   │   │   ├── extract-figures/
│   │   │   │   ├── section-order/
│   │   │   │   └── species-identified/
│   │   │   ├── docconsistency-checklist/
│   │   │   ├── fig-checklist/
│   │   │   │   ├── error-bars-defined/
│   │   │   │   ├── individual-data-points/
│   │   │   │   ├── micrograph-scale-bar/
│   │   │   │   ├── micrograph-symbols-defined/
│   │   │   │   ├── plot-axis-units/
│   │   │   │   ├── plot-gap-labeling/
│   │   │   │   ├── replicates-defined/
│   │   │   │   ├── stat-significance-level/
│   │   │   │   ├── stat-test/
│   │   │   │   └── structure-identified/
│   │   │   └── figclarity-checklist/
│   │   ├── examples/
│   │   │   ├── 10.1038_emboj.2009.312/
│   │   │   ├── 10.1038_emboj.2009.340/
│   │   │   ├── 10.1038_embor.2009.217/
│   │   │   ├── 10.1038_embor.2009.233/
│   │   │   ├── 10.1038_s44318-025-00409-0/
│   │   │   ├── 10.1038_s44318-025-00412-5/
│   │   │   ├── 10.1038_s44318-025-00416-1/
│   │   │   ├── 10.1038_s44319-025-00415-7/
│   │   │   ├── 10.1038_s44319-025-00432-6/
│   │   │   ├── 10.1038_s44319-025-00438-0/
│   │   │   ├── 10.1038_s44320-025-00092-7/
│   │   │   ├── 10.1038_s44320-025-00094-5/
│   │   │   ├── 10.1038_s44320-025-00096-3/
│   │   │   ├── 10.1038_s44321-025-00219-1/
│   │   │   ├── 10.1038_s44321-025-00224-4/
│   │   │   ├── EMBOJ-2024-119734R/
│   │   │   ├── EMBOR-2025-61250V2/
│   │   │   ├── EMM-2025-21341/
│   │   │   └── EMM-2025-21532/
│   │   ├── evaluation/
│   │   │   ├── doc-checklist/
│   │   │   │   └── section-order/
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
│   ├── plots-doc-checklist.ipynb
│   ├── plots-fig-checklist.ipynb
│   ├── images/
│   └── cache/
├── pyproject.toml
├── requirements.txt
├── run_tests.py
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

A checklist is a collection of related checks, organized in a directory structure. The project currently supports four main checklist types:

### 1. Document Checklists (`doc-checklist/`)
Checks that analyze entire documents or manuscript structure:
- **extract-figures**: Extracts and identifies figures from documents
- **section-order**: Validates the order and structure of document sections
- **species-identified**: Checks if species are properly identified in the document

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
- **structure-identified**: Ensures structures are properly identified

### 3. Document Consistency Checklists (`docconsistency-checklist/`)
Checks for consistency across document sections (currently empty)

### 4. Figure Clarity Checklists (`figclarity-checklist/`)
Checks for figure clarity and readability (currently empty)

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
├── docconsistency-checklist/         # Document consistency checks
└── figclarity-checklist/             # Figure clarity checks
```

Each check directory contains:
- `prompts/`: Directory containing the prompt templates used for the check
- `benchmark.json`: Contains the test examples and their expected outputs
- `schema.json`: Defines the structure of the expected output for the check

## Benchmarking data:

The structure of the repository keeps each example as human readable directories, grouping the content files as well as the expected output for each of the checks:

### Examples Structure:

    data/examples/
      ├── 10.1038_emboj.2009.312/           # Figure-based example
      │   ├── content/
      │   │   └── 1/                        # Figure 
      │   │       ├── content/
      │   │       │   ├── embj2009312-fig-0001-m.jpg
      │   │       │   ├── 10.1038-emboj.2009.312Figure20093120001.pptx
      │   │       │   └── caption.txt
      │   │       └── checks/
      │   │           ├── error-bars-defined/
      │   │           ├── individual-data-points/
      │   │           ├── micrograph-scale-bar/
      │   │           ├── micrograph-symbols-defined/
      │   │           ├── plot-axis-units/
      │   │           ├── plot-gap-labeling/
      │   │           ├── replicates-defined/
      │   │           ├── stat-significance-level/
      │   │           └── stat-test/
      │   └── checks/                       # Document-level checks
      │
      ├── EMBOJ-2024-119734R/               # Document-based example
      │   ├── content/
      │   │   └── EMBOJ-2024-119734R-Manuscript_Text-mstxt.docx
      │   └── checks/
      │       └── section-order/
      │           └── expected_output.json
      │
      ├── 10.1038_embor.2009.217/           # Another example
      │   ├── content/
      │   │   └── 4/                        # Figure 
      │   │       ├── content/
      │   │       │   ├── figure.jpg
      │   │       │   ├── figure.pptx
      │   │       │   └── caption.txt
      │   │       └── checks/
      │   └── checks/
      │       └── section-order/
      │           └── expected_output.json
      └── ...

### Checklist Structure:

    data/checklist/
      ├── doc-checklist/                    # A series of checks
      │   ├── extract-figures/
      │   │   ├── prompts/
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

### Evaluation Structure:

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




