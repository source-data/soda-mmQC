# SODA MMQC

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

After installation, you can run the tool using:

```bash
# Run all checklists
soda-mmqc

# Run a specific checklist
soda-mmqc --checklist checklist-name

# Set logging level
soda-mmqc --log-level DEBUG

# Specify custom results directory
soda-mmqc --results-dir path/to/results
```

## Project Structure

```
soda-mmqc/
├── soda_mmqc/
│   ├── __init__.py
│   ├── scripts/
│   │   ├── __init__.py
│   │   └── run.py
│   ├── data/
│   │   ├── __init__.py
│   │   └── checklist/
│   ├── model_api.py
│   └── evaluation.py
├── pyproject.toml
└── README.md
```

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

Several checks can be grouped in a single JSON file, for instance checks related to the statistics, editorial and microscopy standards can be grouped in their respective checklist files:


    data/checklist/
      ├── editorial.json       # Checklist for editorial compliance
      ├── statistics.json      # Checklist for statistical integrity
      ├── microscopy.json      # Checklist for microscopy-specific guidelines


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




