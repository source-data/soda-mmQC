The open multimodal checklist for scientific figures and data (mmQC)
===============================================================

This is an open library of multimodal prompts to verify the quality, rigor and compliance to editorial policies of scientific figures, including the image depicting the results, the respective figure caption and the linked research data. 

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


Benchmarking:

For each of these checks we design a prompt that can be optmized based on a test set assembling examples covering the expected outcomes.

Each check is defined by a JSON file in the `data/checks/` directory.

{
  "id": "CHECK-001",
  "name": "check-experimental-method-mentioned",
  "description": "Checks whether the experimental method, platform, or assay is explicitly mentioned in the figure caption.",
  "prompt_path": "prompts/CHECK-001_prompt.txt",
  "metrics": ["exact_match", "semantic_similarity", "BLEU"],
  "examples": [
    {
      "figure_id": "FIG001",
      "figure_path": "data/figures/FIG001/",
      "expected_output_path": "data/figures/FIG001/check-experimental-method-mentioned/expected_output.txt"
    },
    {
      "figure_id": "FIG002",
      "figure_path": "data/figures/FIG002/",
      "expected_output_path": "data/figures/FIG002/check-experimental-method-mentioned/expected_output.txt"
    }
  ]
}

Checklists:

Several checks can be grouped in a single JSON file, for instance checks related to the statistics, editorial and microscopy standards can be grouped in their respective checklist files:

data/checklist/
  ├── editorial.json       # Checklist for editorial compliance
  ├── statistics.json      # Checklist for statistical integrity
  ├── microscopy.json      # Checklist for microscopy-specific guidelines

Structure of the data:

The structure of the repository keeps each example as human readable directories, grouping the image, the caption as well as the expected output for each of the checks:

data/
  ├── figures/
  │   ├── FIG001/
  │   │   ├── FIG001.png
  │   │   ├── caption.txt
  │   │   ├── check-experimental-method-mentioned/
  │   │   │   ├── expected_output.txt
  │   │
  │   ├── FIG002/
  │   │   ├── FIG002.tiff
  │   │   ├── caption.txt
  │   │   ├── check-experimental-method-mentioned/
  │   │   │   ├── expected_output.txt
  │
  ├── prompts/
  │   ├── CHECK-001_prompt.txt
  │   ├── CHECK-002_prompt.txt
  │
  ├── checklist/
  │   ├── editorial.json       # Checklist for editorial compliance
  │   ├── statistics.json      # Checklist for statistical integrity
  │   ├── microscopy.json      # Checklist for microscopy-specific guidelines
  │
  ├── evaluation/
  │   ├── results/
  │   │   ├── editorial/
  │   │   │   ├── CHECK-001_metrics.json
  │   │   │   ├── CHECK-005_metrics.json
  │   │   ├── statistics/
  │   │   │   ├── CHECK-010_metrics.json
  │   │   │   ├── CHECK-011_metrics.json
  │   │   ├── microscopy/
  │   │   │   ├── CHECK-020_metrics.json

