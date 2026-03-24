#!/usr/bin/env bash
# Create a new figure example directory with content/<figure_id>/content structure.
# Usage: ./make_example_dir.sh <doc_id> [figure_id]
# Example: ./make_example_dir.sh 10.1038_myarticle 1
#
# Run from repo root, or set EXAMPLES_ROOT to soda_mmqc/data/examples.

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXAMPLES_ROOT="${EXAMPLES_ROOT:-${SCRIPT_DIR}/../data/examples}"

DOC_ID="${1:?Usage: $0 <doc_id> [figure_id]}"
FIGURE_ID="${2:-1}"

CONTENT_DIR="${EXAMPLES_ROOT}/${DOC_ID}/content/${FIGURE_ID}/content"
mkdir -p "$CONTENT_DIR"

CAPTION_FILE="${CONTENT_DIR}/caption.txt"
if [[ ! -f "$CAPTION_FILE" ]]; then
  touch "$CAPTION_FILE"
  echo "Created ${CAPTION_FILE} (empty)"
fi

echo "Created: ${CONTENT_DIR}"
echo "Add figure image (e.g. .png, .jpg) and edit caption.txt. Optional: source_data/ or source_data/<panel>/ for western-blot checks."


## doing it by foot....

mkdir -p /Users/sonntag/dev/soda-mmQC/soda_mmqc/data/examples/10.1038_s44319-025-00631-1/content/1/content
mkdir -p /Users/sonntag/dev/soda-mmQC/soda_mmqc/data/examples/10.1038_s44319-025-00631-1/content/2/content
mkdir -p /Users/sonntag/dev/soda-mmQC/soda_mmqc/data/examples/10.1038_s44319-025-00631-1/content/3/content
mkdir -p /Users/sonntag/dev/soda-mmQC/soda_mmqc/data/examples/10.1038_s44319-025-00631-1/content/4/content
mkdir -p /Users/sonntag/dev/soda-mmQC/soda_mmqc/data/examples/10.1038_s44319-025-00631-1/content/5/content
mkdir -p /Users/sonntag/dev/soda-mmQC/soda_mmqc/data/examples/10.1038_s44319-025-00631-1/content/6/content
mkdir -p /Users/sonntag/dev/soda-mmQC/soda_mmqc/data/examples/10.1038_s44319-025-00631-1/content/7/content

mkdir -p /Users/sonntag/dev/soda-mmQC/soda_mmqc/data/examples/10.1038_s44319-025-00631-1/content/1/content/source_data
mkdir -p /Users/sonntag/dev/soda-mmQC/soda_mmqc/data/examples/10.1038_s44319-025-00631-1/content/2/content/source_data
mkdir -p /Users/sonntag/dev/soda-mmQC/soda_mmqc/data/examples/10.1038_s44319-025-00631-1/content/3/content/source_data
mkdir -p /Users/sonntag/dev/soda-mmQC/soda_mmqc/data/examples/10.1038_s44319-025-00631-1/content/4/content/source_data
mkdir -p /Users/sonntag/dev/soda-mmQC/soda_mmqc/data/examples/10.1038_s44319-025-00631-1/content/5/content/source_data
mkdir -p /Users/sonntag/dev/soda-mmQC/soda_mmqc/data/examples/10.1038_s44319-025-00631-1/content/6/content/source_data
mkdir -p /Users/sonntag/dev/soda-mmQC/soda_mmqc/data/examples/10.1038_s44319-025-00631-1/content/7/content/source_data

touch  /Users/sonntag/dev/soda-mmQC/soda_mmqc/data/examples/10.1038_s44319-025-00631-1/content/1/content/caption.txt
touch  /Users/sonntag/dev/soda-mmQC/soda_mmqc/data/examples/10.1038_s44319-025-00631-1/content/2/content/caption.txt
touch  /Users/sonntag/dev/soda-mmQC/soda_mmqc/data/examples/10.1038_s44319-025-00631-1/content/3/content/caption.txt
touch  /Users/sonntag/dev/soda-mmQC/soda_mmqc/data/examples/10.1038_s44319-025-00631-1/content/4/content/caption.txt
touch  /Users/sonntag/dev/soda-mmQC/soda_mmqc/data/examples/10.1038_s44319-025-00631-1/content/5/content/caption.txt
touch  /Users/sonntag/dev/soda-mmQC/soda_mmqc/data/examples/10.1038_s44319-025-00631-1/content/6/content/caption.txt
touch  /Users/sonntag/dev/soda-mmQC/soda_mmqc/data/examples/10.1038_s44319-025-00631-1/content/7/content/caption.txt