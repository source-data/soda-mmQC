#!/usr/bin/env python3
import argparse
from pathlib import Path
from soda_mmqc.tools.visualize import create_checklist_report


def main():
    parser = argparse.ArgumentParser(
        description='Create visualizations for checklist results'
    )
    parser.add_argument(
        'checklist_name',
        help='Name of the checklist to visualize'
    )
    parser.add_argument(
        '--output-dir',
        '-o',
        default='soda_mmqc/data/plots',
        help=(
            'Directory to save the visualizations '
            '(default: soda_mmqc/data/plots)'
        )
    )
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create the visualizations
    create_checklist_report(args.checklist_name, output_dir)


if __name__ == '__main__':
    main() 