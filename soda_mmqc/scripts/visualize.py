#!/usr/bin/env python3
import argparse
from pathlib import Path
from soda_mmqc.tools.visualize import (
    create_checklist_report,
    create_global_checklist_visualization,
    process_check
)
from soda_mmqc.config import get_plots_path


def main():
    parser = argparse.ArgumentParser(
        description='Create visualizations for checklist evaluation results.'
    )
    parser.add_argument(
        'checklist_name',
        help='Name of the checklist (e.g., "mini")'
    )
    parser.add_argument(
        '--output-dir',
        '-o',
        type=str,
        help='Directory to save visualizations (default: from config)'
    )
    parser.add_argument(
        '--check',
        '-c',
        type=str,
        help='Specific check to visualize (if not provided, all checks will be processed)'
    )
    parser.add_argument(
        '--global-only',
        '-g',
        action='store_true',
        help='Only create the global visualization'
    )
    
    args = parser.parse_args()
    
    # Use default plots directory from config if not specified
    output_dir = Path(args.output_dir) if args.output_dir else get_plots_path()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.global_only:
        # Create only the global visualization
        create_global_checklist_visualization(args.checklist_name, output_dir)
    elif args.check:
        # Create visualization for a specific check
        process_check(args.checklist_name, args.check, output_dir)
    else:
        # Create both individual check visualizations and global visualization
        create_checklist_report(args.checklist_name, output_dir)
        create_global_checklist_visualization(args.checklist_name, output_dir)


if __name__ == '__main__':
    main() 