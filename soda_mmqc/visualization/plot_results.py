import json
import numpy as np
import pandas as pd
import plotly.express as px
from pathlib import Path
import argparse
from soda_mmqc.config import get_schema_path, get_evaluation_path
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def load_schema(checklist_name, check_name):
    """Load the schema for a check from the checklist directory.
    
    Args:
        checklist_name: Name of the checklist (e.g., 'mini')
        check_name: Name of the check (e.g., 'error-bars-defined')
        
    Returns:
        Dictionary containing the schema definition
    """
    # Get the checklist directory using the config function
    schema_path = get_schema_path(checklist_name, check_name)
    
    if not schema_path.exists():
        raise ValueError(f"Schema file not found: {schema_path}")
    
    with open(schema_path, 'r') as f:
        schema = json.load(f)
    
    return schema


def get_features_from_schema(schema):
    """Extract feature names from the schema.
    
    Args:
        schema: Dictionary containing the schema definition
        
    Returns:
        List of feature names found in the schema
    """
    try:
        # Navigate to the properties of the first output item
        return schema['format']['schema']['properties']['outputs']['items']['required']
    except (KeyError, TypeError):
        return []


def load_analysis_results(checklist_name, check_name):
    """Load the analysis results from JSON files for all prompts.
    
    Args:
        checklist_name: Name of the checklist (e.g., 'mini')
        check_name: Name of the check (e.g., 'error-bars-defined')
        
    Returns:
        Dictionary mapping prompt names to their analysis results
    """
    # Get the evaluation directory using the config function
    base_path = get_evaluation_path(checklist_name) / check_name

    # Check if we have a summary file
    summary_file = base_path / 'summary.json'
    if summary_file.exists():
        with open(summary_file, 'r') as f:
            return json.load(f)

    # If no summary file, load individual prompt results
    results = {}
    for prompt_dir in base_path.iterdir():
        if prompt_dir.is_dir() and (prompt_dir / 'analysis.json').exists():
            prompt_name = prompt_dir.name
            with open(prompt_dir / 'analysis.json', 'r') as f:
                results[prompt_name] = json.load(f)

    return results


def prepare_data_for_plotting(results_by_prompt, features):
    """Convert the results into a pandas DataFrame for plotting.
    
    Args:
        results_by_prompt: Dictionary mapping prompt names to their analysis results
        features: List of feature names to plot
        
    Returns:
        DataFrame with columns: feature, metric, prompt, mean, std
    """
    if not features:
        print("No features found in the schema")
        return pd.DataFrame()
    
    metrics = ['exact_match', 'semantic_similarity', 'BLEU']
    
    # Create a list to store all data points
    data_points = []
    
    for prompt_name, results in results_by_prompt.items():
        for metric in metrics:
            for feature in features:
                # Skip features that don't exist in the results
                if (not results or 
                    results[0]['analysis'].get(metric, {}).get(feature, None) is None):
                    continue
                values = [result['analysis'][metric][feature] for result in results]
                mean = np.mean(values)
                std = np.std(values)
                
                data_points.append({
                    'feature': feature,
                    'metric': metric,
                    'prompt': prompt_name,
                    'mean': mean,
                    'std': std
                })
    
    # Convert to DataFrame
    df = pd.DataFrame(data_points)
    return df


def create_multipanel_chart(df, output_dir, check_name):
    """Create a multipanel bar chart with one panel for each feature using Plotly Express.
    
    Args:
        df: DataFrame with columns: feature, metric, prompt, mean, std
        output_dir: Directory to save the output file
        check_name: Name of the check (e.g., 'error-bars-defined')
    """
    # Get unique features
    features = df['feature'].unique()

    # Create a combined figure with subplots
    fig = px.bar(
        df,
        x='metric',
        y='mean',
        color='prompt',
        error_y='std',
        facet_row='feature',
        title=f'Tasklist "{check_name.replace("-", " ").title()}" Performance',
        labels={
            'mean': 'Score',
            'metric': 'Metric',
            'prompt': 'Prompt'
        },
        barmode='group',
        color_discrete_sequence=px.colors.qualitative.Set1
    )
    
    # Update layout
    fig.update_layout(
        height=300 * len(features),
        width=1200,
        template='plotly_white',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # Update y-axis for all subplots
    for i in range(1, len(features) + 1):
        fig.update_yaxes(range=[0, 1], row=i, col=1)
    
    # Save the combined figure
    output_path = Path(output_dir) / f'{check_name}_combined_scores.html'
    fig.write_html(str(output_path))
    print(f"Combined visualization saved to {output_path}")


def get_checks_for_checklist(checklist_name):
    """Get all check directories for a given checklist.
    
    Args:
        checklist_name: Name of the checklist (e.g., 'mini')
        
    Returns:
        List of check names
    """
    # Get the evaluation directory using the config function
    checklist_path = get_evaluation_path(checklist_name)
    if not checklist_path.exists():
        raise ValueError(f"Checklist directory not found: {checklist_path}")
    
    # Get all subdirectories that contain analysis results
    checks = []
    for check_dir in checklist_path.iterdir():
        if check_dir.is_dir():
            # Check if this directory contains analysis results
            has_summary = (check_dir / 'summary.json').exists()
            has_analysis = any(
                (d / 'analysis.json').exists() 
                for d in check_dir.iterdir() 
                if d.is_dir()
            )
            if has_summary or has_analysis:
                checks.append(check_dir.name)
    
    return checks


def process_check(checklist_name, check_name, output_dir):
    """Process a single check and create its visualizations.
    
    Args:
        checklist_name: Name of the checklist (e.g., 'mini')
        check_name: Name of the check (e.g., 'error-bars-defined')
        output_dir: Path object for the output directory
        
    Returns:
        bool: True if visualization was created successfully, False otherwise
    """
    print(f"\nProcessing check: {check_name}")
    
    # Load schema and get features
    try:
        schema = load_schema(checklist_name, check_name)
        features = get_features_from_schema(schema)
    except ValueError as e:
        print(f"Error loading schema: {e}")
        return False
    
    # Load and process results
    results_by_prompt = load_analysis_results(checklist_name, check_name)
    if not results_by_prompt:
        print(f"No results found for check: {check_name}")
        return False

    df = prepare_data_for_plotting(results_by_prompt, features)
    if df.empty:
        print(f"No data to plot for check: {check_name}")
        return False

    # Create visualization
    create_multipanel_chart(df, output_dir, check_name)
    return True


def create_checklist_report(checklist_name, output_dir):
    """Create a single report for all checks in a checklist, organized by metric.
    
    Args:
        checklist_name: Name of the checklist (e.g., 'mini')
        output_dir: Directory to save the output file
    """
    # Get all checks for the checklist
    try:
        checks = get_checks_for_checklist(checklist_name)
    except ValueError as e:
        print(f"Error: {e}")
        return False
    
    if not checks:
        print(f"No checks found for checklist: {checklist_name}")
        return False
    
    # Dictionary to store data for each metric
    metric_data = {}
    
    # Process each check
    for check_name in checks:
        print(f"Processing check: {check_name}")
        
        # Load schema and get features
        try:
            schema = load_schema(checklist_name, check_name)
            features = get_features_from_schema(schema)
            
            # Skip if no features found in schema
            if not features:
                print(f"No features found in schema for check: {check_name}")
                continue
                
        except ValueError as e:
            print(f"Error loading schema for check {check_name}: {e}")
            continue
        
        # Load and process results
        results_by_prompt = load_analysis_results(checklist_name, check_name)
        if not results_by_prompt:
            print(f"No results found for check: {check_name}")
            continue
        
        # Prepare data for this check
        df = prepare_data_for_plotting(results_by_prompt, features)
        if df.empty:
            print(f"No data to plot for check: {check_name}")
            continue
        
        # Add check name to the dataframe
        df['check'] = check_name
        
        # Group by metric and append to the appropriate metric data
        for metric in df['metric'].unique():
            metric_df = df[df['metric'] == metric].copy()
            if metric not in metric_data:
                metric_data[metric] = []
            metric_data[metric].append(metric_df)
    
    # Create a report for each metric
    for metric, dfs in metric_data.items():
        if not dfs:
            continue
 
        
        # Create a figure with subplots
        fig = make_subplots(
            rows=1, 
            cols=len(dfs),
            subplot_titles=[df['check'].iloc[0].replace('-', ' ').title() 
                           for df in dfs],
            shared_yaxes=True
        )
        
        # Create individual bar charts for each check and add them to the figure
        for i, df in enumerate(dfs, 1):
            
                   
        # Check if there are multiple prompts across all checks
            prompts = df['prompt'].unique()
            
            has_multiple_prompts = len(prompts) > 1
            # Create a bar chart for this check using Plotly Express
            check_name = df['check'].iloc[0]
            check_df = df.copy()
            
            # Create the bar chart
            check_fig = px.bar(
                check_df,
                x='feature',
                y='mean',
                color='prompt' if has_multiple_prompts else None,
                error_y='std',
                title=f"{check_name.replace('-', ' ').title()}",
                labels={
                    'mean': 'Score',
                    'feature': 'Feature',
                    'prompt': 'Prompt'
                },
                # barmode='group' if has_multiple_prompts else 'relative',
                color_discrete_sequence=px.colors.qualitative.Set1,
            )
            
            # Extract traces from the check figure and add them to the main figure
            for trace in check_fig.data:
                fig.add_trace(trace, row=1, col=i)
        
        # Update layout
        fig.update_layout(
            height=600,  # Increased height to accommodate angled labels
            width=300 * len(dfs),
            template='plotly_white',
            title=(f'Checklist "{checklist_name}" Performance - '
                   f'{metric.replace("_", " ").title()}'),
            barmode='group',
            legend=dict(
                orientation="h",
                yanchor="bottom",  # Use paper (canvas) as reference
                y=-0.5,
                xanchor="right",
                x=0.2  # Position from right edge of the canvas
            ),
            margin=dict(b=100, t=150)  # Add more bottom and top margin
        )
        
        # Update y-axis for all subplots
        for i in range(1, len(dfs) + 1):
            fig.update_yaxes(range=[0, 1.2], row=1, col=i)
            fig.update_xaxes(
                row=1, 
                col=i,
                tickangle=45,  # Angle the tick labels to prevent overlap
                tickfont=dict(size=10)  # Slightly smaller font for tick labels
            )
        
        # Update y-axis title for the first subplot only
        fig.update_yaxes(title_text="Score", row=1, col=1)
        
        # Adjust the position of subplot titles to create more space between titles and plots
        for i in range(1, len(dfs) + 1):
            # Get the current title
            title = fig.layout.annotations[i-1].text
            # Update the title with more space
            fig.layout.annotations[i-1].update(
                text=title,
                y=1.05  # Move titles up to create more space (1.0 is the default)
            )
        
        # Save the combined figure
        output_path = Path(output_dir) / f'{checklist_name}_{metric}_report.html'
        fig.write_html(str(output_path))
        print(f"Report for {metric} saved to {output_path}")
    
    return True


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description='Generate visualization plots for analysis results.'
    )
    parser.add_argument(
        'checklist',
        type=str,
        help='Name of the checklist (e.g., mini)'
    )
    parser.add_argument(
        '--output-dir', 
        type=str, 
        default='soda_mmqc/visualization/results',
        help=('Directory to save the output files '
              '(default: soda_mmqc/visualization/results)')
    )
    parser.add_argument(
        '--report-only',
        action='store_true',
        help=('Generate only the checklist report '
              '(no individual check visualizations)')
    )

    args = parser.parse_args()

    # Create output directory if it doesn't exist
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate the checklist report
    create_checklist_report(args.checklist, output_dir)
    
    # If report-only flag is not set, also generate individual check visualizations
    if not args.report_only:
        # Get all checks for the checklist
        try:
            checks = get_checks_for_checklist(args.checklist)
        except ValueError as e:
            print(f"Error: {e}")
            return

        if not checks:
            print(f"No checks found for checklist: {args.checklist}")
            return

        print(f"Processing {len(checks)} checks for checklist: {args.checklist}")

        # Process each check
        successful_checks = 0
        for check_name in checks:
            if process_check(args.checklist, check_name, output_dir):
                successful_checks += 1
        
        print(f"\nProcessed {successful_checks} out of {len(checks)} checks")
    
    print(f"All visualizations have been saved to: {output_dir}")


if __name__ == '__main__':
    main() 