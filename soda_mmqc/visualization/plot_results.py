import json
import numpy as np
import pandas as pd
import plotly.express as px
from pathlib import Path


def load_analysis_results(checklist_name, check_name):
    """Load the analysis results from JSON files for all prompts.
    
    Args:
        checklist_name: Name of the checklist (e.g., 'mini')
        check_name: Name of the check (e.g., 'error-bars-defined')
        
    Returns:
        Dictionary mapping prompt names to their analysis results
    """
    base_path = Path(
        f'soda_mmqc/data/evaluation/{checklist_name}/{check_name}'
    )
    
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


def prepare_data_for_plotting(results_by_prompt):
    """Convert the results into a pandas DataFrame for plotting.
    
    Args:
        results_by_prompt: Dictionary mapping prompt names to their analysis results
        
    Returns:
        DataFrame with columns: field, metric, prompt, mean, std
    """
    fields = [
        'panel_label',
        'error_bar_on_figure',
        'error_bar_defined_in_caption',
        'from_the_caption'
    ]
    metrics = ['exact_match', 'semantic_similarity', 'BLEU']
    
    # Create a list to store all data points
    data_points = []
    
    for prompt_name, results in results_by_prompt.items():
        for metric in metrics:
            for field in fields:
                # Skip fields that don't exist in the results
                if not results or (results[0]['analysis'].get(metric, {}).get(field, None) is None):
                    continue
                values = [result['analysis'][metric][field] for result in results]
                mean = np.mean(values)
                std = np.std(values)
                
                data_points.append({
                    'field': field,
                    'metric': metric,
                    'prompt': prompt_name,
                    'mean': mean,
                    'std': std
                })
    
    # Convert to DataFrame
    df = pd.DataFrame(data_points)
    return df


def create_multipanel_chart(df, output_dir, check_name):
    """Create a multipanel bar chart with one panel for each field using Plotly Express.
    
    Args:
        df: DataFrame with columns: field, metric, prompt, mean, std
        output_dir: Directory to save the output file
        check_name: Name of the check (e.g., 'error-bars-defined')
    """
    # Get unique fields
    fields = df['field'].unique()
    
    # Create a figure for each field
    for field in fields:
        # Filter data for this field
        field_df = df[df['field'] == field]
        
        # Create the bar chart
        fig = px.bar(
            field_df,
            x='metric',
            y='mean',
            color='prompt',
            error_y='std',
            title=f'Task "{field.replace("_", " ").title()}" Performance',
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
            yaxis_range=[0, 1],
            template='plotly_white',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        # Save the figure
        output_path = Path(output_dir) / f'{check_name}_{field}_scores.html'
        fig.write_html(str(output_path))
        print(f"Visualization saved to {output_path}")
    
    # Create a combined figure with subplots
    fig = px.bar(
        df,
        x='metric',
        y='mean',
        color='prompt',
        error_y='std',
        facet_row='field',
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
        height=300 * len(fields),
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
    for i in range(1, len(fields) + 1):
        fig.update_yaxes(range=[0, 1], row=i, col=1)
    
    # Save the combined figure
    output_path = Path(output_dir) / f'{check_name}_combined_scores.html'
    fig.write_html(str(output_path))
    print(f"Combined visualization saved to {output_path}")


def main():
    # Define paths
    checklist_name = 'mini'
    check_name = 'error-bars-defined'
    output_dir = 'soda_mmqc/visualization/results'
    
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Load and process results
    results_by_prompt = load_analysis_results(checklist_name, check_name)
    df = prepare_data_for_plotting(results_by_prompt)
    
    # Create visualization
    create_multipanel_chart(df, output_dir, check_name)


if __name__ == '__main__':
    main() 