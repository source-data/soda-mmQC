import json
import numpy as np
import pandas as pd
import plotly.express as px
from pathlib import Path
from soda_mmqc.config import get_schema_path, get_evaluation_path, get_plots_path
from soda_mmqc import logger


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
    analysis_file = base_path / 'analysis.json'
    if analysis_file.exists():
        with open(analysis_file, 'r') as f:
            return json.load(f)
    return {}


def prepare_data_for_plotting(results_by_prompt):
    """Convert the results into a pandas DataFrame for plotting.
    
    Args:
        results_by_prompt: Dictionary mapping prompt names to their analysis results
        features: List of feature names to plot
        
    Returns:
        DataFrame with columns: feature, metric, prompt, mean, std
    """

    
    metrics = ['exact_match', 'semantic_similarity', 'BLEU']
    logger.debug(f"Processing data for metrics: {metrics}")
    
    # Create a list to store all data points
    data_points = []
    # {
    # "prompt.1": [
    #     {
    #         "doi": "10.1038_emboj.2009.312",
    #         "figure_id": "1",
    #         "analysis": {
    #             "exact_match": {
    
    for prompt_name, results in results_by_prompt.items():
        logger.debug(f"Processing prompt: {prompt_name}")
        for metric in metrics:
            if not results or metric not in results[0]['analysis']:
                continue

            # Get the overall score for this metric across all examples
            overall_scores = []
            field_scores = {}
            # The structure of the data is as follows:
            # "analysis": {
            #     "exact_match": {
            #         "score": 1.0,
            #         "field_scores": {
            #             "output_0": {
            #                 "score": 1.0,
            #                 "field_scores": {
            #                     "panel_label": {
            #                         "score": 1.0,
            #                         "field_scores": {
            #                             "value": 1.0
            #                         }
            #                     },
            #                     "error_bar_on_figure": {
            #                         "score": 1.0,
            #                         "field_scores": {
            #                             "value": 1.0
            #                         }
            #                     },
            for result in results:
                # 1 result per figure
                # several outputs per figure (one per panel)

                if 'analysis' not in result or metric not in result['analysis']:
                    continue
                
                analysis = result['analysis'][metric]

                # Get overall score for this figure if available
                if 'score' in analysis:
                    overall_scores.append(analysis['score'])
                
                # Get field scores from each output
                if 'field_scores' in analysis:
                    for output_key, output_data in analysis['field_scores'].items():
                        # one output per panel
                        if 'field_scores' in output_data:
                            for field_name, field_data in output_data['field_scores'].items():
                                # each field is a "feature" or one of the "tasks" that makes up the check
                                # Get the score from the field_scores.value
                                if 'score' in field_data:
                                    if field_name not in field_scores:
                                        field_scores[field_name] = []
                                    field_scores[field_name].append(
                                        field_data['score']
                                    )
            
            # Calculate statistics for overall score over all figures
            if overall_scores:
                data_points.append({
                    'feature': 'overall',
                    'metric': metric,
                    'prompt': prompt_name,
                    'mean': np.mean(overall_scores),  # mean over the figures
                    'std': np.std(overall_scores)
                })
            
            # Calculate statistics for each field over all panels
            for feature, scores in field_scores.items():
                if scores:  # Only add if we have scores for this feature
                    data_points.append({
                        'feature': feature,
                        'metric': metric,
                        'prompt': prompt_name,
                        'mean': np.mean(scores),
                        'std': np.std(scores)
                    })
    
    # Convert to DataFrame
    df = pd.DataFrame(data_points)
    logger.debug(f"Created DataFrame with {len(df)} data points")
    return df


def create_multipanel_chart(df, output_dir, check_name):
    """Create a multipanel bar chart with one panel for each feature.
    
    Args:
        df: DataFrame with columns: feature, metric, prompt, mean, std
        output_dir: Directory to save the output file
        check_name: Name of the check (e.g., 'error-bars-defined')
    """
    # Get unique features
    features = df['feature'].unique()
    logger.debug(f"Creating multipanel chart for features: {features}")

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
    logger.info(f"Combined visualization saved to {output_path}")


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
            has_analysis = (check_dir / 'analysis.json').exists()
            if has_analysis:
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
    logger.info(f"Processing check: {check_name}")
    
    # Load schema and get features
    try:
        schema = load_schema(checklist_name, check_name)
        features = get_features_from_schema(schema)
        logger.debug(f"Loaded schema and found features: {features}")
    except ValueError as e:
        logger.error(f"Error loading schema: {e}")
        return False
    
    # Load and process results
    results_by_prompt = load_analysis_results(checklist_name, check_name)
    if not results_by_prompt:
        logger.warning(f"No results found for check: {check_name}")
        return False

    df = prepare_data_for_plotting(results_by_prompt)
    if df.empty:
        logger.warning(f"No data to plot for check: {check_name}")
        return False

    # Create visualization
    create_multipanel_chart(df, output_dir, check_name)
    return True


def create_checklist_report(checklist_name, output_dir):
    """Create a single report for all checks in a checklist.
    
    Args:
        checklist_name: Name of the checklist (e.g., 'mini')
        output_dir: Directory to save the output file
    """
    # Get all checks for this checklist
    checks = get_checks_for_checklist(checklist_name)
    if not checks:
        logger.warning(f"No checks found for checklist: {checklist_name}")
        return

    logger.info(f"Processing {len(checks)} checks for checklist: {checklist_name}")
    # Process each check
    for check_name in checks:
        process_check(checklist_name, check_name, output_dir)


def create_global_checklist_visualization(checklist_name, output_dir):
    """Create a comprehensive visualization of all checks in a checklist.
    
    Args:
        checklist_name: Name of the checklist (e.g., 'mini')
        output_dir: Directory to save the output file
    """
    # Get all checks for this checklist
    checks = get_checks_for_checklist(checklist_name)
    if not checks:
        logger.warning(f"No checks found for checklist: {checklist_name}")
        return

    logger.info(f"Creating global visualization for {len(checks)} checks")
    
    # Collect data from all checks
    all_data = []
    for check_name in checks:
        try:
            results = load_analysis_results(checklist_name, check_name)
            
            if not results:
                continue
                
            df = prepare_data_for_plotting(results)
            if not df.empty:
                df['check'] = check_name
                all_data.append(df)
                
        except Exception as e:
            logger.error(f"Error processing check {check_name}: {e}")
            continue
    
    if not all_data:
        logger.warning("No data collected for visualization")
        return
        
    # Combine all data
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Get unique metrics
    metrics = combined_df['metric'].unique()
    
    # Create a separate plot for each metric
    for metric in metrics:
        # Filter data for this metric
        metric_df = combined_df[combined_df['metric'] == metric]
        
        # Separate overall scores and feature scores
        overall_scores = metric_df[metric_df['feature'] == 'overall']
        feature_scores = metric_df[metric_df['feature'] != 'overall']
        # Create the visualization
        fig = px.bar(
            overall_scores,
            x='check',
            y='mean',
            error_y='std',
            title=(
                f'{metric.replace("_", " ").title()} Scores - '
                f'{checklist_name.title()} Checklist'
            ),
            labels={
                'mean': 'Score',
                'check': 'Check'
            },
            color="check",
            color_discrete_sequence=px.colors.qualitative.Set1,
            template='plotly_dark'
        )
        
        # Add individual feature scores as points
        fig.add_trace(
            px.scatter(
                feature_scores,
                x='check',
                y='mean',
                symbol='feature',
                labels='feature',
                hover_data=['check', 'feature'],
                size=[10] * len(feature_scores),  # Set fixed size for all points
            ).data[0]
        )
        
        # Update layout for better readability
        fig.update_layout(
            height=600,
            width=1200,
            showlegend=False,
            title_x=0.5,
            title_font_size=12,
            barmode='overlay',
        )
        
        # Update axes
        fig.update_xaxes(tickangle=45)
        fig.update_yaxes(range=[0, 1], title="Score")
        
        # Add hover template
        fig.update_traces(
            hovertemplate="<br>".join([
                "Check: %{x}",
                "Score: %{y:.2f}",
                "<extra></extra>"
            ])
        )
        
        # Save the visualization
        output_path = Path(output_dir) / f'{checklist_name}_{metric}_analysis.html'
        fig.write_html(str(output_path))
        logger.info(f"Visualization for {metric} saved to {output_path}")
    
    # Create a summary statistics table
    summary_df = combined_df.groupby(['check', 'metric', 'feature']).agg({
        'mean': ['mean', 'std', 'min', 'max']
    }).round(3)
    
    # Save summary statistics
    summary_path = Path(output_dir) / f'{checklist_name}_summary_stats.csv'
    summary_df.to_csv(summary_path)
    logger.info(f"Summary statistics saved to {summary_path}")
    
    return summary_df 