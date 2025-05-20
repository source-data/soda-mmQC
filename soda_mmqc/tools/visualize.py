import json
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from soda_mmqc.config import (
    get_schema_path, get_evaluation_path, get_plots_path,
    get_content_path, get_image_path, get_caption_path
)
from soda_mmqc import logger
from typing import Dict, Any
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.gridspec as gridspec
from collections import defaultdict
from IPython.display import display, HTML
import base64

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


def prepare_data_for_plotting(results_by_prompt: Dict[str, Any]) -> pd.DataFrame:
    """Convert the results into a pandas DataFrame for plotting.
    
    Args:
        results_by_prompt: Dictionary mapping prompt names to their analysis results
        
    Returns:
        DataFrame with columns: feature, metric, prompt, mean, std, precision, recall, f1_score
    """
    metrics = ['exact_match', 'semantic_similarity', 'BLEU']
    logger.debug(f"Processing data for metrics: {metrics}")
    
    # Create a list to store all data points
    data_points = []

    for prompt_name, results in results_by_prompt.items():
        logger.debug(f"Processing prompt: {prompt_name}")
        for metric in metrics:
            if not results or metric not in results[0]['analysis']:
                continue

            for figure in results:
                doi = figure['doi']
                figure_id = figure['figure_id']
                if 'analysis' not in figure or metric not in figure['analysis']:
                    continue
                
                figure_analysis = figure['analysis'][metric]
                # data structure for one model output corresponding to one figure::
                # "analysis": {
                #     "exact_match": {
                #         "score": 0.5,
                #         "errors": [],
                #         "field_scores": {
                #             "element_0": {    ##### PANEL ID #####
                #                 "score": 1.0,
                #                 "errors": [],
                #                 "field_scores": {    ##### SCORES FOR EACH TASK #####
                #                     "panel_label": {   ##### TASK NAME #####
                #                         "score": 1.0,  
                #                         "errors": [],
                #                         "field_scores": {
                #                             "value": 1.0
                #                         },
                #                         "true_positives": 1
                #                     },
                #                     "error_bar_on_figure": {
                #                         "score": 1.0,
                #                         "errors": [],
                #                         "field_scores": {
                #                             "value": 1.0
                #                         },
                #                         "true_positives": 1
                #                     },
                #                     "error_bar_defined_in_caption": {
                #                         "score": 1.0,
                #                         "errors": [],
                #                         "field_scores": {
                #                             "value": 1.0
                #                         },
                #                         "true_positives": 1
                #                     },
                #                     "from_the_caption": {
                #                         "score": 1.0,
                #                         "errors": [],
                #                         "field_scores": {
                #                             "value": 1.0
                #                         },
                #                         "true_positives": 1
                #                     }
                #                 },
                #                 "std_score": 0.0,
                #                 "true_positives": 4,
                #                 "false_positives": 0,
                #                 "false_negatives": 0,
                #                 "precision": 1.0,
                #                 "recall": 1.0,
                #                 "f1_score": 1.0
                #             }
                #         },
                #         "std_score": 0.13055824196677338,
                #         "true_positives": 42,
                #         "false_positives": 6,
                #         "false_negatives": 0,
                #         "precision": 0.875,
                #         "recall": 1.0,
                #         "f1_score": 0.9333333333333333,
                #         "detailed_scores": {
                #             "panel_label": {
                #                 "avg_score": 1.0,
                #                 "std_score": 0.0,
                #                 "num_matches": 12,
                #                 "num_false_negatives": 0,
                #                 "num_false_positives": 0,
                #                 "precision": 1.0,
                #                 "recall": 1.0,
                #                 "f1_score": 1.0
                #             }
                #         }
                #     }
                # }
                try:
                    # Get overall score for this figure if available
                    new_data_point = {
                        'doi': doi,
                        'figure_id': figure_id,
                        'panel_id': None,
                        'aggregation_level': 'figure',
                        'metric': metric,
                        'prompt': prompt_name,
                        # scores aggregate over all panel for this figure
                        'score': figure_analysis.get('score', None),
                        'std_score': figure_analysis.get('std_score', None),
                        'precision': figure_analysis.get('precision', None),
                        'recall': figure_analysis.get('recall', None),
                        'f1_score': figure_analysis.get('f1_score', None),
                        'semantic_similarity': figure_analysis.get('semantic_similarity', None)
                    }
                    # the scores detailed by TASK
                    detailed_scores = {}
                    for task, task_scores in figure_analysis['detailed_scores'].items():
                        detailed_scores[task] = {
                            'avg_score': task_scores.get('avg_score', None),
                            'std_score': task_scores.get('std_score', None),
                            'num_matches': task_scores.get('num_matches', None),
                            'num_false_negatives': task_scores.get('num_false_negatives', None),
                            'num_false_positives': task_scores.get('num_false_positives', None),
                            'precision': task_scores.get('precision', None),
                            'recall': task_scores.get('recall', None),
                            'f1_score': task_scores.get('f1_score', None),
                            'semantic_similarity': task_scores.get('semantic_similarity', None)
                        }
                    new_data_point['detailed_scores'] = detailed_scores
                    data_points.append(new_data_point)
                except Exception as e:
                    logger.error(f"Error processing figure {figure_id}: {e}")
                    continue

                try:
                    for panel_id, panel_analysis in figure_analysis['field_scores'].items():
                        new_data_point = {
                            'doi': doi,
                            'figure_id': figure_id,
                            'panel_id': panel_id,
                            'aggregation_level': 'panel',
                            'metric': metric,
                            'prompt': prompt_name,
                            # this is not so useful, aggreagation across all tasks for this panel
                            'score': panel_analysis.get('score', None),
                            'std_score': panel_analysis.get('std_score', None),
                            'true_positives': panel_analysis.get('true_positives', None),
                            'false_positives': panel_analysis.get('false_positives', None),
                            'false_negatives': panel_analysis.get('false_negatives', None),
                            'precision': panel_analysis.get('precision', None),
                            'recall': panel_analysis.get('recall', None),
                            'f1_score': panel_analysis.get('f1_score', None),
                            'semantic_similarity': panel_analysis.get('semantic_similarity', None)
                        }
                        # the scores detailed by TASK for this panel
                        # this is the most useful part
                        task_scores = {}
                        for task, scores in panel_analysis['field_scores'].items():
                            task_scores[task] = {
                                'score': scores.get('score', None),
                                'true_positives': scores.get('true_positives', None),
                                'false_positives': scores.get('false_positives', None),
                                'false_negatives': scores.get('false_negatives', None),
                                'precision': scores.get('precision', None),
                                'recall': scores.get('recall', None),
                                'f1_score': scores.get('f1_score', None),
                                'semantic_similarity': scores.get('semantic_similarity', None)
                            }
                        new_data_point['task_scores'] = task_scores
                        data_points.append(new_data_point)
                except Exception as e:
                    logger.error(f"Error processing panel {panel_id}: {e}")
                    continue
    
    # Convert to DataFrame
    df = pd.DataFrame(data_points)
    logger.debug(f"Created DataFrame with {len(df)} data points")
    return df


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


def get_check_data(checklist_name, check_name) -> pd.DataFrame | None:
    """Get data for all checks in a checklist.
    
    Args:
        checklist_name: Name of the checklist (e.g., 'mini')
        check_name: Name of the check (e.g., 'error-bars-defined')
    Returns:
        DataFrame with columns: doi, figure_id, panel_id, aggregation_level, metric, prompt, score, std_score, precision, recall, f1_score, detailed_scores, task_scores
    """

    try:
        results = load_analysis_results(checklist_name, check_name)
    except Exception as e:
        logger.error(f"Error loading analysis results for check {check_name}: {e}")
        return None

    try:
        df = prepare_data_for_plotting(results)
    except Exception as e:
        logger.error(f"Error preparing data for check {check_name}: {e}")
        return None
    
    if not df.empty:
        df['check'] = check_name
        return df
    else:
        logger.warning(f"No data found for check {check_name}")
        return None


def global_checklist_visualization(checklist_name, output_dir=None, metric="semantic_similarity"):
    """Create a comprehensive visualization of all checks in a checklist.
    
    Args:
        checklist_name: Name of the checklist (e.g., 'mini')
        output_dir: Directory to save the output file
        metric: Metric to visualize
    """
    # Get all checks for this checklist
 
    checks = get_checks_for_checklist(checklist_name)

    if not checks:
        logger.warning(f"No checks found for checklist: {checklist_name}")
        return

    logger.info(f"Creating global visualization for {len(checks)} checks")
    
    # Collect data from all checks
    data = []
    for check_name in checks:
        df = get_check_data(checklist_name, check_name)
        if df is not None:
            data.append(df)
            
    if not data:
        logger.warning(f"No data found for checklist: {checklist_name}")
        return

    # Combine all data
    df = pd.concat(data, ignore_index=True)
    
    # Check that chosen metrics is available
    if metric not in df['metric'].unique():
        logger.warning(f"Metric {metric} not found in data")
        return

    prompts = list(df['prompt'].unique())
    checks = list(df['check'].unique())
    
    color_map = {p: px.colors.qualitative.Set1[i % len(px.colors.qualitative.Set1)] for i, p in enumerate(prompts)}
    
    plot = go.Figure()

    # Add scatter plots for each prompt
    for i, prompt in enumerate(prompts):
        logger.info(f"Creating plot for prompt: {prompt}...")
        # Filter data for this prompt and metric
        plotting_panel_data = df.loc[
            (df['prompt'] == prompt) &
            (df['metric'] == metric) &
            (df['aggregation_level'] == 'panel')
        ]
        
        # Add a small offset to x-coordinates to prevent overlapping
        offset_width = 1 / (len(prompts)+1)
        x_offset = (i - (len(prompts) - 1) / 2) * offset_width
        # Create numerical x-positions by mapping checks to numbers and adding offset
        check_to_num = {check: j for j, check in enumerate(checks)}
        x_positions = plotting_panel_data['check'].map(check_to_num) + x_offset
        
        # Add jitter to x positions
        jitter = np.random.normal(0, 0.03, size=len(x_positions))
        x_positions = x_positions + jitter
                
        plot.add_trace(go.Scatter(
            x=x_positions,
            y=plotting_panel_data['score'],
            mode='markers',
            name=prompt,
            marker=dict(
                color="white", #color_map[prompt],
                size=3,
                opacity=0.4,
                line=dict(width=0, color='white')
            ),
            showlegend=True,
            hovertext=plotting_panel_data['doi'] + ' ' + plotting_panel_data['figure_id'] + ' ' + plotting_panel_data['panel_id']
        ))

        average_score = plotting_panel_data.groupby('check')['score'].mean().reset_index()
        std_score = plotting_panel_data.groupby('check')['score'].std().reset_index()
        x_positions = average_score['check'].map(check_to_num) + x_offset
        num_points = plotting_panel_data['check'].value_counts()
        
        plot.add_trace(go.Bar(
            x=x_positions,
            y=average_score['score'],  # Use just the score column
            error_y=dict(
                type='data',
                array=std_score['score'],
                visible=True,
                color="grey",
                thickness=1,
                width=3
            ),
            name=prompt,
            marker_color=color_map[prompt],
            showlegend=True,
            width=offset_width,  # Control the width of the bars
            # add num_points to hovertext
            hovertext=[
                f"Check: {check}<br>Average Score: {score:.3f}<br>Prompt: {prompt}<br>Num Points: {num_points[check]}"
                for check, score in zip(average_score['check'], average_score['score'])
            ]
        ))

    plot.update_layout(
        width=1000,
        height=600,
        title=dict(
            text=f'Benchmarking of "{checklist_name.title()}"<br><span style="font-size: 0.8em; color: #888;">Comparing values with {metric.replace("_", " ")}</span>',
            x=0.5,
            y=0.95
        ),
        title_x=0.5,
        title_font_size=24,
        template='plotly_dark',
        xaxis=dict(
            title='Check',
            tickangle=45,
            ticktext=checks,
            tickvals=list(range(len(checks))),
            range=[-0.5, len(checks) - 0.5]  # Add some padding on the sides
        ),
        yaxis=dict(
            title='Score',
            range=[0, 1.1]  # Adjusted range since scores are typically between 0 and 1
        ),
        barmode='group',
        # boxmode='group',  # This ensures boxes are grouped by check
        # boxgap=0.1,  # Controls spacing between boxes in the same group
        # boxgroupgap=0.3  # Controls spacing between different groups
    )
    
    if output_dir:
        output_path = Path(output_dir) / f'{checklist_name}_{metric}_analysis.html'
        plot.write_html(str(output_path))
        logger.info(f"Visualization for {metric} saved to {output_path}")

    return plot


def remap_task_scores_to_df(plotting_data):
    """
    Remap a DataFrame with a 'task_scores' column (containing dicts) into a flat DataFrame.
    Args:
        task_data (pd.DataFrame): DataFrame with columns ['doi', 'figure_id', 'panel_id', 'task_scores']
    Returns:
        pd.DataFrame: DataFrame with columns ['doi', 'figure_id', 'panel_id', 'task', 'score', 'true_positives', 'false_positives', 'false_negatives']
    """
    task_data = plotting_data[['doi', 'figure_id', 'panel_id', 'task_scores']]
    remapped_task_data = pd.DataFrame(columns=[
        'doi', 'figure_id', 'panel_id', 'task', 'score',
        'true_positives', 'false_positives', 'false_negatives',
        'precision', 'recall', 'f1_score', 'semantic_similarity'
    ])
    for j, row in task_data.iterrows():
        # the task_scores is a dict of task names to their scores
        task_scores = row['task_scores']
        for task_name, task_scores in task_scores.items():
            remapped_task_data.loc[len(remapped_task_data)] = {
                'doi': row['doi'],
                'figure_id': row['figure_id'],
                'panel_id': row['panel_id'],
                'task': task_name,
                'score': task_scores['score'],
                'true_positives': task_scores['true_positives'],
                'false_positives': task_scores['false_positives'],
                'false_negatives': task_scores['false_negatives'],
                'precision': task_scores['precision'],
                'recall': task_scores['recall'],
                'f1_score': task_scores['f1_score'],
                'semantic_similarity': task_scores['semantic_similarity'],
            }
    return remapped_task_data


def check_specific_plot(
    checklist_name, check_name,
    output_dir=None,
    score: str = "score",  # true_positives, false_positives, false_negatives, precision, recall, f1_score, semantic_similarity
    metric: str = "semantic_similarity",
):
    """Create a visualization of a specific check in a checklist.
    
    Args:
        checklist_name: Name of the checklist (e.g., 'mini')
        check_name: Name of the check (e.g., 'error-bars-defined')
        output_dir: Directory to save the output file
        metric: Metric to visualize
    """

    # Get data for this check
    df = get_check_data(checklist_name, check_name)
    if df is None:
        logger.warning(f"No data found for check {check_name}")
        return  
    
    # Check that chosen metrics is available
    if metric not in df['metric'].unique():
        logger.warning(f"Metric {metric} not found in data")
        return
    
    prompts = list(df['prompt'].unique())
    color_map = {p: px.colors.qualitative.Set1[i % len(px.colors.qualitative.Set1)] for i, p in enumerate(prompts)}
    
    plot = go.Figure()
    
    # scatter plot and bar chart for each tasks of the check
    num_points = {}
    for i, prompt in enumerate(prompts):
        panel_data = df.loc[
            (df['prompt'] == prompt) &
            (df['metric'] == metric) &
            (df['aggregation_level'] == 'panel')
        ]
        num_points[prompt] = len(panel_data)
        logger.info(f"Creating plot ({prompt}, {metric}), plotting {score} for {check_name}...")
        logger.info(f"Plotting data: {len(panel_data)} rows")

        remapped_task_data = remap_task_scores_to_df(panel_data)
        tasks = list(remapped_task_data['task'].unique())

        # Add a small offset to x-coordinates to prevent overlapping
        offset_width = 1 / (len(prompts)+1)
        x_offset = (i - (len(prompts) - 1) / 2) * offset_width
        # Create numerical x-positions by mapping checks to numbers and adding offset
        cat_to_num = {task: j for j, task in enumerate(tasks)}
        x_positions = remapped_task_data['task'].map(cat_to_num) + x_offset
        jitter = np.random.normal(0, 0.04, size=len(x_positions))
        x_positions = x_positions + jitter
        plot.add_trace(go.Scatter(
            x=x_positions,
            y=remapped_task_data[score],
            name=prompt,  # Use prompt name instead of task column
            mode='markers',
            marker=dict(
                color="white",
                size=3,
                opacity=0.4,
                line=dict(width=0, color='white')
            ),
            showlegend=True,
            hovertext=remapped_task_data['doi'] + ' fig. ' + remapped_task_data['figure_id'] + ' ' + remapped_task_data['panel_id']
        ))

        # check_data = df.loc[
        #     (df['prompt'] == prompt) &
        #     (df['metric'] == metric) &
        #     (df['aggregation_level'] == 'check')
        # ]

        average_score = remapped_task_data.groupby('task')[score].mean().reset_index()
        std_score = remapped_task_data.groupby('task')[score].std().reset_index()
        x_positions = average_score['task'].map(cat_to_num) + x_offset
        # Add a bar chart for each task
        plot.add_trace(go.Bar(
            x=x_positions,
            y=average_score[score],
            name=prompt,
            error_y=dict(
                type='data',
                array=std_score[score],
                visible=True,
                color="grey",
                thickness=1,
                width=3
            ),
            marker_color=color_map[prompt],
            showlegend=True,
            width=offset_width,  # Control the width of the bars
            hovertext=[
                f"Task: {t}<br>{s}: {s:.3f}<br>Prompt: {prompt}" 
                for t, s in zip(remapped_task_data['task'], average_score[score])
            ]
        ))
    
     
    
    # Format num_points string
    min_points = min(num_points.values())
    max_points = max(num_points.values())
    num_points_str = str(min_points) if min_points == max_points else f"{min_points}-{max_points}"
    
    plot.update_layout(
        width=800,
        height=600,
        title=f'{score.replace("_", " ")} for {check_name.replace("_", " ")} (n={num_points_str})<br><span style="font-size: 0.8em; color: #888;">Comparing values with {metric.replace("_", " ")}</span>',
        xaxis=dict(
            title='Task',
            tickangle=45,
            ticktext=tasks,
            tickvals=list(range(len(tasks))),
            range=[-0.5, len(tasks) - 0.5]  # Add some padding on the sides
        ),
        yaxis_title=f'{score.replace("_", " ")}',
        boxmode='group',  # Group boxes by task
        showlegend=True,
        template='plotly_dark'
    )
    if output_dir:
        output_path = Path(output_dir) / f'{check_name}_{score}_{metric}_analysis.html'
        plot.write_html(str(output_path))
        logger.info(f"Visualization for {score} saved to {output_path}")

    return plot


def check_specific_report_html(
    checklist_name, 
    check_name, 
    k=3, 
    search_doi=None, 
    figure_id=None,
    score: str = "score",  # true_positives, false_positives, false_negatives, precision, recall, f1_score, semantic_similarity
    metric: str = "semantic_similarity",
    prompt: str = "",
):
    """Display a comprehensive report of a specific check in a checklist as HTML in a notebook.
    Args:
        checklist_name: Name of the checklist (e.g., 'mini')
        check_name: Name of the check (e.g., 'error-bars-defined')
        k: Number of worst panels to consider per task
        doi: DOI of the paper to display
        figure_id: Figure ID of the figure to display
    """
    df = get_check_data(checklist_name, check_name)
    if df is None:
        logger.warning(f"No data found for check {check_name}")
        return
    if prompt:
        prompts = [prompt]
    else:
        prompts = list(df['prompt'].unique())
    all_panels = df[df['aggregation_level'] == 'panel']
    tasks = list(all_panels.iloc[0]["task_scores"].keys())

    # Aggregate problematic figures and panels
    problematic_figures = defaultdict(lambda: {
        'panel_tasks': defaultdict(dict),
        # 'prompt': defaultdict(set),
    })
    
    for task in tasks:
        for prompt in prompts:
            panel_level_data = df[
                (df['metric'] == metric) &
                (df['prompt'] == prompt) &
                (df['aggregation_level'] == 'panel')
            ]
            if search_doi is not None:
                # search doi field for search_doi
                doi_match = panel_level_data['doi'].str.contains(search_doi)
                panel_level_data = panel_level_data[doi_match]
            if figure_id is not None:
                panel_level_data = panel_level_data[panel_level_data['figure_id'] == str(figure_id)]
            remapped_task_data = remap_task_scores_to_df(panel_level_data)
            logger.debug(f"Task: {task}")
            task_data = remapped_task_data[remapped_task_data['task'] == task]
            logger.debug(f"Task data: {
                task_data[['doi', 'figure_id', 'panel_id', 'score']].head(6)
            }")
            # check worse panels with score < 1.0
            not_perfect = task_data[task_data[score] < 0.99]
            bad_panels = not_perfect[not_perfect[score] < 0.6]
            if len(bad_panels) > 0:
                worst_panels = bad_panels.sort_values(by=score, ascending=True)
            else:
                worst_panels = not_perfect.sort_values(by=score, ascending=True).head(k)
            logger.debug(f"Worst panels ({prompt}): {worst_panels}")
            for _, row in worst_panels.iterrows():
                fig_key = (row['doi'], row['figure_id'])
                problematic_figures[fig_key]['panel_tasks'][row['panel_id']][task] = row[score]

    for (doi, figure_id), info in problematic_figures.items():
        figure_dict = {'doi': doi, 'figure_id': figure_id}
        image_path = get_image_path(figure_dict)
        caption_path = get_caption_path(figure_dict)
        if image_path is None or not image_path.exists():
            continue
        if caption_path.exists():
            with open(caption_path, 'r') as f:
                caption = f.read()
        else:
            caption = "(No caption found)"
        # Image as base64
        with open(image_path, 'rb') as img_f:
            img_b64 = base64.b64encode(img_f.read()).decode('utf-8')
            
        # make a first table that lists the panels and the tasks that are problematic
        problematic_tasks_table_rows = ""
        for panel, tasks_dict in info['panel_tasks'].items():
            tasks_str = ', '.join([f"{t} [{s:.3f}]" for t, s in tasks_dict.items()])
            problematic_tasks_table_rows += f"<tr><td>{panel}</td><td>{tasks_str}</td></tr>"
        problematic_tasks_table_html = f"""
        <table style='border-collapse: collapse; width: 80%; margin: 20px;'>
            <tr><th>Panel</th><th>Problematic Tasks</th></tr>
            {problematic_tasks_table_rows}
        </table>
        """
        
        prediction_output_table_html = ""
        for prompt in prompts:
            # get the analysis for this figure
            analysis_path = get_evaluation_path(checklist_name) / check_name / 'analysis.json'
            if analysis_path.exists():
                with open(analysis_path, 'r') as f:
                    analysis_data = json.load(f)
                    prompt_data = analysis_data.get(prompt, [])
                    for figure_data in prompt_data:
                        if figure_data.get('doi') == doi and figure_data.get('figure_id') == figure_id:
                            model_outputs = figure_data.get('model_output', {}).get('outputs', [])
                            expected_results = figure_data.get('expected_output', {})
                            break
                    
            prediction_table_rows = "<tr>"
            for task in tasks:
                prediction_table_rows += f"<td>{task}</td>"
            prediction_table_rows += "</tr>"
            for model_out in model_outputs:
                prediction_table_rows += "<tr>"
                for k, v in model_out.items():
                    prediction_table_rows += f"<td>{v}</td>"
                prediction_table_rows += "</tr>"
            logger.info(f"num predictions: {len(model_outputs)}")
            
            prediction_output_table_html += f"""
            <div style='display:flex; flex-direction:row; align-items:flex-start;'>
            <table style='border-collapse: collapse; width: 80%; margin: 20px;'>
                <tr><th>Predictions with {prompt}</th></tr>
                {prediction_table_rows}
            </table>
            </div>
            """
        # Image
        img_html = f"<img src='data:image/png;base64,{img_b64}' style='max-width:400px; vertical-align:top; margin-right:20px;'/>"
        # Caption
        caption_html = f"<div style='display:inline-block; max-width:400px; vertical-align:top; padding:10px; border:1px solid white; background:#000000; color:#FFFFFF;'>{caption}</div>"
        # Layout
        html = f"""
        <h3>Paper: {doi} - Figure: {figure_id}</h3>
        <div style='display:flex; flex-direction:row; align-items:flex-start;'>
            {problematic_tasks_table_html}
        </div>
        {prediction_output_table_html}
        <div style='display:flex; flex-direction:row; align-items:flex-start;'>
            {img_html}
            {caption_html}
        </div>
        <hr>
        """
        display(HTML(html))

