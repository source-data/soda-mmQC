import json
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from soda_mmqc.config import (
    CHECKLIST_DIR,
    EVALUATION_DIR,
)
from soda_mmqc.core.examples import EXAMPLE_TYPES
from soda_mmqc import logger
from typing import Dict, Any


def load_schema(checklist_name, check_name):
    """Load the schema for a check from the checklist directory.
    
    Args:
        checklist_name: Name of the checklist (e.g., 'mini')
        check_name: Name of the check (e.g., 'error-bars-defined')
        
    Returns:
        Dictionary containing the schema definition
    """
    # Get the checklist directory using the config function
    schema_path = CHECKLIST_DIR / checklist_name / check_name / "schema.json"
    
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


def data_to_tabular(analysis, doc_id, item_id, aggregation_level, metric, prompt_name):
    assert isinstance(analysis, dict), f"Analysis is not a dict: {analysis}"
    logger.debug(f"data to tabular for {item_id} at aggregation level {aggregation_level}")
    try:
        # Get overall score for this figure if available
        item_data_point = {
            'doc_id': doc_id,
            'item_id': item_id,
            'aggregation_level': aggregation_level,
            'metric': metric,
            'prompt': prompt_name,
            # scores aggregate over all panel for this figure
            'score': analysis.get('score', None),
            'std_score': analysis.get('std_score', None),
            'precision': analysis.get('precision', None),
            'recall': analysis.get('recall', None),
            'f1_score': analysis.get('f1_score', None),
            'field': 'all_fields_aggregated'
        }
    except Exception as e:
        logger.error(f"Error processing document to tabular data {item_id}: {e}")
        logger.error(f"Analysis: {analysis}")
        return []
    data_points = [item_data_point]
    if 'field_scores' in analysis:
        for field, field_data in analysis['field_scores'].items():
            field_data_point = {
                'doc_id': doc_id,
                'item_id': item_id,
                'aggregation_level': aggregation_level,
                'metric': metric,
                'prompt': prompt_name,
                'score': field_data.get('score', None),
                'std_score': field_data.get('std_score', None),
                'precision': field_data.get('precision', None),
                'recall': field_data.get('recall', None),
                'f1_score': field_data.get('f1_score', None),
                'field': field
            }
            data_points.append(field_data_point)
    # the scores for each element, recursively
    try:
        for subitem_id, subitem_analysis in analysis['element_scores'].items():
            subitem_data_points = data_to_tabular(
                analysis=subitem_analysis,
                doc_id=doc_id,
                item_id=item_id + '/' + subitem_id,
                aggregation_level=aggregation_level + 1,
                metric=metric,
                prompt_name=prompt_name
            )
            if subitem_data_points is not None:
                data_points.extend(subitem_data_points)
    except Exception as e:
        logger.error(f"Error processing subitem scores for document to tabular data {item_id}/{subitem_id}: {e}")
    return data_points


def prepare_data_for_plotting(
    results_by_prompt: Dict[str, Any],
    metric: str = "semantic_similarity"
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Convert the results into a pandas DataFrame for plotting.
    
    Args:
        results_by_prompt: Dictionary mapping prompt names to their analysis results
        
    Returns:
        DataFrame with columns: feature, metric, prompt, mean, std, precision, recall, f1_score
    """

    # Create a list to store all data points
    tabular_data = []
    tabular_metadata = []

    for prompt_name, results in results_by_prompt.items():
        logger.debug(f"Processing prompt: {prompt_name}")
        for document in results:
            analysis = document['analysis']
            metadata = document['metadata']
            new_rows = data_to_tabular(
                analysis,
                doc_id=document['doc_id'],
                item_id=document['doc_id'],
                aggregation_level=0,
                metric=metric,
                prompt_name=prompt_name
            )
            tabular_data.extend(new_rows)
            tabular_metadata.append(metadata)
    # Convert to DataFrame
    try:
        df_data = pd.DataFrame(tabular_data)
        df_metadata = pd.DataFrame(tabular_metadata)
    except Exception as e:
        logger.error(f"Error creating DataFrame: {e}")
        logger.error(f"Tabular data: {tabular_data}")
        return pd.DataFrame(), pd.DataFrame()
    logger.debug(f"Created DataFrame with {len(df_data)} data points")
    logger.debug(f"Created DataFrame with {len(df_metadata)} metadata rows")
    return df_data, df_metadata


def get_checks_for_checklist(checklist_name):
    """Get all check directories for a given checklist.
    
    Args:
        checklist_name: Name of the checklist (e.g., 'mini')
        
    Returns:
        List of check names
    """
    # Get the evaluation directory using the config function
    checklist_path = EVALUATION_DIR / checklist_name
    if not checklist_path.exists():
        raise ValueError(f"Checklist directory not found: {checklist_path}")

    # Get all subdirectories that contain analysis results
    checks = []
    for check_dir in checklist_path.iterdir():
        if check_dir.is_dir():
            checks.append(check_dir.name)

    return checks


def get_check_data(checklist_name, check_name, model) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    """Get data for all checks in a checklist.

    Args:
        checklist_name: Name of the checklist (e.g., 'mini')
        check_name: Name of the check (e.g., 'error-bars-defined')
    Returns:
        DataFrame with columns: doi, figure_id, panel_id, aggregation_level, metric, prompt, score, std_score, precision, recall, f1_score, detailed_scores, task_scores
    """

    analysis_file = EVALUATION_DIR / checklist_name / check_name / model / 'analysis.json'
    if analysis_file.exists():
        with open(analysis_file, 'r') as f:
            results = json.load(f)
    else:
        logger.warning(f"Does not exist: {str(analysis_file)}")
        return None, None

    try:
        df_data, df_metadata = prepare_data_for_plotting(results)
    except Exception as e:
        logger.error(f"Error preparing data for check {check_name}: {e}")
        return None, None

    if not df_data.empty:
        df_data['check'] = check_name
        df_metadata['check'] = check_name
        return df_data, df_metadata
    else:
        logger.warning(f"No data found for check {check_name}")
        return None, None


def checklist_visualization(
    checklist_name, 
    model,
    output_dir=None,
    metric="semantic_similarity",
    score="score",
    aggregation_level=0
):
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
        df_data, df_metadata = get_check_data(checklist_name, check_name, model)
        if df_data is not None:
            data.append(df_data)

    if not data:
        logger.warning(f"No data found for checklist: {checklist_name}")
        return

    # Combine all data
    df_data = pd.concat(data, ignore_index=True)
    
    # Check that chosen metrics is available
    if metric not in df_data['metric'].unique():
        logger.warning(f"Metric {metric} not found in data")
        return

    prompts = list(df_data['prompt'].unique())
    checks = list(df_data['check'].unique())
    
    # Create a global mapping from check names to positions
    global_check_to_num = {check: j for j, check in enumerate(checks)}
    
    color_map = {
        p: px.colors.qualitative.Set1[i % len(px.colors.qualitative.Set1)]
        for i, p in enumerate(prompts)
    }
    
    plot = go.Figure()

    # Add scatter plots for each prompt
    for i, prompt in enumerate(prompts):
        logger.info(f"Creating plot for prompt: {prompt}...")
        offset_width = 1 / (len(prompts)+1)
        x_offset = (i - (len(prompts) - 1) / 2) * offset_width
        # Filter data for this prompt and metric
        plotting_item_data = df_data.loc[
            (df_data['prompt'] == prompt) &
            (df_data['metric'] == metric) &
            (df_data['field'] == 'all_fields_aggregated') &
            (df_data['aggregation_level'] == aggregation_level)
        ]

        # Get the checks that actually have data for this prompt
        checks_with_data = plotting_item_data['check'].unique()
        
        # Calculate aggregated scores for checks that have data
        num_points = plotting_item_data.groupby('check')['score'].count()
        avg_scores = plotting_item_data.groupby('check')['score'].mean()
        std_scores = plotting_item_data.groupby('check')['score'].std()
        
        # Use the checks that actually have data, but in the order they appear in the global checks list
        checks_with_data = [check for check in checks if check in avg_scores.index]
        
        # Add a small offset to x-coordinates to prevent overlapping
        x_positions = [global_check_to_num[check] + x_offset for check in checks_with_data]
        plot.add_trace(go.Bar(
            x=x_positions,
            y=[avg_scores[check] for check in checks_with_data],  # Use scores in the correct order
            error_y=dict(
                type='data',
                array=[std_scores[check] for check in checks_with_data],  # Use std in the correct order
                visible=True,
                color="grey",
                thickness=1,
                width=3
            ),
            name=prompt,
            marker_color=color_map[prompt],
            showlegend=True,
            width=offset_width,  # Control the width of the bars
            hoverinfo='text',
            hovertext=[
                f"Check: {check}<br>Average Score: {avg_scores[check]:.3f}<br>Prompt: {prompt}<br>Num Points: {num_points[check]}"
                for check in checks_with_data
            ]
        ))

        # Add jitter to x positions for individual data points
        jitter = np.random.normal(0, 0.03, size=len(plotting_item_data))
        x_scattered_positions = [
            global_check_to_num[check] + x_offset + j
            for check, j in zip(plotting_item_data['check'], jitter)
        ]

        plot.add_trace(go.Scatter(
            x=x_scattered_positions,
            y=plotting_item_data['score'],
            mode='markers',
            name=prompt,
            marker=dict(
                color="white",  #color_map[prompt],
                size=10,
                opacity=0.4,
                line=dict(width=0, color='white')
            ),
            showlegend=True,
            hovertext=plotting_item_data['item_id']
        ))

    plot.update_layout(
        width=1000,
        height=600,
        title=dict(
            text=f'Benchmarking of "{checklist_name.title()}"<br>'
            f'<span style="font-size: 0.5em; color: #888;">Comparing values with {metric.replace("_", " ")}</span><br>'
            f'<span style="font-size: 0.5em; color: #888;">Model: {model}</span>',
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

    return plot, df_data


def check_visualization(
    checklist_name,
    check_name,
    model,
    output_dir=None,
    score: str = "score",  # true_positives, false_positives, false_negatives, precision, recall, f1_score, semantic_similarity
    metric: str = "semantic_similarity",
    aggregation_level=1
):
    """Create a visualization of a specific check in a checklist.
    
    Args:
        checklist_name: Name of the checklist (e.g., 'mini')
        check_name: Name of the check (e.g., 'error-bars-defined')
        output_dir: Directory to save the output file
        metric: Metric to visualize
    """

    # Get data for this check
    df_data, df_metadata = get_check_data(checklist_name, check_name, model)
    if df_data is None:
        logger.warning(f"No data found for check {check_name}")
        return  
    
    # Check that chosen metrics is available
    if metric not in df_data['metric'].unique():
        logger.warning(f"Metric {metric} not found in data")
        return
    
    prompts = list(df_data['prompt'].unique())
    color_map = {p: px.colors.qualitative.Set1[i % len(px.colors.qualitative.Set1)] for i, p in enumerate(prompts)}
    
    plot = go.Figure()
    
    for i, prompt in enumerate(prompts):
        item_data = df_data.loc[
            (df_data['prompt'] == prompt) &
            (df_data['metric'] == metric) &
            (df_data['field'] != 'all_fields_aggregated') &
            (df_data['aggregation_level'] == aggregation_level)
        ]
        
        logger.info(f"Creating plot ({prompt}, {metric}), plotting {score} for {check_name}...")
        logger.info(f"Plotting data: {len(item_data)} rows")

        # Add a small offset to x-coordinates to prevent overlapping
        offset_width = 1 / (len(prompts)+1)
        x_offset = (i - (len(prompts) - 1) / 2) * offset_width
        # Create numerical x-positions by mapping fields to numbers and adding offset
        fields = item_data['field'].unique()
        field_to_num = {field: j for j, field in enumerate(fields)}
        jitter = np.random.normal(0, 0.04, size=len(item_data))
        x_scattered_positions = [
            field_to_num[field] + x_offset + j
            for field, j in zip(item_data['field'], jitter)
        ]

        plot.add_trace(go.Scatter(
            x=x_scattered_positions,
            y=item_data[score],
            name=prompt,  # Use prompt name instead of task column
            mode='markers',
            marker=dict(
                color="white",
                size=5,
                opacity=0.4,
                line=dict(width=0, color='white')
            ),
            showlegend=True,
            hovertext=item_data['item_id']
        ))

        average_score = item_data.groupby(['field'])[score].mean().reset_index()
        std_score = item_data.groupby(['field'])[score].std().reset_index()
        x_positions = [
            field_to_num[field] + x_offset
            for field in average_score['field']
        ]
          
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
                f"Field: {f}<br>{s}: {s:.3f}<br>Prompt: {prompt}" 
                for f, s in zip(average_score['field'], average_score[score])
            ]
        ))

    # Format num_points string
    num_points = item_data.groupby('field')['item_id'].count()
    min_points = num_points.min()
    max_points = num_points.max()
    num_points_str = str(min_points) if min_points == max_points else f"{min_points} - {max_points}"

    plot.update_layout(
        width=800,
        height=600,
        title=f'{score.replace("_", " ")} for {check_name.replace("_", " ")} (n={num_points_str})<br>'
        f'<span style="font-size: 0.8em; color: #888;">Comparing values with {metric.replace("_", " ")}</span><br>'
        f'<span style="font-size: 0.8em; color: #888;">Model: {model}</span>',
        xaxis=dict(
            title='Fields',
            tickangle=45,
            ticktext=fields,
            tickvals=list(range(len(fields))),
            range=[-0.5, len(fields) - 0.5]  # Add some padding on the sides
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

    return plot, df_data, df_metadata


def check_report(
    checklist_name, 
    check_name, 
    model,
    k=3, 
    search_id=None,
    doc_id=None,
    score: str = "score",  # true_positives, false_positives, false_negatives, precision, recall, f1_score, semantic_similarity
    metric: str = "semantic_similarity",
    prompt: str = "",
    aggregation_level=1
):
    """Display a comprehensive report of a specific check in a checklist as HTML in a notebook.
    Args:
        checklist_name: Name of the checklist (e.g., 'mini')
        check_name: Name of the check (e.g., 'error-bars-defined')
        k: Number of worst panels to consider per task
        doi: DOI of the paper to display
        figure_id: Figure ID of the figure to display
    """
    df_data, df_metadata = get_check_data(checklist_name, check_name, model)
    if df_data is None:
        logger.warning(f"No data found for check {check_name}")
        return
    if prompt:
        prompts = [prompt]
    else:
        prompts = list(df_data['prompt'].unique())
    # item_data = df[df['aggregation_level'] == aggregation_level]
    # fields = item_data['field'].unique()

    # Aggregate problematic figures and panels
    # problematic_items = defaultdict(lambda: {
    #     'fields': defaultdict(dict),
    #     'prompt': defaultdict(set),
    # })

    for prompt in prompts:
        item_data = df_data[
            (df_data['metric'] == metric) &
            (df_data['prompt'] == prompt) &
            (df_data['aggregation_level'] == aggregation_level) &
            (df_data['field'] == 'all_fields_aggregated')
        ]
        if doc_id is not None:
            item_data = item_data[item_data['doc_id'] == str(doc_id)]
        elif search_id is not None:
            # search doi field for search_doi
            id_match = item_data['item_id'].str.contains(search_id)
            item_data = item_data[id_match]

        logger.debug(f"Item data: {
            item_data[['item_id', 'score']].head(6)
        }")
        # check worse items with general score < 1.0
        not_perfect = item_data.loc[
            item_data[score] < 0.99
        ]
        bad_items = not_perfect[not_perfect[score] < 0.6]
        if len(bad_items) > 0:
            worst_items = bad_items.sort_values(by=score, ascending=True)
        else:
            worst_items = not_perfect.sort_values(by=score, ascending=True).head(k)
       
        logger.debug(f"Worst items ({prompt}): {worst_items}")
        field_data = df_data.loc[
            (df_data['metric'] == metric) &
            (df_data['prompt'] == prompt) &
            (df_data['aggregation_level'] == aggregation_level) &
            (df_data['field'] != 'all_fields_aggregated') &
            (df_data['item_id'].isin(worst_items['item_id']))
        ]
        # return worst_items, field_data
        # joing worst_items and metadata based on doc_id
        # this will make it easy to retrieve the example path and the example type
        worst_items = worst_items.merge(df_metadata, on='doc_id', how='left')
        
        # return worst_items, field_data
        html = ""
        for index, row in worst_items.iterrows():
            field_data_for_doc = field_data[field_data['item_id'] == row['item_id']]
        
            # make a first table that lists the elements and the fields that are problematic
            problematic_fields_table_rows = ""
            for i, row_field in field_data_for_doc.iterrows():
                problematic_fields_table_rows += f"<tr><td>{row_field['field']}</td><td>{row_field['score']:.3f}</td><td>{row_field['f1_score']:.3f}</td></tr>"
            problematic_fields_table_html = f"""
            <table style='border-collapse: collapse; width: 80%; margin: 20px;'>
                <tr><th>Field</th><th>Score</th><th>F1 Score</th></tr>
                {problematic_fields_table_rows}
            </table>
            """
            html += f"""
            <h3>Paper: {row['doc_id']} - Element: {row['item_id']} with score {row['score']:.3f}</h3>
            <div style='display:flex; flex-direction:row; align-items:flex-start;'>
                {problematic_fields_table_html}
            </div>
            """
        return html, worst_items, field_data
        
        # example_path = EXAMPLES_DIR / row['source']
        # example_type = row['example_type']
        # example = EXAMPLE_TYPES[example_type](example_path)
        # example.load_from_source()
    #     prediction_output_table_html = ""
    #     for prompt in prompts:
    #         # get the analysis for this figure
    #         analysis_path = get_evaluation_path(checklist_name) / check_name / model / 'analysis.json'
    #         if analysis_path.exists():
    #             with open(analysis_path, 'r') as f:
    #                 analysis_data = json.load(f)
    #                 prompt_data = analysis_data.get(prompt, [])
    #                 for figure_data in prompt_data:
    #                     if figure_data.get('doi') == doi and figure_data.get('figure_id') == figure_id:
    #                         model_outputs = figure_data.get('model_output', {}).get('outputs', [])
    #                         expected_results = figure_data.get('expected_output', {})
    #                         break
                    
    #         prediction_table_rows = "<tr>"
    #         for task in tasks:
    #             prediction_table_rows += f"<td>{task}</td>"
    #         prediction_table_rows += "</tr>"
    #         for model_out in model_outputs:
    #             prediction_table_rows += "<tr>"
    #             for k, v in model_out.items():
    #                 prediction_table_rows += f"<td>{v}</td>"
    #             prediction_table_rows += "</tr>"
    #         logger.info(f"num predictions: {len(model_outputs)}")
            
    #         prediction_output_table_html += f"""
    #         <div style='display:flex; flex-direction:row; align-items:flex-start;'>
    #         <table style='border-collapse: collapse; width: 80%; margin: 20px;'>
    #             <tr><th>Predictions with {prompt}</th></tr>
    #             {prediction_table_rows}
    #         </table>
    #         </div>
    #         """
    #     # Image
    #     img_html = f"<img src='data:image/png;base64,{img_b64}' style='max-width:400px; vertical-align:top; margin-right:20px;'/>"
    #     # Caption
    #     caption_html = f"<div style='display:inline-block; max-width:400px; vertical-align:top; padding:10px; border:1px solid white; background:#000000; color:#FFFFFF;'>{caption}</div>"
    #     # Layout
    
        #     html += f"""
        #     <h3>Paper: {row['doc_id']} - Element: {row['item_id']} with model {row['model']}</h3>
        #     <div style='display:flex; flex-direction:row; align-items:flex-start;'>
        #         {problematic_fields_table_html}
        #     </div>
        #     """
        # return html, worst_items
        # display(HTML(html))
        # """
        # {prediction_output_table_html}
        # <div style='display:flex; flex-direction:row; align-items:flex-start;'>
        #     {img_html}
        #     {caption_html}
        # </div>
        # <hr>
        # """
