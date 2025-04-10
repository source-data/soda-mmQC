# SODA MMQC Visualization

This module provides visualization tools for the SODA MMQC project.

## Error Bars Defined Visualization

The `plot_results.py` script generates an interactive multipanel bar chart showing the performance of the error bars defined checklist across different prompts and metrics.

### Features

- Creates a multipanel figure with one panel for each field (e.g., panel_label, error_bar_on_figure)
- Shows mean scores for each metric and prompt combination with standard deviation error bars
- Organizes results by field, with metrics and prompts on the X-axis
- Creates an interactive HTML file that can be viewed in a web browser
- Allows easy comparison between different prompts and metrics

### Usage

1. Make sure you have the required dependencies installed:
   ```bash
   pip install -r soda_mmqc/visualization/requirements.txt
   ```

2. Run the visualization script:
   ```bash
   python -m soda_mmqc.visualization.plot_results
   ```

3. Open the generated HTML file in a web browser:
   - `soda_mmqc/visualization/results/error-bars-defined_scores.html`

### Customization

You can modify the `plot_results.py` script to:
- Change the color scheme
- Add more interactive features
- Modify the layout or formatting
- Add additional statistics or metrics
- Change the arrangement of panels
- Visualize different checklists by changing the `checklist_name` and `check_name` variables 