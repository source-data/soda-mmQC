import streamlit as st
import json
import yaml
from pathlib import Path
from PIL import Image
import pandas as pd
from datetime import datetime
import argparse
from soda_mmqc.config import get_checklist, EXAMPLES_DIR
from soda_mmqc import logger
from soda_mmqc.examples import FigureExample

# Set page config for wider layout
st.set_page_config(
    page_title="SODA MMQC Example Visualizer",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for smaller caption text
st.markdown("""
    <style>
    /* Override Streamlit's default styles with !important */
    div[data-testid="stMarkdown"] p {
        font-size: 0.8em !important;
    }
    </style>
    """, unsafe_allow_html=True)


def get_workspace_root():
    """Get the workspace root directory."""
    current_dir = Path(__file__).resolve().parent
    return current_dir.parent.parent


def load_example_data(doi, fig, checklist=None):
    """Load all relevant data for an example.
    
    Args:
        doi: The DOI of the example
        fig: Path object pointing to the figure directory
        checklist: Optional dictionary of valid checks
        
    Returns:
        Dictionary containing example data or None if loading fails
    """
    try:
        # Create FigureExample object with the correct dictionary structure
        example_dict = {
            "doi": doi,
            "figure_id": fig.name  # Use the directory name as figure_id
        }
        example = FigureExample(example_dict)
        
        # Convert to dict to maintain compatibility with existing code
        data = example.to_dict()
        
        # Load check outputs if available
        checks_dir = fig / "checks"
        if checks_dir.exists():
            for check_dir in checks_dir.glob("*"):
                if check_dir.is_dir():
                    # Only load check outputs for checks that exist in the checklist
                    if checklist is None or check_dir.name in checklist:
                        try:
                            data["check_outputs"][check_dir.name] = example.get_expected_output(check_dir.name)
                        except FileNotFoundError:
                            st.warning(
                                f"Expected output not found for {check_dir.name}"
                            )
                    else:
                        st.warning(
                            f"Check {check_dir.name} not found in checklist, "
                            "skipping its output"
                        )
        
        return data
    except Exception as e:
        st.error(f"Error loading example data: {e}")
        return None


def save_check_output(example_path, check_name, output_data):
    """Save the updated check output.
    
    Args:
        example_path: Path object pointing to the figure directory
        check_name: Name of the check
        output_data: Dictionary containing the output data
        
    Returns:
        bool: True if save was successful, False otherwise
    """
    try:
        # Create FigureExample object with the correct dictionary structure
        example_dict = {
            "doi": example_path.parent.name,  # Parent directory is the DOI
            "figure_id": example_path.name    # Directory name is the figure_id
        }
        example = FigureExample(example_dict)
        
        # Save using Example class method
        example.save_expected_output(output_data, check_name)
        return True
    except Exception as e:
        st.error(f"Error saving output: {e}")
        return False


def get_example_hierarchy(examples_dir):
    """Get the hierarchical structure of examples."""
    hierarchy = {}
    for doi_dir in examples_dir.glob("*"):
        if doi_dir.is_dir():
            hierarchy[doi_dir.name] = []
            for fig_dir in doi_dir.glob("*"):
                if fig_dir.is_dir():
                    hierarchy[doi_dir.name].append(fig_dir)
    return hierarchy


def load_checklist(checklist_dir):
    """Load the checklist."""
    checklist = {}
    # load the whole checklist structure as a dictionary with the same structure as the checklist_dir
    # there is a schema.json file in this directory
    # and there is a benchmark.json file
    # then there is a prompts/ subdirectorie that includes several *.txt prompt files
    
    for check_dir in checklist_dir.glob("*"):
        if check_dir.is_dir():
            checklist[check_dir.name] = {}
            # load the schema.json file
            schema_path = check_dir / "schema.json"
            checklist[check_dir.name]["schema"] = {}
            if schema_path.exists():
                with open(schema_path, "r") as f:
                    checklist[check_dir.name]["schema"] = json.load(f)
            # verify that the check_dir.name is the same as the name of the schema
            if checklist[check_dir.name]["schema"]["format"]["name"] != check_dir.name:
                st.error(f"The name of the schema {checklist[check_dir.name]['schema']['format']['name']} does not match the name of the check {check_dir.name}")
            # load the benchmark.json file
            benchmark_path = check_dir / "benchmark.json"
            checklist[check_dir.name]["benchmark"] = {}
            if benchmark_path.exists():
                with open(benchmark_path, "r") as f:
                    checklist[check_dir.name]["benchmark"] = json.load(f)
            # load the prompts
            prompts_dir = check_dir / "prompts" 
            checklist[check_dir.name]["prompts"] = {}  # Initialize prompts dictionary
            if prompts_dir.exists():
                for prompt_file in prompts_dir.glob("*.txt"):
                    with open(prompt_file, "r") as f:
                        checklist[check_dir.name]["prompts"][prompt_file.name] = f.read()
    return checklist
   

def serialize_value(value):
    """Serialize a value to a string representation."""
    if value is None:
        return None
    elif isinstance(value, str):
        return value
    elif isinstance(value, (list, tuple, dict)):
        return yaml.dump(value, default_flow_style=False, sort_keys=False, allow_unicode=True)
    else:
        return str(value)


def deserialize_value(value, schema):
    """Deserialize a value according to its schema type.
    
    Args:
        value: The value to deserialize
        schema: The JSON schema defining the value's type and structure
        
    Returns:
        The deserialized value. If value is None, returns an appropriate empty value
        based on the schema type (empty string for strings, empty list for arrays,
        empty dict for objects, None for other types).
    """
    if value is None:
        schema_type = schema.get("type")
        if schema_type == "string":
            return ""
        elif schema_type == "array":
            return []
        elif schema_type == "object":
            return {}
        else:
            return None  # For numbers, booleans, etc.
    
    schema_type = schema.get("type")
    
    if schema_type == "array":
        try:
            # Try to parse as YAML first
            if isinstance(value, str):
                parsed = yaml.safe_load(value)
            else:
                parsed = value
            if isinstance(parsed, list):
                # deserialize each item in the list
                parsed_list = [deserialize_value(item, schema["items"]) for item in parsed]
                return parsed_list
            return value
            # Otherwise try JSON as fallback
        except Exception as e:
            error_msg = f"{e.__class__.__name__} deserializing YAML string: {value}\n{e}"
            st.session_state.deserialization_errors.append(error_msg)
            logger.error(error_msg)
            return value
    elif schema_type == "object":
        try:
            # Try to parse as YAML first
            if isinstance(value, str):
                parsed = yaml.safe_load(value)
            else:
                parsed = value
            if isinstance(parsed, dict):
                # deserialize each item in the dictionary
                parsed_dict = {k: deserialize_value(parsed[k], schema["properties"][k]) for k in schema["required"]}
                return parsed_dict
            return value
        except Exception as e:
            error_msg = f"{e.__class__.__name__} deserializing YAML string: {value} with schema requiring: {schema['required']}"
            st.session_state.deserialization_errors.append(error_msg)
            logger.error(error_msg)
            return value
    elif schema_type == "string":
        return str(value)
    elif schema_type == "number":
        try:
            return float(value)
        except ValueError:
            return value
    elif schema_type == "integer":
        try:
            return int(value)
        except ValueError:
            return value
    elif schema_type == "boolean":
        if isinstance(value, str):
            return value.lower() == "true"
        return bool(value)
    else:
        # If no type specified or unknown type, try YAML first, then JSON
        try:
            return yaml.safe_load(value)
        except yaml.YAMLError:
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                return value


def main(checklist_name):
    st.title("mmQC Benchmark Curation")

    # Initialize session state for tracking saved files and deserialization errors
    if "saved_files" not in st.session_state:
        st.session_state.saved_files = {}
    if "deserialization_errors" not in st.session_state:
        st.session_state.deserialization_errors = []

    # load the checklist
    checklist_dir = get_checklist(checklist_name)
    checklist = load_checklist(checklist_dir)

    st.subheader(checklist_name)
    
    examples_dir = EXAMPLES_DIR
    
    if not examples_dir.exists():
        st.error(f"Examples directory not found at {examples_dir}")
        return
    
    # Get the hierarchical structure
    example_hierarchy = get_example_hierarchy(examples_dir)
    if not example_hierarchy:
        st.warning("No example directories found")
        return
    
    # DOI selection
    col_select_doi, col_select_fig, _ = st.columns([0.2, 0.2, 0.6])

    with col_select_doi:
        selected_doi = st.selectbox(
            "Select Paper",
            list(example_hierarchy.keys()),
            help="Select the paper to curate"
        )
    
    if selected_doi:
        # Figure selection
        with col_select_fig:
            selected_fig = st.selectbox(
                "Select Figure",
                example_hierarchy[selected_doi],
                format_func=lambda x: x.name,
                help="Select the figure to curate"
            )
        
        if selected_fig:
            # Load example data with checklist for validation
            example_data = load_example_data(selected_doi, selected_fig, checklist)
            
            # Create three columns for the layout with adjusted widths
            col1, col2, col3 = st.columns([1, 0.7, 1.3])
            
            # Column 1: Image
            with col1:
                st.header("Figure")
                if example_data["image_path"]:
                    try:
                        image = Image.open(example_data["image_path"])
                        st.image(image)
                    except Exception as e:
                        st.error(f"Error displaying image: {e}")
                else:
                    st.info("No image available")
            
            # Column 2: Caption
            with col2:
                st.header("Caption")
                st.markdown(example_data["caption"])
            
            # Column 3: Check Output
            with col3:
                st.header("Expected Output")
                if example_data["check_outputs"]:
                    # Check selection dropdown
                    selected_check = st.selectbox(
                        "Select Check",
                        list(example_data["check_outputs"].keys()),
                        help=f"Select one of the checks from {checklist_name}"
                    )
                    
                    if selected_check:
                        
                        first_prompt = list(checklist[selected_check]["prompts"].values())[0]
                        # show prompt as collapsible section
                        with st.expander("Prompt"):
                            st.code(
                                first_prompt,
                                language="text",
                                wrap_lines=True
                            )
                        
                        output_data = example_data["check_outputs"][selected_check]
                        if "outputs" in output_data:
                            # Verify that the selected check exists in the checklist
                            if selected_check not in checklist:
                                st.error(f"Check {selected_check} not found in checklist")
                                return
                            
                            processed_outputs = []
                            # Get the schema for the selected check
                            schema_format = checklist[selected_check]["schema"]["format"]
                            schema = schema_format["schema"]
                            schema_props = schema["properties"]["outputs"]
                            schema_path = schema_props["items"]["properties"]
                            
                            for item in output_data["outputs"]:
                                processed_item = item.copy()
                                # Serialize values to strings
                                for key, value in processed_item.items():
                                    if key in schema_path:
                                        processed_item[key] = serialize_value(value)
                                if not all(value is None for value in processed_item.values()):
                                    processed_outputs.append(processed_item)
                            
                            df = pd.DataFrame(processed_outputs)
                            edited_df = st.data_editor(
                                df,
                                num_rows="dynamic",
                                height=300
                            )
                            
                            # Create a unique key for this file
                            file_key = f"{selected_fig}_{selected_check}"
                            
                            # Check if the file has been saved in this session
                            has_been_saved = file_key in st.session_state.saved_files
                            
                            # Show warning if the file has never been checked and not saved in this session
                            if "updated_at" not in output_data and not has_been_saved:
                                st.error("This annotation file has never been checked!")
                            else:
                                st.success(f"Last saved on {output_data['updated_at']}")
                            
                            
                            # Display any deserialization errors from previous run
                            if st.session_state.deserialization_errors:
                                for error in st.session_state.deserialization_errors:
                                    st.error(error)
                                # Clear the errors after displaying them
                                st.session_state.deserialization_errors = []

                            # Update button
                            if st.button(f"Save changes for {selected_check}"):
                                # Clear previous deserialization errors
                                st.session_state.deserialization_errors = []
                                
                                # Parse strings back to their original types
                                processed_records = []
                                for _, row in edited_df.iterrows():
                                    processed_record = row.to_dict()
                                    # Parse each field back to its original type
                                    for key, value in processed_record.items():
                                        if key in schema_path:
                                            # deserialize the value according to schema
                                            processed_record[key] = deserialize_value(
                                                value,
                                                schema_path[key]
                                            )
                                    processed_records.append(processed_record)
                                
                                output_data["outputs"] = processed_records
                                if save_check_output(selected_fig, selected_check, output_data):
                                    # Mark this file as saved in the session state
                                    st.session_state.saved_files[file_key] = True
                                    # Force a rerun to update the UI
                                    st.rerun()
                else:
                    st.info("No check outputs available for this example")


if __name__ == "__main__":
    # takes a command line argument for the checklist file
    # use argparse to parse the command line argument
    parser = argparse.ArgumentParser()
    parser.add_argument("checklist", type=str, help="Name of the checklist to curate.")
    args = parser.parse_args()
    main(args.checklist)