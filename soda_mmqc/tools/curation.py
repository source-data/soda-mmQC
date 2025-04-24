import streamlit as st
import json
from pathlib import Path
from PIL import Image
import pandas as pd
from datetime import datetime
import argparse
from soda_mmqc.config import get_checklist, EXAMPLES_DIR

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


def load_example_data(example_path, checklist=None):
    """Load all relevant data for an example."""
    content_dir = Path(example_path) / "content"
    checks_dir = Path(example_path) / "checks"
    
    # Initialize default values
    data = {
        "caption": "No caption available",
        "image_path": None,
        "check_outputs": {}
    }
    
    # Load caption if available
    caption_file = content_dir / "caption.txt"
    if caption_file.exists():
        try:
            with open(caption_file, "r") as f:
                data["caption"] = f.read().strip()
        except Exception as e:
            st.warning(f"Error reading caption: {e}")
    
    # Load image if available
    try:
        # Find and validate image file
        image_path = None
        for ext in [".png", ".jpg", ".jpeg", ".tiff"]:
            for img_file in content_dir.glob(f"*{ext}"):
                image_path = img_file
                break
            if image_path:
                break
        if image_path:
            data["image_path"] = str(image_path)
    except Exception as e:
        st.warning(f"Error finding images: {e}")
    
    # Load check outputs if available
    if checks_dir.exists():
        for check_dir in checks_dir.glob("*"):
            if check_dir.is_dir():
                # Only load check outputs for checks that exist in the checklist
                if checklist is None or check_dir.name in checklist:
                    expected_output_path = check_dir / "expected_output.json"
                    if expected_output_path.exists():
                        try:
                            with open(expected_output_path, "r") as f:
                                data["check_outputs"][check_dir.name] = json.load(f)
                        except Exception as e:
                            st.warning(
                                f"Error reading {check_dir.name} output: {e}"
                            )
                else:
                    st.warning(
                        f"Check {check_dir.name} not found in checklist, "
                        "skipping its output"
                    )
    
    return data


def save_check_output(example_path, check_name, output_data):
    """Save the updated check output."""
    # update just in time the field "updated_at"
    output_data["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    output_path = Path(example_path) / "checks" / check_name / "expected_output.json"
    try:
        # Ensure the directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(output_data, f, indent=4)
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
   

def main(checklist_name):
    st.title("mmQC Curation")
    
    # Initialize session state for tracking saved files
    if "saved_files" not in st.session_state:
        st.session_state.saved_files = {}
    
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
    selected_doi = st.selectbox(
        "Select Example",
        list(example_hierarchy.keys())
    )
    
    if selected_doi:
        # Figure selection
        selected_fig = st.selectbox(
            "Select Figure",
            example_hierarchy[selected_doi],
            format_func=lambda x: x.name
        )
        
        if selected_fig:
            # Load example data with checklist for validation
            example_data = load_example_data(selected_fig, checklist)
            
            # Create three columns for the layout with adjusted widths
            col1, col2, col3 = st.columns([1, 0.7, 1.3])
            
            # Column 1: Image
            with col1:
                st.header("Figure")
                if example_data["image_path"]:
                    try:
                        image = Image.open(example_data["image_path"])
                        
                        # Define and decorate the dialog function with large width
                        @st.dialog('Zoomed in', width="large")
                        def show_magnified():
                            st.image(image)
                        
                        # Button to trigger dialog
                        if st.button("", icon=":material/zoom_in:"):
                            show_magnified()  # Actually call the function
                        
                        # Show the main image
                        st.image(image, use_container_width=True)
                        
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
                        list(example_data["check_outputs"].keys())
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
                            
                            # Convert list fields to comma-separated strings for editing
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
                                        if not value:
                                            processed_item[key] = None
                                        elif isinstance(value, str):
                                            processed_item[key] = value
                                        elif isinstance(value, list):
                                            # For lists, join with commas
                                            processed_item[key] = ", ".join(str(v) for v in value)
                                        else:
                                            processed_item[key] = json.dumps(value, ensure_ascii=False)
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
                            
                            # Update button
                            if st.button(f"Save changes for {selected_check}"):
                                # Parse strings back to their original types
                                processed_records = []
                                for _, row in edited_df.iterrows():
                                    processed_record = row.to_dict()
                                    # Parse each field back to its original type
                                    for key, value in processed_record.items():
                                        if key in schema_path:
                                            if not value:
                                                processed_record[key] = None
                                            else:
                                                try:
                                                    # First try to parse as JSON
                                                    parsed = json.loads(value)
                                                    processed_record[key] = parsed
                                                except json.JSONDecodeError:
                                                    # If JSON parsing fails, check if it's a comma-separated list
                                                    if "," in value:
                                                        processed_record[key] = [
                                                            v.strip() for v in value.split(",")
                                                        ]
                                                    else:
                                                        processed_record[key] = value
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