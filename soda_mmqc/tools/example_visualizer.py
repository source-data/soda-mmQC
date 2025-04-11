import streamlit as st
import json
from pathlib import Path
from PIL import Image
import pandas as pd


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


def load_example_data(example_path):
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
        image_files = list(content_dir.glob("*.jpg")) + list(content_dir.glob("*.png"))
        if image_files:
            data["image_path"] = str(image_files[0])
    except Exception as e:
        st.warning(f"Error finding images: {e}")
    
    # Load check outputs if available
    if checks_dir.exists():
        for check_dir in checks_dir.glob("*"):
            if check_dir.is_dir():
                expected_output_path = check_dir / "expected_output.json"
                if expected_output_path.exists():
                    try:
                        with open(expected_output_path, "r") as f:
                            data["check_outputs"][check_dir.name] = json.load(f)
                    except Exception as e:
                        st.warning(f"Error reading {check_dir.name} output: {e}")
    
    return data


def save_check_output(example_path, check_name, output_data):
    """Save the updated check output."""
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


def main():
    st.title("SODA MMQC Example Visualizer")
    
    # Example selection
    workspace_root = get_workspace_root()
    examples_dir = workspace_root / "soda_mmqc/data/examples"
    
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
        "Select DOI",
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
            # Load example data
            example_data = load_example_data(selected_fig)
            
            # Create three columns for the layout with adjusted widths
            col1, col2, col3 = st.columns([1, 0.7, 1.3])
            
            # Column 1: Image
            with col1:
                st.header("Figure")
                if example_data["image_path"]:
                    try:
                        image = Image.open(example_data["image_path"])
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
                st.header("Check Output")
                if example_data["check_outputs"]:
                    # Check selection dropdown
                    selected_check = st.selectbox(
                        "Select Check",
                        list(example_data["check_outputs"].keys())
                    )
                    
                    if selected_check:
                        output_data = example_data["check_outputs"][selected_check]
                        if "outputs" in output_data:
                            df = pd.DataFrame(output_data["outputs"])
                            edited_df = st.data_editor(
                                df,
                                num_rows="dynamic",
                                height=300
                            )
                            
                            # Update button
                            if st.button(f"Save changes for {selected_check}"):
                                output_data["outputs"] = edited_df.to_dict("records")
                                if save_check_output(selected_fig, selected_check, output_data):
                                    st.success(f"Saved changes for {selected_check}")
                else:
                    st.info("No check outputs available for this example")


if __name__ == "__main__":
    main() 