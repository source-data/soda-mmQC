import os
import json
from pathlib import Path
from model_api import generate_response  # Placeholder for your model inference function
from evaluation import evaluate_response  # Placeholder for evaluation metrics

def load_json(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

def process_check(check_data, checklist_name, results_dir):
    check_id = check_data["id"]
    prompt_path = check_data["prompt_path"]
    metrics = check_data["metrics"]
    examples = check_data["examples"]
    
    with open(prompt_path, "r", encoding="utf-8") as f:
        prompt_template = f.read()
    
    results = []
    for example in examples:
        figure_id = example["figure_id"]
        figure_dir = Path(example["figure_path"])
        expected_output_path = Path(example["expected_output_path"])
        
        with open(figure_dir / "caption.txt", "r", encoding="utf-8") as f:
            caption = f.read().strip()
        with open(expected_output_path, "r", encoding="utf-8") as f:
            expected_output = f.read().strip()
        
        image_path = next(figure_dir.glob("*.png"), None) or next(figure_dir.glob("*.jpg"), None) or next(figure_dir.glob("*.tiff"), None)
        if not image_path:
            raise ValueError(f"No image found for {figure_id}")
        
        # Generate model response
        model_input = {"image": str(image_path), "caption": caption, "prompt": prompt_template}
        model_output = generate_response(model_input)
        
        # Evaluate response
        evaluation_results = evaluate_response(model_output, expected_output, metrics)
        
        results.append({
            "figure_id": figure_id,
            "expected_output": expected_output,
            "model_output": model_output,
            "evaluation": evaluation_results
        })
    
    # Save results
    checklist_results_dir = Path(results_dir) / checklist_name
    os.makedirs(checklist_results_dir, exist_ok=True)
    results_file = checklist_results_dir / f"{check_id}_metrics.json"
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)
    print(f"Saved results for {check_id} to {results_file}")

def main():
    checklist_dir = "data/checklist/"
    results_dir = "evaluation/results/"
    os.makedirs(results_dir, exist_ok=True)
    
    for checklist_file in Path(checklist_dir).glob("*.json"):
        checklist_name = checklist_file.stem  # Extracts "editorial", "statistics", etc.
        checklist_data = load_json(checklist_file)
        
        for check in checklist_data["checks"]:
            process_check(check, checklist_name, results_dir)

if __name__ == "__main__":
    main()
