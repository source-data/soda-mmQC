import json
from nltk.translate.bleu_score import sentence_bleu
import nltk
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


# Download NLTK data for BLEU score calculation
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:        
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')

SENTENCE_TRANSFORMER = SentenceTransformer('all-MiniLM-L6-v2')


def get_text_fields(json_obj):
    """Extract text fields with semantic content from the JSON object."""
    if isinstance(json_obj, str):
        json_obj = json.loads(json_obj)

    # Initialize dictionaries for each field type
    field_types = {
        "panel_label": [],
        "error_bar_on_figure": [],
        "error_bar_defined_in_caption": [],
        "from_the_caption": []
    }

    # Process panels
    if "outputs" in json_obj:
        for output in json_obj["outputs"]:
            # Add caption text which contains semantic content
            if "panel_label" in output:
                field_types["panel_label"].append(output["panel_label"])
            if "error_bar_on_figure" in output:
                field_types["error_bar_on_figure"].append(
                    output["error_bar_on_figure"]
                )
            if "error_bar_defined_in_caption" in output:
                field_types["error_bar_defined_in_caption"].append(
                    output["error_bar_defined_in_caption"]
                )
            if "from_the_caption" in output:
                field_types["from_the_caption"].append(
                    output["from_the_caption"]
                )
    return field_types


def exact_match_score(predicted, expected):
    """Calculate exact match score between predicted and expected outputs."""
    # Get text fields with semantic content
    pred_fields = get_text_fields(predicted)
    exp_fields = get_text_fields(expected)

    # Calculate exact match score for each field type
    results = {}
    overall_score = 0.0
    total_fields = 0

    for field_type in pred_fields:
        pred_texts = pred_fields[field_type]
        exp_texts = exp_fields[field_type]
        
        # Skip if no fields of this type
        if not pred_texts or not exp_texts:
            results[field_type] = 0.0
            continue
            
        # Calculate score for this field type
        field_score = 0.0
        field_count = 0
        
        for pred_text, exp_text in zip(pred_texts, exp_texts):
            field_score += int(pred_text == exp_text)
            field_count += 1
            
        field_result = field_score / field_count if field_count > 0 else 0.0
        results[field_type] = field_result
        
        # Add to overall score
        overall_score += field_score
        total_fields += field_count
    
    # Add overall score
    results["overall"] = overall_score / total_fields if total_fields > 0 else 0.0
    
    return results


def semantic_similarity_score(predicted, expected, model=SENTENCE_TRANSFORMER):
    """Calculate semantic similarity between predicted and expected outputs."""
    try:
        # Get text fields with semantic content
        pred_fields = get_text_fields(predicted)
        exp_fields = get_text_fields(expected)

        # Calculate semantic similarity for each field type
        results = {}
        overall_score = 0.0
        total_fields = 0

        for field_type in pred_fields:
            pred_texts = pred_fields[field_type]
            exp_texts = exp_fields[field_type]

            # Skip if no fields of this type
            if not pred_texts or not exp_texts:
                results[field_type] = 0.0
                continue

            # Calculate similarity for this field type
            similarities = []

            for pred_text, exp_text in zip(pred_texts, exp_texts):
                pred_embedding = model.encode([pred_text])[0]
                pred_embedding = np.array(pred_embedding).reshape(1, -1)
                exp_embedding = model.encode([exp_text])[0]
                exp_embedding = np.array(exp_embedding).reshape(1, -1)
                similarity = cosine_similarity(
                    pred_embedding, exp_embedding
                )[0][0]
                similarities.append(float(similarity))
            
            # Calculate average for this field type
            field_result = (
                float(sum(similarities) / len(similarities)) 
                if similarities else 0.0
            )
            results[field_type] = field_result
            
            # Add to overall score
            overall_score += sum(similarities)
            total_fields += len(similarities)
        
        # Add overall score
        results["overall"] = (
            overall_score / total_fields if total_fields > 0 else 0.0
        )
        
        return results

    except (json.JSONDecodeError, TypeError, ImportError):
        # Return zeros for all field types in case of error
        return {
            "panel_label": 0.0,
            "error_bar_on_figure": 0.0,
            "error_bar_defined_in_caption": 0.0,
            "from_the_caption": 0.0,
            "overall": 0.0
        }


def bleu_score(predicted, expected):
    """Calculate BLEU score between predicted and expected outputs."""
    try:
        # Get text fields with semantic content
        pred_fields = get_text_fields(predicted)
        exp_fields = get_text_fields(expected)

        # Calculate BLEU score for each field type
        results = {}
        overall_score = 0.0
        total_fields = 0

        for field_type in pred_fields:
            pred_texts = pred_fields[field_type]
            exp_texts = exp_fields[field_type]

            # Skip if no fields of this type
            if not pred_texts or not exp_texts:
                results[field_type] = 0.0
                continue

            # Calculate BLEU scores for this field type
            bleu_scores = []

            for pred_text, exp_text in zip(pred_texts, exp_texts):
                # Case 1: Both empty - perfect match
                if (not exp_text) and (not pred_text):
                    score = 1.0
                # Case 2: Expected empty but predicted has content - wrong
                elif (not exp_text) and pred_text:
                    score = 0.0
                # Case 3: Expected has content and predicted contains it - good
                elif exp_text and (exp_text in pred_text):
                    score = 1.0
                # Case 4: Expected has content but predicted doesn't contain it - use BLEU
                else:
                    # Standard BLEU calculation
                    pred_tokens = nltk.word_tokenize(pred_text.lower())
                    exp_tokens = nltk.word_tokenize(exp_text.lower())
                    score = sentence_bleu([pred_tokens], exp_tokens)
                
                bleu_scores.append(score)
            
            # Calculate average for this field type
            field_result = (
                sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0.0
            )
            results[field_type] = field_result
            
            # Add to overall score
            overall_score += sum(bleu_scores)
            total_fields += len(bleu_scores)
        
        # Add overall score
        results["overall"] = (
            overall_score / total_fields if total_fields > 0 else 0.0
        )
        
        return results

    except (json.JSONDecodeError, TypeError):
        # Return zeros for all field types in case of error
        return {
            "panel_label": 0.0,
            "error_bar_on_figure": 0.0,
            "error_bar_defined_in_caption": 0.0,
            "from_the_caption": 0.0,
            "overall": 0.0
        }


def evaluate_response(model_output, expected_output, metrics):
    """Evaluate model output against expected output using specified metrics."""
    results = {}

    for metric in metrics:
        if metric == "exact_match":
            results[metric] = exact_match_score(model_output, expected_output)
        elif metric == "semantic_similarity":
            results[metric] = semantic_similarity_score(
                model_output, expected_output
            )
        elif metric == "BLEU":
            results[metric] = bleu_score(model_output, expected_output)
        else:
            # For unknown metrics, return zeros for all field types
            results[metric] = {
                "panel_label": 0.0,
                "error_bar_on_figure": 0.0,
                "error_bar_defined_in_caption": 0.0,
                "from_the_caption": 0.0,
                "overall": 0.0
            }

    return results