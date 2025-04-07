import json
import difflib
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


def exact_match_score(predicted, expected):
    """Calculate exact match score between predicted and expected outputs."""
    try:
        # Parse JSON strings if they are strings
        if isinstance(predicted, str):
            predicted = json.loads(predicted)
        if isinstance(expected, str):
            expected = json.loads(expected)
        
        # Compare the JSON objects
        return 1.0 if predicted == expected else 0.0
    except (json.JSONDecodeError, TypeError):
        # If parsing fails, compare as strings
        return 1.0 if predicted == expected else 0.0


def get_text_fields(json_obj):
    """Extract text fields that contain semantic content from the JSON object."""
    if isinstance(json_obj, str):
        json_obj = json.loads(json_obj)
    
    text_fields = []
    
    # Add name field if present
    if "name" in json_obj:
        text_fields.append(json_obj["name"])
    
    # Process panels
    if "panels" in json_obj and isinstance(json_obj["panels"], list):
        for panel in json_obj["panels"]:
            # Add caption text which contains semantic content
            if "from_the_caption" in panel:
                text_fields.append(panel["from_the_caption"])
            
            # Add error bar meaning if it's not a standard enum value
            if "error_bar_meaning" in panel:
                meaning = panel["error_bar_meaning"]
                if meaning not in [
                    "standard deviation",
                    "standard error",
                    "confidence interval",
                    "not applicable"
                ]:
                    text_fields.append(meaning)
    
    return text_fields


def semantic_similarity_score(predicted, expected):
    """Calculate semantic similarity score between text fields of predicted and 
    expected outputs."""
    try:
        # Get text fields with semantic content
        pred_texts = get_text_fields(predicted)
        exp_texts = get_text_fields(expected)
        
        if not pred_texts or not exp_texts:
            return 0.0
            
        # Load the model
        model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Calculate pairwise similarities between all text fields
        similarities = []
        for pred_text in pred_texts:
            pred_embedding = model.encode([pred_text])[0]
            pred_embedding = np.array(pred_embedding).reshape(1, -1)
            
            for exp_text in exp_texts:
                exp_embedding = model.encode([exp_text])[0]
                exp_embedding = np.array(exp_embedding).reshape(1, -1)
                similarity = cosine_similarity(pred_embedding, exp_embedding)[0][0]
                similarities.append(similarity)
        
        # Return maximum similarity found between any pair of texts
        return max(similarities) if similarities else 0.0
        
    except (json.JSONDecodeError, TypeError, ImportError):
        return 0.0


def bleu_score(predicted, expected):
    """Calculate BLEU score between text fields of predicted and expected outputs."""
    try:
        # Get text fields with semantic content
        pred_texts = get_text_fields(predicted)
        exp_texts = get_text_fields(expected)
        
        if not pred_texts or not exp_texts:
            return 0.0
            
        # Calculate BLEU scores between all text pairs
        bleu_scores = []
        for pred_text in pred_texts:
            pred_tokens = nltk.word_tokenize(pred_text.lower())
            
            for exp_text in exp_texts:
                exp_tokens = nltk.word_tokenize(exp_text.lower())
                score = sentence_bleu([exp_tokens], pred_tokens)
                bleu_scores.append(score)
        
        # Return maximum BLEU score found between any pair of texts
        return max(bleu_scores) if bleu_scores else 0.0
        
    except (json.JSONDecodeError, TypeError):
        return 0.0


def structured_match_score(predicted, expected):
    """Calculate structured match score based on the schema requirements."""
    try:
        # Parse JSON strings if they are strings
        if isinstance(predicted, str):
            predicted = json.loads(predicted)
        if isinstance(expected, str):
            expected = json.loads(expected)

        # Check required fields
        required_fields = ["name", "panels"]
        if not all(field in predicted for field in required_fields):
            return 0.0

        # Check panels structure
        if not isinstance(predicted["panels"], list):
            return 0.0

        # Calculate panel-wise scores
        panel_scores = []
        for pred_panel, exp_panel in zip(
            predicted["panels"], expected["panels"]
        ):
            panel_score = 0.0
            total_fields = 0

            # Check required panel fields
            required_panel_fields = [
                "panel_label",
                "error_bar_on_figure",
                "error_bar_defined_in_legend",
                "error_bar_defined_in_caption",
                "error_bar_meaning",
                "from_the_caption"
            ]

            for field in required_panel_fields:
                if field in pred_panel and field in exp_panel:
                    total_fields += 1
                    if pred_panel[field] == exp_panel[field]:
                        panel_score += 1.0

            panel_scores.append(
                panel_score / total_fields if total_fields > 0 else 0.0
            )

        # Return average score across all panels
        return sum(panel_scores) / len(panel_scores) if panel_scores else 0.0

    except (json.JSONDecodeError, TypeError, KeyError, ZeroDivisionError):
        return 0.0


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
        elif metric == "structured_match":
            results[metric] = structured_match_score(
                model_output, expected_output
            )
        else:
            results[metric] = 0.0  # Unknown metric

    return results