import json
from nltk.translate.bleu_score import sentence_bleu
import nltk
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from soda_mmqc.config import DEVICE
import logging

# Suppress progress bars from SentenceTransformer
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)

# Download NLTK data for BLEU score calculation
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:        
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')

SENTENCE_TRANSFORMER = SentenceTransformer(
    'all-MiniLM-L6-v2',
    device=DEVICE
)


def get_text_fields(json_obj, schema):
    """Extract text fields with semantic content from the JSON object.
    
    Args:
        json_obj: The JSON object to extract fields from
        schema: Schema to derive field types from.
        
    Returns:
        Dictionary with field types as keys and lists of values as values.
        
    Raises:
        ValueError: If schema is missing or invalid.
    """
    if isinstance(json_obj, str):
        json_obj = json.loads(json_obj)
        
    if not schema or not isinstance(schema, dict):
        raise ValueError("Schema must be provided and must be a dictionary")

    # Initialize field types from schema
    field_types = {}

    # Extract required fields from schema
    try:
        required_fields = schema.get("format", {}).get("schema", {}).get(
            "properties", {}).get("outputs", {}).get("items", {}).get(
            "required", [])

        if not required_fields:
            raise ValueError("Schema does not contain required fields")
            
        # Initialize empty lists for each required field
        for field in required_fields:
            field_types[field] = []
    except (KeyError, AttributeError) as e:
        raise ValueError(f"Invalid schema format: {str(e)}")

    # Process panels
    if "outputs" in json_obj:
        for output in json_obj["outputs"]:
            # Add caption text which contains semantic content
            for field in field_types:
                if field in output:
                    field_types[field].append(output[field])
    
    return field_types


def exact_match_score(predicted, expected, schema):
    """Calculate exact match score between predicted and expected outputs."""
    # Get text fields with semantic content
    pred_fields = get_text_fields(predicted, schema)
    exp_fields = get_text_fields(expected, schema)

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
    results["overall"] = (overall_score / total_fields
                          if total_fields > 0 else 0.0)
    
    return results


def semantic_similarity_score(predicted, expected, model=SENTENCE_TRANSFORMER,
                             schema=None):
    """Calculate semantic similarity between predicted and expected outputs."""
    try:
        # Get text fields with semantic content
        pred_fields = get_text_fields(predicted, schema)
        exp_fields = get_text_fields(expected, schema)

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
                if not pred_text or not exp_text:
                    logging.warning(f"Empty text field: {field_type}")
                    logging.warning(f"Pred: {pred_text}")
                    logging.warning(f"Exp: {exp_text}")
                    similarities.append(0.0)
                    continue
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

    except (json.JSONDecodeError, TypeError, ImportError, ValueError) as e:
        # Re-raise ValueError, but handle other errors
        if isinstance(e, ValueError):
            raise
        # Return zeros for all field types in case of error
        return {
            "overall": 0.0
        }


def bleu_score(predicted, expected, schema):
    """Calculate BLEU score between predicted and expected outputs."""
    try:
        # Get text fields with semantic content
        pred_fields = get_text_fields(predicted, schema)
        exp_fields = get_text_fields(expected, schema)

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
                # Case 4: Expected has content but predicted doesn't contain it
                # - use BLEU
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

    except (json.JSONDecodeError, TypeError, ValueError) as e:
        # Re-raise ValueError, but handle other errors
        if isinstance(e, ValueError):
            raise
        # Return zeros for all field types in case of error
        return {
            "overall": 0.0
        }


def evaluate_response(model_output, expected_output, metrics, schema):
    """Evaluate model output against expected output using specified metrics."""
    results = {}

    for metric in metrics:
        if metric == "exact_match":
            results[metric] = exact_match_score(model_output, expected_output,
                                               schema)
        elif metric == "semantic_similarity":
            results[metric] = semantic_similarity_score(
                model_output, expected_output, schema=schema
            )
        elif metric == "BLEU":
            results[metric] = bleu_score(model_output, expected_output, schema)
        else:
            # For unknown metrics, return zeros for all field types
            results[metric] = {
                "overall": 0.0
            }

    return results