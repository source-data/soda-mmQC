import json
import difflib
from nltk.translate.bleu_score import sentence_bleu
import nltk
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# Download NLTK data for BLEU score calculation
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')


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


def semantic_similarity_score(predicted, expected):
    """Calculate semantic similarity score between predicted and expected outputs."""
    try:
        # Parse JSON strings if they are strings
        if isinstance(predicted, str):
            predicted = json.loads(predicted)
        if isinstance(expected, str):
            expected = json.loads(expected)
        
        # Convert to string for comparison
        predicted_str = json.dumps(predicted, sort_keys=True)
        expected_str = json.dumps(expected, sort_keys=True)
        
        # Load the model
        model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Encode the sentences
        pred_embedding = model.encode([predicted_str])[0]
        exp_embedding = model.encode([expected_str])[0]
        
        # Calculate cosine similarity
        similarity = cosine_similarity(
            [pred_embedding], 
            [exp_embedding]
        )[0][0]
        
        return float(similarity)
    except (json.JSONDecodeError, TypeError, ImportError):
        # Fallback to string similarity if parsing fails or model not available
        return difflib.SequenceMatcher(
            None, str(predicted), str(expected)
        ).ratio()


def bleu_score(predicted, expected):
    """Calculate BLEU score between predicted and expected outputs."""
    try:
        # Parse JSON strings if they are strings
        if isinstance(predicted, str):
            predicted = json.loads(predicted)
        if isinstance(expected, str):
            expected = json.loads(expected)
        
        # Convert to string for comparison
        predicted_str = json.dumps(predicted, sort_keys=True)
        expected_str = json.dumps(expected, sort_keys=True)
        
        # Tokenize
        predicted_tokens = nltk.word_tokenize(predicted_str.lower())
        expected_tokens = nltk.word_tokenize(expected_str.lower())
        
        # Calculate BLEU score
        return sentence_bleu([expected_tokens], predicted_tokens)
    except (json.JSONDecodeError, TypeError):
        # Fallback to string comparison if JSON parsing fails
        predicted_tokens = nltk.word_tokenize(str(predicted).lower())
        expected_tokens = nltk.word_tokenize(str(expected).lower())
        return sentence_bleu([expected_tokens], predicted_tokens)


def evaluate_response(model_output, expected_output, metrics):
    """Evaluate the model output against the expected output using the specified metrics."""
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
            results[metric] = 0.0  # Unknown metric
    
    return results 