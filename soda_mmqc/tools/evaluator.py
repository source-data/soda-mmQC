"""Tools for evaluating model predictions against expected outputs."""

from typing import Dict, Any, List, Callable, Optional, Union
from dataclasses import dataclass
import torch
from sentence_transformers import SentenceTransformer
from nltk.translate.bleu_score import sentence_bleu
from nltk.tokenize import word_tokenize
import nltk
from soda_mmqc.config import DEVICE
import logging
from soda_mmqc import logger

# Suppress progress bars from SentenceTransformer
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)

# Download required NLTK data
nltk.download('punkt')

# Initialize SentenceTransformer model
MODEL = SentenceTransformer('all-MiniLM-L6-v2', device=DEVICE)


@dataclass
class ComparisonResult:
    """Result of a comparison operation.
    
    Attributes:
        score: Float between 0 and 1 indicating the match quality
        errors: List of error messages if any
        field_scores: Dictionary of scores for each field in the comparison
    """
    score: float
    errors: List[str]
    field_scores: Dict[str, Any]
    
    def __init__(
        self, 
        score: float, 
        errors: Optional[List[str]] = None,
        field_scores: Optional[Dict[str, Any]] = None
    ):
        self.score = max(0.0, min(1.0, score))
        self.errors = errors or []
        self.field_scores = field_scores or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the ComparisonResult to a dictionary.
        
        Returns:
            Dictionary representation of the ComparisonResult
        """
        return {
            "score": self.score,
            "errors": self.errors,
            "field_scores": {
                k: v.to_dict() if isinstance(v, ComparisonResult) else v
                for k, v in self.field_scores.items()
            }
        }


class JSONEvaluator:
    def __init__(
        self, 
        schema: Dict[str, Any],
        string_metric: str = "perfect_match"
    ):
        """Initialize the evaluator with a schema file.
        
        Args:
            schema: JSON schema
            string_metric: String comparison metric to use. One of:
                - "perfect_match": Exact string matching
                - "bleu": BLEU score
                - "semantic_similarity": Using SentenceTransformer
        """
        self.schema = schema
        
        # Set string comparison function based on metric
        if string_metric in ["perfect_match", "exact_match"]:
            self.string_comparator = self._exact_string_match
        elif string_metric.lower() == "bleu":
            self.string_comparator = self._bleu_score
        elif string_metric == "semantic_similarity":
            self.string_comparator = self._semantic_similarity
        else:
            raise ValueError(
                f"Invalid string_metric: {string_metric}. Must be one of: "
                "'perfect_match', 'exact_match', 'bleu', 'semantic_similarity'"
            )
    
    def _get_schema_for_path(self, path: List[str]) -> Dict[str, Any]:
        """Get the schema definition for a specific path in the structure.
        
        Args:
            path: List of keys representing the path to the schema definition
            
        Returns:
            Schema definition for the given path
        """
        current = self.schema["format"]["schema"]
        for key in path:
            if key == "items":
                current = current.get("items", {})
            elif key == "properties":
                current = current.get("properties", {})
            else:
                current = current.get("properties", {}).get(key, {})
        return current
    
    def _semantic_similarity(self, pred: str, exp: str) -> float:
        """Calculate semantic similarity between two strings using 
        SentenceTransformer.
        
        Args:
            pred: Prediction string
            exp: Expected string
            
        Returns:
            Score between 0 and 1 indicating semantic similarity
        """
        # Encode the sentences
        embeddings = MODEL.encode([pred, exp], convert_to_tensor=True)
        
        # Calculate cosine similarity
        similarity = torch.nn.functional.cosine_similarity(
            embeddings[0].unsqueeze(0),
            embeddings[1].unsqueeze(0)
        )
        
        # Convert to float and map from [-1, 1] to [0, 1]
        score = float(similarity.item())
        logger.debug(f"cos_sim({pred}, {exp}): {score}")
        return (score + 1) / 2  # Map from [-1, 1] to [0, 1]
    
    def _exact_string_match(self, pred: str, exp: str) -> float:
        """Calculate exact string match between two strings.
        
        Args:
            pred: Prediction string
            exp: Expected string
        """
        logger.debug(f"exact_match({pred}, {exp}): {1.0 if pred == exp else 0.0}")
        return 1.0 if pred == exp else 0.0
    
    def _bleu_score(self, pred: str, exp: str) -> float:
        """Calculate BLEU score between two strings.
        
        Args:
            pred: Prediction string
            exp: Expected string
            
        Returns:
            Score between 0 and 1 indicating BLEU score
        """
        # Tokenize the sentences
        pred_tokens = [word_tokenize(pred)]
        exp_tokens = word_tokenize(exp)
        
        # Calculate BLEU score with smoothing
        try:
            score = float(sentence_bleu([exp_tokens], pred_tokens))
        except (ZeroDivisionError, TypeError):
            # Handle case where there are no matching n-grams
            score = 0.0
        
        return score
    
    def evaluate(
        self, 
        prediction: Dict[str, Any], 
        expected: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Compare prediction against expected output.
        
        Args:
            prediction: The model's prediction as a JSON object
            expected: The expected output as a JSON object
            
        Returns:
            Dictionary containing:
            - score: Overall average score between 0 and 1
            - field_scores: Dictionary of scores for each field
            - errors: List of error messages if any
        """
        # Get the schema structure
        schema_structure = self.schema["format"]["schema"]
        
        # Compare the outputs array
        if "outputs" in schema_structure["properties"]:
            pred_outputs = prediction.get("outputs", [])
            exp_outputs = expected.get("outputs", [])
            
            # Get the outputs schema
            outputs_schema = self._get_schema_for_path(["outputs"])
            
            # Compare outputs using _compare_values_with_schema
            outputs_result = self._compare_values_with_schema(
                pred_outputs,
                exp_outputs,
                outputs_schema,
                ["outputs"]
            )
            
            # Convert the entire ComparisonResult to a dictionary
            return outputs_result.to_dict()
        
        return {
            "score": 0.0,
            "field_scores": {},
            "errors": []
        }


    def _compare_values_with_schema(
        self,
        pred_value: Any,
        exp_value: Any,
        schema: Dict[str, Any],
        schema_path: List[str]
    ) -> ComparisonResult:
        """Compare values using schema information.
        
        Args:
            pred_value: Prediction value
            exp_value: Expected value
            schema: Schema definition for this value
            schema_path: Path to the schema definition
            
        Returns:
            ComparisonResult with the comparison score
        """
        value_type = schema.get("type")
        logger.debug(f"Comparing values of type {value_type}")

        if value_type == "string":
            pred_value = pred_value if pred_value is not None else ""
            exp_value = exp_value if exp_value is not None else ""
            if "enum" in schema:
                if pred_value not in schema["enum"]:
                    return ComparisonResult(
                        0.0,
                        [f"Value mismatch: '{pred_value}' not in {schema['enum']}"],
                        {"value": 0.0}
                    )
            return self._compare_strings(pred_value, exp_value)
        
        elif value_type == "array":
            pred_value = pred_value if pred_value is not None else []
            exp_value = exp_value if exp_value is not None else []
            if not isinstance(pred_value, list) or not isinstance(exp_value, list):
                return ComparisonResult(
                    0.0,
                    [f"Type mismatch: expected array, got {type(pred_value)}, {type(exp_value)}"],
                    {}
                )
            
            return self._compare_lists(pred_value, exp_value, schema_path)
        
        elif value_type == "object":
            pred_value = pred_value if pred_value is not None else {}
            exp_value = exp_value if exp_value is not None else {}
            if not isinstance(pred_value, dict) or not isinstance(exp_value, dict):
                return ComparisonResult(
                    0.0,
                    [f"Type mismatch: expected object, got {type(pred_value)}, {type(exp_value)}"],
                    {}
                )
            return self._compare_objects(pred_value, exp_value, schema_path)
        
        else:
            # For other types (number, boolean), use exact match
            score = 1.0 if pred_value == exp_value else 0.0
            if score == 0.0:
                return ComparisonResult(
                    score,
                    [f"Value mismatch: '{pred_value}' != '{exp_value}'"],
                    {"value": score}
                )
            return ComparisonResult(score, [], {"value": score})

    def _compare_objects(
        self, 
        pred: Dict[str, Any], 
        exp: Dict[str, Any],
        schema_path: List[str]
    ) -> ComparisonResult:
        """Compare two objects field by field using schema information.
        
        Args:
            pred: Prediction object
            exp: Expected object
            schema_path: Path to the schema definition for this object
            
        Returns:
            ComparisonResult with:
            - score: Average score across matched elements
            - field_scores: Dictionary of ComparisonResults for each element
            - errors: List of error messages
        """
        logger.debug(f"Comparing objects:\n\texpected: {exp}\n\tprediction: {pred}")
        
        field_scores = {}
        errors = []
        
        # Get schema for this object
        schema = self._get_schema_for_path(schema_path)
        
        # Get required fields from schema
        required_fields = schema.get("required", [])
        
        # Check all required fields
        for field in required_fields:
            field_scores[field] = {}
            logger.debug(f"Analyzing object field: {field}")
            if field not in pred:
                errors.append(f"Missing field in prediction: {field}")
                field_scores[field] = ComparisonResult(0.0, [f"Missing field in prediction: {field}"], {})
                continue

            if field not in exp:
                errors.append(f"Missing field in expected: {field}")
                field_scores[field] = ComparisonResult(0.0, [f"Missing field in expected: {field}"], {})
                continue

            pred_value = pred[field]
            exp_value = exp[field]

            # Get field schema
            field_schema = self._get_schema_for_path(schema_path + [field])
            
            # Compare values based on schema type
            field_result = self._compare_values_with_schema(
                pred_value, 
                exp_value, 
                field_schema,
                schema_path + [field]
            )
            field_scores[field] = field_result
            errors.extend(field_result.errors)
        
        if not field_scores:
            return ComparisonResult(0.0, errors, {})
        
        # Calculate average score across all fields
        avg_score = sum(f.score for f in field_scores.values()) / len(field_scores)
        return ComparisonResult(avg_score, errors, field_scores)

    
    def _compare_lists(
        self, 
        pred: List[Any], 
        exp: List[Any],
        schema_path: List[str]
    ) -> ComparisonResult:
        """Compare two list values using schema information.
        
        Args:
            pred: Prediction list
            exp: Expected list
            schema_path: Path to the schema definition
            
        Returns:
            ComparisonResult with:
            - score: Average score across matched elements
            - field_scores: Dictionary of ComparisonResults for each element
            - errors: List of error messages
        """
        logger.debug(f"Comparing lists:\n\texpected: {exp}\n\tprediction: {pred}")
        if not exp:
            if not pred:
                logger.debug("Comparing 2 empty lists")
                return ComparisonResult(1.0, [], {
                    "empty_list": ComparisonResult(1.0, [], {})
                })
            return ComparisonResult(
                0.0,
                ["Expected empty list"], 
                {"empty_list": ComparisonResult(0.0, ["Expected empty list"], {})}
            )
        
        if not pred:
            logger.debug(f"Comparing {exp} with empty pred")
            missing_elements = {
                f"missing_element_{i}": ComparisonResult(0.0, [f"Missing element at index {i}"], {})
                for i in range(len(exp))
            }
            return ComparisonResult(
                0.0, 
                ["Prediction list is empty"], 
                missing_elements
            )
        
        # Track which prediction elements have been matched
        matched_pred_indices = set()
        element_scores = {}
        errors = []

        # Get item schema
        item_schema = self._get_schema_for_path(schema_path + ["items"])

        # For each expected element, find the best matching prediction element
        for i, exp_item in enumerate(exp):
            best_score = 0.0
            best_pred_idx = None
            best_result = None

            # Try to match with each unmatched prediction element
            for j, pred_item in enumerate(pred):
                if j in matched_pred_indices:
                    continue

                item_result = self._compare_values_with_schema(
                    pred_item,
                    exp_item,
                    item_schema,
                    schema_path + ["items"]
                )
                if item_result.score > best_score:
                    best_score = item_result.score
                    best_pred_idx = j
                    best_result = item_result

            if best_pred_idx is not None:
                matched_pred_indices.add(best_pred_idx)
                element_scores[f"element_{i}"] = best_result
                logger.debug(f"best match for {exp_item}: {pred[best_pred_idx]} with score {best_score}")
                
            else:
                element_scores[f"missing_element_{i}"] = ComparisonResult(
                    0.0,
                    [f"Missing element at index {i}"],
                    {}
                )
                errors.append(f"Missing element at index {i}")

        # Check for extra elements in prediction
        for j in range(len(pred)):
            if j not in matched_pred_indices:
                element_scores[f"extra_element_{j}"] = ComparisonResult(
                    0.0,
                    [f"Extra element at index {j}"],
                    {}
                )
                errors.append(f"Extra element at index {j}")

        if not element_scores:
            logger.debug(f"No elements from {exp} matched {pred}")
            return ComparisonResult(0.0, errors, {})

        # Calculate average score across all elements (including missing/extra)
        avg_score = sum(e.score for e in element_scores.values()) / len(element_scores)
        return ComparisonResult(avg_score, errors, element_scores)

    def _compare_strings(self, pred: str, exp: str) -> ComparisonResult:
        """Compare two strings using the configured string comparator.

        Args:
            pred: Prediction string
            exp: Expected string    
        """
        logger.debug(f"Comparing strings:\n\texpected: {exp}\n\tprediction: {pred}")
        score = self.string_comparator(pred, exp)
        return ComparisonResult(score, [], {
            "value": ComparisonResult(score, [], {})
        })
