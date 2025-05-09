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
import statistics

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
        std_score: Standard deviation of scores (for lists and objects)
        true_positives: Number of matches with score >= threshold (for lists)
        false_positives: Number of extra elements in prediction (for lists)
        false_negatives: Number of missing elements (for lists)
        precision: Precision score (for lists)
        recall: Recall score (for lists)
        f1_score: F1 score (for lists)
        detailed_scores: Dictionary with average and std scores per field (for lists)
    """
    score: float
    errors: List[str]
    field_scores: Dict[str, Any]
    std_score: Optional[float] = None
    true_positives: Optional[int] = None
    false_positives: Optional[int] = None
    false_negatives: Optional[int] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    detailed_scores: Optional[Dict[str, Dict[str, Any]]] = None
    
    def __init__(
        self, 
        score: float, 
        errors: Optional[List[str]] = None,
        field_scores: Optional[Dict[str, Any]] = None,
        std_score: Optional[float] = None,
        true_positives: Optional[int] = None,
        false_positives: Optional[int] = None,
        false_negatives: Optional[int] = None,
        precision: Optional[float] = None,
        recall: Optional[float] = None,
        f1_score: Optional[float] = None,
        detailed_scores: Optional[Dict[str, Dict[str, Any]]] = None
    ):
        self.score = max(0.0, min(1.0, score))
        self.errors = errors or []
        self.field_scores = field_scores or {}
        self.std_score = std_score
        self.true_positives = true_positives
        self.false_positives = false_positives
        self.false_negatives = false_negatives
        self.precision = precision
        self.recall = recall
        self.f1_score = f1_score
        self.detailed_scores = detailed_scores
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the ComparisonResult to a dictionary.
        
        Returns:
            Dictionary representation of the ComparisonResult
        """
        result = {
            "score": self.score,
            "errors": self.errors,
            "field_scores": {
                k: v.to_dict() if isinstance(v, ComparisonResult) else v
                for k, v in self.field_scores.items()
            }
        }
        
        # Add optional metrics if they exist
        if self.std_score is not None:
            result["std_score"] = self.std_score
        if self.true_positives is not None:
            result["true_positives"] = self.true_positives
        if self.false_positives is not None:
            result["false_positives"] = self.false_positives
        if self.false_negatives is not None:
            result["false_negatives"] = self.false_negatives
        if self.precision is not None:
            result["precision"] = self.precision
        if self.recall is not None:
            result["recall"] = self.recall
        if self.f1_score is not None:
            result["f1_score"] = self.f1_score
        if self.detailed_scores is not None:
            result["detailed_scores"] = self.detailed_scores
            
        return result


class JSONEvaluator:
    def __init__(
        self, 
        schema: Dict[str, Any],
        string_metric: str = "perfect_match",
        match_threshold: float = 0.5
    ):
        """Initialize the evaluator with a schema file.
        
        Args:
            schema: JSON schema
            string_metric: String comparison metric to use. One of:
                - "perfect_match": Exact string matching
                - "bleu": BLEU score
                - "semantic_similarity": Using SentenceTransformer
            match_threshold: Threshold for considering a match (0-1)
        """
        self.schema = schema
        self.match_threshold = max(0.0, min(1.0, match_threshold))
        
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
            if "enum" in schema:
                if pred_value not in schema["enum"]:
                    return ComparisonResult(
                        0.0,
                        [f"Value mismatch: '{pred_value}' not in {schema['enum']}"],
                        {"value": 0.0},
                        false_positives=1
                    )
            return self._compare_strings(pred_value, exp_value)
        
        elif value_type == "array":
            if not isinstance(pred_value, list) or not isinstance(exp_value, list):
                return ComparisonResult(
                    0.0,
                    [f"Type mismatch: expected array, got {type(pred_value)}, {type(exp_value)}"],
                    {},
                    false_positives=1 if pred_value is not None else 0,
                    false_negatives=1 if exp_value is not None else 0
                )
            
            return self._compare_lists(pred_value, exp_value, schema_path)
        
        elif value_type == "object":
            if not isinstance(pred_value, dict) or not isinstance(exp_value, dict):
                return ComparisonResult(
                    0.0,
                    [f"Type mismatch: expected object, got {type(pred_value)}, {type(exp_value)}"],
                    {},
                    false_positives=1 if pred_value is not None else 0,
                    false_negatives=1 if exp_value is not None else 0
                )
            return self._compare_objects(pred_value, exp_value, schema_path)
        
        else:
            # For other types (number, boolean), use exact match
            score = 1.0 if pred_value == exp_value else 0.0
            if score == 0.0:
                return ComparisonResult(
                    score,
                    [f"Value mismatch: '{pred_value}' != '{exp_value}'"],
                    {"value": score},
                    false_positives=1 if pred_value is not None else 0,
                    false_negatives=1 if exp_value is not None else 0
                )
            return ComparisonResult(score, [], {"value": score}, true_positives=1)

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
            - std_score: Standard deviation of scores
            - true_positives: Number of matches with score >= threshold
            - false_positives: Number of extra elements
            - false_negatives: Number of missing elements
            - precision: Precision score
            - recall: Recall score
            - f1_score: F1 score
            - field_scores: Dictionary of ComparisonResults for each element
            - errors: List of error messages
        """
        logger.debug(f"Comparing objects:\n\texpected: {exp}\n\tprediction: {pred}")
        
        field_scores = {}
        errors = []
        field_scores_list = []
        
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
                field_scores[field] = ComparisonResult(0.0, [f"Missing field in prediction: {field}"], {}, false_negatives=1)
                continue

            if field not in exp:
                errors.append(f"Missing field in expected: {field}")
                field_scores[field] = ComparisonResult(0.0, [f"Missing field in expected: {field}"], {}, false_positives=1)
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
            field_scores_list.append(field_result.score)
            errors.extend(field_result.errors)
        
        if not field_scores:
            return ComparisonResult(0.0, errors, {})
        
        # Aggregate metrics from all field comparisons
        true_positives = sum(
            result.true_positives or 0 
            for result in field_scores.values()
        )
        false_positives = sum(
            result.false_positives or 0 
            for result in field_scores.values()
        )
        false_negatives = sum(
            result.false_negatives or 0 
            for result in field_scores.values()
        )
        
        # Calculate precision, recall, F1
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # Calculate average score and standard deviation
        avg_score = sum(field_scores_list) / len(field_scores_list)
        std_score = statistics.stdev(field_scores_list) if len(field_scores_list) > 1 else 0
        
        return ComparisonResult(
            avg_score,
            errors,
            field_scores,
            std_score=std_score,
            true_positives=true_positives,
            false_positives=false_positives,
            false_negatives=false_negatives,
            precision=precision,
            recall=recall,
            f1_score=f1_score
        )

    
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
            - std_score: Standard deviation of scores
            - true_positives: Number of matches with score >= threshold
            - false_positives: Number of extra elements
            - false_negatives: Number of missing elements
            - precision: Precision score
            - recall: Recall score
            - f1_score: F1 score
            - field_scores: Dictionary of ComparisonResults for each element
            - detailed_scores: Dictionary with average and std scores per field
            - errors: List of error messages
        """
        logger.debug(f"Comparing lists:\n\texpected: {exp}\n\tprediction: {pred}")
        if not exp:
            if not pred:
                logger.debug("Comparing 2 empty lists")
                return ComparisonResult(1.0, [], {
                    "empty_list": ComparisonResult(1.0, [], {}, true_positives=1)
                })
            return ComparisonResult(
                0.0,
                ["Expected empty list"], 
                {"empty_list": ComparisonResult(0.0, ["Expected empty list"], {}, false_positives=len(pred))},
                false_positives=len(pred)
            )
        
        if not pred:
            logger.debug(f"Comparing {exp} with empty pred")
            missing_elements = {
                f"missing_element_{i}": ComparisonResult(0.0, [f"Missing element at index {i}"], {}, false_negatives=1)
                for i in range(len(exp))
            }
            return ComparisonResult(
                0.0, 
                ["Prediction list is empty"], 
                missing_elements,
                false_negatives=len(exp)
            )
        
        # Track which prediction elements have been matched
        matched_pred_indices = set()
        element_scores = {}
        errors = []
        matched_scores = []
        
        # Track detailed scores per field
        detailed_scores = {}
        field_scores_by_field = {}
        field_false_positives = {}

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
                matched_scores.append(best_score)
                logger.debug(f"best match for {exp_item}: {pred[best_pred_idx]} with score {best_score}")
                
                # Track field scores for matched elements
                if isinstance(best_result.field_scores, dict):
                    for field, field_result in best_result.field_scores.items():
                        if field not in field_scores_by_field:
                            field_scores_by_field[field] = []
                            field_false_positives[field] = 0
                        if isinstance(field_result, ComparisonResult):
                            field_scores_by_field[field].append(field_result.score)
                        else:
                            field_scores_by_field[field].append(field_result)
                
            else:
                element_scores[f"missing_element_{i}"] = ComparisonResult(
                    0.0,
                    [f"Missing expected element at index {i}"],
                    {},
                    false_negatives=1
                )
                errors.append(f"Missing expected element at index {i}")

        # Check for extra elements in prediction
        for j in range(len(pred)):
            if j not in matched_pred_indices:
                element_scores[f"extra_element_{j}"] = ComparisonResult(
                    0.0,
                    [f"Extra predicted element at index {j}"],
                    {},
                    false_positives=1
                )
                errors.append(f"Extra predicted element at index {j}")
                
                # Track false positives per field for extra elements
                extra_result = self._compare_values_with_schema(
                    pred[j],
                    None,  # No expected value to compare against
                    item_schema,
                    schema_path + ["items"]
                )
                if isinstance(extra_result.field_scores, dict):
                    for field in extra_result.field_scores:
                        if field not in field_false_positives:
                            field_false_positives[field] = 0
                        field_false_positives[field] += 1

        if not element_scores:
            logger.debug(f"No elements from {exp} matched {pred}")
            return ComparisonResult(0.0, errors, {})

        # Aggregate metrics from all element comparisons
        true_positives = sum(
            result.true_positives or 0 
            for result in element_scores.values()
        )
        false_positives = sum(
            result.false_positives or 0 
            for result in element_scores.values()
        )
        false_negatives = sum(
            result.false_negatives or 0 
            for result in element_scores.values()
        )
        
        # Calculate precision, recall, F1
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # Calculate average score and standard deviation
        avg_score = sum(matched_scores) / len(matched_scores) if matched_scores else 0
        std_score = statistics.stdev(matched_scores) if len(matched_scores) > 1 else 0

        # Calculate detailed scores per field
        for field, scores in field_scores_by_field.items():
            if scores:
                # Calculate average including false negatives (0 scores)
                total_expected = len(scores) + false_negatives
                avg_score = sum(scores) / total_expected if total_expected > 0 else 0
                
                # Calculate field-level metrics
                field_precision = len(scores) / (len(scores) + field_false_positives.get(field, 0)) if (len(scores) + field_false_positives.get(field, 0)) > 0 else 0
                field_recall = len(scores) / total_expected if total_expected > 0 else 0
                field_f1 = 2 * (field_precision * field_recall) / (field_precision + field_recall) if (field_precision + field_recall) > 0 else 0
                
                detailed_scores[field] = {
                    "avg_score": avg_score,
                    "std_score": statistics.stdev(scores) if len(scores) > 1 else 0,
                    "num_matches": len(scores),
                    "num_false_negatives": false_negatives,
                    "num_false_positives": field_false_positives.get(field, 0),
                    "precision": field_precision,
                    "recall": field_recall,
                    "f1_score": field_f1
                }

        return ComparisonResult(
            avg_score,
            errors,
            element_scores,
            std_score=std_score,
            true_positives=true_positives,
            false_positives=false_positives,
            false_negatives=false_negatives,
            precision=precision,
            recall=recall,
            f1_score=f1_score,
            detailed_scores=detailed_scores
        )

    def _compare_strings(self, pred: str, exp: str) -> ComparisonResult:
        """Compare two strings using the configured string comparator.

        Args:
            pred: Prediction string
            exp: Expected string    
        """
        logger.debug(f"Comparing strings:\n\texpected: {exp}\n\tprediction: {pred}")
        
        # Handle None cases first
        if pred is None and exp is None:
            return ComparisonResult(1.0, [], {"value": 1.0}, true_positives=1)
        if pred is None and exp is not None:
            return ComparisonResult(0.0, ["Missing value"], {"value": 0.0}, false_negatives=1)
        if pred is not None and exp is None:
            return ComparisonResult(0.0, ["Extra value"], {"value": 0.0}, false_positives=1)
            
        # Both are strings, compare them
        score = self.string_comparator(pred, exp)
        if score >= self.match_threshold:
            return ComparisonResult(score, [], {"value": score}, true_positives=1)
        else:
            return ComparisonResult(score, [], {"value": score}, false_positives=1)
