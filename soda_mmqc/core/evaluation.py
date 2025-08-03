"""Tools for evaluating model predictions against expected outputs."""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
import torch
from sentence_transformers import SentenceTransformer
from soda_mmqc.config import DEVICE
import logging
from soda_mmqc import logger
import statistics
from scipy.optimize import linear_sum_assignment
import numpy as np

# Suppress progress bars from SentenceTransformer
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)

# Initialize SentenceTransformer model
MODEL = SentenceTransformer('all-MiniLM-L6-v2', device=DEVICE)


def lcs_length(s1: str, s2: str) -> int:
    """Calculate the length of the Longest Common Subsequence between two strings.
    
    Args:
        s1: First string
        s2: Second string
        
    Returns:
        Length of the longest common subsequence
    """
    m, n = len(s1), len(s2)
    
    # Create a 2D array to store lengths of common subsequence
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    # Build the LCS table in bottom-up manner
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    
    return dp[m][n]


def lcs_ratio(s1: str, s2: str) -> float:
    """Calculate LCS ratio between two strings.
    
    The LCS ratio is: (2 * LCS_length) / (len(s1) + len(s2))
    This gives a value between 0 and 1, where 1 means the strings are identical.
    
    Args:
        s1: First string
        s2: Second string
        
    Returns:
        LCS ratio between 0 and 1
    """
    if not s1 and not s2:
        return 1.0
    
    if not s1 or not s2:
        return 0.0
    
    lcs_len = lcs_length(s1, s2)
    return (2.0 * lcs_len) / (len(s1) + len(s2))


@dataclass
class ComparisonResult:
    """Result of a comparison operation.
    
    Attributes:
        score: Float between 0 and 1 indicating the match quality
        element_scores: Dictionary of scores for each element in the comparison
        field_scores: Dictionary of scores for each field in the comparison 
            aggregated over the elements
        std_score: Standard deviation of scores (for lists and objects)
        true_positive: Number of matches with score >= threshold (for lists)
        false_positive: Number of extra elements in prediction (for lists)
        false_negative: Number of missing elements (for lists)
        precision: Precision score (for lists)
        recall: Recall score (for lists)
        f1_score: F1 score (for lists)
        element_scores: Dictionary with average and std scores per field 
            (for lists)
    """
    score: float = 0.0
    element_scores: Dict[str, Any] = field(default_factory=dict)
    field_scores: Dict[str, Any] = field(default_factory=dict)
    std_score: float = 0.0
    true_positive: bool = False
    false_positive: bool = False
    false_negative: bool = False
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0

    def __init__(
        self,
        score: float,
        element_scores: Dict[str, Any] = {},
        field_scores: Dict[str, Any] = {},
        std_score: float = 0.0,
        true_positive: bool = False,
        false_positive: bool = False,
        false_negative: bool = False,
        precision: float = 0.0,
        recall: float = 0.0,
        f1_score: float = 0.0,
    ):
        self.score = max(0.0, min(1.0, score))
        self.element_scores = element_scores
        self.field_scores = field_scores
        self.std_score = std_score
        self.true_positive = true_positive
        self.false_positive = false_positive
        self.false_negative = false_negative
        self.precision = precision
        self.recall = recall
        self.f1_score = f1_score

    def to_dict(self) -> Dict[str, Any]:
        """Convert the ComparisonResult to a dictionary.
        
        Returns:
            Dictionary representation of the ComparisonResult
        """
        return {
            "score": self.score,
            "element_scores": {
                k: v.to_dict() if isinstance(v, ComparisonResult) else v
                for k, v in self.element_scores.items()
            },
            "field_scores": {
                k: v.to_dict() if isinstance(v, ComparisonResult) else v
                for k, v in self.field_scores.items()
            },
            "std_score": self.std_score,
            "true_positive": self.true_positive,
            "false_positive": self.false_positive,
            "false_negative": self.false_negative,
            "precision": self.precision,
            "recall": self.recall,
            "f1_score": self.f1_score,
        }


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
                - "semantic_similarity": Using SentenceTransformer
                - "longest_common_subsequence": LCS-based similarity
            match_threshold: Threshold for considering a match (0-1)
        """
        self.schema = schema
        self.match_threshold = max(0.0, min(1.0, match_threshold))

        # Set string comparison function based on metric
        if string_metric in ["perfect_match", "exact_match"]:
            self.string_comparator = self._exact_string_match
        elif string_metric == "semantic_similarity":
            self.string_comparator = self._semantic_similarity
        elif string_metric in ["longest_common_subsequence", "fuzzy_match"]:
            self.string_comparator = self._fuzzy_string_match
        else:
            raise ValueError(
                f"Invalid string_metric: {string_metric}. Must be one of: "
                "'perfect_match', 'exact_match', 'semantic_similarity', "
                "'longest_common_subsequence'"
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
        logger.debug(f"raw cos_sim({pred}, {exp}): {score}")
        # negative scores are antonymic, so anything below 0 is essentially 
        # zero from a practical standpoint
        # cap it to 1.0 just to be safe
        return min(max(score, 0), 1.0)

    def _exact_string_match(self, pred: str, exp: str) -> float:
        """Calculate exact string match between two strings.
        
        Args:
            pred: Prediction string
            exp: Expected string
        """
        score = 1.0 if pred == exp else 0.0
        logger.debug(f"exact_match({pred}, {exp}): {score}")
        return score

    def _fuzzy_string_match(self, pred: str, exp: str) -> float:
        """Calculate fuzzy string match using LCS (Longest Common Subsequence) ratio.
        
        This method is good for both short words/keywords and longer text segments.
        It finds the longest common subsequence and normalizes by average length.
        
        Args:
            pred: Prediction string
            exp: Expected string
            
        Returns:
            Score between 0 and 1 indicating LCS-based similarity
        """
        # Normalize strings: lowercase and strip whitespace
        pred_norm = pred.lower().strip()
        exp_norm = exp.lower().strip()
        
        # Calculate LCS ratio
        score = lcs_ratio(pred_norm, exp_norm)
        
        logger.debug(f"fuzzy_match LCS({pred}, {exp}): {score:.3f}")
        
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
        logger.debug(f"Comparing values with schema: {schema_path}")
        value_type = schema.get("type")
        logger.debug(f"Comparing values of type {value_type}")

        if value_type == "string":
            if "enum" in schema:
                if pred_value not in schema["enum"]:
                    return ComparisonResult(
                        score=0.0,
                        element_scores={},
                        field_scores={},
                        false_positive=True
                    )
            return self._compare_strings(pred_value, exp_value)

        elif value_type == "array":
            if not isinstance(pred_value, list) or not isinstance(exp_value, list):
                return ComparisonResult(
                    score=0.0,
                    element_scores={},
                    field_scores={},
                    false_positive=True if pred_value is not None else False,
                    false_negative=True if exp_value is not None else False
                )
            return self._compare_lists(pred_value, exp_value, schema_path)

        elif value_type == "object":
            if not isinstance(pred_value, dict) or not isinstance(exp_value, dict):
                return ComparisonResult(
                    score=0.0,
                    element_scores={},
                    field_scores={},
                    false_positive=True if pred_value is not None else False,
                    false_negative=True if exp_value is not None else False
                )
            return self._compare_objects(pred_value, exp_value, schema_path)

        else:
            # For other types (number, boolean), use exact match
            score = 1.0 if pred_value == exp_value else 0.0
            if score == 0.0:
                return ComparisonResult(
                    score=score,
                    element_scores={},
                    field_scores={},
                    false_positive=True if pred_value is not None else False,
                    false_negative=True if exp_value is not None else False
                )
            return ComparisonResult(
                score=score, 
                element_scores={}, 
                field_scores={}, 
                true_positive=True
            )

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
            - score: Average score across matched fields
            - std_score: Standard deviation of scores
            - true_positive: True if all expected fields matched above threshold
            - false_positive: True if there are any extra fields or mismatches
            - false_negative: True if there are any missing fields or mismatches
            - precision: Precision score
            - recall: Recall score
            - f1_score: F1 score
            - field_scores: Dictionary of ComparisonResults for each field
        """
        logger.debug(f"Comparing objects:\n\texpected: {exp}\n\tprediction: {pred}")

        field_scores = {}
        field_scores_list = []

        # Get schema for this object
        schema = self._get_schema_for_path(schema_path)

        # Get required fields from schema
        required_fields = schema.get("required", [])
        all_expected_fields = set(required_fields)
        all_predicted_fields = set(pred.keys())

        # Track field-level metrics
        field_true_positives = 0
        field_false_positives = 0
        field_false_negatives = 0

        # Compare all required fields
        for field_key in required_fields:
            logger.debug(f"Analyzing object field: {field_key}")
            
            if field_key not in pred:
                # Missing field in prediction
                field_scores[field_key] = ComparisonResult(
                    score=0.0, 
                    element_scores={}, 
                    field_scores={}, 
                    true_positive=False,
                    false_positive=False,
                    false_negative=True
                )
                field_false_negatives += 1
                continue

            if field_key not in exp:
                # Unexpected field in prediction (shouldn't happen with required fields)
                field_scores[field_key] = ComparisonResult(
                    score=0.0, 
                    element_scores={}, 
                    field_scores={}, 
                    true_positive=False,
                    false_positive=True,
                    false_negative=False
                )
                field_false_positives += 1
                continue

            # Both objects have this field, compare values
            pred_value = pred[field_key]
            exp_value = exp[field_key]

            # Get field schema
            field_schema = self._get_schema_for_path(schema_path + [field_key])

            # Compare values based on schema type
            field_result = self._compare_values_with_schema(
                pred_value,
                exp_value,
                field_schema,
                schema_path + [field_key]
            )
            field_scores[field_key] = field_result
            field_scores_list.append(field_result.score)

            # Track field-level metrics from the ComparisonResult
            if field_result.true_positive:
                field_true_positives += 1
            if field_result.false_positive:
                field_false_positives += 1
            if field_result.false_negative:
                field_false_negatives += 1

        # Handle extra fields in prediction (not in required fields)
        extra_fields = all_predicted_fields - all_expected_fields
        for field_key in extra_fields:
            field_scores[field_key] = ComparisonResult(
                score=0.0,
                element_scores={},
                field_scores={},
                true_positive=False,
                false_positive=True,
                false_negative=False
            )
            field_false_positives += 1

        # Handle missing fields in prediction (not in predicted fields)
        missing_fields = all_expected_fields - all_predicted_fields
        for field_key in missing_fields:
            if field_key not in field_scores:  # Avoid double counting
                field_scores[field_key] = ComparisonResult(
                    score=0.0,
                    element_scores={},
                    field_scores={},
                    true_positive=False,
                    false_positive=False,
                    false_negative=True
                )
                field_false_negatives += 1

        if not field_scores:
            return ComparisonResult(0.0, {}, {})

        # Object-level booleans: true_positive only if ALL expected fields 
        # matched above threshold
        total_expected_fields = len(all_expected_fields)
        
        if total_expected_fields == 0:
            # No required fields - empty objects
            true_positive = len(all_predicted_fields) == 0  # True only if both are empty
            false_positive = len(all_predicted_fields) > 0  # Any extra fields
            false_negative = False  # No expected fields to miss
        else:
            # True positive: all expected fields matched above threshold AND no extra fields
            true_positive = (field_true_positives == total_expected_fields and
                           field_false_positives == 0)
            
            # False positive: any extra fields or mismatches
            false_positive = field_false_positives > 0
            
            # False negative: any missing fields or mismatches
            false_negative = field_false_negatives > 0

        # Calculate precision, recall, F1 using field counts
        if total_expected_fields == 0:
            # No required fields - empty objects
            precision = 1.0 if len(all_predicted_fields) == 0 else 0.0
            recall = 1.0  # No expected fields to recall
            f1_score = 1.0 if len(all_predicted_fields) == 0 else 0.0
        else:
            precision = (field_true_positives / (field_true_positives + field_false_positives) 
                        if (field_true_positives + field_false_positives) > 0 else 0)
            recall = (field_true_positives / (field_true_positives + field_false_negatives) 
                     if (field_true_positives + field_false_negatives) > 0 else 0)
            f1_score = (2 * (precision * recall) / (precision + recall) 
                       if (precision + recall) > 0 else 0)

        # Calculate average score and standard deviation
        # Include all expected fields in the score calculation (missing fields count as 0)
        total_expected_fields = len(all_expected_fields)
        if total_expected_fields == 0:
            # No required fields - empty objects
            avg_score = 1.0 if len(all_predicted_fields) == 0 else 0.0
            std_score = 0.0
        else:
            # Build complete score list including missing fields as 0
            complete_scores = []
            for field_key in all_expected_fields:
                if field_key in field_scores:
                    complete_scores.append(field_scores[field_key].score)
                else:
                    complete_scores.append(0.0)  # Missing field = 0 score
            
            avg_score = sum(complete_scores) / len(complete_scores)
            std_score = (statistics.stdev(complete_scores) 
                        if len(complete_scores) > 1 else 0)

        return ComparisonResult(
            score=avg_score,
            element_scores={},
            field_scores=field_scores,
            std_score=std_score,
            true_positive=true_positive,
            false_positive=false_positive,
            false_negative=false_negative,
            precision=precision,
            recall=recall,
            f1_score=f1_score,
        )

    def _compare_lists(
        self,
        pred: List[Any],
        exp: List[Any],
        schema_path: List[str]
    ) -> ComparisonResult:
        """Compare two list values using schema information.
        
        Args:
            pred: Predicted results list
            exp: Expected results list
            schema_path: Path to the schema definition
            
        Returns:
            ComparisonResult with:
            - score: Average score across matched elements
            - std_score: Standard deviation of scores
            - true_positive: Number of matches with score >= threshold
            - false_positive: Number of extra elements
            - false_negative: Number of missing elements
            - precision: Precision score
            - recall: Recall score
            - f1_score: F1 score
            - field_scores: Dictionary of ComparisonResults for each element
            - element_scores: Dictionary with average and std scores per field
        """
        logger.debug(f"Comparing lists:\n\texpected: {exp}\n\tprediction: {pred}")
        # Get item schema for field-level tracking
        item_schema = self._get_schema_for_path(schema_path + ["items"])
        
        if not exp:
            if not pred:
                logger.debug("Comparing 2 empty lists")
                two_empty_lists_result = ComparisonResult(
                    score=1.0,
                    element_scores={},
                    field_scores={},
                    true_positive=True, false_positive=False, 
                    false_negative=False,
                    precision=1.0, recall=1.0, f1_score=1.0,
                )
                
                # For empty lists, field_scores should be empty or perfect scores
                field_scores = {}
                if item_schema.get("type") == "object":
                    for field_key in item_schema.get("required", []):
                        field_scores[field_key] = ComparisonResult(
                            score=1.0,
                            element_scores={},
                            field_scores={},
                            true_positive=True,
                            false_positive=False,
                            false_negative=False,
                            precision=1.0,
                            recall=1.0,
                            f1_score=1.0
                        )
                
                return ComparisonResult(
                    score=1.0,
                    element_scores={
                        "two_empty_lists": two_empty_lists_result
                    },
                    field_scores=field_scores,
                    true_positive=True, false_positive=False, 
                    false_negative=False,
                    precision=1.0, recall=1.0, f1_score=1.0,
                )
            else:
                logger.debug(f"Comparing empty exp with {pred}")
                element_results = {
                    f"unexpected_predicted_element_{i}": ComparisonResult(
                        score=0.0,
                        element_scores={},
                        field_scores={},
                        true_positive=False, false_positive=True, 
                        false_negative=False,
                        precision=0.0, recall=0.0, f1_score=0.0,
                    )
                    for i in range(len(pred))
                }
                
                # Initialize field-level tracking for extra predicted elements
                field_scores = {}
                if item_schema.get("type") == "object":
                    for field_key in item_schema.get("required", []):
                        field_scores[field_key] = ComparisonResult(
                            score=0.0,
                            element_scores={},
                            field_scores={},
                            true_positive=False,
                            false_positive=True,
                            false_negative=False,
                            precision=0.0,
                            recall=0.0,
                            f1_score=0.0
                        )
                
                return ComparisonResult(
                    score=0.0,
                    element_scores=element_results,
                    field_scores=field_scores,
                    true_positive=False, false_positive=True, 
                    false_negative=False,
                    precision=0.0, recall=0.0, f1_score=0.0,
                )
        elif not pred:
            logger.debug(f"Comparing {exp} with empty pred")
            element_results = {
                f"missing_element_{i}": ComparisonResult(
                    score=0.0,
                    element_scores={},
                    field_scores={},
                    true_positive=False, false_positive=False, 
                    false_negative=True
                )
                for i in range(len(exp))
            }
            
            # Initialize field-level tracking for missing expected elements
            field_scores = {}
            if item_schema.get("type") == "object":
                for field_key in item_schema.get("required", []):
                    # Each missing expected element contributes a false negative for each field
                    field_scores[field_key] = ComparisonResult(
                        score=0.0,
                        element_scores={},
                        field_scores={},
                        true_positive=False,
                        false_positive=False,
                        false_negative=True,
                        precision=0.0,
                        recall=0.0,
                        f1_score=0.0
                    )

            return ComparisonResult(
                score=0.0,
                element_scores=element_results,
                field_scores=field_scores,
                true_positive=False, false_positive=False, 
                false_negative=True,
                precision=0.0, recall=0.0, f1_score=0.0,
            )
        else:
            # Get item schema
            item_schema = self._get_schema_for_path(schema_path + ["items"])

            # Use optimal assignment (Hungarian algorithm) to find best matches
            element_results = {}  # these are elementwise scores
            field_scores = {}  # these are fieldwise scores aggregated over the elements
            
            # Initialize field-level tracking if list items are objects
            field_data = {}  # temporary storage for field aggregation
            if item_schema.get("type") == "object":
                # Only track required fields from the schema
                required_fields = item_schema.get("required", [])
                for field_key in required_fields:
                    field_data[field_key] = {
                        'cum_score': 0.0,       # Cumulative score from all elements
                        'false_positives': 0,   # Count of FP contributions
                        'false_negatives': 0,   # Count of FN contributions
                        'true_positives': 0     # Count of TP contributions
                    }
            # Create similarity matrix between all predicted and expected elements
            n_pred = len(pred)
            n_exp = len(exp)
            # first we have to match the elements of the expected and predicted lists
            # to catch nasty bugs, we first assert that the lists are not empty
            assert n_pred > 0 and n_exp > 0, "Lists must not be empty at this point!"
            # Calculate similarity matrix
            similarity_matrix = np.zeros((n_pred, n_exp))
            # create a matrix to store the ComparisonResults objects
            # we do not want to run these expensive comparisons twice
            comparison_results_matrix = np.zeros((n_pred, n_exp), dtype=object)
            for i, pred_item in enumerate(pred):
                for j, exp_item in enumerate(exp):
                    item_result = self._compare_values_with_schema(
                        pred_item,
                        exp_item,
                        item_schema,
                        schema_path + ["items"]
                    )
                    similarity_matrix[i, j] = item_result.score
                    comparison_results_matrix[i, j] = item_result
            # Find optimal assignment using similarity matrix
            assigned_pred_indices, assigned_exp_indices = linear_sum_assignment(-similarity_matrix)
            # Now:
            # - assigned_pred_indices/assigned_exp_indices contain the optimal assignment
            # - We need to check if each assignment meets the threshold to determine if it's TP or mismatch
            # - Unmatched predicted elements are FP
            # - Unmatched expected elements are FN
            # Track which assigned elements are true positives vs mismatches
            true_positive_pred_indices = set()
            true_positive_exp_indices = set()

            # Process assigned pairs to keep only the true positives that are above the threshold
            # mismatches are ignored, they will automatically appear as false positives and false negatives
            
            assert len(assigned_pred_indices) == len(assigned_exp_indices), "Assigned indices count mismatch"
            for pred_idx, exp_idx in zip(assigned_pred_indices, assigned_exp_indices):
                comparison_result = comparison_results_matrix[pred_idx, exp_idx]
                original_score = similarity_matrix[pred_idx, exp_idx]
                
                # Only treat as match if the original score meets the threshold
                if original_score >= self.match_threshold:
                    element_results[f"match_{pred_idx}_{exp_idx}"] = comparison_result
                    true_positive_pred_indices.add(pred_idx)
                    true_positive_exp_indices.add(exp_idx)
                    
                    # Aggregate field scores from this match
                    for field_key, field_result in comparison_result.field_scores.items():
                        if field_key in field_data:
                            field_data[field_key]['cum_score'] += field_result.score
                            field_data[field_key]['true_positives'] += 1
            num_mismatches = len(assigned_pred_indices) - len(true_positive_pred_indices)
            # Handle unmatched predicted elements (extra elements)
            # These are elements that weren't assigned either because
            # hungarian algo did not assign them or
            # because they are sub-threshold
            false_positive_pred_indices = set(range(n_pred)) - set(true_positive_pred_indices)
            for pred_idx in false_positive_pred_indices:
                element_results[f"unexpected_element_{pred_idx}"] = ComparisonResult(
                    score=0.0,
                    element_scores={},
                    field_scores={},
                    true_positive=False,
                    false_positive=True,
                    false_negative=False
                )
                
                # For field aggregation: extra objects contribute to false positives
                if item_schema.get("type") == "object":
                    # Extra objects contribute false positives for all required fields
                    for field_key in field_data.keys():
                        # field_data[field_key]['cum_score'] += 0.0  # False positive = 0 score
                        field_data[field_key]['false_positives'] += 1

            # Handle unmatched expected elements (missing elements)
            false_negative_exp_indices = set(range(n_exp)) - set(true_positive_exp_indices)
            for exp_idx in false_negative_exp_indices:
                element_results[f"missing_element_{exp_idx}"] = ComparisonResult(
                    score=0.0,
                    element_scores={},
                    field_scores={},
                    true_positive=False,
                    false_positive=False,
                    false_negative=True
                )
                
                # For field aggregation: missing objects contribute to false negatives
                if item_schema.get("type") == "object":
                    # Missing objects contribute false negatives for all required fields
                    for field_key in field_data.keys():
                        # field_data[field_key]['cum_score'] += 0.0  # False negative = 0 score
                        field_data[field_key]['false_negatives'] += 1
            
            # NOTE: an index can correspond to both a missing element and an unexpected element
            # there was a prediction, but it could not be matched to an expected element
            # There will be two comparison results for these; not sure if it's so good
            
            # Aggregate metrics from all element comparisons
            # Sum up TP/FP/FN
            true_positive_count = len(true_positive_pred_indices)
            false_positive_count = len(false_positive_pred_indices)
            false_negative_count = len(false_negative_exp_indices)
            
            # List-level booleans: true_positive only if ALL elements matched above threshold
            # false_positive/false_negative if there are any issues

            # True positive: all expected elements matched above threshold AND no extra elements
            true_positive = (
                true_positive_count == n_exp and
                false_positive_count == 0
            )
            
            # False positive: any extra elements or mismatches
            false_positive = false_positive_count > 0
            
            # False negative: any missing elements or mismatches  
            false_negative = false_negative_count > 0
            # if we sum true positive, false positive and false negative, the mismatched elements are counted twice!
            n_all = max(n_pred, n_exp)
            
            assert n_all == true_positive_count + false_positive_count + false_negative_count - num_mismatches, "Mistake with calculating the number of elements"
            # note that scores of mismatches are zero
            all_scores = [
                result.score
                for result in element_results.values()
            ]
            assert len(all_scores) >= n_all, "Mistake with calculating the number of elements"

            # Calculate list-level average, std, precision, recall, F1
            avg_score = sum(all_scores) / n_all if all_scores else 0
            std_score = statistics.stdev(all_scores) if len(all_scores) > 1 else 0
            precision = (true_positive_count / (true_positive_count + false_positive_count) 
                        if (true_positive_count + false_positive_count) > 0 else 0)
            recall = (true_positive_count / (true_positive_count + false_negative_count) 
                     if (true_positive_count + false_negative_count) > 0 else 0)
            f1_score = (2 * (precision * recall) / (precision + recall) 
                       if (precision + recall) > 0 else 0)

            # Aggregate field-level statistics if list items are objects
            if item_schema.get("type") == "object":
                for field_key, data in field_data.items():
                    # Calculate aggregated metrics - be consistent with list-level aggregation
                    field_avg_score = data['cum_score'] / n_all if n_all > 0 else 0
                    
                    field_precision = (data['true_positives'] / (data['true_positives'] + data['false_positives']) 
                                     if (data['true_positives'] + data['false_positives']) > 0 else 0)
                    field_recall = data['true_positives'] / n_all if n_all > 0 else 0
                    field_f1 = (2 * (field_precision * field_recall) / (field_precision + field_recall) 
                               if (field_precision + field_recall) > 0 else 0)
                    
                    field_scores[field_key] = ComparisonResult(
                        score=field_avg_score,
                        element_scores={},
                        field_scores={},
                        std_score=0.0,  # We don't track individual scores anymore
                        true_positive=field_precision == 1.0 and field_recall == 1.0,
                        false_positive=data['false_positives'] > 0,
                        false_negative=data['false_negatives'] > 0,
                        precision=field_precision,
                        recall=field_recall,
                        f1_score=field_f1
                    )

            return ComparisonResult(
                score=avg_score,
                element_scores=element_results,
                field_scores=field_scores,
                std_score=std_score,
                true_positive=true_positive,
                false_positive=false_positive,
                false_negative=false_negative,
                precision=precision,
                recall=recall,
                f1_score=f1_score,
            )

    def _compare_strings(self, pred: Optional[str], exp: Optional[str]) -> ComparisonResult:
        """Compare two strings using the configured string comparator.

        Args:
            pred: Prediction string
            exp: Expected string    
        """
        logger.debug(f"Comparing strings:\n\texpected: {exp}\n\tprediction: {pred}")

        # Handle None cases first
        if pred is None and exp is None:
            return ComparisonResult(
                score=1.0,
                element_scores={},
                field_scores={},
                true_positive=True, false_positive=False, false_negative=False,
                precision=1.0, recall=1.0, f1_score=1.0,
            )
        if pred is None and exp is not None:
            return ComparisonResult(
                score=0.0,
                element_scores={},
                field_scores={},
                true_positive=False, false_positive=False, false_negative=True,
                precision=0.0, recall=0.0, f1_score=0.0,
            )
        if pred is not None and exp is None:
            return ComparisonResult(
                score=0.0,
                element_scores={},
                field_scores={},
                true_positive=False, false_positive=True, false_negative=False,
                precision=0.0, recall=0.0, f1_score=0.0,
            )

        # Both are strings, compare them
        # We know pred and exp are not None here due to the checks above
        score = self.string_comparator(str(pred), str(exp))
        if score >= self.match_threshold:
            return ComparisonResult(
                score=score,
                element_scores={},
                field_scores={},
                true_positive=True, false_positive=False, false_negative=False,
                precision=1.0, recall=1.0, f1_score=1.0,
            )
        else:
            return ComparisonResult(
                score=score,
                element_scores={},
                field_scores={},
                true_positive=False, false_positive=True, false_negative=False,
                precision=0.0, recall=0.0, f1_score=0.0,
            )
