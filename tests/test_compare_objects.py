import unittest
from unittest.mock import patch

from soda_mmqc.core.evaluation import JSONEvaluator


class TestCompareObjects(unittest.TestCase):
    """Test cases for the _compare_objects method."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a schema for testing object comparisons
        self.schema = {
            "format": {
                "schema": {
                    "type": "object",
                    "properties": {
                        "outputs": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "value": {"type": "string"},
                                "score": {"type": "number"},
                                "description": {"type": "string"}
                            },
                            "required": ["name", "value", "score"]
                        }
                    }
                }
            }
        }
        
        # Create evaluator with exact matching for predictable tests
        # Use threshold 0.6 so that only perfect matches (score=1.0) count as TP
        self.evaluator = JSONEvaluator(
            self.schema, 
            string_metric="perfect_match", 
            match_threshold=0.6
        )

    def test_perfect_match_all_fields(self):
        """Test when all fields match exactly."""
        pred = {
            "name": "test_item",
            "value": "test_value", 
            "score": 42.5
        }
        exp = {
            "name": "test_item",
            "value": "test_value",
            "score": 42.5
        }
        result = self.evaluator._compare_objects(pred, exp, ["outputs"])
        
        self.assertEqual(result.score, 1.0)
        self.assertEqual(result.true_positive, True)
        self.assertEqual(result.false_positive, False)
        self.assertEqual(result.false_negative, False)
        self.assertEqual(result.precision, 1.0)
        self.assertEqual(result.recall, 1.0)
        self.assertEqual(result.f1_score, 1.0)
        self.assertIn("name", result.field_scores)
        self.assertIn("value", result.field_scores)
        self.assertIn("score", result.field_scores)

    def test_partial_match_some_fields(self):
        """Test when some fields match and others don't."""
        pred = {
            "name": "test_item",
            "value": "wrong_value",  # Mismatch
            "score": 42.5
        }
        exp = {
            "name": "test_item",
            "value": "test_value",
            "score": 42.5
        }
        result = self.evaluator._compare_objects(pred, exp, ["outputs"])
        
        # 2 fields match (2 TP), 1 field mismatch (1 FP)
        self.assertEqual(result.score, 2/3)  # Average of field scores (1.0 + 0.0 + 1.0) / 3
        self.assertEqual(result.true_positive, False)  # Not all fields matched perfectly
        self.assertEqual(result.false_positive, True)
        self.assertEqual(result.false_negative, False)  # Mismatch only counts as false positive
        self.assertEqual(result.precision, 2/3)
        self.assertEqual(result.recall, 1.0)  # 2 TP / (2 TP + 0 FN) = 1.0

    def test_missing_required_field(self):
        """Test when a required field is missing from prediction."""
        pred = {
            "name": "test_item",
            "value": "test_value"
            # Missing "score" field
        }
        exp = {
            "name": "test_item",
            "value": "test_value",
            "score": 42.5
        }
        result = self.evaluator._compare_objects(pred, exp, ["outputs"])
        
        # 2 fields match (2 TP), 1 missing field (1 FN)
        self.assertEqual(result.score, 2/3)  # Average of field scores (1.0 + 1.0 + 0.0) / 3
        self.assertEqual(result.true_positive, False)  # Missing required field
        self.assertEqual(result.false_positive, False)
        self.assertEqual(result.false_negative, True)
        self.assertEqual(result.precision, 1.0)
        self.assertEqual(result.recall, 2/3)
        self.assertIn("score", result.field_scores)

    def test_extra_field_in_prediction(self):
        """Test when prediction has extra fields not in expected."""
        pred = {
            "name": "test_item",
            "value": "test_value",
            "score": 42.5,
            "extra_field": "extra_value"  # Extra field
        }
        exp = {
            "name": "test_item",
            "value": "test_value",
            "score": 42.5
        }
        result = self.evaluator._compare_objects(pred, exp, ["outputs"])
        
        # 3 fields match (3 TP), 1 extra field (1 FP)
        self.assertEqual(result.score, 1.0)  # All expected fields match perfectly (1.0 + 1.0 + 1.0) / 3
        self.assertEqual(result.true_positive, False)  # Extra field means not perfect
        self.assertEqual(result.false_positive, True)
        self.assertEqual(result.false_negative, False)
        self.assertEqual(result.precision, 3/4)
        self.assertEqual(result.recall, 1.0)
        self.assertIn("extra_field", result.field_scores)

    def test_mixed_scenario(self):
        """Test a complex scenario with matches, mismatches, missing and extra fields."""
        pred = {
            "name": "test_item",      # Perfect match
            "value": "wrong_value",   # Mismatch
            "score": 42.5,           # Perfect match
            "extra_field": "extra"    # Extra field
        }
        exp = {
            "name": "test_item",
            "value": "test_value",
            "score": 42.5
            # Missing "description" field (optional)
        }
        result = self.evaluator._compare_objects(pred, exp, ["outputs"])
        
        # 2 fields match (2 TP), 1 mismatch (1 FP), 1 extra (1 FP)
        self.assertEqual(result.score, 2/3)  # Average of expected field scores (1.0 + 0.0 + 1.0) / 3
        self.assertEqual(result.true_positive, False)  # Not all fields matched perfectly
        self.assertEqual(result.false_positive, True)
        self.assertEqual(result.false_negative, False)  # No missing fields, only extra/mismatched
        self.assertEqual(result.precision, 2/4)
        self.assertEqual(result.recall, 1.0)  # 2 TP / (2 TP + 0 FN) = 1.0

    def test_empty_objects(self):
        """Test comparison of empty objects."""
        pred = {}
        exp = {}
        result = self.evaluator._compare_objects(pred, exp, ["outputs"])
        
        # Schema requires ["name", "value", "score"] fields, so empty objects are missing all required fields
        self.assertEqual(result.score, 0.0)  # All required fields missing = 0 score
        self.assertEqual(result.true_positive, False)  # Missing all required fields
        self.assertEqual(result.false_positive, False)
        self.assertEqual(result.false_negative, True)  # All required fields missing
        self.assertEqual(result.precision, 0.0)
        self.assertEqual(result.recall, 0.0)

    def test_prediction_empty_expected_not_empty(self):
        """Test when prediction is empty but expected has fields."""
        pred = {}
        exp = {
            "name": "test_item",
            "value": "test_value",
            "score": 42.5
        }
        result = self.evaluator._compare_objects(pred, exp, ["outputs"])
        
        # All required fields are missing
        self.assertEqual(result.score, 0.0)
        self.assertEqual(result.true_positive, False)
        self.assertEqual(result.false_positive, False)
        self.assertEqual(result.false_negative, True)
        self.assertEqual(result.precision, 0.0)
        self.assertEqual(result.recall, 0.0)

    def test_expected_empty_prediction_not_empty(self):
        """Test when expected is empty but prediction has fields."""
        pred = {
            "name": "test_item",
            "value": "test_value",
            "score": 42.5
        }
        exp = {}
        result = self.evaluator._compare_objects(pred, exp, ["outputs"])
        
        # All fields are extra
        self.assertEqual(result.score, 0.0)
        self.assertEqual(result.true_positive, False)
        self.assertEqual(result.false_positive, True)
        self.assertEqual(result.false_negative, False)
        self.assertEqual(result.precision, 0.0)
        self.assertEqual(result.recall, 0.0)

    def test_nested_object_fields(self):
        """Test comparison with nested object fields."""
        # Create schema with nested object
        nested_schema = {
            "format": {
                "schema": {
                    "type": "object",
                    "properties": {
                        "outputs": {
                            "type": "object",
                            "properties": {
                                "metadata": {
                                    "type": "object",
                                    "properties": {
                                        "author": {"type": "string"},
                                        "version": {"type": "string"}
                                    },
                                    "required": ["author", "version"]
                                },
                                "data": {"type": "string"}
                            },
                            "required": ["metadata", "data"]
                        }
                    }
                }
            }
        }
        nested_evaluator = JSONEvaluator(nested_schema, string_metric="perfect_match")
        
        pred = {
            "metadata": {
                "author": "John Doe",
                "version": "1.0"
            },
            "data": "test_data"
        }
        exp = {
            "metadata": {
                "author": "John Doe",
                "version": "1.0"
            },
            "data": "test_data"
        }
        result = nested_evaluator._compare_objects(pred, exp, ["outputs"])
        
        self.assertEqual(result.score, 1.0)
        self.assertEqual(result.true_positive, True)
        self.assertEqual(result.false_positive, False)
        self.assertEqual(result.false_negative, False)

    def test_array_field_comparison(self):
        """Test comparison with array fields."""
        # Create schema with array field
        array_schema = {
            "format": {
                "schema": {
                    "type": "object",
                    "properties": {
                        "outputs": {
                            "type": "object",
                            "properties": {
                                "tags": {
                                    "type": "array",
                                    "items": {"type": "string"}
                                },
                                "name": {"type": "string"}
                            },
                            "required": ["tags", "name"]
                        }
                    }
                }
            }
        }
        array_evaluator = JSONEvaluator(array_schema, string_metric="perfect_match")
        
        pred = {
            "tags": ["tag1", "tag2"],
            "name": "test_item"
        }
        exp = {
            "tags": ["tag1", "tag2"],
            "name": "test_item"
        }
        result = array_evaluator._compare_objects(pred, exp, ["outputs"])
        
        self.assertEqual(result.score, 1.0)
        self.assertEqual(result.true_positive, True)
        self.assertEqual(result.false_positive, False)
        self.assertEqual(result.false_negative, False)

    def test_field_scores_structure(self):
        """Test that field_scores contains the expected structure."""
        pred = {
            "name": "test_item",
            "value": "wrong_value",
            "score": 42.5
        }
        exp = {
            "name": "test_item",
            "value": "test_value",
            "score": 42.5
        }
        result = self.evaluator._compare_objects(pred, exp, ["outputs"])
        
        # Check that field_scores contains the expected keys
        self.assertIn("name", result.field_scores)
        self.assertIn("value", result.field_scores)
        self.assertIn("score", result.field_scores)
        
        # Each field should be a ComparisonResult
        for key, value in result.field_scores.items():
            self.assertIsInstance(value, type(result))  # ComparisonResult type

    def test_standard_deviation_calculation(self):
        """Test that standard deviation is calculated correctly."""
        pred = {
            "name": "test_item",      # Perfect match
            "value": "wrong_value",   # Mismatch
            "score": 42.5            # Perfect match
        }
        exp = {
            "name": "test_item",
            "value": "test_value",
            "score": 42.5
        }
        result = self.evaluator._compare_objects(pred, exp, ["outputs"])
        
        # Scores should be [1.0, 0.0, 1.0] for the three fields
        # Standard deviation should be calculated from these scores
        self.assertIsInstance(result.std_score, float)
        self.assertGreaterEqual(result.std_score, 0.0)

if __name__ == "__main__":
    unittest.main() 