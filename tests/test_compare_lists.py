import os
import sys
import unittest
from unittest.mock import patch

# Add the project root to the path
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from soda_mmqc.evaluation import JSONEvaluator


class TestCompareLists(unittest.TestCase):
    """Test cases for the _compare_lists method."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a schema for testing list comparisons
        self.schema = {
            "format": {
                "schema": {
                    "type": "object",
                    "properties": {
                        "outputs": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "name": {"type": "string"},
                                    "value": {"type": "string"},
                                    "score": {"type": "number"}
                                },
                                "required": ["name", "value"]
                            }
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

    def test_both_empty_lists(self):
        """Test when both prediction and expected lists are empty."""
        result = self.evaluator._compare_lists([], [], ["outputs"])
        
        self.assertEqual(result.score, 1.0)
        self.assertEqual(result.true_positive, True)
        self.assertEqual(result.false_positive, False)
        self.assertEqual(result.false_negative, False)
        self.assertEqual(result.precision, 1.0)
        self.assertEqual(result.recall, 1.0)
        self.assertEqual(result.f1_score, 1.0)
        self.assertIn("two_empty_lists", result.element_scores)

    def test_expected_empty_prediction_not_empty(self):
        """Test when expected list is empty but prediction is not."""
        pred = [{"name": "test", "value": "data"}]
        result = self.evaluator._compare_lists(pred, [], ["outputs"])
        
        self.assertEqual(result.score, 0.0)
        self.assertEqual(result.true_positive, False)
        self.assertEqual(result.false_positive, True)
        self.assertEqual(result.false_negative, False)
        self.assertEqual(result.precision, 0.0)
        self.assertEqual(result.recall, 0.0)
        self.assertEqual(result.f1_score, 0.0)
        self.assertIn("unexpected_predicted_element_0", result.element_scores)

    def test_prediction_empty_expected_not_empty(self):
        """Test when prediction list is empty but expected is not."""
        exp = [{"name": "test", "value": "data"}]
        result = self.evaluator._compare_lists([], exp, ["outputs"])
        
        self.assertEqual(result.score, 0.0)
        self.assertEqual(result.true_positive, False)
        self.assertEqual(result.false_positive, False)
        self.assertEqual(result.false_negative, True)
        self.assertEqual(result.precision, 0.0)
        self.assertEqual(result.recall, 0.0)
        self.assertEqual(result.f1_score, 0.0)
        self.assertIn("missing_element_0", result.element_scores)

    def test_perfect_match_same_length(self):
        """Test when lists have same length and all elements match."""
        pred = [
            {"name": "item1", "value": "data1"},
            {"name": "item2", "value": "data2"}
        ]
        exp = [
            {"name": "item1", "value": "data1"},
            {"name": "item2", "value": "data2"}
        ]
        result = self.evaluator._compare_lists(pred, exp, ["outputs"])
        self.assertEqual(result.score, 1.0)
        # Both elements are perfect matches (score = 1.0 >= 0.6)
        self.assertEqual(result.true_positive, True)
        self.assertEqual(result.false_positive, False)
        self.assertEqual(result.false_negative, False)
        self.assertEqual(result.precision, 1.0)
        self.assertEqual(result.recall, 1.0)
        self.assertEqual(result.f1_score, 1.0)
        self.assertIn("match_0_0", result.element_scores)
        self.assertIn("match_1_1", result.element_scores)

    def test_partial_match_same_length(self):
        """Test when lists have same length but some elements don't match."""
        pred = [
            {"name": "item1", "value": "data1"},
            {"name": "item2", "value": "wrong_data"},
            {"name": "item3", "value": "other_wrong_data"}
        ]
        exp = [
            {"name": "item1", "value": "data1"},
            {"name": "item2", "value": "data2"},
            {"name": "item3", "value": "data3"}
        ]
        result = self.evaluator._compare_lists(pred, exp, ["outputs"])
        
        # First element: perfect match (score >= threshold) = 1 TP
        # Second element: mismatch (score < threshold) = 1 FP, 1 FN
        # Third element: mismatch (score < threshold) = 1 FP, 1 FN
        # num_all = 3
        # true_positive should be False because not all elements matched perfectly
        self.assertEqual(result.score, 1/3)  # Average of element scores (1.0 + 0 + 0) / 3
        self.assertEqual(result.true_positive, False)  # Not all elements matched perfectly
        self.assertEqual(result.false_positive, True)
        self.assertEqual(result.false_negative, True)  # 1 mismatch
        self.assertEqual(result.precision, 1/3)
        self.assertEqual(result.recall, 1/3)
        self.assertEqual(result.f1_score, 1/3)  # 2 * (1/3 * 1/3) / (1/3 + 1/3)

    def test_prediction_shorter_than_expected(self):
        """Test when prediction list is shorter than expected (missing elements)."""
        pred = [{"name": "item1", "value": "data1"}]
        exp = [
            {"name": "item1", "value": "data1"},
            {"name": "item2", "value": "data2"},
            {"name": "item3", "value": "data3"}
        ]
        result = self.evaluator._compare_lists(pred, exp, ["outputs"])
        
        # One element matches (1 TP), two missing elements (2 FN)
        # true_positive should be False because not all expected elements matched
        self.assertEqual(result.score, 1/3)  # 1 match out of 3 total elements
        self.assertEqual(result.true_positive, False)  # Not all elements matched
        self.assertEqual(result.false_positive, False)
        self.assertEqual(result.false_negative, True)  # 2 missing elements
        self.assertEqual(result.precision, 1.0)
        self.assertEqual(result.recall, 1/3)
        self.assertIn("match_0_0", result.element_scores)
        self.assertIn("missing_element_1", result.element_scores)
        self.assertIn("missing_element_2", result.element_scores)

    def test_prediction_longer_than_expected(self):
        """Test when prediction list is longer than expected (extra elements)."""
        pred = [
            {"name": "item1", "value": "data1"},
            {"name": "item2", "value": "data2"},
            {"name": "extra", "value": "extra_data"}
        ]
        exp = [
            {"name": "item1", "value": "data1"},
            {"name": "item2", "value": "data2"}
        ]
        result = self.evaluator._compare_lists(pred, exp, ["outputs"])
        
        # Two elements match (2 TP), one extra element (1 FP)
        # true_positive should be False because there's an extra element
        self.assertEqual(result.score, 2/3)  # 2 matches out of 3 total elements
        self.assertEqual(result.true_positive, False)  # Extra element means not perfect
        self.assertEqual(result.false_positive, True)
        self.assertEqual(result.false_negative, False)
        self.assertEqual(result.precision, 2/3)
        self.assertEqual(result.recall, 1.0)
        self.assertIn("match_0_0", result.element_scores)
        self.assertIn("match_1_1", result.element_scores)
        self.assertIn("unexpected_element_2", result.element_scores)

    def test_string_elements(self):
        """Test list comparison with string elements."""
        pred = ["hello", "world", "extra"]
        exp = ["hello", "world", "missing"]
        
        # Create a simple schema for string lists
        string_schema = {
            "format": {
                "schema": {
                    "type": "object",
                    "properties": {
                        "outputs": {
                            "type": "array",
                            "items": {"type": "string"}
                        }
                    }
                }
            }
        }
        string_evaluator = JSONEvaluator(string_schema, string_metric="perfect_match")
        
        result = string_evaluator._compare_lists(pred, exp, ["outputs"])
        
        # 2 matches (2 TP), 1 mismatch (1 FP and FN)
        # num element = 3 (we don't count mismatches twice)
        # true_positive should be False because not all elements matched
        self.assertEqual(result.score, 0.6666666666666666)  # Average of element scores (1.0 + 1.0 + 0.0 + 0.0) / 3
        self.assertEqual(result.true_positive, False)  # Not all elements matched
        self.assertEqual(result.false_positive, True)
        self.assertEqual(result.false_negative, True)  # 1 mismatch

    def test_number_elements(self):
        """Test list comparison with number elements."""
        pred = [1, 2, 3, 4]
        exp = [1, 2, 5]
        
        # Create a simple schema for number lists
        number_schema = {
            "format": {
                "schema": {
                    "type": "object",
                    "properties": {
                        "outputs": {
                            "type": "array",
                            "items": {"type": "number"}
                        }
                    },
                    "required": ["outputs"]
                }
            }
        }
        number_evaluator = JSONEvaluator(number_schema, string_metric="perfect_match")
        
        result = number_evaluator._compare_lists(pred, exp, ["outputs"])
        
        # 2 matches (2 TP), 1 mismatch (1 FP), 1 extra (1 FP), 1 missing (1 FN)
        # true_positive should be False because not all elements matched perfectly
        self.assertEqual(result.score, 0.5)  # Average of element scores (1.0 + 1.0 + 0.0 + 0.0) / 4
        self.assertEqual(result.true_positive, False)  # Not all elements matched perfectly
        self.assertEqual(result.false_positive, True)
        self.assertEqual(result.false_negative, True)  # 1 mismatch

    def test_field_scores_structure(self):
        """Test that field_scores has the expected structure."""
        pred = [
            {"name": "item1", "value": "data1"},
            {"name": "item2", "value": "wrong"}
        ]
        exp = [
            {"name": "item1", "value": "data1"},
            {"name": "item2", "value": "data2"}
        ]
        result = self.evaluator._compare_lists(pred, exp, ["outputs"])
        
        # Check that field_scores exists and has the right structure
        self.assertIsInstance(result.field_scores, dict)
        
        # If there are field scores, check their structure
        if result.field_scores:
            for field, field_result in result.field_scores.items():
                self.assertIsInstance(field_result, type(result))  # ComparisonResult type
                # Check that the ComparisonResult has the expected attributes
                self.assertIsInstance(field_result.score, float)
                self.assertIsInstance(field_result.precision, float)
                self.assertIsInstance(field_result.recall, float)
                self.assertIsInstance(field_result.f1_score, float)

    def test_standard_deviation_calculation(self):
        """Test that standard deviation is calculated correctly."""
        pred = [
            {"name": "item1", "value": "data1"},  # Perfect match
            {"name": "item2", "value": "wrong"},  # Mismatch
            {"name": "item3", "value": "data3"}   # Perfect match
        ]
        exp = [
            {"name": "item1", "value": "data1"},
            {"name": "item2", "value": "data2"},
            {"name": "item3", "value": "data3"}
        ]
        result = self.evaluator._compare_lists(pred, exp, ["outputs"])
        
        # Scores should be [1.0, 0.0, 1.0] for the three elements
        # Standard deviation should be calculated from these scores
        self.assertIsInstance(result.std_score, float)
        self.assertGreaterEqual(result.std_score, 0.0)

    def test_error_messages(self):
        """Test that appropriate error messages are generated."""
        pred = [{"name": "item1", "value": "data1"}]
        exp = [
            {"name": "item1", "value": "data1"},
            {"name": "item2", "value": "data2"}
        ]
        result = self.evaluator._compare_lists(pred, exp, ["outputs"])
        

    def test_element_scores_structure(self):
        """Test that field_scores contains the expected structure."""
        pred = [
            {"name": "item1", "value": "data1"},
            {"name": "item2", "value": "wrong"}
        ]
        exp = [
            {"name": "item1", "value": "data1"},
            {"name": "item2", "value": "data2"}
        ]
        result = self.evaluator._compare_lists(pred, exp, ["outputs"])
        
        # Check that element_scores contains the expected keys
        # First element: perfect match (score >= threshold)
        self.assertIn("match_0_0", result.element_scores)
        # Second element: subthreshold assignment becomes unmatched
        # The prediction element becomes a false positive (unexpected)
        # The expected element becomes a false negative (missing)
        self.assertIn("unexpected_element_1", result.element_scores)
        self.assertIn("missing_element_1", result.element_scores)
        
        # Each element in element_scores should be a ComparisonResult
        for key, value in result.element_scores.items():
            self.assertIsInstance(value, type(result))  # ComparisonResult type
        
        # field_scores should contain ComparisonResult objects
        if result.field_scores:
            for field, field_result in result.field_scores.items():
                self.assertIsInstance(field_result, type(result))  # ComparisonResult type


if __name__ == "__main__":
    unittest.main() 