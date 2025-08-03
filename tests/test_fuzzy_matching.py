import unittest
from soda_mmqc.core.evaluation import JSONEvaluator, lcs_ratio


class TestFuzzyMatching(unittest.TestCase):
    def setUp(self):
        # Create a simple schema for testing
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
                                    "text": {"type": "string"}
                                },
                                "required": ["text"]
                            }
                        }
                    },
                    "required": ["outputs"]
                }
            }
        }
        
        # Create evaluator with fuzzy matching for testing
        self.fuzzy_evaluator = JSONEvaluator(
            self.schema, 
            string_metric="longest_common_subsequence", 
            match_threshold=0.5
        )

    def test_lcs_ratio_directly(self):
        """Test the LCS ratio function directly."""
        # Identical strings
        self.assertEqual(lcs_ratio("yes", "yes"), 1.0)
        self.assertEqual(lcs_ratio("hello world", "hello world"), 1.0)
        
        # Completely different strings
        self.assertEqual(lcs_ratio("yes", "no"), 0.0)
        self.assertEqual(lcs_ratio("cat", "dog"), 0.0)
        
        # Empty strings
        self.assertEqual(lcs_ratio("", ""), 1.0)
        self.assertEqual(lcs_ratio("", "hello"), 0.0)
        self.assertEqual(lcs_ratio("hello", ""), 0.0)
        
        # Partial matches
        # "hello" and "helo" have LCS "helo" (length 4)
        # LCS ratio = 2*4 / (5+4) = 8/9 ≈ 0.889
        self.assertAlmostEqual(lcs_ratio("hello", "helo"), 8/9, places=3)
        
        # Text segments - expected use case
        # "the cat sat" vs "the big cat sat" 
        # LCS = "the cat sat" (length 11, including spaces)
        # LCS ratio = 2*11 / (11+15) = 22/26 ≈ 0.846
        expected = lcs_ratio("the cat sat", "the big cat sat")
        self.assertGreater(expected, 0.8)

    def test_exact_match(self):
        """Test that exact matches return perfect score."""
        result = self.fuzzy_evaluator._compare_strings("yes", "yes")
        self.assertEqual(result.score, 1.0)
        self.assertEqual(result.true_positive, True)

    def test_case_insensitive_match(self):
        """Test that case differences are handled correctly."""
        result = self.fuzzy_evaluator._compare_strings("YES", "yes")
        self.assertEqual(result.score, 1.0)
        self.assertEqual(result.true_positive, True)

    def test_whitespace_handling(self):
        """Test that whitespace differences are handled correctly."""
        result = self.fuzzy_evaluator._compare_strings(" yes ", "yes")
        self.assertEqual(result.score, 1.0)
        self.assertEqual(result.true_positive, True)

    def test_yes_no_variations(self):
        """Test common yes/no variations."""
        # Perfect matches after normalization
        result1 = self.fuzzy_evaluator._compare_strings("Yes", "yes")
        self.assertEqual(result1.score, 1.0)
        
        result2 = self.fuzzy_evaluator._compare_strings("YES", "yes")
        self.assertEqual(result2.score, 1.0)
        
        result3 = self.fuzzy_evaluator._compare_strings("No", "no")
        self.assertEqual(result3.score, 1.0)
        
        result4 = self.fuzzy_evaluator._compare_strings("NO", "no")
        self.assertEqual(result4.score, 1.0)

    def test_completely_different_answers(self):
        """Test that completely different answers get zero score."""
        result = self.fuzzy_evaluator._compare_strings("yes", "no")
        self.assertEqual(result.score, 0.0)
        self.assertEqual(result.true_positive, False)

    def test_text_segment_extraction_cases(self):
        """Test cases where model extracts different text segments."""
        # Model extracts more text than expected
        result1 = self.fuzzy_evaluator._compare_strings(
            "the cat sat on the mat", 
            "I saw that the cat sat on the mat yesterday"
        )
        # Should have good similarity due to common subsequence
        self.assertGreater(result1.score, 0.6)
        
        # Model extracts less text than expected  
        result2 = self.fuzzy_evaluator._compare_strings(
            "the big cat sat on the mat",
            "cat sat on mat" 
        )
        # Should have moderate similarity
        self.assertGreater(result2.score, 0.5)

    def test_partial_word_overlap(self):
        """Test cases with partial word overlap."""
        result = self.fuzzy_evaluator._compare_strings("present", "presentation")
        # LCS would find "present" (7 chars) in "presentation" (12 chars)
        # LCS ratio = 2*7 / (7+12) = 14/19 ≈ 0.737
        self.assertGreater(result.score, 0.7)

    def test_word_order_changes(self):
        """Test that word order changes affect the score appropriately."""
        result = self.fuzzy_evaluator._compare_strings("red blue", "blue red")
        # LCS would find either "red" or "blue" but not both due to order
        # Should have moderate score
        self.assertGreater(result.score, 0.4)
        self.assertLess(result.score, 0.8)

    def test_empty_strings(self):
        """Test comparison of empty strings."""
        result = self.fuzzy_evaluator._compare_strings("", "")
        self.assertEqual(result.score, 1.0)
        self.assertEqual(result.true_positive, True)

    def test_empty_vs_non_empty(self):
        """Test comparison of empty string vs non-empty string."""
        result = self.fuzzy_evaluator._compare_strings("", "yes")
        self.assertEqual(result.score, 0.0)
        self.assertEqual(result.true_positive, False)

    def test_none_values(self):
        """Test handling of None values."""
        result = self.fuzzy_evaluator._compare_strings(None, None)
        self.assertEqual(result.score, 1.0)
        self.assertEqual(result.true_positive, True)

    def test_none_vs_string(self):
        """Test comparison of None vs string."""
        result = self.fuzzy_evaluator._compare_strings(None, "yes")
        self.assertEqual(result.score, 0.0)
        self.assertEqual(result.true_positive, False)

    def test_threshold_behavior(self):
        """Test that threshold behavior works correctly."""
        # Test with high threshold
        high_threshold_evaluator = JSONEvaluator(
            self.schema, 
            string_metric="longest_common_subsequence", 
            match_threshold=0.9
        )
        
        # Create a case that has moderate similarity
        result = high_threshold_evaluator._compare_strings("hello", "helo")
        # Should be above 0.8 but below 0.9, so false positive
        self.assertGreater(result.score, 0.8)
        self.assertLess(result.score, 0.9)
        self.assertEqual(result.true_positive, False)

    def test_comparison_result_structure(self):
        """Test that ComparisonResult has the expected structure."""
        result = self.fuzzy_evaluator._compare_strings("test", "test")
        
        # Check that all expected attributes exist
        self.assertIsInstance(result.score, float)
        self.assertIsInstance(result.element_scores, dict)
        self.assertIsInstance(result.field_scores, dict)
        self.assertIsInstance(result.std_score, float)
        self.assertIsInstance(result.true_positive, bool)
        self.assertIsInstance(result.false_positive, bool)
        self.assertIsInstance(result.false_negative, bool)
        self.assertIsInstance(result.precision, float)
        self.assertIsInstance(result.recall, float)
        self.assertIsInstance(result.f1_score, float)


if __name__ == "__main__":
    unittest.main() 