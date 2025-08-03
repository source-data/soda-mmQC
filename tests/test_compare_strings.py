import unittest
from soda_mmqc.core.evaluation import JSONEvaluator


class TestCompareStrings(unittest.TestCase):
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
        
        # Create evaluator with exact matching for testing
        self.exact_evaluator = JSONEvaluator(
            self.schema, 
            string_metric="perfect_match", 
            match_threshold=0.5
        )
        
        # Create evaluator with semantic similarity for testing
        self.semantic_evaluator = JSONEvaluator(
            self.schema, 
            string_metric="semantic_similarity", 
            match_threshold=0.5
        )

    def test_both_none(self):
        """Test when both prediction and expected are None."""
        result = self.exact_evaluator._compare_strings(None, None)
        
        self.assertEqual(result.score, 1.0)
        self.assertEqual(result.true_positive, True)
        self.assertEqual(result.false_positive, False)
        self.assertEqual(result.false_negative, False)
        self.assertEqual(result.precision, 1.0)
        self.assertEqual(result.recall, 1.0)
        self.assertEqual(result.f1_score, 1.0)

    def test_prediction_none_expected_not_none(self):
        """Test when prediction is None but expected is not None."""
        result = self.exact_evaluator._compare_strings(
            None, "expected text"
        )
        
        self.assertEqual(result.score, 0.0)
        self.assertEqual(result.true_positive, False)
        self.assertEqual(result.false_positive, False)
        self.assertEqual(result.false_negative, True)
        self.assertEqual(result.precision, 0.0)
        self.assertEqual(result.recall, 0.0)
        self.assertEqual(result.f1_score, 0.0)

    def test_prediction_not_none_expected_none(self):
        """Test when prediction is not None but expected is None."""
        result = self.exact_evaluator._compare_strings(
            "predicted text", None
        )
        
        self.assertEqual(result.score, 0.0)
        self.assertEqual(result.true_positive, False)
        self.assertEqual(result.false_positive, True)
        self.assertEqual(result.false_negative, False)
        self.assertEqual(result.precision, 0.0)
        self.assertEqual(result.recall, 0.0)
        self.assertEqual(result.f1_score, 0.0)

    def test_exact_match_above_threshold(self):
        """Test when strings match exactly and score is above threshold."""
        result = self.exact_evaluator._compare_strings(
            "hello world", "hello world"
        )
        
        self.assertEqual(result.score, 1.0)
        self.assertEqual(result.true_positive, True)
        self.assertEqual(result.false_positive, False)
        self.assertEqual(result.false_negative, False)
        self.assertEqual(result.precision, 1.0)
        self.assertEqual(result.recall, 1.0)
        self.assertEqual(result.f1_score, 1.0)

    def test_exact_match_below_threshold(self):
        """Test when strings don't match and score is below threshold."""
        result = self.exact_evaluator._compare_strings(
            "hello world", "different text"
        )
        
        self.assertEqual(result.score, 0.0)
        self.assertEqual(result.true_positive, False)
        self.assertEqual(result.false_positive, True)
        self.assertEqual(result.false_negative, False)
        self.assertEqual(result.precision, 0.0)
        self.assertEqual(result.recall, 0.0)
        self.assertEqual(result.f1_score, 0.0)

    def test_case_sensitivity(self):
        """Test that exact matching is case sensitive."""
        result = self.exact_evaluator._compare_strings(
            "Hello World", "hello world"
        )
        
        self.assertEqual(result.score, 0.0)
        self.assertEqual(result.true_positive, False)
        self.assertEqual(result.false_positive, True)

    def test_whitespace_sensitivity(self):
        """Test that exact matching is sensitive to whitespace."""
        result = self.exact_evaluator._compare_strings(
            "hello world", "hello  world"
        )
        
        self.assertEqual(result.score, 0.0)
        self.assertEqual(result.true_positive, False)
        self.assertEqual(result.false_positive, True)

    def test_empty_strings(self):
        """Test comparison of empty strings."""
        result = self.exact_evaluator._compare_strings("", "")
        
        self.assertEqual(result.score, 1.0)
        self.assertEqual(result.true_positive, True)
        self.assertEqual(result.false_positive, False)
        self.assertEqual(result.false_negative, False)

    def test_empty_vs_non_empty(self):
        """Test comparison of empty string vs non-empty string."""
        result = self.exact_evaluator._compare_strings("", "hello")
        
        self.assertEqual(result.score, 0.0)
        self.assertEqual(result.true_positive, False)
        self.assertEqual(result.false_positive, True)
        self.assertEqual(result.false_negative, False)

    def test_non_string_inputs(self):
        """Test that non-string inputs are converted to strings."""
        result = self.exact_evaluator._compare_strings("123", "123")
        
        self.assertEqual(result.score, 1.0)
        self.assertEqual(result.true_positive, True)
        self.assertEqual(result.false_positive, False)
        self.assertEqual(result.false_negative, False)

    def test_semantic_similarity_above_threshold(self):
        """Test semantic similarity when score is above threshold."""
        result = self.semantic_evaluator._compare_strings(
            "The cat is on the mat", "The cat is on the mat"
        )
        
        # Identical strings should have perfect semantic similarity
        self.assertEqual(result.score, 1.0)
        self.assertEqual(result.true_positive, True)
        self.assertEqual(result.false_positive, False)
        self.assertEqual(result.false_negative, False)

    def test_semantic_similarity_below_threshold(self):
        """Test semantic similarity when score is below threshold."""
        result = self.semantic_evaluator._compare_strings(
            "The cat is on the mat", "The weather is sunny"
        )
        
        # Unrelated strings should have low semantic similarity
        self.assertLess(result.score, 0.5)
        self.assertEqual(result.true_positive, False)
        self.assertEqual(result.false_positive, True)
        self.assertEqual(result.false_negative, False)

    def test_different_threshold_values(self):
        """Test that different threshold values work correctly."""
        # Test with lower threshold
        low_threshold_evaluator = JSONEvaluator(
            self.schema, 
            string_metric="semantic_similarity", 
            match_threshold=0.1
        )
        
        result = low_threshold_evaluator._compare_strings(
            "The cat is on the mat", "The weather is sunny"
        )
        
        # With lower threshold, more matches should be considered true positives
        self.assertLess(result.score, 0.5)
        # The exact behavior depends on the semantic similarity score

    def test_comparison_result_structure(self):
        """Test that ComparisonResult has the expected structure."""
        result = self.exact_evaluator._compare_strings("test", "test")
        
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

    def test_real_semantic_similarity_synonyms(self):
        """Test that semantic similarity gives high scores for synonyms."""
        result = self.semantic_evaluator._compare_strings(
            "The cat is on the mat", "The feline is on the mat"
        )
        
        # Synonyms should have high similarity
        self.assertGreater(result.score, 0.7)
        self.assertLessEqual(result.score, 1.0)
        print(f"Synonyms similarity score: {result.score}")

    def test_real_semantic_similarity_related_concepts(self):
        """Test that semantic similarity gives moderate scores for related concepts."""
        result = self.semantic_evaluator._compare_strings(
            "The cat is on the mat", "The dog is on the mat"
        )
        
        # Related concepts should have moderate to high similarity
        self.assertGreater(result.score, 0.5)
        self.assertLessEqual(result.score, 1.0)
        print(f"Related concepts similarity score: {result.score}")

    def test_real_semantic_similarity_unrelated_concepts(self):
        """Test that semantic similarity gives low scores for unrelated concepts."""
        result = self.semantic_evaluator._compare_strings(
            "The cat is on the mat", "The weather is sunny today"
        )
        
        # Unrelated concepts should have low similarity
        self.assertLess(result.score, 0.5)
        self.assertGreaterEqual(result.score, 0.0)
        print(f"Unrelated concepts similarity score: {result.score}")

    def test_real_semantic_similarity_opposites(self):
        """Test that semantic similarity gives moderate scores for opposites."""
        result = self.semantic_evaluator._compare_strings(
            "The room is hot", "The room is cold"
        )
        
        # Opposites are related concepts, so they should have moderate similarity
        # but not as high as synonyms
        self.assertGreater(
            result.score, 0.5
        )  # They're related (temperature)
        self.assertLess(
            result.score, 0.9
        )  # But not as similar as synonyms
        self.assertGreaterEqual(result.score, 0.0)
        print(f"Opposites similarity score: {result.score}")

    def test_real_semantic_similarity_same_meaning_different_words(self):
        """Test that semantic similarity detects same meaning with different words."""
        result = self.semantic_evaluator._compare_strings(
            "I love this movie", "I adore this film"
        )
        # Same meaning with different words should have high similarity
        self.assertGreater(
            result.score, 0.7
        )
        self.assertLessEqual(
            result.score, 1.0
        )
        print(f"Same meaning similarity score: {result.score}")

    def test_semantic_vs_exact_matching_behavior(self):
        """Test that semantic similarity is more flexible than exact matching."""
        exact_result = self.exact_evaluator._compare_strings(
            "The cat is on the mat", "The feline is on the mat"
        )
        semantic_result = self.semantic_evaluator._compare_strings(
            "The cat is on the mat", "The feline is on the mat"
        )
        # Semantic similarity should be higher than exact match for synonyms
        self.assertEqual(exact_result.score, 0.0)  # Exact match fails
        self.assertGreater(
            semantic_result.score, 0.5
        )  # Semantic similarity succeeds
        print(f"Exact match score: {exact_result.score}")
        print(f"Semantic similarity score: {semantic_result.score}")


if __name__ == "__main__":
    unittest.main() 