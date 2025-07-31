"""Tests for the evaluation module."""

import unittest

from soda_mmqc.evaluation import JSONEvaluator


class TestEvaluate(unittest.TestCase):
    """Test cases for the evaluation functionality."""

    def generate_analysis(self, num_matches, num_missing, num_unexpected,
                         doc_id="TEST-DOC"):
        """
        Generate an analysis dictionary algorithmically.
        
        Args:
            num_matches: Number of correctly matched elements
            num_missing: Number of missing elements in model output
            num_unexpected: Number of unexpected elements in model output
            doc_id: Document ID for the test case
            
        Returns:
            dict: Analysis dictionary with expected_output and model_output
        """
        # Standard section names for testing
        section_names = [
            "Summary", "Introduction", "Results", "Discussion",
            "Materials and Methods", "Data availability", "Acknowledgements",
            "Author contributions", "Conflict of interest", "References",
            "Figure legends", "Supplementary figure legends"
        ]
        
        correct_names = [
            "Abstract", "Introduction", "Results", "Discussion",
            "Methods", "Data Availability", "Acknowledgements",
            "Author contributions", 
            "Disclosure and competing interests statement",
            "References", "Figure legends", "Expanded View legends"
        ]
        
        # Generate expected output (ground truth)
        expected_sections = []
        for i in range(num_matches + num_missing):
            expected_sections.append({
                "name": section_names[i],
                "correct_name": correct_names[i],
                "correct_position": "yes",
                "to_remove": "no"
            })
        
        expected_output = {
            "outputs": [{
                "sections_as_in_manuscript": expected_sections,
                "missing_sections": []
            }]
        }
        
        # Generate model output (prediction)
        model_sections = []
        
        # Add matched sections (first num_matches)
        for i in range(num_matches):
            model_sections.append({
                "name": section_names[i],
                "correct_name": correct_names[i],
                "correct_position": "yes",
                "to_remove": "no"
            })
        
        # Add unexpected sections (if any)
        for i in range(num_unexpected):
            unexpected_idx = num_matches + num_missing + i
            if unexpected_idx < len(section_names):
                model_sections.append({
                    "name": f"Unexpected_{section_names[unexpected_idx]}",
                    "correct_name": f"Unexpected_{correct_names[unexpected_idx]}",
                    "correct_position": "no",
                    "to_remove": "yes"
                })
        
        # Generate missing sections list
        missing_sections = []
        for i in range(num_missing):
            missing_idx = num_matches + i
            if missing_idx < len(correct_names):
                missing_sections.append({
                    "name": correct_names[missing_idx]
                })
        
        model_output = {
            "outputs": [{
                "sections_as_in_manuscript": model_sections,
                "missing_sections": missing_sections
            }]
        }
        
        return {
            "doc_id": doc_id,
            "expected_output": expected_output,
            "model_output": model_output
        }

    def test_json_evaluator_with_generated_data(self):
        """Test JSONEvaluator with algorithmically generated test data."""
        # Create a simple schema for testing
        schema = {
            "format": {
                "type": "json_schema",
                "name": "section-order-alt",
                "schema": {
                    "type": "object",
                    "properties": {
                        "outputs": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "sections_as_in_manuscript": {
                                        "type": "array",
                                        "items": {
                                            "type": "object",
                                            "properties": {
                                                "name": {"type": "string"},
                                                "correct_name": {"type": "string"},
                                                "correct_position": {"type": "string"},
                                                "to_remove": {"type": "string"}
                                            },
                                            "required": ["name", "correct_name", 
                                                        "correct_position", "to_remove"]
                                        }
                                    },
                                    "missing_sections": {
                                        "type": "array",
                                        "items": {
                                            "type": "object",
                                            "properties": {
                                                "name": {"type": "string"}
                                            },
                                            "required": ["name"]
                                        }
                                    }
                                },
                                "required": ["sections_as_in_manuscript", 
                                           "missing_sections"]
                            }
                        }
                    },
                    "required": ["outputs"]
                }
            }
        }
        
        # Generate test data with known characteristics
        analysis = self.generate_analysis(
            num_matches=3, 
            num_missing=1, 
            num_unexpected=1, 
            doc_id="TEST-GENERATED"
        )
        
        # Create evaluator and run evaluation
        evaluator = JSONEvaluator(schema, string_metric="perfect_match", 
                                 match_threshold=0.1)
        result = evaluator.evaluate(analysis["model_output"], 
                                  analysis["expected_output"])
        
        # Verify the evaluation results
        self.assertIn("element_scores", result)
        
        # When there are matches, we should have match_0_0 key
        self.assertIn("match_0_0", result["element_scores"])
        
        # Check that we have the expected number of matches
        # (3 matches should result in 3 matched elements)
        selection = (result["element_scores"]["match_0_0"]["field_scores"]
                    ["sections_as_in_manuscript"]["element_scores"])
        matches = [el for el in selection if el.startswith("match_")]
        self.assertEqual(len(matches), 3)
        
        # Check that we have the expected number of missing elements
        missing = [el for el in selection if el.startswith("missing_")]
        self.assertEqual(len(missing), 1)

    def test_perfect_match_scenario(self):
        """Test JSONEvaluator with perfect match scenario."""
        schema = {
            "format": {
                "type": "json_schema",
                "name": "section-order-alt",
                "schema": {
                    "type": "object",
                    "properties": {
                        "outputs": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "sections_as_in_manuscript": {
                                        "type": "array",
                                        "items": {
                                            "type": "object",
                                            "properties": {
                                                "name": {"type": "string"},
                                                "correct_name": {"type": "string"},
                                                "correct_position": {"type": "string"},
                                                "to_remove": {"type": "string"}
                                            },
                                            "required": ["name", "correct_name", 
                                                        "correct_position", "to_remove"]
                                        }
                                    },
                                    "missing_sections": {
                                        "type": "array",
                                        "items": {
                                            "type": "object",
                                            "properties": {
                                                "name": {"type": "string"}
                                            },
                                            "required": ["name"]
                                        }
                                    }
                                },
                                "required": ["sections_as_in_manuscript", 
                                           "missing_sections"]
                            }
                        }
                    },
                    "required": ["outputs"]
                }
            }
        }
        
        # Generate test data with perfect matches only
        analysis = self.generate_analysis(
            num_matches=4, 
            num_missing=0, 
            num_unexpected=0, 
            doc_id="PERFECT-MATCH"
        )
        
        # Create evaluator and run evaluation
        evaluator = JSONEvaluator(schema, string_metric="perfect_match", 
                                 match_threshold=0.1)
        result = evaluator.evaluate(analysis["model_output"], 
                                  analysis["expected_output"])
        
        # Verify the evaluation results
        self.assertIn("element_scores", result)
        self.assertIn("match_0_0", result["element_scores"])
        
        # Check that we have the expected number of matches (4 perfect matches)
        selection = (result["element_scores"]["match_0_0"]["field_scores"]
                    ["sections_as_in_manuscript"]["element_scores"])
        matches = [el for el in selection if el.startswith("match_")]
        self.assertEqual(len(matches), 4)
        
        # Check that we have no missing elements
        missing = [el for el in selection if el.startswith("missing_")]
        self.assertEqual(len(missing), 0)

    def test_all_missing_scenario(self):
        """Test JSONEvaluator with all sections missing scenario."""
        schema = {
            "format": {
                "type": "json_schema",
                "name": "section-order-alt",
                "schema": {
                    "type": "object",
                    "properties": {
                        "outputs": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "sections_as_in_manuscript": {
                                        "type": "array",
                                        "items": {
                                            "type": "object",
                                            "properties": {
                                                "name": {"type": "string"},
                                                "correct_name": {"type": "string"},
                                                "correct_position": {"type": "string"},
                                                "to_remove": {"type": "string"}
                                            },
                                            "required": ["name", "correct_name", 
                                                        "correct_position", "to_remove"]
                                        }
                                    },
                                    "missing_sections": {
                                        "type": "array",
                                        "items": {
                                            "type": "object",
                                            "properties": {
                                                "name": {"type": "string"}
                                            },
                                            "required": ["name"]
                                        }
                                    }
                                },
                                "required": ["sections_as_in_manuscript", 
                                           "missing_sections"]
                            }
                        }
                    },
                    "required": ["outputs"]
                }
            }
        }
        
        # Generate test data with all sections missing
        analysis = self.generate_analysis(
            num_matches=0, 
            num_missing=3, 
            num_unexpected=0, 
            doc_id="ALL-MISSING"
        )
        
        # Create evaluator and run evaluation
        evaluator = JSONEvaluator(schema, string_metric="perfect_match", 
                                 match_threshold=0.1)
        result = evaluator.evaluate(analysis["model_output"], 
                                  analysis["expected_output"])
        
        # Verify the evaluation results
        self.assertIn("element_scores", result)
        
        # When there are no matches, there won't be a match_0_0 key
        # Instead, we should have missing_element keys
        missing_keys = [key for key in result["element_scores"].keys() 
                       if key.startswith("missing_")]
        self.assertEqual(len(missing_keys), 1)  # It groups them into one missing_element
        
        # Check that we have no matches
        match_keys = [key for key in result["element_scores"].keys() 
                     if key.startswith("match_")]
        self.assertEqual(len(match_keys), 0)
        
        # Now let's look deeper into the structure to find the actual elements
        # The top level is just a summary, we need to go deeper
        if "missing_element_0" in result["element_scores"]:
            missing_element = result["element_scores"]["missing_element_0"]
            if "field_scores" in missing_element:
                field_scores = missing_element["field_scores"]
                if "sections_as_in_manuscript" in field_scores:
                    section_scores = field_scores["sections_as_in_manuscript"]
                    if "element_scores" in section_scores:
                        # This should contain the individual missing elements
                        individual_missing = [key for key in section_scores["element_scores"].keys() 
                                           if key.startswith("missing_")]
                        self.assertEqual(len(individual_missing), 3)

    def test_synthetic_unexpected_missing_section(self):
        """Test JSONEvaluator with synthetic data: model predicts missing section that shouldn't be there."""
        schema = {
            "format": {
                "type": "json_schema",
                "name": "section-order-alt",
                "schema": {
                    "type": "object",
                    "properties": {
                        "outputs": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "sections_as_in_manuscript": {
                                        "type": "array",
                                        "items": {
                                            "type": "object",
                                            "properties": {
                                                "name": {"type": "string"},
                                                "correct_name": {"type": "string"},
                                                "correct_position": {"type": "string"},
                                                "to_remove": {"type": "string"}
                                            },
                                            "required": ["name", "correct_name", 
                                                        "correct_position", "to_remove"]
                                        }
                                    },
                                    "missing_sections": {
                                        "type": "array",
                                        "items": {
                                            "type": "object",
                                            "properties": {
                                                "name": {"type": "string"}
                                            },
                                            "required": ["name"]
                                        }
                                    }
                                },
                                "required": ["sections_as_in_manuscript", 
                                           "missing_sections"]
                            }
                        }
                    },
                    "required": ["outputs"]
                }
            }
        }
        
        # Simplified synthetic data
        analysis = {
            "doc_id": "SYNTHETIC-TEST",
            "expected_output": {
                "outputs": [
                    {
                        "sections_as_in_manuscript": [
                            {
                                "name": "Introduction",
                                "correct_name": "Introduction",
                                "correct_position": "yes",
                                "to_remove": "no"
                            }
                        ],
                        "missing_sections": []  # Empty - no sections should be missing
                    }
                ]
            },
            "model_output": {
                "outputs": [
                    {
                        "sections_as_in_manuscript": [
                            {
                                "name": "Introduction",
                                "correct_name": "Introduction",
                                "correct_position": "yes",
                                "to_remove": "no"
                            }
                        ],
                        "missing_sections": [
                            {
                                "name": "Methods"  # Model incorrectly predicts this as missing
                            }
                        ]
                    }
                ]
            }
        }
        
        # Create evaluator and run evaluation
        evaluator = JSONEvaluator(schema, string_metric="perfect_match", 
                                 match_threshold=0.1)
        result = evaluator.evaluate(analysis["model_output"], 
                                  analysis["expected_output"])
        
        # Check for unexpected_predicted_element in the result
        # It's nested inside match_0_0.field_scores.missing_sections.element_scores
        match_0_0 = result["element_scores"]["match_0_0"]
        missing_sections_field = match_0_0["field_scores"]["missing_sections"]
        unexpected_keys = [key for key in missing_sections_field["element_scores"].keys() 
                          if key.startswith("unexpected_predicted_element")]
        self.assertGreater(len(unexpected_keys), 0, 
                          "Should have unexpected_predicted_element for incorrectly predicted missing section")
        
        # Also check that we have the expected matches
        match_keys = [key for key in result["element_scores"].keys() 
                     if key.startswith("match_")]
        self.assertGreater(len(match_keys), 0, "Should have match elements")

    def test_synthetic_field_level_scoring(self):
        """Test JSONEvaluator with synthetic data: model predicts wrong field value."""
        schema = {
            "format": {
                "type": "json_schema",
                "name": "section-order-alt",
                "schema": {
                    "type": "object",
                    "properties": {
                        "outputs": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "sections_as_in_manuscript": {
                                        "type": "array",
                                        "items": {
                                            "type": "object",
                                            "properties": {
                                                "name": {"type": "string"},
                                                "correct_name": {"type": "string"},
                                                "correct_position": {"type": "string"},
                                                "to_remove": {"type": "string"}
                                            },
                                            "required": ["name", "correct_name", 
                                                        "correct_position", "to_remove"]
                                        }
                                    },
                                    "missing_sections": {
                                        "type": "array",
                                        "items": {
                                            "type": "object",
                                            "properties": {
                                                "name": {"type": "string"}
                                            },
                                            "required": ["name"]
                                        }
                                    }
                                },
                                "required": ["sections_as_in_manuscript", 
                                           "missing_sections"]
                            }
                        }
                    },
                    "required": ["outputs"]
                }
            }
        }
        
        # Simplified synthetic data with field-level error
        analysis = {
            "doc_id": "FIELD-LEVEL-TEST",
            "expected_output": {
                "outputs": [
                    {
                        "sections_as_in_manuscript": [
                            {
                                "name": "Materials and Methods",
                                "correct_name": "Methods",
                                "correct_position": "yes",  # Expected: yes
                                "to_remove": "no"
                            }
                        ],
                        "missing_sections": []
                    }
                ]
            },
            "model_output": {
                "outputs": [
                    {
                        "sections_as_in_manuscript": [
                            {
                                "name": "Materials and Methods",
                                "correct_name": "Methods",
                                "correct_position": "no",   # Model predicts: no (WRONG!)
                                "to_remove": "no"
                            }
                        ],
                        "missing_sections": []
                    }
                ]
            }
        }
        
        # Create evaluator and run evaluation
        evaluator = JSONEvaluator(schema, string_metric="perfect_match", 
                                 match_threshold=0.1)
        result = evaluator.evaluate(analysis["model_output"], 
                                  analysis["expected_output"])
        
        # Verify the evaluation results
        self.assertIn("element_scores", result)
        self.assertIn("match_0_0", result["element_scores"])
        
        # Check that we have the expected match
        match_keys = [key for key in result["element_scores"].keys() 
                     if key.startswith("match_")]
        self.assertEqual(len(match_keys), 1, "Should have one match element")
        
        # Look at the field-level scores within the matched element
        match_0_0 = result["element_scores"]["match_0_0"]
        sections_field = match_0_0["field_scores"]["sections_as_in_manuscript"]
        
        # Check that we have one section element
        section_elements = [key for key in sections_field["element_scores"].keys() 
                          if key.startswith("match_")]
        self.assertEqual(len(section_elements), 1, "Should have one section element")
        
        # Get the section element and check its field scores
        section_element = sections_field["element_scores"]["match_0_0"]
        field_scores = section_element["field_scores"]
        
        # Check that correct_position field has a score of 0.0 (wrong value)
        self.assertIn("correct_position", field_scores)
        correct_position_score = field_scores["correct_position"]["score"]
        self.assertEqual(correct_position_score, 0.0, 
                        "correct_position should have score 0.0 for wrong value")
        
        # Check that other fields have score 1.0 (correct values)
        self.assertIn("name", field_scores)
        name_score = field_scores["name"]["score"]
        self.assertEqual(name_score, 1.0, "name should have score 1.0")
        
        self.assertIn("correct_name", field_scores)
        correct_name_score = field_scores["correct_name"]["score"]
        self.assertEqual(correct_name_score, 1.0, "correct_name should have score 1.0")
        
        self.assertIn("to_remove", field_scores)
        to_remove_score = field_scores["to_remove"]["score"]
        self.assertEqual(to_remove_score, 1.0, "to_remove should have score 1.0")
        
        # Check that the section element has a score of 3/4 (3 correct fields, 1 wrong)
        section_score = section_element["score"]
        self.assertEqual(section_score, 3/4, "Section should have score 3/4 (3 correct, 1 wrong)")
        
        # Check that the overall match has a score less than 1.0 due to the field error
        match_score = match_0_0["score"]
        self.assertLess(match_score, 1.0, "Overall match should have score < 1.0 due to field error")

    def test_real_life_materials_methods_position_error(self):
        """Test the specific case from real-life example: Materials and Methods has wrong correct_position."""
        schema = {
            "format": {
                "type": "json_schema",
                "name": "section-order-alt",
                "schema": {
                    "type": "object",
                    "properties": {
                        "outputs": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "sections_as_in_manuscript": {
                                        "type": "array",
                                        "items": {
                                            "type": "object",
                                            "properties": {
                                                "name": {"type": "string"},
                                                "correct_name": {"type": "string"},
                                                "correct_position": {"type": "string"},
                                                "to_remove": {"type": "string"}
                                            },
                                            "required": ["name", "correct_name", 
                                                        "correct_position", "to_remove"]
                                        }
                                    },
                                    "missing_sections": {
                                        "type": "array",
                                        "items": {
                                            "type": "object",
                                            "properties": {
                                                "name": {"type": "string"}
                                            },
                                            "required": ["name"]
                                        }
                                    }
                                },
                                "required": ["sections_as_in_manuscript", 
                                           "missing_sections"]
                            }
                        }
                    },
                    "required": ["outputs"]
                }
            }
        }
        
        # Extract just the Materials and Methods section from the real-life example
        analysis = {
            "doc_id": "REAL-LIFE-MATERIALS-METHODS",
            "expected_output": {
                "outputs": [
                    {
                        "sections_as_in_manuscript": [
                            {
                                "name": "Materials and Methods",
                                "correct_name": "Methods",
                                "correct_position": "yes",  # Expected: yes
                                "to_remove": "no"
                            }
                        ],
                        "missing_sections": []
                    }
                ]
            },
            "model_output": {
                "outputs": [
                    {
                        "sections_as_in_manuscript": [
                            {
                                "name": "Materials and Methods",
                                "correct_name": "Methods",
                                "correct_position": "no",   # Model predicts: no (WRONG!)
                                "to_remove": "no"
                            }
                        ],
                        "missing_sections": []
                    }
                ]
            }
        }
        
        # Create evaluator and run evaluation
        evaluator = JSONEvaluator(schema, string_metric="perfect_match", 
                                 match_threshold=0.1)
        result = evaluator.evaluate(analysis["model_output"], 
                                  analysis["expected_output"])
        
        # Verify the evaluation results
        self.assertIn("element_scores", result)
        self.assertIn("match_0_0", result["element_scores"])
        
        # Check that we have the expected match
        match_keys = [key for key in result["element_scores"].keys() 
                     if key.startswith("match_")]
        self.assertEqual(len(match_keys), 1, "Should have one match element")
        
        # Look at the field-level scores within the matched element
        match_0_0 = result["element_scores"]["match_0_0"]
        sections_field = match_0_0["field_scores"]["sections_as_in_manuscript"]
        
        # Check that we have one section element
        section_elements = [key for key in sections_field["element_scores"].keys() 
                          if key.startswith("match_")]
        self.assertEqual(len(section_elements), 1, "Should have one section element")
        
        # Get the section element and check its field scores
        section_element = sections_field["element_scores"]["match_0_0"]
        field_scores = section_element["field_scores"]
        
        # Check that correct_position field has a score of 0.0 (wrong value)
        self.assertIn("correct_position", field_scores)
        correct_position_score = field_scores["correct_position"]["score"]
        self.assertEqual(correct_position_score, 0.0, 
                        "correct_position should have score 0.0 for wrong value")
        
        # Check that other fields have score 1.0 (correct values)
        self.assertIn("name", field_scores)
        name_score = field_scores["name"]["score"]
        self.assertEqual(name_score, 1.0, "name should have score 1.0")
        
        self.assertIn("correct_name", field_scores)
        correct_name_score = field_scores["correct_name"]["score"]
        self.assertEqual(correct_name_score, 1.0, "correct_name should have score 1.0")
        
        self.assertIn("to_remove", field_scores)
        to_remove_score = field_scores["to_remove"]["score"]
        self.assertEqual(to_remove_score, 1.0, "to_remove should have score 1.0")
        
        # Check that the section element has a score of 3/4 (3 correct fields, 1 wrong)
        section_score = section_element["score"]
        self.assertEqual(section_score, 3/4, "Section should have score 3/4 (3 correct, 1 wrong)")
        
        # Check that the overall match has a score less than 1.0 due to the field error
        match_score = match_0_0["score"]
        self.assertLess(match_score, 1.0, "Overall match should have score < 1.0 due to field error")

    def test_real_life_comprehensive(self):
        """Test the complete real-life example with both unexpected missing section and field-level error."""
        schema = {
            "format": {
                "type": "json_schema",
                "name": "section-order-alt",
                "schema": {
                    "type": "object",
                    "properties": {
                        "outputs": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "sections_as_in_manuscript": {
                                        "type": "array",
                                        "items": {
                                            "type": "object",
                                            "properties": {
                                                "name": {"type": "string"},
                                                "correct_name": {"type": "string"},
                                                "correct_position": {"type": "string"},
                                                "to_remove": {"type": "string"}
                                            },
                                            "required": ["name", "correct_name", 
                                                        "correct_position", "to_remove"]
                                        }
                                    },
                                    "missing_sections": {
                                        "type": "array",
                                        "items": {
                                            "type": "object",
                                            "properties": {
                                                "name": {"type": "string"}
                                            },
                                            "required": ["name"]
                                        }
                                    }
                                },
                                "required": ["sections_as_in_manuscript", 
                                           "missing_sections"]
                            }
                        }
                    },
                    "required": ["outputs"]
                }
            }
        }
        
        # Simplified version of the real-life example with both issues
        analysis = {
            "doc_id": "REAL-LIFE-COMPREHENSIVE",
            "expected_output": {
                "outputs": [
                    {
                        "sections_as_in_manuscript": [
                            {
                                "name": "Materials and Methods",
                                "correct_name": "Methods",
                                "correct_position": "yes",  # Expected: yes
                                "to_remove": "no"
                            }
                        ],
                        "missing_sections": []  # Empty - no sections should be missing
                    }
                ]
            },
            "model_output": {
                "outputs": [
                    {
                        "sections_as_in_manuscript": [
                            {
                                "name": "Materials and Methods",
                                "correct_name": "Methods",
                                "correct_position": "no",   # Model predicts: no (WRONG!)
                                "to_remove": "no"
                            }
                        ],
                        "missing_sections": [
                            {
                                "name": "Methods"  # Model incorrectly predicts this as missing
                            }
                        ]
                    }
                ]
            }
        }
        
        # Create evaluator and run evaluation
        evaluator = JSONEvaluator(schema, string_metric="perfect_match", 
                                 match_threshold=0.1)
        result = evaluator.evaluate(analysis["model_output"], 
                                  analysis["expected_output"])
        
        # Verify the evaluation results
        self.assertIn("element_scores", result)
        self.assertIn("match_0_0", result["element_scores"])
        
        # Check for unexpected_predicted_element in the missing_sections field
        match_0_0 = result["element_scores"]["match_0_0"]
        missing_sections_field = match_0_0["field_scores"]["missing_sections"]
        unexpected_keys = [key for key in missing_sections_field["element_scores"].keys() 
                          if key.startswith("unexpected_predicted_element")]
        self.assertGreater(len(unexpected_keys), 0, 
                          "Should have unexpected_predicted_element for incorrectly predicted missing section")
        
        # Check that we have the expected match
        match_keys = [key for key in result["element_scores"].keys() 
                     if key.startswith("match_")]
        self.assertEqual(len(match_keys), 1, "Should have one match element")
        
        # Look at the field-level scores within the matched element
        sections_field = match_0_0["field_scores"]["sections_as_in_manuscript"]
        
        # Check that we have one section element
        section_elements = [key for key in sections_field["element_scores"].keys() 
                          if key.startswith("match_")]
        self.assertEqual(len(section_elements), 1, "Should have one section element")
        
        # Get the section element and check its field scores
        section_element = sections_field["element_scores"]["match_0_0"]
        field_scores = section_element["field_scores"]
        
        # Check that correct_position field has a score of 0.0 (wrong value)
        self.assertIn("correct_position", field_scores)
        correct_position_score = field_scores["correct_position"]["score"]
        self.assertEqual(correct_position_score, 0.0, 
                        "correct_position should have score 0.0 for wrong value")
        
        # Check that other fields have score 1.0 (correct values)
        self.assertIn("name", field_scores)
        name_score = field_scores["name"]["score"]
        self.assertEqual(name_score, 1.0, "name should have score 1.0")
        
        self.assertIn("correct_name", field_scores)
        correct_name_score = field_scores["correct_name"]["score"]
        self.assertEqual(correct_name_score, 1.0, "correct_name should have score 1.0")
        
        self.assertIn("to_remove", field_scores)
        to_remove_score = field_scores["to_remove"]["score"]
        self.assertEqual(to_remove_score, 1.0, "to_remove should have score 1.0")
        
        # Check that the section element has a score of 3/4 (3 correct fields, 1 wrong)
        section_score = section_element["score"]
        self.assertEqual(section_score, 3/4, "Section should have score 3/4 (3 correct, 1 wrong)")
        
        # Check that the overall match has a score less than 1.0 due to both errors
        match_score = match_0_0["score"]
        self.assertLess(match_score, 1.0, "Overall match should have score < 1.0 due to field error and unexpected missing section")

    def test_synthetic_top_level_scoring(self):
        """Test JSONEvaluator with synthetic data: verify top-level score aggregation."""
        schema = {
            "format": {
                "type": "json_schema",
                "name": "section-order-alt",
                "schema": {
                    "type": "object",
                    "properties": {
                        "outputs": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "sections_as_in_manuscript": {
                                        "type": "array",
                                        "items": {
                                            "type": "object",
                                            "properties": {
                                                "name": {"type": "string"},
                                                "correct_name": {"type": "string"},
                                                "correct_position": {"type": "string"},
                                                "to_remove": {"type": "string"}
                                            },
                                            "required": ["name", "correct_name", 
                                                        "correct_position", "to_remove"]
                                        }
                                    },
                                    "missing_sections": {
                                        "type": "array",
                                        "items": {
                                            "type": "object",
                                            "properties": {
                                                "name": {"type": "string"}
                                            },
                                            "required": ["name"]
                                        }
                                    }
                                },
                                "required": ["sections_as_in_manuscript", 
                                           "missing_sections"]
                            }
                        }
                    },
                    "required": ["outputs"]
                }
            }
        }
        
        # Synthetic data with one section that has a field-level error
        analysis = {
            "doc_id": "TOP-LEVEL-SCORING-TEST",
            "expected_output": {
                "outputs": [
                    {
                        "sections_as_in_manuscript": [
                            {
                                "name": "Introduction",
                                "correct_name": "Introduction",
                                "correct_position": "yes",  # Expected: yes
                                "to_remove": "no"
                            }
                        ],
                        "missing_sections": []
                    }
                ]
            },
            "model_output": {
                "outputs": [
                    {
                        "sections_as_in_manuscript": [
                            {
                                "name": "Introduction",
                                "correct_name": "Introduction",
                                "correct_position": "no",   # Model predicts: no (WRONG!)
                                "to_remove": "no"
                            }
                        ],
                        "missing_sections": []
                    }
                ]
            }
        }
        
        # Create evaluator and run evaluation
        evaluator = JSONEvaluator(schema, string_metric="perfect_match", 
                                 match_threshold=0.1)
        result = evaluator.evaluate(analysis["model_output"], 
                                  analysis["expected_output"])
        
        # Verify the evaluation results
        self.assertIn("element_scores", result)
        self.assertIn("match_0_0", result["element_scores"])
        
        # Check that we have the expected match
        match_keys = [key for key in result["element_scores"].keys() 
                     if key.startswith("match_")]
        self.assertEqual(len(match_keys), 1, "Should have one match element")
        
        # Get the top-level match element
        match_0_0 = result["element_scores"]["match_0_0"]
        
        # Check the field-level scores for the two main fields
        field_scores = match_0_0["field_scores"]
        
        # Check sections_as_in_manuscript field score
        self.assertIn("sections_as_in_manuscript", field_scores)
        sections_score = field_scores["sections_as_in_manuscript"]["score"]
        # Should be 3/4 because one section has 3 correct fields and 1 wrong field
        self.assertEqual(sections_score, 3/4, 
                        "sections_as_in_manuscript should have score 3/4 (3 correct, 1 wrong)")
        
        # Check missing_sections field score
        self.assertIn("missing_sections", field_scores)
        missing_sections_score = field_scores["missing_sections"]["score"]
        # Should be 1.0 because both expected and predicted are empty
        self.assertEqual(missing_sections_score, 1.0, 
                        "missing_sections should have score 1.0 (both empty)")
        
        # Check the overall top-level score
        # It should be the average of the two field scores: (3/4 + 1.0) / 2 = 7/8
        top_level_score = match_0_0["score"]
        expected_score = (3/4 + 1.0) / 2  # 7/8 = 0.875
        self.assertEqual(top_level_score, expected_score, 
                        f"Top-level score should be {expected_score} (average of field scores)")
        
        # Verify the detailed structure for sections_as_in_manuscript
        sections_field = field_scores["sections_as_in_manuscript"]
        self.assertIn("element_scores", sections_field)
        
        # Check that we have one section element
        section_elements = [key for key in sections_field["element_scores"].keys() 
                          if key.startswith("match_")]
        self.assertEqual(len(section_elements), 1, "Should have one section element")
        
        # Get the section element and verify its score
        section_element = sections_field["element_scores"]["match_0_0"]
        section_score = section_element["score"]
        self.assertEqual(section_score, 3/4, "Section should have score 3/4 (3 correct, 1 wrong)")
        
        # Verify the field-level scores within the section
        section_field_scores = section_element["field_scores"]
        
        # Check that correct_position field has a score of 0.0 (wrong value)
        self.assertIn("correct_position", section_field_scores)
        correct_position_score = section_field_scores["correct_position"]["score"]
        self.assertEqual(correct_position_score, 0.0, 
                        "correct_position should have score 0.0 for wrong value")
        
        # Check that other fields have score 1.0 (correct values)
        for field_name in ["name", "correct_name", "to_remove"]:
            self.assertIn(field_name, section_field_scores)
            field_score = section_field_scores[field_name]["score"]
            self.assertEqual(field_score, 1.0, f"{field_name} should have score 1.0")

    def test_real_life_top_level_score(self):
        """Test the top-level score of the real-life example with field-level errors."""
        schema = {
            "format": {
                "type": "json_schema",
                "name": "section-order-alt",
                "schema": {
                    "type": "object",
                    "properties": {
                        "outputs": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "sections_as_in_manuscript": {
                                        "type": "array",
                                        "items": {
                                            "type": "object",
                                            "properties": {
                                                "name": {"type": "string"},
                                                "correct_name": {"type": "string"},
                                                "correct_position": {"type": "string"},
                                                "to_remove": {"type": "string"}
                                            },
                                            "required": ["name", "correct_name", 
                                                        "correct_position", "to_remove"]
                                        }
                                    },
                                    "missing_sections": {
                                        "type": "array",
                                        "items": {
                                            "type": "object",
                                            "properties": {
                                                "name": {"type": "string"}
                                            },
                                            "required": ["name"]
                                        }
                                    }
                                },
                                "required": ["sections_as_in_manuscript", 
                                           "missing_sections"]
                            }
                        }
                    },
                    "required": ["outputs"]
                }
            }
        }
        
        # Use the real-life example data
        analysis = {
            "doc_id": "EMBOJ-2024-119734R",
            "expected_output": {
                "outputs": [
                    {
                        "sections_as_in_manuscript": [
                            {
                                "name": "Summary",
                                "correct_name": "Abstract",
                                "correct_position": "yes",
                                "to_remove": "no"
                            },
                            {
                                "name": "Introduction",
                                "correct_name": "Introduction",
                                "correct_position": "yes",
                                "to_remove": "no"
                            },
                            {
                                "name": "Results",
                                "correct_name": "Results",
                                "correct_position": "yes",
                                "to_remove": "no"
                            },
                            {
                                "name": "Discussion",
                                "correct_name": "Discussion",
                                "correct_position": "yes",
                                "to_remove": "no"
                            },
                            {
                                "name": "Materials and Methods",
                                "correct_name": "Methods",
                                "correct_position": "yes",  # Expected: yes
                                "to_remove": "no"
                            },
                            {
                                "name": "Data availability",
                                "correct_name": "Data Availability",
                                "correct_position": "yes",
                                "to_remove": "no"
                            },
                            {
                                "name": "Acknowledgements",
                                "correct_name": "Acknowledgements",
                                "correct_position": "yes",
                                "to_remove": "no"
                            },
                            {
                                "name": "Author contributions",
                                "correct_name": "Author contributions",
                                "correct_position": "yes",
                                "to_remove": "no"
                            },
                            {
                                "name": "Conflict of interest",
                                "correct_name": "Disclosure and competing interests statement",
                                "correct_position": "yes",
                                "to_remove": "no"
                            },
                            {
                                "name": "References",
                                "correct_name": "References",
                                "correct_position": "yes",
                                "to_remove": "no"
                            },
                            {
                                "name": "Figure legends",
                                "correct_name": "Figure legends",
                                "correct_position": "yes",
                                "to_remove": "no"
                            },
                            {
                                "name": "Supplementary figure legends",
                                "correct_name": "Expanded View legends",
                                "correct_position": "yes",
                                "to_remove": "no"
                            }
                        ],
                        "missing_sections": []
                    }
                ]
            },
            "model_output": {
                "outputs": [
                    {
                        "sections_as_in_manuscript": [
                            {
                                "name": "Summary",
                                "correct_name": "Abstract",
                                "correct_position": "yes",
                                "to_remove": "no"
                            },
                            {
                                "name": "Introduction",
                                "correct_name": "Introduction",
                                "correct_position": "yes",
                                "to_remove": "no"
                            },
                            {
                                "name": "Results",
                                "correct_name": "Results",
                                "correct_position": "yes",
                                "to_remove": "no"
                            },
                            {
                                "name": "Discussion",
                                "correct_name": "Discussion",
                                "correct_position": "yes",
                                "to_remove": "no"
                            },
                            {
                                "name": "Materials and Methods",
                                "correct_name": "Methods",
                                "correct_position": "no",   # Model predicts: no (WRONG!)
                                "to_remove": "no"
                            },
                            {
                                "name": "Data availability",
                                "correct_name": "Data Availability",
                                "correct_position": "yes",
                                "to_remove": "no"
                            },
                            {
                                "name": "Acknowledgements",
                                "correct_name": "Acknowledgments",  # Typo in model output
                                "correct_position": "yes",
                                "to_remove": "no"
                            },
                            {
                                "name": "Author contributions",
                                "correct_name": "Author contributions",
                                "correct_position": "yes",
                                "to_remove": "no"
                            },
                            {
                                "name": "Conflict of interest",
                                "correct_name": "Disclosure and competing interests statement",
                                "correct_position": "yes",
                                "to_remove": "no"
                            },
                            {
                                "name": "References",
                                "correct_name": "References",
                                "correct_position": "yes",
                                "to_remove": "no"
                            },
                            {
                                "name": "Figure legends",
                                "correct_name": "Figure legends",
                                "correct_position": "yes",
                                "to_remove": "no"
                            }
                        ],
                        "missing_sections": [
                            {
                                "name": "Expanded View legends"  # Model incorrectly predicts this as missing
                            }
                        ]
                    }
                ]
            }
        }
        
        # Create evaluator and run evaluation
        evaluator = JSONEvaluator(schema, string_metric="perfect_match", 
                                 match_threshold=0.1)
        result = evaluator.evaluate(analysis["model_output"], 
                                  analysis["expected_output"])
        
        # Verify the evaluation results
        self.assertIn("element_scores", result)
        self.assertIn("match_0_0", result["element_scores"])
        
        # Get the top-level match element
        match_0_0 = result["element_scores"]["match_0_0"]
        
        # Check the field-level scores for the two main fields
        field_scores = match_0_0["field_scores"]
        
        # Check sections_as_in_manuscript field score
        self.assertIn("sections_as_in_manuscript", field_scores)
        sections_score = field_scores["sections_as_in_manuscript"]["score"]
        # Should be less than 1.0 due to field-level errors in some sections
        self.assertLess(sections_score, 1.0, 
                        "sections_as_in_manuscript should have score < 1.0 due to field errors")
        
        # Check missing_sections field score
        self.assertIn("missing_sections", field_scores)
        missing_sections_score = field_scores["missing_sections"]["score"]
        # Should be less than 1.0 due to unexpected predicted element
        self.assertLess(missing_sections_score, 1.0, 
                        "missing_sections should have score < 1.0 due to unexpected element")
        
        # Check the overall top-level score
        # It should be less than 1.0 due to both field errors
        top_level_score = match_0_0["score"]
        self.assertLess(top_level_score, 1.0, 
                        "Top-level score should be < 1.0 due to field errors in both sections")
        
        # Verify that the score is reasonable (not too low)
        self.assertGreater(top_level_score, 0.4, 
                          "Top-level score should be > 0.4 (most fields are correct)")

    def test_emm_2025_21341_bug_reproduction(self):
        """Test reproduction of the EMM-2025-21341 bug in JSONEvaluator."""
        schema = {
            "format": {
                "type": "json_schema",
                "name": "section-order-alt",
                "schema": {
                    "type": "object",
                    "properties": {
                        "outputs": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "sections_as_in_manuscript": {
                                        "type": "array",
                                        "items": {
                                            "type": "object",
                                            "properties": {
                                                "name": {"type": "string"},
                                                "correct_name": {"type": "string"},
                                                "correct_position": {"type": "string"},
                                                "to_remove": {"type": "string"}
                                            },
                                            "required": ["name", "correct_name", 
                                                        "correct_position", "to_remove"]
                                        }
                                    },
                                    "missing_sections": {
                                        "type": "array",
                                        "items": {
                                            "type": "object",
                                            "properties": {
                                                "name": {"type": "string"}
                                            },
                                            "required": ["name"]
                                        }
                                    }
                                },
                                "required": ["sections_as_in_manuscript", 
                                           "missing_sections"]
                            }
                        }
                    },
                    "required": ["outputs"]
                }
            }
        }
        
        # Reproduce the exact EMM-2025-21341 scenario
        analysis = {
            "doc_id": "EMM-2025-21341",
            "expected_output": {
                "outputs": [
                    {
                        "sections_as_in_manuscript": [
                            {
                                "name": "Abstract",
                                "correct_name": "Abstract",
                                "correct_position": "yes",
                                "to_remove": "no"
                            },
                            {
                                "name": "Introduction",
                                "correct_name": "Introduction",
                                "correct_position": "yes",
                                "to_remove": "no"
                            },
                            {
                                "name": "Results",
                                "correct_name": "Results",
                                "correct_position": "yes",
                                "to_remove": "no"
                            },
                            {
                                "name": "Discussion",
                                "correct_name": "Discussion",
                                "correct_position": "yes",
                                "to_remove": "no"
                            },
                            {
                                "name": "Materials and methods",
                                "correct_name": "Methods",
                                "correct_position": "yes",  # Expected: yes
                                "to_remove": "no"
                            },
                            {
                                "name": "Acknowledgments",
                                "correct_name": "Acknowledgments",
                                "correct_position": "yes",
                                "to_remove": "no"
                            },
                            {
                                "name": "Author contributions",
                                "correct_name": "Author contributions",
                                "correct_position": "yes",
                                "to_remove": "no"
                            },
                            {
                                "name": "Disclosure and competing interests statement",
                                "correct_name": "Disclosure and competing interests statement",
                                "correct_position": "yes",
                                "to_remove": "no"
                            },
                            {
                                "name": "Data Availability",
                                "correct_name": "Data Availability",
                                "correct_position": "yes",
                                "to_remove": "no"
                            },
                            {
                                "name": "The paper explained",
                                "correct_name": "",
                                "correct_position": "no",
                                "to_remove": "yes"
                            },
                            {
                                "name": "References",
                                "correct_name": "References",
                                "correct_position": "yes",
                                "to_remove": "no"
                            },
                            {
                                "name": "Figure legends",
                                "correct_name": "Figure legends",
                                "correct_position": "yes",
                                "to_remove": "no"
                            },
                            {
                                "name": "Expanded View legends",
                                "correct_name": "Expanded View legends",
                                "correct_position": "yes",
                                "to_remove": "no"
                            }
                        ],
                        "missing_sections": []
                    }
                ]
            },
            "model_output": {
                "outputs": [
                    {
                        "sections_as_in_manuscript": [
                            {
                                "name": "Abstract",
                                "correct_name": "Abstract",
                                "correct_position": "yes",
                                "to_remove": "no"
                            },
                            {
                                "name": "Introduction",
                                "correct_name": "Introduction",
                                "correct_position": "yes",
                                "to_remove": "no"
                            },
                            {
                                "name": "Results",
                                "correct_name": "Results",
                                "correct_position": "yes",
                                "to_remove": "no"
                            },
                            {
                                "name": "Discussion",
                                "correct_name": "Discussion",
                                "correct_position": "yes",
                                "to_remove": "no"
                            },
                            {
                                "name": "Materials and methods",
                                "correct_name": "Methods",
                                "correct_position": "no",   # Model predicts: no (WRONG!)
                                "to_remove": "no"
                            },
                            {
                                "name": "Acknowledgments",
                                "correct_name": "Acknowledgements",  # Typo in model output
                                "correct_position": "yes",
                                "to_remove": "no"
                            },
                            {
                                "name": "Author contributions",
                                "correct_name": "Author contributions",
                                "correct_position": "yes",
                                "to_remove": "no"
                            },
                            {
                                "name": "Disclosure and competing interests statement",
                                "correct_name": "Disclosure and competing interests statement",
                                "correct_position": "yes",
                                "to_remove": "no"
                            },
                            {
                                "name": "Data Availability",
                                "correct_name": "Data Availability",
                                "correct_position": "yes",
                                "to_remove": "no"
                            },
                            {
                                "name": "The paper explained",
                                "correct_name": "",
                                "correct_position": "no",
                                "to_remove": "yes"
                            },
                            {
                                "name": "References",
                                "correct_name": "References",
                                "correct_position": "yes",
                                "to_remove": "no"
                            },
                            {
                                "name": "Figure legends",
                                "correct_name": "Figure legends",
                                "correct_position": "yes",
                                "to_remove": "no"
                            },
                            {
                                "name": "Expanded View legends",
                                "correct_name": "Expanded View legends",
                                "correct_position": "yes",
                                "to_remove": "no"
                            }
                        ],
                        "missing_sections": [
                            {
                                "name": "Data Availability"  # Model incorrectly predicts this as missing
                            }
                        ]
                    }
                ]
            }
        }
        
        # Create evaluator and run evaluation
        evaluator = JSONEvaluator(schema, string_metric="perfect_match", 
                                 match_threshold=0.1)
        result = evaluator.evaluate(analysis["model_output"], 
                                  analysis["expected_output"])
        
        # Debug: Print the result structure to understand what's happening
        print("DEBUG: Result structure:")
        print(f"Top-level score: {result.get('score', 'N/A')}")
        print(f"Element scores keys: {list(result.get('element_scores', {}).keys())}")
        
        # The bug: This should have match_0_0, not unexpected_element_0 and missing_element_0
        # Most sections should match, with only field-level errors
        self.assertIn("element_scores", result)
        
        # Check if we have the expected match structure
        element_scores = result["element_scores"]
        match_keys = [key for key in element_scores.keys() 
                     if key.startswith("match_")]
        
        # This should have matches since most sections are correct
        self.assertGreater(len(match_keys), 0, 
                          "Should have match elements since most sections are correct")
        
        # The score should be greater than 0 since most fields are correct
        top_level_score = result.get("score", 0)
        self.assertGreater(top_level_score, 0, 
                          "Score should be > 0 since most fields are correct")

    def test_emm_2025_21341_exact_data_reproduction(self):
        """Test reproduction of the exact EMM-2025-21341 data differences."""
        schema = {
            "format": {
                "type": "json_schema",
                "name": "section-order-alt",
                "schema": {
                    "type": "object",
                    "properties": {
                        "outputs": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "sections_as_in_manuscript": {
                                        "type": "array",
                                        "items": {
                                            "type": "object",
                                            "properties": {
                                                "name": {"type": "string"},
                                                "correct_name": {"type": "string"},
                                                "correct_position": {"type": "string"},
                                                "to_remove": {"type": "string"}
                                            },
                                            "required": ["name", "correct_name", 
                                                        "correct_position", "to_remove"]
                                        }
                                    },
                                    "missing_sections": {
                                        "type": "array",
                                        "items": {
                                            "type": "object",
                                            "properties": {
                                                "name": {"type": "string"}
                                            },
                                            "required": ["name"]
                                        }
                                    }
                                },
                                "required": ["sections_as_in_manuscript", 
                                           "missing_sections"]
                            }
                        }
                    },
                    "required": ["outputs"]
                }
            }
        }
        
        # Reproduce the EXACT data from the real case
        analysis = {
            "doc_id": "EMM-2025-21341",
            "expected_output": {
                "outputs": [
                    {
                        "sections_as_in_manuscript": [
                            {
                                "name": "Abstract",
                                "correct_name": "Abstract",
                                "correct_position": "yes",
                                "to_remove": "no"
                            },
                            {
                                "name": "Introduction",
                                "correct_name": "Introduction",
                                "correct_position": "yes",
                                "to_remove": "no"
                            },
                            {
                                "name": "Results",
                                "correct_name": "Results",
                                "correct_position": "yes",
                                "to_remove": "no"
                            },
                            {
                                "name": "Discussion",
                                "correct_name": "Discussion",
                                "correct_position": "yes",
                                "to_remove": "no"
                            },
                            {
                                "name": "Materials and methods",
                                "correct_name": "Methods",
                                "correct_position": "yes",
                                "to_remove": "no"
                            },
                            {
                                "name": "Acknowledgments",  # Note: NO 'e'
                                "correct_name": "Acknowledgments",
                                "correct_position": "yes",
                                "to_remove": "no"
                            },
                            {
                                "name": "Author contributions",
                                "correct_name": "Author contributions",
                                "correct_position": "yes",
                                "to_remove": "no"
                            },
                            {
                                "name": "Disclosure and competing interests statement",
                                "correct_name": "Disclosure and competing interests statement",
                                "correct_position": "yes",
                                "to_remove": "no"
                            },
                            {
                                "name": "Data Availability",
                                "correct_name": "Data Availability",
                                "correct_position": "yes",
                                "to_remove": "no"
                            },
                            {
                                "name": "The paper explained",
                                "correct_name": "",
                                "correct_position": "no",
                                "to_remove": "yes"
                            },
                            {
                                "name": "References",
                                "correct_name": "References",
                                "correct_position": "yes",
                                "to_remove": "no"
                            },
                            {
                                "name": "Figure legends",
                                "correct_name": "Figure legends",
                                "correct_position": "yes",
                                "to_remove": "no"
                            },
                            {
                                "name": "Expanded View legends",
                                "correct_name": "Expanded View legends",
                                "correct_position": "yes",
                                "to_remove": "no"
                            }
                        ],
                        "missing_sections": []
                    }
                ]
            },
            "model_output": {
                "outputs": [
                    {
                        "sections_as_in_manuscript": [
                            {
                                "name": "Abstract",
                                "correct_name": "Abstract",
                                "correct_position": "yes",
                                "to_remove": "no"
                            },
                            {
                                "name": "Introduction",
                                "correct_name": "Introduction",
                                "correct_position": "yes",
                                "to_remove": "no"
                            },
                            {
                                "name": "Results",
                                "correct_name": "Results",
                                "correct_position": "yes",
                                "to_remove": "no"
                            },
                            {
                                "name": "Discussion",
                                "correct_name": "Discussion",
                                "correct_position": "yes",
                                "to_remove": "no"
                            },
                            {
                                "name": "Materials and methods",
                                "correct_name": "Methods",
                                "correct_position": "no",  # Field error
                                "to_remove": "no"
                            },
                            {
                                "name": "Acknowledgments",
                                "correct_name": "Acknowledgements",  # Note: WITH 'e'
                                "correct_position": "yes",
                                "to_remove": "no"
                            },
                            {
                                "name": "Author contributions",
                                "correct_name": "Author contributions",
                                "correct_position": "yes",
                                "to_remove": "no"
                            },
                            {
                                "name": "Disclosure and competing interests statement",
                                "correct_name": "Disclosure and competing interests statement",
                                "correct_position": "yes",
                                "to_remove": "no"
                            },
                            {
                                "name": "Data Availability",
                                "correct_name": "Data Availability",
                                "correct_position": "yes",
                                "to_remove": "no"
                            },
                            {
                                "name": "The paper explained",
                                "correct_name": "",
                                "correct_position": "no",
                                "to_remove": "yes"
                            },
                            {
                                "name": "References",
                                "correct_name": "References",
                                "correct_position": "yes",
                                "to_remove": "no"
                            },
                            {
                                "name": "Figure legends",
                                "correct_name": "Figure legends",
                                "correct_position": "yes",
                                "to_remove": "no"
                            },
                            {
                                "name": "Expanded View legends",
                                "correct_name": "Expanded View legends",
                                "correct_position": "yes",
                                "to_remove": "no"
                            }
                        ],
                        "missing_sections": [
                            {
                                "name": "Data Availability"
                            }
                        ]
                    }
                ]
            }
        }
        
        # Create evaluator and run evaluation
        evaluator = JSONEvaluator(schema, string_metric="perfect_match", 
                                 match_threshold=0.1)
        result = evaluator.evaluate(analysis["model_output"], 
                                  analysis["expected_output"])
        
        # Debug: Print the result structure
        print("DEBUG: Exact data reproduction result:")
        print(f"Top-level score: {result.get('score', 'N/A')}")
        print(f"Element scores keys: {list(result.get('element_scores', {}).keys())}")
        
        # Check what we actually get
        self.assertIn("element_scores", result)
        element_scores = result["element_scores"]
        
        # Print more details about the structure
        for key, value in element_scores.items():
            print(f"  {key}: score={value.get('score', 'N/A')}")
            if 'field_scores' in value:
                for field_key, field_value in value.get('field_scores', {}).items():
                    print(f"    {field_key}: score={field_value.get('score', 'N/A')}")

    def test_emm_2025_21341_with_semantic_similarity(self):
        """Test EMM-2025-21341 with semantic_similarity metric like the actual pipeline."""
        schema = {
            "format": {
                "type": "json_schema",
                "name": "section-order-alt",
                "schema": {
                    "type": "object",
                    "properties": {
                        "outputs": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "sections_as_in_manuscript": {
                                        "type": "array",
                                        "items": {
                                            "type": "object",
                                            "properties": {
                                                "name": {"type": "string"},
                                                "correct_name": {"type": "string"},
                                                "correct_position": {"type": "string"},
                                                "to_remove": {"type": "string"}
                                            },
                                            "required": ["name", "correct_name", 
                                                        "correct_position", "to_remove"]
                                        }
                                    },
                                    "missing_sections": {
                                        "type": "array",
                                        "items": {
                                            "type": "object",
                                            "properties": {
                                                "name": {"type": "string"}
                                            },
                                            "required": ["name"]
                                        }
                                    }
                                },
                                "required": ["sections_as_in_manuscript", 
                                           "missing_sections"]
                            }
                        }
                    },
                    "required": ["outputs"]
                }
            }
        }
        
        # Use the exact data from the real case
        analysis = {
            "doc_id": "EMM-2025-21341",
            "expected_output": {
                "outputs": [
                    {
                        "sections_as_in_manuscript": [
                            {
                                "name": "Abstract",
                                "correct_name": "Abstract",
                                "correct_position": "yes",
                                "to_remove": "no"
                            },
                            {
                                "name": "Introduction",
                                "correct_name": "Introduction",
                                "correct_position": "yes",
                                "to_remove": "no"
                            },
                            {
                                "name": "Results",
                                "correct_name": "Results",
                                "correct_position": "yes",
                                "to_remove": "no"
                            },
                            {
                                "name": "Discussion",
                                "correct_name": "Discussion",
                                "correct_position": "yes",
                                "to_remove": "no"
                            },
                            {
                                "name": "Materials and methods",
                                "correct_name": "Methods",
                                "correct_position": "yes",
                                "to_remove": "no"
                            },
                            {
                                "name": "Acknowledgments",
                                "correct_name": "Acknowledgments",
                                "correct_position": "yes",
                                "to_remove": "no"
                            },
                            {
                                "name": "Author contributions",
                                "correct_name": "Author contributions",
                                "correct_position": "yes",
                                "to_remove": "no"
                            },
                            {
                                "name": "Disclosure and competing interests statement",
                                "correct_name": "Disclosure and competing interests statement",
                                "correct_position": "yes",
                                "to_remove": "no"
                            },
                            {
                                "name": "Data Availability",
                                "correct_name": "Data Availability",
                                "correct_position": "yes",
                                "to_remove": "no"
                            },
                            {
                                "name": "The paper explained",
                                "correct_name": "",
                                "correct_position": "no",
                                "to_remove": "yes"
                            },
                            {
                                "name": "References",
                                "correct_name": "References",
                                "correct_position": "yes",
                                "to_remove": "no"
                            },
                            {
                                "name": "Figure legends",
                                "correct_name": "Figure legends",
                                "correct_position": "yes",
                                "to_remove": "no"
                            },
                            {
                                "name": "Expanded View legends",
                                "correct_name": "Expanded View legends",
                                "correct_position": "yes",
                                "to_remove": "no"
                            }
                        ],
                        "missing_sections": []
                    }
                ]
            },
            "model_output": {
                "outputs": [
                    {
                        "sections_as_in_manuscript": [
                            {
                                "name": "Abstract",
                                "correct_name": "Abstract",
                                "correct_position": "yes",
                                "to_remove": "no"
                            },
                            {
                                "name": "Introduction",
                                "correct_name": "Introduction",
                                "correct_position": "yes",
                                "to_remove": "no"
                            },
                            {
                                "name": "Results",
                                "correct_name": "Results",
                                "correct_position": "yes",
                                "to_remove": "no"
                            },
                            {
                                "name": "Discussion",
                                "correct_name": "Discussion",
                                "correct_position": "yes",
                                "to_remove": "no"
                            },
                            {
                                "name": "Materials and methods",
                                "correct_name": "Methods",
                                "correct_position": "no",
                                "to_remove": "no"
                            },
                            {
                                "name": "Acknowledgments",
                                "correct_name": "Acknowledgements",
                                "correct_position": "yes",
                                "to_remove": "no"
                            },
                            {
                                "name": "Author contributions",
                                "correct_name": "Author contributions",
                                "correct_position": "yes",
                                "to_remove": "no"
                            },
                            {
                                "name": "Disclosure and competing interests statement",
                                "correct_name": "Disclosure and competing interests statement",
                                "correct_position": "yes",
                                "to_remove": "no"
                            },
                            {
                                "name": "Data Availability",
                                "correct_name": "Data Availability",
                                "correct_position": "yes",
                                "to_remove": "no"
                            },
                            {
                                "name": "The paper explained",
                                "correct_name": "",
                                "correct_position": "no",
                                "to_remove": "yes"
                            },
                            {
                                "name": "References",
                                "correct_name": "References",
                                "correct_position": "yes",
                                "to_remove": "no"
                            },
                            {
                                "name": "Figure legends",
                                "correct_name": "Figure legends",
                                "correct_position": "yes",
                                "to_remove": "no"
                            },
                            {
                                "name": "Expanded View legends",
                                "correct_name": "Expanded View legends",
                                "correct_position": "yes",
                                "to_remove": "no"
                            }
                        ],
                        "missing_sections": [
                            {
                                "name": "Data Availability"
                            }
                        ]
                    }
                ]
            }
        }
        
        # Create evaluator with semantic_similarity like the actual pipeline
        evaluator = JSONEvaluator(schema, string_metric="semantic_similarity")
        result = evaluator.evaluate(analysis["model_output"], 
                                  analysis["expected_output"])
        
        # Debug: Print the result structure
        print("DEBUG: Semantic similarity result:")
        print(f"Top-level score: {result.get('score', 'N/A')}")
        print(f"Element scores keys: {list(result.get('element_scores', {}).keys())}")
        
        # Check what we actually get
        self.assertIn("element_scores", result)
        element_scores = result["element_scores"]
        
        # Print more details about the structure
        for key, value in element_scores.items():
            print(f"  {key}: score={value.get('score', 'N/A')}")
            if 'field_scores' in value:
                for field_key, field_value in value.get('field_scores', {}).items():
                    print(f"    {field_key}: score={field_value.get('score', 'N/A')}")

    def test_real_life_example(self):
        schema_str = """
        {
            "format": {
                "type": "json_schema",
                "name": "section-order-alt",
                "schema": {
                    "type": "object",
                    "properties": {
                        "outputs": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "sections_as_in_manuscript": {
                                        "type": "array",
                                        "items": {
                                            "type": "object",
                                            "properties": {
                                                "name": {
                                                    "type": "string",
                                                    "description": "Section name as it appears in the manuscript"
                                                },
                                                "correct_name": {
                                                    "type": "string",
                                                    "description": "Correct standardized name for the section (empty string if should be removed)"
                                                },
                                                "correct_position": {
                                                    "type": "string",
                                                    "enum": ["yes", "no"],
                                                    "description": "Whether the section is in the correct position"
                                                },
                                                "to_remove": {
                                                    "type": "string",
                                                    "enum": ["yes", "no"],
                                                    "description": "Whether the section should be removed"
                                                }
                                            },
                                            "required": ["name", "correct_name", "correct_position", "to_remove"],
                                            "additionalProperties": false
                                        },
                                        "description": "List of sections as they appear in the manuscript with analysis"
                                    },
                                    "missing_sections": {
                                        "type": "array",
                                        "items": {
                                            "type": "object",
                                            "properties": {
                                                "name": {
                                                    "type": "string",
                                                    "description": "Name of the missing section"
                                                }
                                            },
                                            "required": ["name"],
                                            "additionalProperties": false
                                        },
                                        "description": "List of sections that are missing from the manuscript"
                                    }
                                },
                                "required": [
                                    "sections_as_in_manuscript",
                                    "missing_sections"
                                ],
                                "additionalProperties": false
                            }
                        }
                    },
                    "required": ["outputs"],
                    "additionalProperties": false
                },
                "strict": true
            }
        }"""
        import json
        schema = json.loads(schema_str)
        analysis = {
            "doc_id": "EMBOJ-2024-119734R",
            "expected_output": {
                "outputs": [
                    {
                        "sections_as_in_manuscript": [
                            {
                                "name": "Summary",
                                "correct_name": "Abstract",
                                "correct_position": "yes",
                                "to_remove": "no"
                            },
                            {
                                "name": "Introduction",
                                "correct_name": "Introduction",
                                "correct_position": "yes",
                                "to_remove": "no"
                            },
                            {
                                "name": "Results",
                                "correct_name": "Results",
                                "correct_position": "yes",
                                "to_remove": "no"
                            },
                            {
                                "name": "Discussion",
                                "correct_name": "Discussion",
                                "correct_position": "yes",
                                "to_remove": "no"
                            },
                            {
                                "name": "Materials and Methods",
                                "correct_name": "Methods",
                                "correct_position": "yes",
                                "to_remove": "no"
                            },
                            {
                                "name": "Data availability",
                                "correct_name": "Data Availability",
                                "correct_position": "yes",
                                "to_remove": "no"
                            },
                            {
                                "name": "Acknowledgements",
                                "correct_name": "Acknowledgements",
                                "correct_position": "yes",
                                "to_remove": "no"
                            },
                            {
                                "name": "Author contributions",
                                "correct_name": "Author contributions",
                                "correct_position": "yes",
                                "to_remove": "no"
                            },
                            {
                                "name": "Conflict of interest",
                                "correct_name": "Disclosure and competing interests statement",
                                "correct_position": "yes",
                                "to_remove": "no"
                            },
                            {
                                "name": "References",
                                "correct_name": "References",
                                "correct_position": "yes",
                                "to_remove": "no"
                            },
                            {
                                "name": "Figure legends",
                                "correct_name": "Figure legends",
                                "correct_position": "yes",
                                "to_remove": "no"
                            },
                            {
                                "name": "Supplementary figure legends",
                                "correct_name": "Expanded View legends",
                                "correct_position": "yes",
                                "to_remove": "no"
                            }
                        ],
                        "missing_sections": []
                    }
                ]
            },
            "model_output": {
                "outputs": [
                    {
                        "sections_as_in_manuscript": [
                            {
                                "name": "Summary",
                                "correct_name": "Abstract",
                                "correct_position": "yes",
                                "to_remove": "no"
                            },
                            {
                                "name": "Introduction",
                                "correct_name": "Introduction",
                                "correct_position": "yes",
                                "to_remove": "no"
                            },
                            {
                                "name": "Results",
                                "correct_name": "Results",
                                "correct_position": "yes",
                                "to_remove": "no"
                            },
                            {
                                "name": "Discussion",
                                "correct_name": "Discussion",
                                "correct_position": "yes",
                                "to_remove": "no"
                            },
                            {
                                "name": "Materials and Methods",
                                "correct_name": "Methods",
                                "correct_position": "no",
                                "to_remove": "no"
                            },
                            {
                                "name": "Data availability",
                                "correct_name": "Data Availability",
                                "correct_position": "yes",
                                "to_remove": "no"
                            },
                            {
                                "name": "Acknowledgements",
                                "correct_name": "Acknowledgments",
                                "correct_position": "yes",
                                "to_remove": "no"
                            },
                            {
                                "name": "Author contributions",
                                "correct_name": "Author contributions",
                                "correct_position": "yes",
                                "to_remove": "no"
                            },
                            {
                                "name": "Conflict of interest",
                                "correct_name": "Disclosure and competing interests statement",
                                "correct_position": "yes",
                                "to_remove": "no"
                            },
                            {
                                "name": "References",
                                "correct_name": "References",
                                "correct_position": "yes",
                                "to_remove": "no"
                            },
                            {
                                "name": "Figure legends",
                                "correct_name": "Figure legends",
                                "correct_position": "yes",
                                "to_remove": "no"
                            }
                        ],
                        "missing_sections": [
                            {
                                "name": "Expanded View legends"
                            }
                        ]
                    }
                ]
            }
        }
        pred = analysis["model_output"]
        exp = analysis["expected_output"]
        this_evaluator = JSONEvaluator(schema, string_metric="perfect_match", match_threshold=0.1)
        result = this_evaluator.evaluate(pred, exp)
        
        # Test the specific case: model predicts "Expanded View legends" as missing
        # but expected output has empty missing_sections list
        # This should result in an unexpected_predicted_element
        self.assertIn("element_scores", result)
        
        # Check for unexpected_predicted_element in the result
        # It's nested inside match_0_0.field_scores.missing_sections.element_scores
        match_0_0 = result["element_scores"]["match_0_0"]
        missing_sections_field = match_0_0["field_scores"]["missing_sections"]
        unexpected_keys = [key for key in missing_sections_field["element_scores"].keys() 
                          if key.startswith("unexpected_predicted_element")]
        self.assertGreater(len(unexpected_keys), 0, 
                          "Should have unexpected_predicted_element for incorrectly predicted missing section")
        
        # Also check that we have the expected matches
        match_keys = [key for key in result["element_scores"].keys() 
                     if key.startswith("match_")]
        self.assertGreater(len(match_keys), 0, "Should have match elements")

    def test_emm_2025_21341_with_lower_threshold(self):
        """Test EMM-2025-21341 with semantic_similarity and lower match_threshold."""
        schema = {
            "format": {
                "type": "json_schema",
                "name": "section-order-alt",
                "schema": {
                    "type": "object",
                    "properties": {
                        "outputs": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "sections_as_in_manuscript": {
                                        "type": "array",
                                        "items": {
                                            "type": "object",
                                            "properties": {
                                                "name": {"type": "string"},
                                                "correct_name": {"type": "string"},
                                                "correct_position": {"type": "string"},
                                                "to_remove": {"type": "string"}
                                            },
                                            "required": ["name", "correct_name", 
                                                        "correct_position", "to_remove"]
                                        }
                                    },
                                    "missing_sections": {
                                        "type": "array",
                                        "items": {
                                            "type": "object",
                                            "properties": {
                                                "name": {"type": "string"}
                                            },
                                            "required": ["name"]
                                        }
                                    }
                                },
                                "required": ["sections_as_in_manuscript", 
                                           "missing_sections"]
                            }
                        }
                    },
                    "required": ["outputs"]
                }
            }
        }
        
        # Use the exact data from the real case
        analysis = {
            "doc_id": "EMM-2025-21341",
            "expected_output": {
                "outputs": [
                    {
                        "sections_as_in_manuscript": [
                            {
                                "name": "Abstract",
                                "correct_name": "Abstract",
                                "correct_position": "yes",
                                "to_remove": "no"
                            },
                            {
                                "name": "Introduction",
                                "correct_name": "Introduction",
                                "correct_position": "yes",
                                "to_remove": "no"
                            },
                            {
                                "name": "Results",
                                "correct_name": "Results",
                                "correct_position": "yes",
                                "to_remove": "no"
                            },
                            {
                                "name": "Discussion",
                                "correct_name": "Discussion",
                                "correct_position": "yes",
                                "to_remove": "no"
                            },
                            {
                                "name": "Materials and methods",
                                "correct_name": "Methods",
                                "correct_position": "yes",
                                "to_remove": "no"
                            },
                            {
                                "name": "Acknowledgments",
                                "correct_name": "Acknowledgments",
                                "correct_position": "yes",
                                "to_remove": "no"
                            },
                            {
                                "name": "Author contributions",
                                "correct_name": "Author contributions",
                                "correct_position": "yes",
                                "to_remove": "no"
                            },
                            {
                                "name": "Disclosure and competing interests statement",
                                "correct_name": "Disclosure and competing interests statement",
                                "correct_position": "yes",
                                "to_remove": "no"
                            },
                            {
                                "name": "Data Availability",
                                "correct_name": "Data Availability",
                                "correct_position": "yes",
                                "to_remove": "no"
                            },
                            {
                                "name": "The paper explained",
                                "correct_name": "",
                                "correct_position": "no",
                                "to_remove": "yes"
                            },
                            {
                                "name": "References",
                                "correct_name": "References",
                                "correct_position": "yes",
                                "to_remove": "no"
                            },
                            {
                                "name": "Figure legends",
                                "correct_name": "Figure legends",
                                "correct_position": "yes",
                                "to_remove": "no"
                            },
                            {
                                "name": "Expanded View legends",
                                "correct_name": "Expanded View legends",
                                "correct_position": "yes",
                                "to_remove": "no"
                            }
                        ],
                        "missing_sections": []
                    }
                ]
            },
            "model_output": {
                "outputs": [
                    {
                        "sections_as_in_manuscript": [
                            {
                                "name": "Abstract",
                                "correct_name": "Abstract",
                                "correct_position": "yes",
                                "to_remove": "no"
                            },
                            {
                                "name": "Introduction",
                                "correct_name": "Introduction",
                                "correct_position": "yes",
                                "to_remove": "no"
                            },
                            {
                                "name": "Results",
                                "correct_name": "Results",
                                "correct_position": "yes",
                                "to_remove": "no"
                            },
                            {
                                "name": "Discussion",
                                "correct_name": "Discussion",
                                "correct_position": "yes",
                                "to_remove": "no"
                            },
                            {
                                "name": "Materials and methods",
                                "correct_name": "Methods",
                                "correct_position": "no",
                                "to_remove": "no"
                            },
                            {
                                "name": "Acknowledgments",
                                "correct_name": "Acknowledgements",
                                "correct_position": "yes",
                                "to_remove": "no"
                            },
                            {
                                "name": "Author contributions",
                                "correct_name": "Author contributions",
                                "correct_position": "yes",
                                "to_remove": "no"
                            },
                            {
                                "name": "Disclosure and competing interests statement",
                                "correct_name": "Disclosure and competing interests statement",
                                "correct_position": "yes",
                                "to_remove": "no"
                            },
                            {
                                "name": "Data Availability",
                                "correct_name": "Data Availability",
                                "correct_position": "yes",
                                "to_remove": "no"
                            },
                            {
                                "name": "The paper explained",
                                "correct_name": "",
                                "correct_position": "no",
                                "to_remove": "yes"
                            },
                            {
                                "name": "References",
                                "correct_name": "References",
                                "correct_position": "yes",
                                "to_remove": "no"
                            },
                            {
                                "name": "Figure legends",
                                "correct_name": "Figure legends",
                                "correct_position": "yes",
                                "to_remove": "no"
                            },
                            {
                                "name": "Expanded View legends",
                                "correct_name": "Expanded View legends",
                                "correct_position": "yes",
                                "to_remove": "no"
                            }
                        ],
                        "missing_sections": [
                            {
                                "name": "Data Availability"
                            }
                        ]
                    }
                ]
            }
        }
        
        # Create evaluator with semantic_similarity and lower threshold
        evaluator = JSONEvaluator(schema, string_metric="semantic_similarity", 
                                 match_threshold=0.1)  # Lower threshold
        result = evaluator.evaluate(analysis["model_output"], 
                                  analysis["expected_output"])
        
        # Debug: Print the result structure
        print("DEBUG: Lower threshold result:")
        print(f"Top-level score: {result.get('score', 'N/A')}")
        print(f"Element scores keys: {list(result.get('element_scores', {}).keys())}")
        
        # Check what we actually get
        self.assertIn("element_scores", result)
        element_scores = result["element_scores"]
        
        # Print more details about the structure
        for key, value in element_scores.items():
            print(f"  {key}: score={value.get('score', 'N/A')}")
            if 'field_scores' in value:
                for field_key, field_value in value.get('field_scores', {}).items():
                    print(f"    {field_key}: score={field_value.get('score', 'N/A')}")
        
        # This should now have match_0_0 instead of unexpected_element_0
        match_keys = [key for key in element_scores.keys() 
                     if key.startswith("match_")]
        self.assertGreater(len(match_keys), 0, 
                          "Should have match elements with lower threshold")


if __name__ == "__main__":
    unittest.main()
