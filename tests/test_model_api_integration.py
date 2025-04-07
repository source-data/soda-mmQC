import os
import unittest
import tempfile
import json
from PIL import Image
from dotenv import load_dotenv

# Add the project root to the path
import sys
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from soda_mmqc.model_api import generate_response_openai, encode_image

class TestModelApiIntegration(unittest.TestCase):
    """Integration tests for the model_api module."""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures that are reused across all tests."""
        load_dotenv()
        if not os.getenv("OPENAI_API_KEY"):
            raise unittest.SkipTest("OPENAI_API_KEY not found in environment")

    def setUp(self):
        """Set up test fixtures."""
        # Create a test image with a simple plot and error bars
        self.test_image = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
        img = Image.new('RGB', (400, 300), color='white')
        img.save(self.test_image.name)
        self.test_image.close()

        # Test data
        self.test_caption = "Figure 1: Bar plot showing mean values with error bars representing standard deviation."
        self.test_prompt = "Check if error bars are properly defined in this figure."

    def tearDown(self):
        """Clean up test fixtures."""
        os.unlink(self.test_image.name)

    def test_generate_response_openai_integration(self):
        """Test the generate_response_openai function with real API."""
        # Encode the test image
        encoded_image = encode_image(self.test_image.name)

        # Call the function
        result = generate_response_openai(
            encoded_image,
            self.test_caption,
            self.test_prompt
        )

        # Parse and verify the structure of the response
        try:
            response_json = json.loads(result)
            
            # Check basic structure
            self.assertIn("name", response_json)
            self.assertIn("panels", response_json)
            self.assertIsInstance(response_json["panels"], list)
            
            # Check panel structure if any panels are returned
            if response_json["panels"]:
                panel = response_json["panels"][0]
                required_fields = [
                    "panel_label",
                    "error_bar_on_figure",
                    "error_bar_defined_in_legend",
                    "error_bar_meaning"
                ]
                for field in required_fields:
                    self.assertIn(field, panel)

        except json.JSONDecodeError:
            self.fail("Response is not valid JSON")

if __name__ == "__main__":
    unittest.main() 