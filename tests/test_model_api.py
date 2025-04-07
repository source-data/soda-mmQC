import os
import json
import unittest
import base64
import sys
from unittest.mock import patch, MagicMock
import tempfile

# Add the project root to the path
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

# Import our mock modules
from tests.mock_modules import OpenAI, load_dotenv

# Create mock modules
mock_openai = MagicMock()
mock_dotenv = MagicMock()

# Patch the modules before importing the code under test
with patch.dict('sys.modules', {
    'openai': mock_openai,
    'dotenv': mock_dotenv
}):
    from soda_mmqc.model_api import generate_response_openai, encode_image


class TestModelApi(unittest.TestCase):
    """Test cases for the model_api module."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary test image file
        self.test_image = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
        # Create a minimal valid JPEG image data
        minimal_jpeg = b'\xFF\xD8\xFF\xE0\x00\x10JFIF\x00\x01\x01\x01\x00H\x00H\x00\x00\xFF\xDB\x00C\x00\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xC0\x00\x0B\x08\x00\x01\x00\x01\x01\x01\x11\x00\xFF\xC4\x00\x14\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xFF\xDA\x00\x08\x01\x01\x00\x00\x3F\x00?\xFF\xD9'
        self.test_image.write(minimal_jpeg)
        self.test_image.close()
        
        # Test data
        self.test_caption = "Test figure caption"
        self.test_prompt = "Test prompt"
        self.test_json_schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "panels": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "panel_label": {"type": "string"},
                            "error_bar_on_figure": {"type": "string"},
                            "error_bar_defined_in_legend": {"type": "string"},
                            "error_bar_meaning": {"type": "string"}
                        },
                        "required": [
                            "panel_label",
                            "error_bar_on_figure",
                            "error_bar_defined_in_legend",
                            "error_bar_meaning"
                        ]
                    }
                }
            },
            "required": ["name", "panels"]
        }

    def tearDown(self):
        """Clean up test fixtures."""
        # Clean up the temporary file
        os.unlink(self.test_image.name)

    def test_generate_response_openai(self):
        """Test the generate_response_openai function."""
        # Mock OpenAI response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '{"name": "test", "panels": []}'
        mock_openai.OpenAI.return_value.chat.completions.create.return_value = mock_response

        # Encode the test image
        encoded_image = encode_image(self.test_image.name)

        # Call the function
        result = generate_response_openai(
            encoded_image,
            self.test_caption,
            self.test_prompt
        )

        # Verify the result
        self.assertEqual(result, '{"name": "test", "panels": []}')
        mock_openai.OpenAI.return_value.chat.completions.create.assert_called_once()


if __name__ == "__main__":
    unittest.main() 