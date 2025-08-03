import os
import json
import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

from soda_mmqc.lib.api import generate_response_openai, generate_response
from soda_mmqc.core.examples import FigureExample
from soda_mmqc.scripts.run import ModelInput


class TestModelApi(unittest.TestCase):
    """Test cases for the model_api module."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary test image file
        self.test_image = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
        # Create a minimal valid JPEG image data
        minimal_jpeg = (
            b'\xFF\xD8\xFF\xE0\x00\x10JFIF\x00\x01\x01\x01\x00H\x00H\x00\x00'
            b'\xFF\xDB\x00C\x00\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF'
            b'\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF'
            b'\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF'
            b'\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF'
            b'\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF'
            b'\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF'
            b'\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF'
            b'\xFF\xC0\x00\x0B\x08\x00\x01\x00\x01\x01\x01\x11\x00\xFF\xC4\x00'
            b'\x14\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
            b'\x00\x00\x00\x00\x00\xFF\xDA\x00\x08\x01\x01\x00\x00\x3F\x00?'
            b'\xFF\xD9'
        )
        self.test_image.write(minimal_jpeg)
        self.test_image.close()
        
        # Create a temporary directory structure for the example
        self.test_dir = tempfile.mkdtemp()
        self.content_dir = Path(self.test_dir) / "content"
        self.content_dir.mkdir()
        
        # Move the test image to the content directory
        import shutil
        shutil.move(self.test_image.name, self.content_dir / "test_image.jpg")
        
        # Create a caption file
        caption_file = self.content_dir / "caption.txt"
        caption_file.write_text("Test figure caption")
        
        # Test data
        self.test_prompt = "Analyze this figure for error bars."
        self.test_schema = {
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
        self.test_model = "gpt-4o-2024-08-06"
        self.test_metadata = {"test": "metadata"}

    def tearDown(self):
        """Clean up test fixtures."""
        # Clean up the temporary directory
        import shutil
        shutil.rmtree(self.test_dir)

    @patch('soda_mmqc.lib.api.OpenAI')
    @patch('soda_mmqc.lib.api.os.getenv')
    def test_generate_response_openai_success(self, mock_getenv, mock_openai):
        """Test successful response generation with OpenAI API."""
        # Mock environment variables
        mock_getenv.return_value = "test-api-key"
        
        # Mock OpenAI client and response
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        
        # Mock the response object
        mock_response = MagicMock()
        mock_response.output_text = '{"name": "test", "panels": []}'
        mock_response.metadata = {"test": "metadata"}
        mock_response.id = "test-response-id"
        mock_response.model = "gpt-4o-2024-08-06"
        mock_client.responses.create.return_value = mock_response
        
        # Create a test example
        example = FigureExample(str(self.test_dir))
        example.load_from_source()
        
        # Call the function
        result, metadata = generate_response_openai(
            example=example,
            prompt=self.test_prompt,
            schema=self.test_schema,
            model=self.test_model,
            metadata=self.test_metadata
        )
        
        # Verify the result
        self.assertEqual(result, {"name": "test", "panels": []})
        self.assertIn("response_id", metadata)
        self.assertIn("model", metadata)
        self.assertEqual(metadata["response_id"], "test-response-id")
        self.assertEqual(metadata["model"], "gpt-4o-2024-08-06")
        
        # Verify the API was called correctly
        mock_client.responses.create.assert_called_once()
        call_args = mock_client.responses.create.call_args
        self.assertEqual(call_args[1]["model"], self.test_model)
        self.assertEqual(call_args[1]["text"], self.test_schema)
        self.assertEqual(call_args[1]["metadata"], self.test_metadata)

    @patch('soda_mmqc.lib.api.OpenAI')
    @patch('soda_mmqc.lib.api.os.getenv')
    def test_generate_response_openai_json_error(self, mock_getenv, mock_openai):
        """Test handling of JSON parsing errors."""
        # Mock environment variables
        mock_getenv.return_value = "test-api-key"
        
        # Mock OpenAI client and response
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        
        # Mock the response object with invalid JSON
        mock_response = MagicMock()
        mock_response.output_text = '{"invalid": json}'
        mock_client.responses.create.return_value = mock_response
        
        # Create a test example
        example = FigureExample(str(self.test_dir))
        example.load_from_source()
        
        # Call the function and expect it to raise an exception
        with self.assertRaises(json.JSONDecodeError):
            generate_response_openai(
                example=example,
                prompt=self.test_prompt,
                schema=self.test_schema,
                model=self.test_model,
                metadata=self.test_metadata
            )

    @patch('soda_mmqc.lib.api.OpenAI')
    @patch('soda_mmqc.lib.api.os.getenv')
    def test_generate_response_openai_missing_metadata(self, mock_getenv, mock_openai):
        """Test handling of missing metadata in response."""
        # Mock environment variables
        mock_getenv.return_value = "test-api-key"
        
        # Mock OpenAI client and response
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        
        # Mock the response object with missing metadata attributes
        mock_response = MagicMock()
        mock_response.output_text = '{"name": "test", "panels": []}'
        # Explicitly set metadata to None to simulate missing attributes
        mock_response.metadata = None
        mock_response.id = None
        mock_response.model = None
        mock_client.responses.create.return_value = mock_response
        
        # Create a test example
        example = FigureExample(str(self.test_dir))
        example.load_from_source()
        
        # Call the function
        result, metadata = generate_response_openai(
            example=example,
            prompt=self.test_prompt,
            schema=self.test_schema,
            model=self.test_model,
            metadata=self.test_metadata
        )
        
        # Verify the result
        self.assertEqual(result, {"name": "test", "panels": []})
        # Should have empty strings for response_id and model when attributes are None
        self.assertEqual(metadata, {"response_id": "", "model": ""})

    @patch('soda_mmqc.lib.api.generate_response_openai')
    def test_generate_response_wrapper(self, mock_generate_openai):
        """Test the generate_response wrapper function."""
        # Mock the underlying function
        mock_generate_openai.return_value = ({"test": "result"}, {"test": "metadata"})
        
        # Create a test example
        example = FigureExample(str(self.test_dir))
        example.load_from_source()
        
        # Create ModelInput object
        model_input = ModelInput(
            example=example,
            prompt=self.test_prompt,
            schema=self.test_schema
        )
        
        # Call the wrapper function
        result, metadata = generate_response(
            model_input=model_input,
            model=self.test_model,
            metadata=self.test_metadata
        )
        
        # Verify the result
        self.assertEqual(result, {"test": "result"})
        self.assertEqual(metadata, {"test": "metadata"})
        
        # Verify the underlying function was called correctly
        mock_generate_openai.assert_called_once_with(
            example=example,
            prompt=self.test_prompt,
            schema=self.test_schema,
            model=self.test_model,
            metadata=self.test_metadata
        )

    @patch('soda_mmqc.lib.api.OpenAI')
    @patch('soda_mmqc.lib.api.os.getenv')
    def test_generate_response_openai_with_retry(self, mock_getenv, mock_openai):
        """Test that the function retries on JSON errors."""
        # Mock environment variables
        mock_getenv.return_value = "test-api-key"
        
        # Mock OpenAI client
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        
        # Mock the response object to fail twice then succeed
        mock_response1 = MagicMock()
        mock_response1.output_text = '{"invalid": json}'  # Invalid JSON
        
        mock_response2 = MagicMock()
        mock_response2.output_text = '{"invalid": json}'  # Invalid JSON again
        
        mock_response3 = MagicMock()
        mock_response3.output_text = '{"name": "test", "panels": []}'  # Valid JSON
        mock_response3.metadata = {"test": "metadata"}
        mock_response3.id = "test-response-id"
        mock_response3.model = "gpt-4o-2024-08-06"
        
        # Set up the mock to return different responses on subsequent calls
        mock_client.responses.create.side_effect = [mock_response1, mock_response2, mock_response3]
        
        # Create a test example
        example = FigureExample(str(self.test_dir))
        example.load_from_source()
        
        # Call the function - it should retry and eventually succeed
        result, metadata = generate_response_openai(
            example=example,
            prompt=self.test_prompt,
            schema=self.test_schema,
            model=self.test_model,
            metadata=self.test_metadata
        )
        
        # Verify the result
        self.assertEqual(result, {"name": "test", "panels": []})
        self.assertEqual(metadata["response_id"], "test-response-id")
        
        # Verify the API was called 3 times (2 failures + 1 success)
        self.assertEqual(mock_client.responses.create.call_count, 3)


if __name__ == "__main__":
    unittest.main()
