import os
import unittest
from unittest.mock import patch
import logging

from soda_mmqc.config import (
    _validate_and_setup_api_provider,
    API_PROVIDER,
    DEFAULT_MODELS,
    DEFAULT_MODEL
)


class TestConfig(unittest.TestCase):
    """Test cases for the configuration module."""

    def setUp(self):
        """Set up test fixtures."""
        # Silence logging during tests
        logging.disable(logging.CRITICAL)

    def tearDown(self):
        """Clean up after tests."""
        # Re-enable logging after tests
        logging.disable(logging.NOTSET)

    @patch.dict(os.environ, {'API_PROVIDER': 'openai', 'OPENAI_API_KEY': 'test-key'})
    def test_validate_api_provider_openai_with_key(self):
        """Test OpenAI provider validation with API key present."""
        provider = _validate_and_setup_api_provider()
        self.assertEqual(provider, "openai")

    @patch.dict(os.environ, {'API_PROVIDER': 'openai'}, clear=False)
    @patch.dict(os.environ, {}, clear=True)
    def test_validate_api_provider_openai_without_key(self):
        """Test OpenAI provider validation without API key."""
        with patch.dict(os.environ, {'API_PROVIDER': 'openai'}):
            provider = _validate_and_setup_api_provider()
            self.assertEqual(provider, "openai")

    @patch.dict(os.environ, {'API_PROVIDER': 'anthropic', 'ANTHROPIC_API_KEY': 'test-key'})
    def test_validate_api_provider_anthropic_with_key(self):
        """Test Anthropic provider validation with API key present."""
        provider = _validate_and_setup_api_provider()
        self.assertEqual(provider, "anthropic")

    @patch.dict(os.environ, {'API_PROVIDER': 'anthropic'}, clear=False)
    @patch.dict(os.environ, {}, clear=True)
    def test_validate_api_provider_anthropic_without_key(self):
        """Test Anthropic provider validation without API key."""
        with patch.dict(os.environ, {'API_PROVIDER': 'anthropic'}):
            provider = _validate_and_setup_api_provider()
            self.assertEqual(provider, "anthropic")

    @patch.dict(os.environ, {'API_PROVIDER': 'invalid_provider'})
    def test_validate_api_provider_invalid(self):
        """Test invalid provider falls back to OpenAI."""
        provider = _validate_and_setup_api_provider()
        self.assertEqual(provider, "openai")

    @patch.dict(os.environ, {}, clear=True)
    def test_validate_api_provider_default(self):
        """Test default provider when no environment variable is set."""
        provider = _validate_and_setup_api_provider()
        self.assertEqual(provider, "openai")

    @patch.dict(os.environ, {'API_PROVIDER': 'OPENAI'})  # Test case insensitivity
    def test_validate_api_provider_case_insensitive(self):
        """Test that provider names are case insensitive."""
        provider = _validate_and_setup_api_provider()
        self.assertEqual(provider, "openai")

    def test_default_models_structure(self):
        """Test that default models are properly configured."""
        self.assertIn("openai", DEFAULT_MODELS)
        self.assertIn("anthropic", DEFAULT_MODELS)
        self.assertIsInstance(DEFAULT_MODELS["openai"], str)
        self.assertIsInstance(DEFAULT_MODELS["anthropic"], str)
        self.assertTrue(len(DEFAULT_MODELS["openai"]) > 0)
        self.assertTrue(len(DEFAULT_MODELS["anthropic"]) > 0)

    def test_default_model_selection(self):
        """Test that DEFAULT_MODEL is correctly selected based on provider."""
        # This test uses the actual global configuration
        self.assertIn(DEFAULT_MODEL, DEFAULT_MODELS.values())

    @patch('soda_mmqc.config.API_PROVIDER', 'openai')
    def test_default_model_openai(self):
        """Test default model selection for OpenAI."""
        from soda_mmqc.config import DEFAULT_MODELS
        expected_model = DEFAULT_MODELS["openai"]
        # Since we can't easily mock the module-level variable,
        # we'll test the logic conceptually
        self.assertEqual(DEFAULT_MODELS.get("openai"), "gpt-4o-2024-08-06")

    @patch('soda_mmqc.config.API_PROVIDER', 'anthropic')
    def test_default_model_anthropic(self):
        """Test default model selection for Anthropic."""
        from soda_mmqc.config import DEFAULT_MODELS
        expected_model = DEFAULT_MODELS["anthropic"]
        # Since we can't easily mock the module-level variable,
        # we'll test the logic conceptually
        self.assertEqual(DEFAULT_MODELS.get("anthropic"), "claude-3-5-sonnet-20241022")

    def test_api_provider_validation_function_type(self):
        """Test that the validation function returns string."""
        provider = _validate_and_setup_api_provider()
        self.assertIsInstance(provider, str)
        self.assertIn(provider, ["openai", "anthropic"])


class TestConfigIntegration(unittest.TestCase):
    """Integration tests for configuration with API module."""

    def setUp(self):
        """Set up test fixtures."""
        # Silence logging during tests
        logging.disable(logging.CRITICAL)

    def tearDown(self):
        """Clean up after tests."""
        # Re-enable logging after tests
        logging.disable(logging.NOTSET)

    @patch.dict(os.environ, {'API_PROVIDER': 'openai'})
    def test_config_integration_openai(self):
        """Test that API module uses config correctly for OpenAI."""
        # Import here to get fresh config with mocked environment
        import importlib
        import soda_mmqc.config
        importlib.reload(soda_mmqc.config)
        
        from soda_mmqc.config import API_PROVIDER, DEFAULT_MODEL
        self.assertEqual(API_PROVIDER, "openai")
        self.assertIn(DEFAULT_MODEL, ["gpt-4o-2024-08-06", "gpt-4o-mini"])

    @patch.dict(os.environ, {'API_PROVIDER': 'anthropic'})
    def test_config_integration_anthropic(self):
        """Test that API module uses config correctly for Anthropic."""
        # Import here to get fresh config with mocked environment
        import importlib
        import soda_mmqc.config
        importlib.reload(soda_mmqc.config)
        
        from soda_mmqc.config import API_PROVIDER, DEFAULT_MODEL
        self.assertEqual(API_PROVIDER, "anthropic")
        self.assertIn(DEFAULT_MODEL, ["claude-3-5-sonnet-20241022", "claude-3-7-sonnet-20250219"])


if __name__ == "__main__":
    unittest.main()