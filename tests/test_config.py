import os
import unittest
from unittest.mock import patch
import logging

from soda_mmqc.config import (
    _validate_and_setup_api_provider,
    API_PROVIDER,
    DEFAULT_MODELS,
    DEFAULT_MODEL,
    STRING_COMPARE_MODES,
)
from soda_mmqc.core.leaves import StringCompareMode


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

    def test_default_model_openai(self):
        """Test that OpenAI default model is configured."""
        from soda_mmqc.config import DEFAULT_MODELS
        openai_model = DEFAULT_MODELS["openai"]
        self.assertIsInstance(openai_model, str)
        self.assertTrue(len(openai_model) > 0)

    def test_default_model_anthropic(self):
        """Test that Anthropic default model is configured."""
        from soda_mmqc.config import DEFAULT_MODELS
        anthropic_model = DEFAULT_MODELS["anthropic"]
        self.assertIsInstance(anthropic_model, str)
        self.assertTrue(len(anthropic_model) > 0)

    def test_api_provider_validation_function_type(self):
        """Test that the validation function returns string."""
        provider = _validate_and_setup_api_provider()
        self.assertIsInstance(provider, str)
        self.assertIn(provider, ["openai", "anthropic"])

    def test_string_compare_modes_match_leaves_enum(self):
        """STRING_COMPARE_MODES aligns with manifest string_compare values."""
        expected = {mode.value for mode in StringCompareMode}
        self.assertEqual(set(STRING_COMPARE_MODES), expected)


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
        import importlib
        import soda_mmqc.config
        importlib.reload(soda_mmqc.config)
        from soda_mmqc.config import API_PROVIDER, DEFAULT_MODEL, DEFAULT_MODELS
        self.assertEqual(API_PROVIDER, "openai")
        self.assertEqual(DEFAULT_MODEL, DEFAULT_MODELS["openai"])

    @patch.dict(os.environ, {'API_PROVIDER': 'anthropic'})
    def test_config_integration_anthropic(self):
        """Test that API module uses config correctly for Anthropic."""
        import importlib
        import soda_mmqc.config
        importlib.reload(soda_mmqc.config)
        from soda_mmqc.config import API_PROVIDER, DEFAULT_MODEL, DEFAULT_MODELS
        self.assertEqual(API_PROVIDER, "anthropic")
        self.assertEqual(DEFAULT_MODEL, DEFAULT_MODELS["anthropic"])


if __name__ == "__main__":
    unittest.main()