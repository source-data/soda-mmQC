#!/usr/bin/env python3
"""Test script for Pandoc integration with WordExample class."""

import subprocess
import sys
from pathlib import Path
import pytest


def test_pandoc_installation():
    """Test if Pandoc is installed and working."""
    try:
        result = subprocess.run(
            ["pandoc", "--version"],
            capture_output=True,
            text=True,
            check=True
        )
        print("✅ Pandoc is installed and working")
        print(f"Version: {result.stdout.split()[1]}")
    except FileNotFoundError:
        pytest.skip("Pandoc not found. Install from: https://pandoc.org/installing.html")
    except subprocess.CalledProcessError as e:
        pytest.fail(f"Pandoc error: {e.stderr}")


def test_word_file_conversion():
    """Test converting a Word file to text."""
    # Look for a test Word file in the tests/documents/content directory
    test_dir = Path(__file__).parent / "documents" / "content"
    test_files = list(test_dir.glob("*.docx"))
    if not test_files:
        pytest.skip("No .docx files found in tests/documents/content for testing")
    test_file = test_files[0]
    print(f"Testing with file: {test_file}")
    try:
        result = subprocess.run(
            ["pandoc", str(test_file), "-t", "plain"],
            capture_output=True,
            text=True,
            check=True
        )
        text_content = result.stdout.strip()
        print(f"✅ Successfully converted Word file to text ({len(text_content)} characters)")
        print(f"First 200 characters: {text_content[:200]}...")
        assert len(text_content) > 0, "Extracted text should not be empty"
        assert isinstance(text_content, str), "Extracted content should be a string"
    except subprocess.CalledProcessError as e:
        pytest.fail(f"Pandoc conversion error: {e.stderr}")


def test_wordexample_integration():
    """Test the WordExample class with Pandoc integration."""
    try:
        from soda_mmqc.examples import WordExample
    except ImportError:
        pytest.skip("soda_mmqc.examples module not available")
    # Use tests/documents as the example directory
    example_dir = Path(__file__).parent / "documents"
    test_files = list((example_dir / "content").glob("*.docx"))
    if not test_files:
        pytest.skip("No .docx files found in tests/documents/content for testing")
    try:
        word_example = WordExample(str(example_dir))
        assert hasattr(word_example, 'content'), "WordExample should have content attribute"
        assert word_example.content is not None, "Content should not be None"
        assert len(word_example.content) > 0, "Content should not be empty"
        content_hash = word_example.get_content_hash()
        assert isinstance(content_hash, str), "Content hash should be a string"
        assert len(content_hash) == 64, "SHA-256 hash should be 64 characters"
        model_input = word_example.prepare_model_input("Test prompt")
        assert "content" in model_input, "Model input should have content"
        assert "metadata" in model_input, "Model input should have metadata"
        assert model_input["metadata"]["doc_id"] == "test-doi-123"
    except Exception as e:
        pytest.fail(f"WordExample integration test failed: {str(e)}")


if __name__ == "__main__":
    # Allow running as standalone script
    print("Testing Pandoc integration for WordExample class...\n")

    # Test 1: Pandoc installation
    try:
        test_pandoc_installation()
        print("✅ Pandoc installation test passed")
    except Exception as e:
        print(f"❌ Pandoc installation test failed: {e}")
        sys.exit(1)

    print()

    # Test 2: Word file conversion
    try:
        test_word_file_conversion()
        print("✅ Word file conversion test passed")
    except Exception as e:
        print(f"❌ Word file conversion test failed: {e}")
        sys.exit(1)

    print()
    print("🎉 All tests passed! Pandoc integration is ready.")
    print("\nYou can now use WordExample class with Pandoc for text extraction.") 