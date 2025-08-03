#!/usr/bin/env python3
"""Test project architecture and imports after reorganization."""

import unittest

from soda_mmqc import logger
from soda_mmqc.config import (
    PACKAGE_ROOT,
    DATA_DIR,
    CHECKLIST_DIR,
    EXAMPLES_DIR,
    EVALUATION_DIR,
    PLOTS_DIR,
    CACHE_DIR
)


class TestProjectArchitecture(unittest.TestCase):
    """Test the project architecture and imports after reorganization."""
    
    def test_package_structure(self):
        """Test that the package structure is correct."""
        # Check that core modules exist
        self.assertTrue((PACKAGE_ROOT / "core").exists())
        self.assertTrue((PACKAGE_ROOT / "core" / "examples.py").exists())
        self.assertTrue((PACKAGE_ROOT / "core" / "evaluation.py").exists())
        self.assertTrue((PACKAGE_ROOT / "core" / "curation.py").exists())
        
        # Check that lib modules exist
        self.assertTrue((PACKAGE_ROOT / "lib").exists())
        self.assertTrue((PACKAGE_ROOT / "lib" / "api.py").exists())
        self.assertTrue((PACKAGE_ROOT / "lib" / "cache.py").exists())
        
        # Check that scripts exist (CLI entry points)
        self.assertTrue((PACKAGE_ROOT / "scripts").exists())
        self.assertTrue((PACKAGE_ROOT / "scripts" / "run.py").exists())
        self.assertTrue((PACKAGE_ROOT / "scripts" / "curate.py").exists())
        self.assertTrue((PACKAGE_ROOT / "scripts" / "visualize.py").exists())
        
        # Check that utils exist
        self.assertTrue((PACKAGE_ROOT / "utils").exists())
        self.assertTrue((PACKAGE_ROOT / "utils" / "hash_utils.py").exists())
        
        # Check that data directory exists
        self.assertTrue((PACKAGE_ROOT / "data").exists())
        
        # Check that config and init files exist
        self.assertTrue((PACKAGE_ROOT / "config.py").exists())
        self.assertTrue((PACKAGE_ROOT / "__init__.py").exists())
    
    def test_core_imports(self):
        """Test that core modules can be imported."""
        try:
            from soda_mmqc.core import examples, evaluation, curation
            self.assertTrue(hasattr(examples, 'EXAMPLE_FACTORY'))
            self.assertTrue(hasattr(evaluation, 'JSONEvaluator'))
            self.assertTrue(hasattr(curation, 'load_example_data'))
            logger.info("Core imports successful")
        except ImportError as e:
            self.fail(f"Failed to import core modules: {e}")
    
    def test_lib_imports(self):
        """Test that lib modules can be imported."""
        try:
            from soda_mmqc.lib import api, cache
            self.assertTrue(hasattr(api, 'generate_response'))
            self.assertTrue(hasattr(cache, 'ModelCache'))
            logger.info("Lib imports successful")
        except ImportError as e:
            self.fail(f"Failed to import lib modules: {e}")
    
    def test_scripts_imports(self):
        """Test that script modules can be imported."""
        try:
            from soda_mmqc.scripts import run, curate, visualize
            self.assertTrue(hasattr(run, 'main'))
            self.assertTrue(hasattr(curate, 'main'))
            self.assertTrue(hasattr(visualize, 'checklist_visualization'))
            logger.info("Scripts imports successful")
        except ImportError as e:
            self.fail(f"Failed to import script modules: {e}")
    
    def test_utils_imports(self):
        """Test that utils modules can be imported."""
        try:
            from soda_mmqc.utils import hash_utils
            self.assertTrue(hasattr(hash_utils, 'hash_document_and_json'))
            self.assertTrue(hasattr(hash_utils, 'verify_hash'))
            logger.info("Utils imports successful")
        except ImportError as e:
            self.fail(f"Failed to import utils modules: {e}")
    
    def test_config_paths(self):
        """Test that configuration paths are correct."""
        # Check that all config paths exist or can be created
        self.assertTrue(PACKAGE_ROOT.exists())
        self.assertTrue(DATA_DIR.exists())
        self.assertTrue(CHECKLIST_DIR.exists())
        self.assertTrue(EXAMPLES_DIR.exists())
        self.assertTrue(EVALUATION_DIR.exists())
        self.assertTrue(PLOTS_DIR.exists())
        
        # CACHE_DIR might not exist initially, but should be creatable
        CACHE_DIR.mkdir(exist_ok=True)
        self.assertTrue(CACHE_DIR.exists())
    
    def test_no_duplicate_modules(self):
        """Test that there are no duplicate modules between core and scripts."""
        core_files = set()
        scripts_files = set()
        
        # Get core module names
        core_dir = PACKAGE_ROOT / "core"
        for file in core_dir.glob("*.py"):
            if file.name != "__init__.py":
                core_files.add(file.stem)
        
        # Get scripts module names
        scripts_dir = PACKAGE_ROOT / "scripts"
        for file in scripts_dir.glob("*.py"):
            if file.name != "__init__.py":
                scripts_files.add(file.stem)
        
        # Check for duplicates
        duplicates = core_files.intersection(scripts_files)
        self.assertEqual(duplicates, set(), 
                        f"Found duplicate modules between core and scripts: {duplicates}")
    
    def test_cli_entry_points(self):
        """Test that CLI entry points are properly configured."""
        # Check that the main CLI functions exist
        try:
            from soda_mmqc.scripts.run import main as run_main
            from soda_mmqc.scripts.run import initialize_main
            from soda_mmqc.scripts.curate import main as curate_main
            
            self.assertTrue(callable(run_main))
            self.assertTrue(callable(initialize_main))
            self.assertTrue(callable(curate_main))
            logger.info("CLI entry points exist and are callable")
        except ImportError as e:
            self.fail(f"Failed to import CLI entry points: {e}")
    
    def test_data_structure(self):
        """Test that the data directory structure is correct."""
        # Check that key data subdirectories exist
        self.assertTrue((DATA_DIR / "checklist").exists())
        self.assertTrue((DATA_DIR / "examples").exists())
        self.assertTrue((DATA_DIR / "evaluation").exists())
        
        # Check that there are some checklists
        checklist_dir = DATA_DIR / "checklist"
        self.assertTrue((checklist_dir / "doc-checklist").exists())
        self.assertTrue((checklist_dir / "fig-checklist").exists())
        
        # Check that there are some examples
        examples_dir = DATA_DIR / "examples"
        example_dirs = list(examples_dir.glob("*"))
        self.assertGreater(len(example_dirs), 0, "No examples found")
    
    def test_logging_setup(self):
        """Test that logging is properly configured."""
        # Check that logger is available
        self.assertIsNotNone(logger)
        self.assertTrue(hasattr(logger, 'info'))
        self.assertTrue(hasattr(logger, 'error'))
        self.assertTrue(hasattr(logger, 'debug'))
        
        # Test that we can log messages
        try:
            logger.info("Test logging message")
            logger.debug("Test debug message")
            logger.error("Test error message")
        except Exception as e:
            self.fail(f"Logging failed: {e}")


if __name__ == "__main__":
    unittest.main() 