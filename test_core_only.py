#!/usr/bin/env python3
"""
Core-only test script for essential functionality verification.
Tests only the core components without heavy ML libraries.
"""

import sys
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def test_configuration_loading():
    """Test configuration loading without importing ML libraries."""
    logger.info("Testing configuration loading...")
    
    try:
        import yaml
        
        # Test config file exists and loads
        config_file = Path("config/model_config.yaml")
        if not config_file.exists():
            logger.error("Config file not found")
            return False
            
        with open(config_file) as f:
            config = yaml.safe_load(f)
        
        # Test structure
        required_sections = ["models", "runtime", "model_priority"]
        for section in required_sections:
            if section not in config:
                logger.error(f"Missing section: {section}")
                return False
        
        # Test models section  
        models = config["models"]
        if not isinstance(models, dict) or len(models) == 0:
            logger.error("No models configured")
            return False
        
        logger.info(f"✓ Found {len(models)} configured models: {list(models.keys())}")
        
        # Test first model config structure
        first_model = next(iter(models.values()))
        required_fields = ["model_name", "model_type", "max_memory_gb"]
        for field in required_fields:
            if field not in first_model:
                logger.error(f"Missing model field: {field}")
                return False
        
        logger.info("✓ Model configuration structure valid")
        return True
        
    except Exception as e:
        logger.error(f"Configuration test failed: {e}")
        return False


def test_memory_utilities():
    """Test memory utilities without ML libraries."""
    logger.info("Testing memory utilities...")
    
    try:
        import psutil
        
        # Test basic memory info 
        memory = psutil.virtual_memory()
        total_gb = memory.total / (1024**3)
        available_gb = memory.available / (1024**3)
        
        if total_gb <= 0 or available_gb <= 0:
            logger.error("Invalid memory values")
            return False
            
        logger.info(f"✓ System memory: {available_gb:.1f}GB available / {total_gb:.1f}GB total")
        
        # Test memory checking logic (without importing our utilities to avoid torch)
        def check_memory_simple(required_gb, buffer_gb=2.0):
            return available_gb >= (required_gb + buffer_gb)
        
        # Test reasonable requirements
        can_load_4gb = check_memory_simple(4.0)
        can_load_huge = check_memory_simple(99999.0)
        
        logger.info(f"✓ Memory checks: 4GB={can_load_4gb}, 99999GB={can_load_huge}")
        
        if can_load_huge:
            logger.warning("Unexpected: system claims to have 100TB+ memory")
        
        return True
        
    except Exception as e:
        logger.error(f"Memory utilities test failed: {e}")
        return False


def test_project_structure():
    """Test project structure and file organization."""
    logger.info("Testing project structure...")
    
    essential_files = [
        "requirements.txt",
        "pyproject.toml",
        "config/model_config.yaml", 
        "config/test_config.yaml"
    ]
    
    essential_dirs = [
        "src",
        "src/models", 
        "src/utils",
        "tests"
    ]
    
    # Check files
    for file_path in essential_files:
        if not Path(file_path).is_file():
            logger.error(f"Missing file: {file_path}")
            return False
    
    # Check directories
    for dir_path in essential_dirs:
        if not Path(dir_path).is_dir():
            logger.error(f"Missing directory: {dir_path}")
            return False
    
    logger.info("✓ All essential files and directories exist")
    
    # Check Python package structure
    python_files = [
        "src/__init__.py",
        "src/models/__init__.py", 
        "src/utils/__init__.py",
        "tests/__init__.py"
    ]
    
    for py_file in python_files:
        if not Path(py_file).is_file():
            logger.error(f"Missing Python package file: {py_file}")
            return False
    
    logger.info("✓ Python package structure correct")
    return True


def test_yaml_configs():
    """Test YAML configuration files can be loaded."""
    logger.info("Testing YAML configuration files...")
    
    try:
        import yaml
        
        config_files = [
            ("config/model_config.yaml", ["models", "runtime"]),
            ("config/test_config.yaml", ["test_data", "scenarios"])
        ]
        
        for config_file, required_sections in config_files:
            with open(config_file) as f:
                config = yaml.safe_load(f)
            
            for section in required_sections:
                if section not in config:
                    logger.error(f"Missing section '{section}' in {config_file}")
                    return False
            
            logger.info(f"✓ {config_file} loads correctly")
        
        return True
        
    except Exception as e:
        logger.error(f"YAML config test failed: {e}")
        return False


def main():
    """Run core functionality tests."""
    logger.info("=== VLM Adversarial Patch Finder - Core Tests ===")
    logger.info("Testing essential functionality without ML libraries...")
    logger.info("")
    
    tests = [
        ("Project Structure", test_project_structure),
        ("YAML Configurations", test_yaml_configs),
        ("Model Configuration", test_configuration_loading),
        ("Memory Utilities", test_memory_utilities)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        logger.info(f"Running: {test_name}")
        try:
            if test_func():
                passed += 1
                logger.info("PASSED")
            else:
                logger.error("FAILED")
        except Exception as e:
            logger.error(f"FAILED with exception: {e}")
        logger.info("")
    
    logger.info("=" * 50)
    logger.info(f"Results: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 Core foundation is ready!")
        logger.info("Next step: Install ML dependencies and test model loading")
        return True
    else:
        logger.error("❌ Core foundation has issues that need fixing")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)