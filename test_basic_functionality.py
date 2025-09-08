#!/usr/bin/env python3
"""
Basic functionality test script for VLM Adversarial Patch Finder.

This script tests the foundation components without loading actual models,
suitable for development environment verification.
"""

import sys
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_imports():
    """Test that all core modules can be imported."""
    logger.info("Testing module imports...")
    
    try:
        from src.models.vlm_loader import VLMLoader, ModelConfig
        from src.models.gemma_vlm import GemmaVLM
        from src.utils.memory_utils import get_memory_info, check_memory_requirements
        from src.utils.image_utils import create_test_image, preprocess_image
        logger.info("✓ All core modules imported successfully")
        return True
    except ImportError as e:
        logger.error(f"✗ Import failed: {e}")
        return False


def test_configuration():
    """Test configuration loading."""
    logger.info("Testing configuration loading...")
    
    try:
        vlm = GemmaVLM()
        
        # Test config structure
        assert "models" in vlm.config
        assert "runtime" in vlm.config
        assert "model_priority" in vlm.config
        
        # Test model availability
        models = vlm.get_available_models()
        assert len(models) > 0
        logger.info(f"✓ Found {len(models)} available model configurations: {models}")
        
        # Test model config retrieval
        first_model = models[0]
        model_config = vlm.get_model_config(first_model)
        assert isinstance(model_config, ModelConfig)
        logger.info(f"✓ Successfully loaded config for {first_model}")
        
        return True
    except Exception as e:
        logger.error(f"✗ Configuration test failed: {e}")
        return False


def test_memory_utilities():
    """Test memory management utilities."""
    logger.info("Testing memory utilities...")
    
    try:
        from src.utils.memory_utils import get_memory_info, check_memory_requirements
        
        # Test memory info retrieval
        memory_info = get_memory_info()
        required_keys = ["total_gb", "available_gb", "used_gb", "percent_used"]
        
        for key in required_keys:
            assert key in memory_info
            assert isinstance(memory_info[key], (int, float))
        
        logger.info(f"✓ System memory: {memory_info['used_gb']:.1f}GB / {memory_info['total_gb']:.1f}GB ({memory_info['percent_used']:.1f}%)")
        
        # Test memory requirement checking
        can_load_small, msg_small = check_memory_requirements(1.0)  # 1GB
        can_load_huge, msg_huge = check_memory_requirements(99999.0)  # Impossible
        
        assert isinstance(can_load_small, bool)
        assert isinstance(can_load_huge, bool)
        assert not can_load_huge  # Should be false for huge requirement
        
        logger.info("✓ Memory requirement checking works correctly")
        return True
        
    except Exception as e:
        logger.error(f"✗ Memory utilities test failed: {e}")
        return False


def test_image_processing():
    """Test image processing utilities.""" 
    logger.info("Testing image processing...")
    
    try:
        from src.utils.image_utils import create_test_image, preprocess_image
        import torch
        
        # Test image creation
        image = create_test_image((128, 128))
        assert image.size == (128, 128)
        assert image.mode == "RGB"
        logger.info("✓ Test image creation successful")
        
        # Test image preprocessing
        tensor = preprocess_image(image, target_size=(224, 224), normalize=False)
        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (1, 3, 224, 224)
        logger.info("✓ Image preprocessing successful")
        
        # Test with normalization
        tensor_norm = preprocess_image(image, target_size=(224, 224), normalize=True)
        assert isinstance(tensor_norm, torch.Tensor)
        assert tensor_norm.shape == (1, 3, 224, 224)
        logger.info("✓ Image preprocessing with normalization successful")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Image processing test failed: {e}")
        return False


def test_model_loader_structure():
    """Test model loader structure without loading actual models."""
    logger.info("Testing model loader structure...")
    
    try:
        vlm = GemmaVLM()
        
        # Test initial state
        assert not vlm.is_model_loaded()
        logger.info("✓ Initial model state correct (not loaded)")
        
        # Test model info when no model loaded
        info = vlm.get_model_info()
        assert info["status"] == "no_model_loaded"
        logger.info("✓ Model info reporting works correctly")
        
        # Test generation config
        gen_config = vlm.generation_config
        expected_keys = ["max_new_tokens", "temperature", "do_sample", "pad_token_id"]
        for key in expected_keys:
            assert key in gen_config
        logger.info("✓ Generation config structure correct")
        
        # Test memory checking for models
        models = vlm.get_available_models()
        if models:
            can_load, message = vlm.check_model_memory_requirements(models[0])
            assert isinstance(can_load, bool)
            assert isinstance(message, str)
            logger.info(f"✓ Memory check for {models[0]}: {message}")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Model loader structure test failed: {e}")
        return False


def test_file_structure():
    """Test that required files and directories exist."""
    logger.info("Testing file structure...")
    
    required_files = [
        "requirements.txt",
        "pyproject.toml", 
        "config/model_config.yaml",
        "config/test_config.yaml",
        "src/__init__.py",
        "src/models/__init__.py",
        "src/utils/__init__.py",
        "tests/__init__.py"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        logger.error(f"✗ Missing required files: {missing_files}")
        return False
    
    logger.info("✓ All required files and directories exist")
    return True


def main():
    """Run all basic functionality tests."""
    logger.info("Starting VLM Adversarial Patch Finder basic functionality tests...")
    logger.info("=" * 60)
    
    tests = [
        test_file_structure,
        test_imports,
        test_configuration,
        test_memory_utilities,
        test_image_processing,
        test_model_loader_structure
    ]
    
    passed_tests = 0
    total_tests = len(tests)
    
    for test_func in tests:
        try:
            if test_func():
                passed_tests += 1
            logger.info("-" * 40)
        except Exception as e:
            logger.error(f"✗ Test {test_func.__name__} failed with exception: {e}")
            logger.info("-" * 40)
    
    # Summary
    logger.info("=" * 60)
    logger.info(f"Test Summary: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        logger.info("🎉 All foundation components are working correctly!")
        logger.info("Ready for model loading and inference testing.")
        return True
    else:
        logger.error(f"❌ {total_tests - passed_tests} tests failed.")
        logger.error("Please fix the issues before proceeding.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)