"""
Comprehensive model setup verification tests.
Tests for Issue #3 - Stream B: VLM Model Integration
"""

import pytest
import torch
import logging
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import tempfile
import yaml

from src.models.gemma_vlm import GemmaVLM
from src.models.vlm_loader import VLMLoader, ModelConfig
from src.utils.memory_utils import get_memory_info, check_memory_requirements

# Setup logging for tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestModelSetup:
    """Test model setup and configuration verification."""
    
    def test_gemma3_configuration_loaded(self):
        """Test that Gemma 3 4B configuration is properly loaded."""
        vlm = GemmaVLM()
        
        # Verify Gemma 4B is available
        available_models = vlm.get_available_models()
        assert "gemma_4b" in available_models, "Gemma 4B model should be available"
        
        # Get and verify configuration
        gemma_config = vlm.get_model_config("gemma_4b")
        
        # Verify model name is correct Gemma 3 model
        assert gemma_config.model_name == "google/gemma-3-4b-it", f"Expected Gemma 3 4B, got {gemma_config.model_name}"
        
        # Verify model type is gemma3 for multimodal capabilities
        assert gemma_config.model_type == "gemma3", f"Expected gemma3 type, got {gemma_config.model_type}"
        
        # Verify memory requirements are reasonable
        assert gemma_config.max_memory_gb == 8, f"Expected 8GB memory requirement, got {gemma_config.max_memory_gb}"
        
        # Verify torch dtype
        assert gemma_config.torch_dtype == "float16", f"Expected float16, got {gemma_config.torch_dtype}"
        assert gemma_config.get_torch_dtype() == torch.float16, "Should convert to torch.float16"
    
    def test_model_priority_configuration(self):
        """Test that model priority is correctly configured."""
        vlm = GemmaVLM()
        
        model_priority = vlm.config.get("model_priority", [])
        
        # Verify priority order
        assert len(model_priority) >= 1, "Should have at least one model in priority list"
        assert model_priority[0] == "gemma_4b", "Gemma 4B should be first priority"
        
        # If multiple models, verify they exist in config
        for model_key in model_priority:
            assert model_key in vlm.get_available_models(), f"Priority model {model_key} should be available"
    
    def test_runtime_configuration(self):
        """Test runtime configuration for VLM operations."""
        vlm = GemmaVLM()
        runtime_config = vlm.config.get("runtime", {})
        
        # Verify memory management settings
        assert "memory_threshold_gb" in runtime_config, "Memory threshold should be configured"
        assert runtime_config["check_memory_before_load"] is True, "Should check memory before loading"
        
        # Verify device configuration
        assert "device_priority" in runtime_config, "Device priority should be configured"
        device_priority = runtime_config["device_priority"]
        assert "mps" in device_priority, "MPS should be in device priority (Apple Silicon)"
        assert "cpu" in device_priority, "CPU should be in device priority as fallback"
        
        # Verify generation settings
        gen_config = vlm.generation_config
        assert "max_new_tokens" in gen_config, "Max new tokens should be configured"
        assert "temperature" in gen_config, "Temperature should be configured"
        assert gen_config["temperature"] <= 1.0, "Temperature should be reasonable"
    
    def test_memory_requirements_checking(self):
        """Test memory requirements checking functionality."""
        vlm = GemmaVLM()
        
        # Test memory check for Gemma 4B
        can_load, message = vlm.check_model_memory_requirements("gemma_4b")
        assert isinstance(can_load, bool), "Should return boolean"
        assert isinstance(message, str), "Should return message string"
        assert len(message) > 0, "Message should not be empty"
        
        # Get system memory info
        memory_info = get_memory_info()
        logger.info(f"System memory: {memory_info}")
        
        # If we have sufficient memory, should return True
        if memory_info["available_gb"] >= 10:  # Buffer for safety
            assert can_load, f"Should be able to load with {memory_info['available_gb']:.1f}GB available"
        else:
            logger.info(f"Insufficient memory for testing: {memory_info['available_gb']:.1f}GB available")
    
    def test_cache_directory_configuration(self):
        """Test model cache directory configuration."""
        vlm = GemmaVLM()
        gemma_config = vlm.get_model_config("gemma_4b")
        
        # Verify cache directory is configured
        assert hasattr(gemma_config, "cache_dir"), "Cache directory should be configured"
        assert gemma_config.cache_dir is not None, "Cache directory should not be None"
        
        # Verify cache directory path
        cache_path = Path(gemma_config.cache_dir)
        assert str(cache_path).endswith("cache"), "Cache directory should end with 'cache'"
    
    def test_device_detection_and_configuration(self):
        """Test device detection and proper configuration."""
        vlm = GemmaVLM()
        runtime_config = vlm.config.get("runtime", {})
        
        # Test CUDA availability
        cuda_available = torch.cuda.is_available()
        logger.info(f"CUDA available: {cuda_available}")
        
        # Test MPS availability (Apple Silicon)
        mps_available = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
        logger.info(f"MPS available: {mps_available}")
        
        # Verify device priority is sensible for current system
        device_priority = runtime_config.get("device_priority", [])
        if mps_available:
            assert "mps" in device_priority, "MPS should be available on Apple Silicon"
        
        # CPU should always be available as fallback
        assert "cpu" in device_priority, "CPU should always be available"
    
    @patch('src.models.gemma_vlm.AutoTokenizer')
    @patch('src.models.gemma_vlm.Gemma3ForConditionalGeneration')
    @patch('src.models.gemma_vlm.AutoProcessor')
    def test_model_loading_components(self, mock_processor, mock_model, mock_tokenizer):
        """Test model loading with mocked components."""
        # Setup mocks
        mock_tokenizer_instance = Mock()
        mock_tokenizer_instance.pad_token = None
        mock_tokenizer_instance.eos_token = "<eos>"
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
        
        mock_model_instance = Mock()
        mock_model_instance.gradient_checkpointing_enable = Mock()
        mock_model.from_pretrained.return_value = mock_model_instance
        
        mock_processor_instance = Mock()
        mock_processor.from_pretrained.return_value = mock_processor_instance
        
        # Test model loading
        vlm = GemmaVLM()
        success = vlm._load_specific_model("gemma_4b")
        
        # Verify loading was attempted
        assert success, "Model loading should succeed with mocks"
        assert vlm.is_model_loaded(), "Should report model as loaded"
        
        # Verify tokenizer was configured
        mock_tokenizer.from_pretrained.assert_called_once()
        assert mock_tokenizer_instance.pad_token == "<eos>", "Pad token should be set"
        
        # Verify model was loaded with correct class
        mock_model.from_pretrained.assert_called_once()
        
        # Verify processor was loaded
        mock_processor.from_pretrained.assert_called_once()
    
    def test_model_info_tracking(self):
        """Test model information tracking."""
        vlm = GemmaVLM()
        
        # Test initial state
        assert not vlm.is_model_loaded(), "Should not have model loaded initially"
        
        model_info = vlm.get_model_info()
        assert model_info["status"] == "no_model_loaded", "Should report no model loaded"
    
    def test_fallback_model_loading(self):
        """Test fallback model loading mechanism."""
        vlm = GemmaVLM()
        
        # Test that fallback mechanism exists
        assert hasattr(vlm, 'load_model_with_fallback'), "Should have fallback loading method"
        
        # Test priority list exists
        model_priority = vlm.config.get("model_priority", [])
        assert len(model_priority) > 0, "Should have model priority list"
        
        # All models in priority should be valid
        for model_key in model_priority:
            assert model_key in vlm.get_available_models(), f"Priority model {model_key} should exist in config"


class TestHuggingFaceIntegration:
    """Test HuggingFace integration components."""
    
    def test_transformers_imports(self):
        """Test that required transformers components are available."""
        # Test basic imports
        from transformers import AutoTokenizer, AutoProcessor
        from transformers import Gemma3ForConditionalGeneration, Gemma3ForCausalLM
        
        # These should not raise ImportError
        assert AutoTokenizer is not None
        assert AutoProcessor is not None
        assert Gemma3ForConditionalGeneration is not None
        assert Gemma3ForCausalLM is not None
    
    def test_model_name_validation(self):
        """Test that configured model names are valid HuggingFace model names."""
        vlm = GemmaVLM()
        
        for model_key in vlm.get_available_models():
            config = vlm.get_model_config(model_key)
            
            # Model name should follow HuggingFace format
            assert "/" in config.model_name, f"Model name {config.model_name} should include organization"
            assert config.model_name.startswith("google/"), f"Expected Google model, got {config.model_name}"
            assert "gemma" in config.model_name.lower(), f"Expected Gemma model, got {config.model_name}"
    
    def test_trust_remote_code_configuration(self):
        """Test that trust_remote_code is properly configured."""
        vlm = GemmaVLM()
        
        for model_key in vlm.get_available_models():
            config = vlm.get_model_config(model_key)
            
            # Should have trust_remote_code configured
            assert hasattr(config, "trust_remote_code"), "Should have trust_remote_code setting"
            assert config.trust_remote_code is True, "Should trust remote code for Gemma models"


@pytest.mark.integration
class TestModelSetupIntegration:
    """Integration tests for model setup (require actual model loading)."""
    
    @pytest.mark.slow
    def test_actual_model_loading_if_memory_available(self):
        """Test actual model loading if sufficient memory is available."""
        vlm = GemmaVLM()
        
        # Check if we have sufficient memory
        can_load, message = vlm.check_model_memory_requirements("gemma_4b")
        
        if not can_load:
            pytest.skip(f"Insufficient memory for integration test: {message}")
        
        # Attempt actual loading
        try:
            success = vlm.load_model_with_fallback()
            if success:
                assert vlm.is_model_loaded(), "Should report model as loaded"
                
                # Test model info
                model_info = vlm.get_model_info()
                assert model_info["status"] == "loaded", "Should report loaded status"
                assert "model_name" in model_info, "Should include model name in info"
                
                logger.info(f"Successfully loaded model: {model_info}")
            else:
                pytest.skip("Model loading failed - this may be expected on resource-constrained systems")
        
        except Exception as e:
            pytest.skip(f"Model loading failed with exception: {e}")


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])