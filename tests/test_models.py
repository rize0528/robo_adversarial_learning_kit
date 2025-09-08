"""Unit tests for VLM model loading and inference functionality."""

import pytest
import torch
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import tempfile
import yaml
import logging

# Setup logging for tests
logging.basicConfig(level=logging.INFO)

# Import modules to test
from src.models.vlm_loader import VLMLoader, ModelConfig
from src.models.gemma_vlm import GemmaVLM
from src.utils.memory_utils import get_memory_info, check_memory_requirements
from src.utils.image_utils import create_test_image, preprocess_image


class TestModelConfig:
    """Test ModelConfig dataclass functionality."""
    
    def test_model_config_creation(self):
        """Test basic ModelConfig creation."""
        config = ModelConfig(
            model_name="test-model",
            model_type="gemma",
            max_memory_gb=8.0
        )
        
        assert config.model_name == "test-model"
        assert config.model_type == "gemma"
        assert config.max_memory_gb == 8.0
        assert config.torch_dtype == "float16"  # default
        assert config.device_map == "auto"  # default
    
    def test_torch_dtype_conversion(self):
        """Test torch dtype string to tensor dtype conversion."""
        config = ModelConfig(
            model_name="test",
            model_type="test",
            max_memory_gb=1.0,
            torch_dtype="float32"
        )
        
        assert config.get_torch_dtype() == torch.float32
        
        # Test default fallback
        config.torch_dtype = "invalid_dtype"
        assert config.get_torch_dtype() == torch.float16


class TestVLMLoader:
    """Test base VLMLoader functionality."""
    
    def test_default_config_loading(self):
        """Test loading default config when file doesn't exist."""
        loader = VLMLoader(config_path="nonexistent_config.yaml")
        
        assert "models" in loader.config
        assert "runtime" in loader.config
        assert "model_priority" in loader.config
        assert len(loader.config["model_priority"]) > 0
    
    def test_config_file_loading(self):
        """Test loading config from actual file."""
        # Create temporary config file
        config_data = {
            "models": {
                "test_model": {
                    "model_name": "test/model",
                    "model_type": "test",
                    "max_memory_gb": 4.0
                }
            },
            "runtime": {
                "memory_threshold_gb": 8,
                "max_new_tokens": 256
            },
            "model_priority": ["test_model"]
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            temp_path = f.name
        
        try:
            loader = VLMLoader(config_path=temp_path)
            assert loader.config["models"]["test_model"]["model_name"] == "test/model"
            assert loader.config["runtime"]["max_new_tokens"] == 256
        finally:
            Path(temp_path).unlink()
    
    def test_get_available_models(self):
        """Test getting list of available model configurations."""
        loader = VLMLoader()
        models = loader.get_available_models()
        
        assert isinstance(models, list)
        assert len(models) > 0
        assert "gemma_4b" in models
    
    def test_get_model_config(self):
        """Test getting specific model configuration."""
        loader = VLMLoader()
        
        # Test valid model
        config = loader.get_model_config("gemma_4b")
        assert isinstance(config, ModelConfig)
        assert config.model_type == "gemma3"
        
        # Test invalid model
        with pytest.raises(ValueError, match="not found in configuration"):
            loader.get_model_config("nonexistent_model")
    
    def test_check_model_memory_requirements(self):
        """Test memory requirement checking."""
        loader = VLMLoader()
        
        can_load, message = loader.check_model_memory_requirements("gemma_4b")
        assert isinstance(can_load, bool)
        assert isinstance(message, str)
        assert len(message) > 0
    
    def test_model_status_tracking(self):
        """Test model loading status tracking."""
        loader = VLMLoader()
        
        # Initially no model loaded
        assert not loader.is_model_loaded()
        
        info = loader.get_model_info()
        assert info["status"] == "no_model_loaded"


class TestMemoryUtils:
    """Test memory utility functions."""
    
    def test_get_memory_info(self):
        """Test memory information retrieval."""
        memory_info = get_memory_info()
        
        required_keys = ["total_gb", "available_gb", "used_gb", "percent_used"]
        for key in required_keys:
            assert key in memory_info
            assert isinstance(memory_info[key], (int, float))
            assert memory_info[key] >= 0
        
        # Check logical relationships
        assert memory_info["total_gb"] > 0
        assert memory_info["used_gb"] <= memory_info["total_gb"]
        assert 0 <= memory_info["percent_used"] <= 100
    
    def test_check_memory_requirements(self):
        """Test memory requirement checking."""
        # Test with reasonable requirement
        can_load, message = check_memory_requirements(1.0)  # 1GB
        assert isinstance(can_load, bool)
        assert isinstance(message, str)
        
        # Test with impossible requirement
        can_load_huge, message_huge = check_memory_requirements(999999.0)  # 1M GB
        assert not can_load_huge
        assert "Insufficient memory" in message_huge


class TestImageUtils:
    """Test image processing utilities."""
    
    def test_create_test_image(self):
        """Test test image creation."""
        image = create_test_image((100, 100))
        
        assert image.size == (100, 100)
        assert image.mode == "RGB"
    
    def test_preprocess_image_basic(self):
        """Test basic image preprocessing."""
        image = create_test_image((128, 128))
        
        # Test basic preprocessing
        tensor = preprocess_image(image, target_size=(224, 224), normalize=False)
        
        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (1, 3, 224, 224)  # Batch, Channels, Height, Width
        assert 0 <= tensor.min() <= 1
        assert 0 <= tensor.max() <= 1
    
    def test_preprocess_image_normalized(self):
        """Test image preprocessing with normalization."""
        image = create_test_image((64, 64))
        
        tensor = preprocess_image(image, target_size=(224, 224), normalize=True)
        
        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (1, 3, 224, 224)
        # Normalized values can be negative due to ImageNet normalization


class TestGemmaVLMUnit:
    """Unit tests for GemmaVLM (without actual model loading)."""
    
    def test_gemma_vlm_initialization(self):
        """Test GemmaVLM initialization without loading models."""
        vlm = GemmaVLM()
        
        assert not vlm.is_model_loaded()
        assert vlm.generation_config is not None
        assert "max_new_tokens" in vlm.generation_config
    
    def test_generation_config_loading(self):
        """Test generation configuration loading."""
        vlm = GemmaVLM()
        config = vlm.generation_config
        
        expected_keys = ["max_new_tokens", "temperature", "do_sample", "pad_token_id"]
        for key in expected_keys:
            assert key in config
    
    @patch('src.models.gemma_vlm.AutoTokenizer')
    @patch('src.models.gemma_vlm.AutoModelForCausalLM')
    def test_model_loading_success(self, mock_model, mock_tokenizer):
        """Test successful model loading (mocked)."""
        # Setup mocks
        mock_tokenizer.from_pretrained.return_value = Mock()
        mock_model.from_pretrained.return_value = Mock()
        
        vlm = GemmaVLM()
        
        # Mock the model config to use available model
        available_models = vlm.get_available_models()
        if available_models:
            success = vlm._load_specific_model(available_models[0])
            assert success
            assert vlm.is_model_loaded()
    
    def test_text_generation_without_model(self):
        """Test text generation fails gracefully without loaded model."""
        vlm = GemmaVLM()
        
        with pytest.raises(RuntimeError, match="No model loaded"):
            vlm.generate_text("test prompt")
    
    def test_inference_test_without_model(self):
        """Test inference test fails gracefully without loaded model."""
        vlm = GemmaVLM()
        
        result = vlm.test_inference()
        assert not result["success"]
        assert "No model loaded" in result["error"]


# Integration test markers for tests that require actual model loading
@pytest.mark.integration
class TestGemmaVLMIntegration:
    """Integration tests that require actual model loading (run separately)."""
    
    @pytest.mark.slow
    def test_actual_model_loading(self):
        """Test actual model loading - only run if explicitly requested."""
        pytest.skip("Actual model loading test - requires significant resources")
        
        # This test would actually load a model if enabled:
        # vlm = GemmaVLM()
        # success = vlm.load_model_with_fallback()
        # assert success
        # assert vlm.is_model_loaded()
    
    @pytest.mark.slow  
    def test_actual_inference(self):
        """Test actual inference - only run if explicitly requested."""
        pytest.skip("Actual inference test - requires loaded model")
        
        # This test would actually run inference if enabled:
        # vlm = GemmaVLM()
        # vlm.load_model_with_fallback()
        # result = vlm.test_inference("Hello, world!")
        # assert result["success"]
        # assert len(result["response"]) > 0


if __name__ == "__main__":
    # Run unit tests only by default
    pytest.main([__file__, "-v", "-m", "not integration"])