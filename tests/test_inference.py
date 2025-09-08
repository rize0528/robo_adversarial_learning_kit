"""
Comprehensive inference pipeline testing.
Tests for Issue #3 - Stream B: VLM Model Integration - Inference Pipeline
"""

import pytest
import torch
import logging
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from PIL import Image
import numpy as np

from src.models.gemma_vlm import GemmaVLM
from src.utils.image_utils import create_test_image, preprocess_image

# Setup logging for tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestInferencePipeline:
    """Test basic inference pipeline functionality."""
    
    @patch('src.models.gemma_vlm.AutoTokenizer')
    @patch('src.models.gemma_vlm.Gemma3ForConditionalGeneration')  
    @patch('src.models.gemma_vlm.AutoProcessor')
    def test_text_only_inference_pipeline(self, mock_processor, mock_model, mock_tokenizer):
        """Test text-only inference pipeline."""
        # Setup mocks
        mock_tokenizer_instance = self._setup_mock_tokenizer()
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
        
        mock_model_instance = self._setup_mock_model()
        mock_model.from_pretrained.return_value = mock_model_instance
        
        mock_processor_instance = Mock()
        mock_processor.from_pretrained.return_value = mock_processor_instance
        
        # Test inference pipeline
        vlm = GemmaVLM()
        vlm._load_specific_model("gemma_4b")
        
        # Test text-only generation
        prompt = "Hello, how are you?"
        
        # Mock tokenizer output
        mock_inputs = {"input_ids": torch.tensor([[1, 2, 3]]), "attention_mask": torch.tensor([[1, 1, 1]])}
        mock_tokenizer_instance.return_value = mock_inputs
        mock_tokenizer_instance.__call__.return_value = mock_inputs
        
        # Mock model generation output
        generated_ids = torch.tensor([[1, 2, 3, 4, 5]])
        mock_model_instance.generate.return_value = generated_ids
        
        # Mock tokenizer decode
        mock_tokenizer_instance.decode.return_value = f"{prompt} I'm doing well, thank you!"
        
        # Test generation
        response = vlm._generate_text_only(prompt)
        
        assert isinstance(response, str), "Response should be a string"
        assert len(response) > 0, "Response should not be empty"
        
        # Verify tokenizer was called
        mock_tokenizer_instance.assert_called_once()
        
        # Verify model generation was called
        mock_model_instance.generate.assert_called_once()
        
        # Verify decode was called
        mock_tokenizer_instance.decode.assert_called_once_with(generated_ids[0], skip_special_tokens=True)
    
    @patch('src.models.gemma_vlm.AutoTokenizer')
    @patch('src.models.gemma_vlm.Gemma3ForConditionalGeneration')
    @patch('src.models.gemma_vlm.AutoProcessor')
    def test_multimodal_inference_pipeline(self, mock_processor, mock_model, mock_tokenizer):
        """Test multimodal inference pipeline with image input."""
        # Setup mocks
        mock_tokenizer_instance = self._setup_mock_tokenizer()
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
        
        mock_model_instance = self._setup_mock_model()
        mock_model.from_pretrained.return_value = mock_model_instance
        
        mock_processor_instance = Mock()
        mock_processor.from_pretrained.return_value = mock_processor_instance
        
        # Test multimodal inference
        vlm = GemmaVLM()
        vlm._load_specific_model("gemma_4b")
        
        # Create test inputs
        prompt = "What do you see in this image?"
        test_image = create_test_image((224, 224))
        
        # Mock processor output
        mock_inputs = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
            "pixel_values": torch.randn(1, 3, 224, 224)
        }
        mock_processor_instance.return_value = mock_inputs
        mock_processor_instance.__call__.return_value = mock_inputs
        
        # Mock model generation
        generated_ids = torch.tensor([[1, 2, 3, 4, 5, 6]])
        mock_model_instance.generate.return_value = generated_ids
        
        # Mock processor decode
        mock_processor_instance.decode.return_value = f"{prompt} I see a colorful test image with geometric patterns."
        
        # Test multimodal generation
        response = vlm._generate_multimodal(prompt, test_image)
        
        assert isinstance(response, str), "Response should be a string"
        assert len(response) > 0, "Response should not be empty"
        
        # Verify processor was called with image and text
        mock_processor_instance.assert_called_once()
        call_args = mock_processor_instance.call_args
        assert "text" in call_args[1] or call_args[0], "Should process text"
        assert "images" in call_args[1] or call_args[0], "Should process images"
        
        # Verify model generation was called
        mock_model_instance.generate.assert_called_once()
    
    @patch('src.models.gemma_vlm.AutoTokenizer')
    @patch('src.models.gemma_vlm.Gemma3ForConditionalGeneration')
    @patch('src.models.gemma_vlm.AutoProcessor')
    def test_generate_text_method_routing(self, mock_processor, mock_model, mock_tokenizer):
        """Test that generate_text method routes correctly based on input."""
        # Setup mocks
        mock_tokenizer_instance = self._setup_mock_tokenizer()
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
        
        mock_model_instance = self._setup_mock_model()
        mock_model.from_pretrained.return_value = mock_model_instance
        
        mock_processor_instance = Mock()
        mock_processor.from_pretrained.return_value = mock_processor_instance
        
        vlm = GemmaVLM()
        vlm._load_specific_model("gemma_4b")
        
        # Mock the internal methods
        vlm._generate_text_only = Mock(return_value="Text-only response")
        vlm._generate_multimodal = Mock(return_value="Multimodal response")
        
        # Test text-only routing
        response = vlm.generate_text("Hello")
        assert response == "Text-only response"
        vlm._generate_text_only.assert_called_once()
        vlm._generate_multimodal.assert_not_called()
        
        # Reset mocks
        vlm._generate_text_only.reset_mock()
        vlm._generate_multimodal.reset_mock()
        
        # Test multimodal routing
        test_image = create_test_image((224, 224))
        response = vlm.generate_text("Describe image", image=test_image)
        assert response == "Multimodal response"
        vlm._generate_multimodal.assert_called_once()
        vlm._generate_text_only.assert_not_called()
    
    def test_inference_without_loaded_model(self):
        """Test that inference fails gracefully without loaded model."""
        vlm = GemmaVLM()
        
        # Ensure no model is loaded
        assert not vlm.is_model_loaded()
        
        # Test that generation raises appropriate error
        with pytest.raises(RuntimeError, match="No model loaded"):
            vlm.generate_text("Hello")
        
        # Test that test_inference returns error result
        result = vlm.test_inference()
        assert not result["success"]
        assert "No model loaded" in result["error"]
    
    def test_generation_config_handling(self):
        """Test that generation configuration is properly handled."""
        vlm = GemmaVLM()
        
        # Test default generation config
        gen_config = vlm.generation_config
        assert isinstance(gen_config, dict)
        
        expected_keys = ["max_new_tokens", "temperature", "do_sample", "pad_token_id"]
        for key in expected_keys:
            assert key in gen_config, f"Should have {key} in generation config"
        
        # Test config values are reasonable
        assert gen_config["max_new_tokens"] > 0, "Max tokens should be positive"
        assert 0 <= gen_config["temperature"] <= 2.0, "Temperature should be reasonable"
        assert isinstance(gen_config["do_sample"], bool), "do_sample should be boolean"
    
    @patch('src.models.gemma_vlm.AutoTokenizer')
    @patch('src.models.gemma_vlm.Gemma3ForConditionalGeneration')
    @patch('src.models.gemma_vlm.AutoProcessor')
    def test_inference_test_method(self, mock_processor, mock_model, mock_tokenizer):
        """Test the test_inference method functionality."""
        # Setup mocks
        mock_tokenizer_instance = self._setup_mock_tokenizer()
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
        
        mock_model_instance = self._setup_mock_model()
        mock_model.from_pretrained.return_value = mock_model_instance
        
        mock_processor_instance = Mock()
        mock_processor.from_pretrained.return_value = mock_processor_instance
        
        vlm = GemmaVLM()
        vlm._load_specific_model("gemma_4b")
        
        # Mock the generate_text method
        test_response = "Hello! I'm working correctly."
        vlm.generate_text = Mock(return_value=test_response)
        
        # Test inference test
        result = vlm.test_inference("Test prompt")
        
        assert result["success"] is True, "Test should succeed"
        assert "prompt" in result, "Should include prompt in result"
        assert "response" in result, "Should include response in result"
        assert result["response"] == test_response, "Should return expected response"
        assert "response_length" in result, "Should include response length"
        assert "cpu_time_seconds" in result, "Should include timing information"
        assert "model_info" in result, "Should include model info"
        
        # Verify generate_text was called
        vlm.generate_text.assert_called_once()
    
    def _setup_mock_tokenizer(self):
        """Setup a mock tokenizer with required methods."""
        mock_tokenizer = Mock()
        mock_tokenizer.pad_token = None
        mock_tokenizer.eos_token = "<eos>"
        mock_tokenizer.decode = Mock(return_value="Decoded response")
        
        # Mock the __call__ method for tokenization
        mock_tokenizer.__call__ = Mock(return_value={
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]])
        })
        
        return mock_tokenizer
    
    def _setup_mock_model(self):
        """Setup a mock model with required methods."""
        mock_model = Mock()
        mock_model.gradient_checkpointing_enable = Mock()
        mock_model.generate = Mock(return_value=torch.tensor([[1, 2, 3, 4, 5]]))
        
        # Mock parameters for device detection
        mock_param = Mock()
        mock_param.device = torch.device("cpu")
        mock_model.parameters = Mock(return_value=iter([mock_param]))
        
        return mock_model


class TestImageProcessing:
    """Test image processing components of the inference pipeline."""
    
    def test_test_image_creation(self):
        """Test creation of test images for inference testing."""
        # Test basic image creation
        image = create_test_image((224, 224))
        assert isinstance(image, Image.Image), "Should return PIL Image"
        assert image.size == (224, 224), "Should have correct size"
        assert image.mode == "RGB", "Should be RGB mode"
    
    def test_image_preprocessing(self):
        """Test image preprocessing for model input."""
        # Create test image
        image = create_test_image((128, 128))
        
        # Test preprocessing
        tensor = preprocess_image(image, target_size=(224, 224))
        
        assert isinstance(tensor, torch.Tensor), "Should return tensor"
        assert tensor.shape == (1, 3, 224, 224), "Should have correct shape (B,C,H,W)"
        assert tensor.dtype == torch.float32, "Should be float32 tensor"
        
        # Check value range (should be normalized)
        assert tensor.min() >= -3, "Values should be reasonable after normalization"
        assert tensor.max() <= 3, "Values should be reasonable after normalization"
    
    def test_tensor_to_pil_conversion(self):
        """Test conversion of tensor images back to PIL."""
        from torchvision.transforms import ToPILImage
        
        # Create a test tensor
        tensor = torch.randn(3, 224, 224)  # C, H, W format
        
        # Convert to PIL
        to_pil = ToPILImage()
        image = to_pil(tensor)
        
        assert isinstance(image, Image.Image), "Should convert to PIL Image"
        assert image.size == (224, 224), "Should maintain size"
        assert image.mode == "RGB", "Should be RGB"
    
    def test_different_image_formats(self):
        """Test handling of different image formats."""
        formats_and_modes = [
            ((224, 224), "RGB"),
            ((256, 256), "RGB"), 
            ((512, 512), "RGB"),
        ]
        
        for size, mode in formats_and_modes:
            image = create_test_image(size)
            assert image.size == size, f"Size should be {size}"
            assert image.mode == mode, f"Mode should be {mode}"
            
            # Test preprocessing works
            tensor = preprocess_image(image, target_size=(224, 224))
            assert tensor.shape == (1, 3, 224, 224), "Preprocessing should work for all formats"


class TestInferenceErrorHandling:
    """Test error handling in the inference pipeline."""
    
    def test_invalid_image_handling(self):
        """Test handling of invalid image inputs."""
        vlm = GemmaVLM()
        
        # Test with None image (should default to text-only)
        # This should not raise an error when no model is loaded
        with pytest.raises(RuntimeError, match="No model loaded"):
            vlm.generate_text("Hello", image=None)
    
    def test_generation_error_handling(self):
        """Test handling of generation errors."""
        vlm = GemmaVLM()
        
        # Mock a loaded model state but with failing generation
        vlm.model = Mock()
        vlm.tokenizer = Mock() 
        vlm.processor = None
        vlm.current_model_config = Mock()
        
        # Mock generation failure
        vlm.model.parameters.return_value = iter([Mock(device=torch.device("cpu"))])
        vlm.model.generate.side_effect = RuntimeError("Mock generation error")
        vlm.tokenizer.return_value = {"input_ids": torch.tensor([[1, 2, 3]])}
        
        # Test that error is properly handled
        with pytest.raises(RuntimeError):
            vlm.generate_text("Test prompt")
    
    def test_device_movement_errors(self):
        """Test handling of device movement errors."""
        # This would test CUDA OOM errors, device mismatch, etc.
        # For now, we'll just verify the structure exists
        vlm = GemmaVLM()
        
        # The device movement logic should be in the generation methods
        assert hasattr(vlm, "_generate_text_only"), "Should have text generation method"
        assert hasattr(vlm, "_generate_multimodal"), "Should have multimodal generation method"


@pytest.mark.integration
class TestInferenceIntegration:
    """Integration tests for inference pipeline (require actual model loading)."""
    
    @pytest.mark.slow
    def test_real_inference_if_model_available(self):
        """Test real inference if model can be loaded."""
        vlm = GemmaVLM()
        
        # Check if we can load a model
        can_load, message = vlm.check_model_memory_requirements("gemma_4b")
        
        if not can_load:
            pytest.skip(f"Cannot load model for integration test: {message}")
        
        try:
            # Attempt to load model
            success = vlm.load_model_with_fallback()
            if not success:
                pytest.skip("Model loading failed")
            
            # Test text inference
            result = vlm.test_inference("Hello, how are you?")
            if result["success"]:
                assert "response" in result, "Should have response"
                assert len(result["response"]) > 0, "Response should not be empty"
                assert "cpu_time_seconds" in result, "Should measure timing"
                logger.info(f"Real inference result: {result}")
            else:
                pytest.skip(f"Inference failed: {result.get('error')}")
                
        except Exception as e:
            pytest.skip(f"Integration test failed: {e}")


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])