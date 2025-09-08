"""Gemma VLM implementation for adversarial testing."""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoProcessor
from PIL import Image
from typing import Optional, List, Dict, Any, Union
import logging

from .vlm_loader import VLMLoader, ModelConfig
from ..utils.image_utils import preprocess_image

logger = logging.getLogger(__name__)


class GemmaVLM(VLMLoader):
    """Gemma VLM implementation with HuggingFace Transformers."""
    
    def __init__(self, config_path: str = "config/model_config.yaml"):
        """Initialize Gemma VLM loader."""
        super().__init__(config_path)
        self.generation_config = self._get_generation_config()
    
    def _get_generation_config(self) -> Dict[str, Any]:
        """Get text generation configuration."""
        runtime = self.config.get("runtime", {})
        return {
            "max_new_tokens": runtime.get("max_new_tokens", 512),
            "temperature": runtime.get("temperature", 0.1),
            "do_sample": runtime.get("do_sample", False),
            "pad_token_id": runtime.get("pad_token_id", 0),
        }
    
    def _load_specific_model(self, model_key: str) -> bool:
        """Load a specific Gemma model.
        
        Args:
            model_key: Model configuration key
            
        Returns:
            True if loading successful
        """
        try:
            model_config = self.get_model_config(model_key)
            
            logger.info(f"Loading Gemma model: {model_config.model_name}")
            
            # Load tokenizer
            logger.info("Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_config.model_name,
                trust_remote_code=model_config.trust_remote_code,
                cache_dir=model_config.cache_dir
            )
            
            # Set pad token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model
            logger.info("Loading model...")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_config.model_name,
                torch_dtype=model_config.get_torch_dtype(),
                device_map=model_config.device_map,
                trust_remote_code=model_config.trust_remote_code,
                cache_dir=model_config.cache_dir,
                low_cpu_mem_usage=True
            )
            
            # Enable gradient checkpointing if configured
            runtime = self.config.get("runtime", {})
            if runtime.get("enable_gradient_checkpointing", True):
                self.model.gradient_checkpointing_enable()
            
            # Try to load processor if available (for multimodal models)
            try:
                logger.info("Attempting to load processor...")
                self.processor = AutoProcessor.from_pretrained(
                    model_config.model_name,
                    trust_remote_code=model_config.trust_remote_code,
                    cache_dir=model_config.cache_dir
                )
                logger.info("Processor loaded successfully")
            except Exception as e:
                logger.info(f"No processor available for this model: {e}")
                self.processor = None
            
            logger.info(f"Successfully loaded Gemma model: {model_config.model_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load Gemma model {model_key}: {e}")
            self._cleanup_model()
            return False
    
    def generate_text(
        self, 
        prompt: str, 
        image: Optional[Union[Image.Image, torch.Tensor]] = None,
        **generation_kwargs
    ) -> str:
        """Generate text response from prompt and optional image.
        
        Args:
            prompt: Text prompt
            image: Optional image input
            **generation_kwargs: Additional generation parameters
            
        Returns:
            Generated text response
        """
        if not self.is_model_loaded():
            raise RuntimeError("No model loaded. Call load_model_with_fallback() first.")
        
        try:
            # Merge generation config
            gen_config = {**self.generation_config, **generation_kwargs}
            
            # Handle multimodal input if processor is available
            if self.processor is not None and image is not None:
                return self._generate_multimodal(prompt, image, **gen_config)
            else:
                return self._generate_text_only(prompt, **gen_config)
                
        except Exception as e:
            logger.error(f"Text generation failed: {e}")
            raise
    
    def _generate_multimodal(
        self, 
        prompt: str, 
        image: Union[Image.Image, torch.Tensor],
        **generation_kwargs
    ) -> str:
        """Generate text with multimodal input."""
        # Ensure image is PIL Image
        if isinstance(image, torch.Tensor):
            # Convert tensor to PIL Image
            from torchvision.transforms import ToPILImage
            to_pil = ToPILImage()
            image = to_pil(image.squeeze(0) if image.dim() == 4 else image)
        
        # Process inputs
        inputs = self.processor(
            text=prompt,
            images=image,
            return_tensors="pt",
            padding=True
        )
        
        # Move to model device
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                 for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                **generation_kwargs
            )
        
        # Decode response
        response = self.processor.decode(outputs[0], skip_special_tokens=True)
        
        # Clean up prompt from response
        if prompt in response:
            response = response.replace(prompt, "").strip()
        
        return response
    
    def _generate_text_only(self, prompt: str, **generation_kwargs) -> str:
        """Generate text from text-only prompt."""
        # Tokenize input
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        
        # Move to model device
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                **generation_kwargs
            )
        
        # Decode response
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Clean up prompt from response
        if prompt in response:
            response = response.replace(prompt, "").strip()
        
        return response
    
    def test_inference(self, test_prompt: str = "Hello, how are you?") -> Dict[str, Any]:
        """Test basic inference functionality.
        
        Args:
            test_prompt: Simple test prompt
            
        Returns:
            Dict with test results
        """
        if not self.is_model_loaded():
            return {
                "success": False,
                "error": "No model loaded"
            }
        
        try:
            logger.info(f"Testing inference with prompt: '{test_prompt}'")
            
            start_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
            end_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
            
            import time
            cpu_start = time.time()
            
            if torch.cuda.is_available() and start_time:
                start_time.record()
            
            # Generate response
            response = self.generate_text(
                test_prompt,
                max_new_tokens=50,  # Keep it short for testing
                temperature=0.1
            )
            
            if torch.cuda.is_available() and end_time:
                end_time.record()
                torch.cuda.synchronize()
                gpu_time = start_time.elapsed_time(end_time) / 1000.0  # Convert to seconds
            else:
                gpu_time = None
            
            cpu_time = time.time() - cpu_start
            
            result = {
                "success": True,
                "prompt": test_prompt,
                "response": response,
                "response_length": len(response),
                "cpu_time_seconds": cpu_time,
                "model_info": self.get_model_info()
            }
            
            if gpu_time is not None:
                result["gpu_time_seconds"] = gpu_time
            
            logger.info(f"Inference test successful. Response: '{response[:100]}...'")
            return result
            
        except Exception as e:
            logger.error(f"Inference test failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "prompt": test_prompt
            }