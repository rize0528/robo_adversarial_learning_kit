"""Base VLM model loader with memory management and fallback support."""

import yaml
import torch
from dataclasses import dataclass
from typing import Optional, Dict, Any, List
import logging
from pathlib import Path

from ..utils.memory_utils import check_memory_requirements, log_memory_usage, clear_gpu_memory

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuration for VLM model loading."""
    model_name: str
    model_type: str
    max_memory_gb: float
    torch_dtype: str = "float16"
    device_map: str = "auto"
    trust_remote_code: bool = True
    cache_dir: str = "./models/cache"
    
    def get_torch_dtype(self) -> torch.dtype:
        """Convert string dtype to torch dtype."""
        dtype_map = {
            "float16": torch.float16,
            "float32": torch.float32,
            "bfloat16": torch.bfloat16
        }
        return dtype_map.get(self.torch_dtype, torch.float16)


class VLMLoader:
    """Base class for loading and managing VLM models with memory management."""
    
    def __init__(self, config_path: str = "config/model_config.yaml"):
        """Initialize VLM loader with configuration.
        
        Args:
            config_path: Path to model configuration YAML file
        """
        self.config_path = config_path
        self.config = self._load_config()
        self.model = None
        self.processor = None
        self.tokenizer = None
        self.current_model_config = None
        
        # Setup logging
        self._setup_logging()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            config_file = Path(self.config_path)
            if not config_file.exists():
                logger.warning(f"Config file {self.config_path} not found, using defaults")
                return self._get_default_config()
                
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Loaded config from {self.config_path}")
            return config
            
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration if config file is missing."""
        return {
            "models": {
                "gemma_4b": {
                    "model_name": "google/gemma-2-9b-it",
                    "model_type": "gemma", 
                    "max_memory_gb": 12,
                    "torch_dtype": "float16"
                }
            },
            "runtime": {
                "memory_threshold_gb": 16,
                "check_memory_before_load": True,
                "max_new_tokens": 512,
                "temperature": 0.1
            },
            "model_priority": ["gemma_4b"]
        }
    
    def _setup_logging(self):
        """Setup logging configuration."""
        log_config = self.config.get("logging", {})
        log_level = log_config.get("level", "INFO")
        
        # Create logs directory if specified
        log_file = log_config.get("file")
        if log_file:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Configure logger
        logger.setLevel(getattr(logging, log_level))
        
    def get_available_models(self) -> List[str]:
        """Get list of available model configurations."""
        return list(self.config.get("models", {}).keys())
    
    def get_model_config(self, model_key: str) -> ModelConfig:
        """Get model configuration by key."""
        model_configs = self.config.get("models", {})
        if model_key not in model_configs:
            raise ValueError(f"Model {model_key} not found in configuration")
        
        model_data = model_configs[model_key]
        return ModelConfig(**model_data)
    
    def check_model_memory_requirements(self, model_key: str) -> tuple[bool, str]:
        """Check if system can load the specified model.
        
        Args:
            model_key: Model configuration key
            
        Returns:
            Tuple of (can_load, message)
        """
        try:
            model_config = self.get_model_config(model_key)
            return check_memory_requirements(model_config.max_memory_gb)
        except Exception as e:
            return False, f"Failed to check requirements: {e}"
    
    def load_model_with_fallback(self) -> bool:
        """Load model using priority list with fallback.
        
        Returns:
            True if any model loaded successfully
        """
        model_priority = self.config.get("model_priority", [])
        runtime_config = self.config.get("runtime", {})
        
        if runtime_config.get("check_memory_before_load", True):
            log_memory_usage("before_model_loading")
        
        for model_key in model_priority:
            logger.info(f"Attempting to load model: {model_key}")
            
            # Check memory requirements
            can_load, memory_msg = self.check_model_memory_requirements(model_key)
            logger.info(f"Memory check for {model_key}: {memory_msg}")
            
            if not can_load:
                logger.warning(f"Skipping {model_key} due to memory constraints")
                continue
            
            # Attempt to load model
            try:
                success = self._load_specific_model(model_key)
                if success:
                    logger.info(f"Successfully loaded model: {model_key}")
                    self.current_model_config = self.get_model_config(model_key)
                    
                    if runtime_config.get("check_memory_before_load", True):
                        log_memory_usage("after_model_loading")
                    
                    return True
                    
            except Exception as e:
                logger.error(f"Failed to load {model_key}: {e}")
                # Clear any partial loading
                self._cleanup_model()
                clear_gpu_memory()
                continue
        
        logger.error("Failed to load any model from priority list")
        return False
    
    def _load_specific_model(self, model_key: str) -> bool:
        """Load a specific model. To be implemented by subclasses.
        
        Args:
            model_key: Model configuration key
            
        Returns:
            True if loading successful
        """
        raise NotImplementedError("Subclasses must implement _load_specific_model")
    
    def _cleanup_model(self):
        """Clean up partially loaded model resources."""
        if self.model is not None:
            del self.model
            self.model = None
        if self.processor is not None:
            del self.processor
            self.processor = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
    
    def is_model_loaded(self) -> bool:
        """Check if a model is currently loaded."""
        return self.model is not None
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about currently loaded model."""
        if not self.is_model_loaded():
            return {"status": "no_model_loaded"}
        
        info = {
            "status": "loaded",
            "model_type": self.current_model_config.model_type if self.current_model_config else "unknown",
            "model_name": self.current_model_config.model_name if self.current_model_config else "unknown"
        }
        
        # Add device information
        if hasattr(self.model, 'device'):
            info["device"] = str(self.model.device)
        
        return info