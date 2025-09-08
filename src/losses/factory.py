"""Factory for creating and configuring loss functions.

This module provides a factory pattern implementation for easy creation
and configuration of different loss functions used in adversarial patch
generation.
"""

import torch
from typing import Dict, Any, Optional, Type, List
import logging
from pathlib import Path
import yaml

from .base import (
    LossFunction, 
    BatchLossFunction, 
    LossConfig, 
    RegularizationTerm,
    TotalVariationLoss,
    SmoothnessPenalty
)

logger = logging.getLogger(__name__)


class LossFactory:
    """Factory for creating and configuring loss functions."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize loss factory.
        
        Args:
            config_path: Optional path to loss configuration YAML file
        """
        self.config_path = config_path
        self.loss_registry: Dict[str, Type[LossFunction]] = {}
        self.regularization_registry: Dict[str, Type[RegularizationTerm]] = {}
        self.default_config = self._get_default_config()
        
        # Register built-in regularization terms
        self._register_builtin_regularizers()
        
        # Register built-in loss functions
        self._register_builtin_loss_functions()
        
        # Load configuration if provided
        if config_path and Path(config_path).exists():
            self.config = self._load_config()
        else:
            self.config = self.default_config
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default loss function configuration."""
        return {
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "dtype": "float32",
            "reduction": "mean",
            "regularization_weight": 1e-3,
            "gradient_clipping": None,
            "batch_size": 8,
            "regularization_terms": {
                "total_variation": {
                    "enabled": True,
                    "weight": 1e-4
                },
                "smoothness": {
                    "enabled": True,
                    "weight": 1e-3
                }
            }
        }
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Loaded loss config from {self.config_path}")
            
            # Merge with defaults
            merged_config = self.default_config.copy()
            merged_config.update(config)
            return merged_config
            
        except Exception as e:
            logger.error(f"Failed to load config from {self.config_path}: {e}")
            logger.info("Using default configuration")
            return self.default_config
    
    def _register_builtin_regularizers(self):
        """Register built-in regularization terms."""
        self.regularization_registry["total_variation"] = TotalVariationLoss
        self.regularization_registry["smoothness"] = SmoothnessPenalty
        self.regularization_registry["tv"] = TotalVariationLoss  # Alias
        
        logger.info(f"Registered {len(self.regularization_registry)} built-in regularization terms")
    
    def _register_builtin_loss_functions(self):
        """Register built-in loss functions."""
        try:
            # Import loss functions here to avoid circular imports
            from .targeted import TargetedLoss
            from .non_targeted import NonTargetedLoss
            from .composite import CompositeLoss, BatchCompositeLoss
            
            # Register targeted loss functions
            self.loss_registry["targeted"] = TargetedLoss
            self.loss_registry["targeted_attack"] = TargetedLoss  # Alias
            
            # Register non-targeted loss functions
            self.loss_registry["non_targeted"] = NonTargetedLoss
            self.loss_registry["suppression"] = NonTargetedLoss  # Alias
            self.loss_registry["confidence_reduction"] = NonTargetedLoss  # Alias
            
            # Register composite loss functions
            self.loss_registry["composite"] = CompositeLoss
            self.loss_registry["multi_objective"] = CompositeLoss  # Alias
            self.loss_registry["batch_composite"] = BatchCompositeLoss
            
            logger.info(f"Registered {len(self.loss_registry)} built-in loss functions")
            
        except ImportError as e:
            logger.warning(f"Failed to register some built-in loss functions: {e}")
    
    def register_loss_function(self, name: str, loss_class: Type[LossFunction]):
        """Register a new loss function class.
        
        Args:
            name: Name to register the loss function under
            loss_class: Loss function class to register
        """
        if not issubclass(loss_class, LossFunction):
            raise ValueError(f"Loss class must be a subclass of LossFunction, got {loss_class}")
        
        self.loss_registry[name] = loss_class
        logger.info(f"Registered loss function: {name} -> {loss_class.__name__}")
    
    def register_regularization_term(self, name: str, reg_class: Type[RegularizationTerm]):
        """Register a new regularization term class.
        
        Args:
            name: Name to register the regularization term under
            reg_class: Regularization term class to register
        """
        if not issubclass(reg_class, RegularizationTerm):
            raise ValueError(f"Regularization class must be a subclass of RegularizationTerm, got {reg_class}")
        
        self.regularization_registry[name] = reg_class
        logger.info(f"Registered regularization term: {name} -> {reg_class.__name__}")
    
    def create_loss_config(self, **overrides) -> LossConfig:
        """Create a loss configuration with optional overrides.
        
        Args:
            **overrides: Configuration values to override defaults
            
        Returns:
            LossConfig instance
        """
        config_dict = self.config.copy()
        config_dict.update(overrides)
        
        # Convert dtype string to torch.dtype
        dtype_str = config_dict.get("dtype", "float32")
        dtype_map = {
            "float16": torch.float16,
            "float32": torch.float32, 
            "bfloat16": torch.bfloat16
        }
        config_dict["dtype"] = dtype_map.get(dtype_str, torch.float32)
        
        # Remove keys that aren't LossConfig fields
        valid_keys = {"device", "dtype", "reduction", "regularization_weight", "gradient_clipping"}
        filtered_config = {k: v for k, v in config_dict.items() if k in valid_keys}
        
        return LossConfig(**filtered_config)
    
    def create_regularization_terms(self, device: str = None) -> List[RegularizationTerm]:
        """Create regularization terms from configuration.
        
        Args:
            device: Device to create regularization terms on
            
        Returns:
            List of configured regularization terms
        """
        if device is None:
            device = self.config.get("device", "cpu")
        
        regularization_terms = []
        reg_config = self.config.get("regularization_terms", {})
        
        for name, settings in reg_config.items():
            if not settings.get("enabled", True):
                continue
                
            if name not in self.regularization_registry:
                logger.warning(f"Unknown regularization term: {name}. Skipping.")
                continue
            
            reg_class = self.regularization_registry[name]
            weight = settings.get("weight", 1e-3)
            
            try:
                reg_term = reg_class(weight=weight, device=device)
                regularization_terms.append(reg_term)
                logger.info(f"Created regularization term: {name} (weight={weight})")
            except Exception as e:
                logger.error(f"Failed to create regularization term {name}: {e}")
        
        return regularization_terms
    
    def create_loss_function(self, 
                           loss_type: str, 
                           config: Optional[LossConfig] = None,
                           add_regularization: bool = True,
                           **kwargs) -> LossFunction:
        """Create a loss function of the specified type.
        
        Args:
            loss_type: Type of loss function to create
            config: Optional loss configuration. If None, creates from factory config.
            add_regularization: Whether to add configured regularization terms
            **kwargs: Additional arguments for loss function constructor
            
        Returns:
            Configured loss function instance
        """
        if loss_type not in self.loss_registry:
            raise ValueError(f"Unknown loss type: {loss_type}. Available: {list(self.loss_registry.keys())}")
        
        # Create configuration if not provided
        if config is None:
            config = self.create_loss_config()
        
        # Get loss function class
        loss_class = self.loss_registry[loss_type]
        
        # Determine if we need batch processing
        batch_size = kwargs.pop("batch_size", self.config.get("batch_size", 8))
        
        # Create loss function instance
        try:
            if issubclass(loss_class, BatchLossFunction):
                loss_fn = loss_class(config=config, batch_size=batch_size, **kwargs)
            else:
                loss_fn = loss_class(config=config, **kwargs)
            
            logger.info(f"Created loss function: {loss_type}")
            
        except Exception as e:
            logger.error(f"Failed to create loss function {loss_type}: {e}")
            raise
        
        # Add regularization terms if requested
        if add_regularization:
            reg_terms = self.create_regularization_terms(device=config.device)
            for reg_term in reg_terms:
                loss_fn.add_regularization(reg_term)
        
        return loss_fn
    
    def get_available_loss_types(self) -> List[str]:
        """Get list of available loss function types."""
        return list(self.loss_registry.keys())
    
    def get_available_regularization_terms(self) -> List[str]:
        """Get list of available regularization terms."""
        return list(self.regularization_registry.keys())
    
    def save_config(self, config_path: str):
        """Save current configuration to YAML file.
        
        Args:
            config_path: Path to save configuration to
        """
        try:
            # Convert torch dtypes back to strings for serialization
            config_to_save = self.config.copy()
            if "dtype" in config_to_save and isinstance(config_to_save["dtype"], torch.dtype):
                dtype_reverse_map = {
                    torch.float16: "float16",
                    torch.float32: "float32",
                    torch.bfloat16: "bfloat16"
                }
                config_to_save["dtype"] = dtype_reverse_map.get(config_to_save["dtype"], "float32")
            
            config_path = Path(config_path)
            config_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(config_path, 'w') as f:
                yaml.dump(config_to_save, f, default_flow_style=False)
            
            logger.info(f"Saved configuration to {config_path}")
            
        except Exception as e:
            logger.error(f"Failed to save configuration to {config_path}: {e}")
            raise
    
    def get_factory_info(self) -> Dict[str, Any]:
        """Get information about the factory state.
        
        Returns:
            Dictionary with factory information
        """
        return {
            "config_path": self.config_path,
            "available_loss_types": self.get_available_loss_types(),
            "available_regularization_terms": self.get_available_regularization_terms(),
            "current_device": self.config.get("device"),
            "current_dtype": str(self.config.get("dtype", torch.float32)),
            "regularization_weight": self.config.get("regularization_weight")
        }


# Global factory instance for convenience
_default_factory = None

def get_default_factory() -> LossFactory:
    """Get the default global loss factory instance.
    
    Returns:
        Default LossFactory instance
    """
    global _default_factory
    if _default_factory is None:
        _default_factory = LossFactory()
    return _default_factory


def create_loss_function(loss_type: str, **kwargs) -> LossFunction:
    """Convenience function to create loss function using default factory.
    
    Args:
        loss_type: Type of loss function to create
        **kwargs: Additional arguments for loss function creation
        
    Returns:
        Configured loss function instance
    """
    factory = get_default_factory()
    return factory.create_loss_function(loss_type, **kwargs)