"""Abstract base classes for loss functions in adversarial patch generation.

This module provides the foundation for implementing various loss functions used
in adversarial attacks on Vision Language Models, with built-in support for
GPU optimization, regularization, and batch processing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class LossConfig:
    """Configuration for loss functions."""
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: torch.dtype = torch.float32
    reduction: str = "mean"  # 'mean', 'sum', 'none'
    regularization_weight: float = 1e-3
    gradient_clipping: Optional[float] = None
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.reduction not in ["mean", "sum", "none"]:
            raise ValueError(f"reduction must be one of ['mean', 'sum', 'none'], got {self.reduction}")
        if self.regularization_weight < 0:
            raise ValueError(f"regularization_weight must be non-negative, got {self.regularization_weight}")


class RegularizationTerm(ABC):
    """Abstract base class for regularization terms."""
    
    def __init__(self, weight: float = 1e-3, device: str = "cuda"):
        """Initialize regularization term.
        
        Args:
            weight: Weight of the regularization term
            device: Device to perform computations on
        """
        self.weight = weight
        self.device = device
    
    @abstractmethod
    def compute(self, patch: torch.Tensor) -> torch.Tensor:
        """Compute regularization loss for the given patch.
        
        Args:
            patch: Adversarial patch tensor of shape (batch_size, channels, height, width)
            
        Returns:
            Regularization loss tensor
        """
        pass
    
    def __call__(self, patch: torch.Tensor) -> torch.Tensor:
        """Make the regularization term callable."""
        return self.weight * self.compute(patch)


class TotalVariationLoss(RegularizationTerm):
    """Total variation regularization for patch smoothness."""
    
    def compute(self, patch: torch.Tensor) -> torch.Tensor:
        """Compute total variation loss.
        
        Args:
            patch: Patch tensor of shape (batch_size, channels, height, width)
            
        Returns:
            Total variation loss
        """
        if patch.dim() != 4:
            raise ValueError(f"Expected 4D tensor (batch_size, channels, height, width), got {patch.dim()}D")
        
        # Compute differences along height and width dimensions
        diff_h = torch.abs(patch[:, :, 1:, :] - patch[:, :, :-1, :])
        diff_w = torch.abs(patch[:, :, :, 1:] - patch[:, :, :, :-1])
        
        # Sum over spatial dimensions and channels
        tv_loss = torch.sum(diff_h) + torch.sum(diff_w)
        
        # Normalize by patch size
        batch_size = patch.size(0)
        return tv_loss / (batch_size * patch.numel())


class SmoothnessPenalty(RegularizationTerm):
    """L2 smoothness penalty using convolution with Laplacian kernel."""
    
    def __init__(self, weight: float = 1e-3, device: str = "cuda"):
        super().__init__(weight, device)
        # Create Laplacian kernel for edge detection
        self.laplacian_kernel = torch.tensor([[0, -1, 0], 
                                             [-1, 4, -1], 
                                             [0, -1, 0]], 
                                           dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
    
    def compute(self, patch: torch.Tensor) -> torch.Tensor:
        """Compute smoothness penalty using Laplacian filter.
        
        Args:
            patch: Patch tensor of shape (batch_size, channels, height, width)
            
        Returns:
            Smoothness penalty loss
        """
        if patch.dim() != 4:
            raise ValueError(f"Expected 4D tensor (batch_size, channels, height, width), got {patch.dim()}D")
        
        batch_size, channels, height, width = patch.shape
        
        # Apply Laplacian kernel to each channel
        smoothness_loss = 0
        for c in range(channels):
            channel_patch = patch[:, c:c+1, :, :]
            # Apply convolution with Laplacian kernel (padding=1 to maintain size)
            edges = F.conv2d(channel_patch, self.laplacian_kernel, padding=1)
            smoothness_loss += torch.mean(edges ** 2)
        
        return smoothness_loss / channels


class LossFunction(ABC):
    """Abstract base class for adversarial loss functions.
    
    This class provides the foundation for implementing various loss functions
    used in adversarial patch generation, with built-in support for GPU 
    optimization, regularization terms, and batch processing.
    """
    
    def __init__(self, config: Optional[LossConfig] = None):
        """Initialize loss function with configuration.
        
        Args:
            config: Loss function configuration. If None, uses default config.
        """
        self.config = config or LossConfig()
        self.regularization_terms: List[RegularizationTerm] = []
        self.device = self.config.device
        
        # Ensure we have a valid device
        if torch.cuda.is_available() and self.config.device == "cuda":
            self.device = "cuda"
        else:
            self.device = "cpu"
            
        logger.info(f"Initialized {self.__class__.__name__} on device: {self.device}")
    
    def add_regularization(self, regularization: RegularizationTerm):
        """Add a regularization term to the loss function.
        
        Args:
            regularization: Regularization term to add
        """
        # Move regularization to correct device if needed
        if hasattr(regularization, 'laplacian_kernel'):
            regularization.laplacian_kernel = regularization.laplacian_kernel.to(self.device)
        regularization.device = self.device
        self.regularization_terms.append(regularization)
        logger.info(f"Added regularization term: {regularization.__class__.__name__}")
    
    def _ensure_tensor_on_device(self, tensor: torch.Tensor) -> torch.Tensor:
        """Ensure tensor is on the correct device and has the right dtype.
        
        Args:
            tensor: Input tensor
            
        Returns:
            Tensor moved to correct device and dtype
        """
        if tensor.device != torch.device(self.device):
            tensor = tensor.to(self.device)
        if tensor.dtype != self.config.dtype:
            tensor = tensor.to(self.config.dtype)
        return tensor
    
    def _compute_regularization(self, patch: torch.Tensor) -> torch.Tensor:
        """Compute total regularization loss from all regularization terms.
        
        Args:
            patch: Adversarial patch tensor
            
        Returns:
            Total regularization loss
        """
        if not self.regularization_terms:
            return torch.tensor(0.0, device=self.device, dtype=self.config.dtype)
        
        total_reg = torch.tensor(0.0, device=self.device, dtype=self.config.dtype)
        for reg_term in self.regularization_terms:
            total_reg = total_reg + reg_term(patch)
        
        return total_reg
    
    def _apply_reduction(self, loss: torch.Tensor) -> torch.Tensor:
        """Apply reduction to loss tensor.
        
        Args:
            loss: Loss tensor
            
        Returns:
            Reduced loss tensor
        """
        if self.config.reduction == "mean":
            return torch.mean(loss)
        elif self.config.reduction == "sum":
            return torch.sum(loss)
        else:  # reduction == "none"
            return loss
    
    def _apply_gradient_clipping(self, loss: torch.Tensor) -> torch.Tensor:
        """Apply gradient clipping if configured.
        
        Args:
            loss: Loss tensor with gradients
            
        Returns:
            Loss tensor with potentially clipped gradients
        """
        if self.config.gradient_clipping is not None and loss.requires_grad:
            # Register hook for gradient clipping
            def clip_grad_hook(grad):
                return torch.clamp(grad, -self.config.gradient_clipping, self.config.gradient_clipping)
            loss.register_hook(clip_grad_hook)
        return loss
    
    @abstractmethod
    def compute_loss(self, 
                    model_outputs: Dict[str, Any], 
                    patch: torch.Tensor, 
                    targets: Optional[torch.Tensor] = None,
                    **kwargs) -> torch.Tensor:
        """Compute the main loss (without regularization).
        
        This method must be implemented by subclasses to define the specific
        loss computation logic.
        
        Args:
            model_outputs: Dictionary containing model outputs (logits, features, etc.)
            patch: Adversarial patch tensor of shape (batch_size, channels, height, width)
            targets: Optional target labels/values
            **kwargs: Additional arguments specific to the loss function
            
        Returns:
            Main loss tensor (before regularization)
        """
        pass
    
    def forward(self, 
               model_outputs: Dict[str, Any], 
               patch: torch.Tensor, 
               targets: Optional[torch.Tensor] = None,
               **kwargs) -> Dict[str, torch.Tensor]:
        """Compute total loss including regularization.
        
        Args:
            model_outputs: Dictionary containing model outputs
            patch: Adversarial patch tensor
            targets: Optional target labels/values
            **kwargs: Additional arguments for loss computation
            
        Returns:
            Dictionary with 'total_loss', 'main_loss', and 'regularization_loss'
        """
        # Ensure tensors are on correct device
        patch = self._ensure_tensor_on_device(patch)
        if targets is not None:
            targets = self._ensure_tensor_on_device(targets)
        
        # Move model outputs to correct device if needed
        for key, value in model_outputs.items():
            if isinstance(value, torch.Tensor):
                model_outputs[key] = self._ensure_tensor_on_device(value)
        
        # Compute main loss
        main_loss = self.compute_loss(model_outputs, patch, targets, **kwargs)
        main_loss = self._apply_reduction(main_loss)
        
        # Compute regularization
        reg_loss = self._compute_regularization(patch)
        
        # Total loss
        total_loss = main_loss + self.config.regularization_weight * reg_loss
        total_loss = self._apply_gradient_clipping(total_loss)
        
        return {
            "total_loss": total_loss,
            "main_loss": main_loss,
            "regularization_loss": reg_loss
        }
    
    def __call__(self, 
                model_outputs: Dict[str, Any], 
                patch: torch.Tensor, 
                targets: Optional[torch.Tensor] = None,
                **kwargs) -> Dict[str, torch.Tensor]:
        """Make the loss function callable."""
        return self.forward(model_outputs, patch, targets, **kwargs)
    
    def get_device(self) -> str:
        """Get the device this loss function is configured for."""
        return self.device
    
    def get_info(self) -> Dict[str, Any]:
        """Get information about this loss function.
        
        Returns:
            Dictionary with loss function information
        """
        return {
            "class": self.__class__.__name__,
            "device": self.device,
            "dtype": str(self.config.dtype),
            "reduction": self.config.reduction,
            "regularization_weight": self.config.regularization_weight,
            "num_regularization_terms": len(self.regularization_terms),
            "regularization_terms": [reg.__class__.__name__ for reg in self.regularization_terms]
        }


class BatchLossFunction(LossFunction):
    """Base class for loss functions optimized for batch processing.
    
    This class extends the base LossFunction with utilities specifically
    designed for efficient batch processing of multiple patches.
    """
    
    def __init__(self, config: Optional[LossConfig] = None, batch_size: int = 8):
        """Initialize batch loss function.
        
        Args:
            config: Loss function configuration
            batch_size: Expected batch size for optimization
        """
        super().__init__(config)
        self.batch_size = batch_size
        logger.info(f"Initialized {self.__class__.__name__} with batch_size={batch_size}")
    
    def process_batch(self, 
                     model_outputs_batch: List[Dict[str, Any]], 
                     patches_batch: torch.Tensor,
                     targets_batch: Optional[torch.Tensor] = None,
                     **kwargs) -> Dict[str, torch.Tensor]:
        """Process a batch of model outputs and patches.
        
        Args:
            model_outputs_batch: List of model output dictionaries
            patches_batch: Batch of patches with shape (batch_size, channels, height, width)  
            targets_batch: Optional batch of targets
            **kwargs: Additional arguments
            
        Returns:
            Aggregated loss results
        """
        batch_size = len(model_outputs_batch)
        
        # Validate batch consistency
        if patches_batch.size(0) != batch_size:
            raise ValueError(f"Batch size mismatch: {patches_batch.size(0)} patches vs {batch_size} outputs")
        
        if targets_batch is not None and targets_batch.size(0) != batch_size:
            raise ValueError(f"Batch size mismatch: {targets_batch.size(0)} targets vs {batch_size} outputs")
        
        # Combine model outputs into single batch dictionary
        combined_outputs = {}
        for key in model_outputs_batch[0].keys():
            if isinstance(model_outputs_batch[0][key], torch.Tensor):
                # Stack tensors along batch dimension
                combined_outputs[key] = torch.stack([outputs[key] for outputs in model_outputs_batch])
            else:
                # Keep non-tensor values as lists
                combined_outputs[key] = [outputs[key] for outputs in model_outputs_batch]
        
        # Compute loss using standard forward method
        return self.forward(combined_outputs, patches_batch, targets_batch, **kwargs)
    
    def get_optimal_batch_size(self) -> int:
        """Get the optimal batch size for this loss function."""
        return self.batch_size