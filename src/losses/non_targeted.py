"""Non-targeted attack loss functions for adversarial patch generation.

This module implements loss functions for non-targeted attacks that suppress
object detection, reduce confidence, or create general misclassifications
without forcing specific outputs.
"""

import torch
import torch.nn.functional as F
from typing import Dict, Any, Optional, Union, List
from enum import Enum
import logging

from .base import LossFunction, LossConfig

logger = logging.getLogger(__name__)


class SuppressionMode(Enum):
    """Modes for non-targeted suppression attacks."""
    CONFIDENCE_REDUCTION = "confidence_reduction"    # Reduce overall confidence
    ENTROPY_MAXIMIZATION = "entropy_maximization"   # Maximize prediction entropy
    LOGIT_MINIMIZATION = "logit_minimization"       # Minimize target logits
    DETECTION_SUPPRESSION = "detection_suppression" # Suppress object detection


class NonTargetedLoss(LossFunction):
    """Non-targeted loss function for general object suppression.
    
    This loss function implements various strategies for non-targeted attacks
    including confidence reduction, entropy maximization, and object detection
    suppression without forcing specific misclassifications.
    """
    
    def __init__(self, 
                 config: Optional[LossConfig] = None,
                 suppression_mode: SuppressionMode = SuppressionMode.CONFIDENCE_REDUCTION,
                 confidence_target: float = 0.1,
                 entropy_weight: float = 1.0,
                 temperature: float = 1.0,
                 suppress_classes: Optional[List[int]] = None,
                 gradient_smoothing: bool = True,
                 smoothing_factor: float = 0.1):
        """Initialize non-targeted loss function.
        
        Args:
            config: Loss function configuration
            suppression_mode: Mode for suppression attack
            confidence_target: Target confidence level for reduction (0-1)
            entropy_weight: Weight for entropy maximization
            temperature: Temperature for softmax scaling
            suppress_classes: Specific classes to suppress (None for all)
            gradient_smoothing: Whether to apply gradient smoothing
            smoothing_factor: Factor for gradient smoothing (0-1)
        """
        super().__init__(config)
        self.suppression_mode = suppression_mode
        self.confidence_target = confidence_target
        self.entropy_weight = entropy_weight
        self.temperature = temperature
        self.suppress_classes = suppress_classes
        self.gradient_smoothing = gradient_smoothing
        self.smoothing_factor = smoothing_factor
        
        # Validate parameters
        self._validate_parameters()
        
        logger.info(f"Initialized NonTargetedLoss with mode={suppression_mode.value}, "
                   f"confidence_target={confidence_target}, entropy_weight={entropy_weight}")
    
    def _validate_parameters(self):
        """Validate initialization parameters."""
        if not (0.0 <= self.confidence_target <= 1.0):
            raise ValueError(f"confidence_target must be in [0, 1], got {self.confidence_target}")
        
        if self.entropy_weight < 0:
            raise ValueError(f"entropy_weight must be non-negative, got {self.entropy_weight}")
        
        if self.temperature <= 0:
            raise ValueError(f"temperature must be positive, got {self.temperature}")
        
        if not (0.0 <= self.smoothing_factor <= 1.0):
            raise ValueError(f"smoothing_factor must be in [0, 1], got {self.smoothing_factor}")
        
        if self.suppress_classes is not None:
            if not isinstance(self.suppress_classes, (list, tuple)):
                raise ValueError("suppress_classes must be a list or tuple of integers")
            if not all(isinstance(c, int) and c >= 0 for c in self.suppress_classes):
                raise ValueError("suppress_classes must contain non-negative integers")
    
    def compute_loss(self, 
                    model_outputs: Dict[str, Any], 
                    patch: torch.Tensor, 
                    targets: Optional[torch.Tensor] = None,
                    original_outputs: Optional[Dict[str, Any]] = None,
                    **kwargs) -> torch.Tensor:
        """Compute non-targeted attack loss.
        
        Args:
            model_outputs: Dictionary containing model outputs (must include 'logits')
            patch: Adversarial patch tensor
            targets: Optional original class labels to suppress
            original_outputs: Optional original model outputs for comparison
            **kwargs: Additional arguments
            
        Returns:
            Computed loss tensor
        """
        # Extract logits from model outputs
        logits = model_outputs.get("logits")
        if logits is None:
            raise ValueError("model_outputs must contain 'logits' key")
        
        # Ensure tensors are on correct device and have correct dtype
        logits = self._ensure_tensor_on_device(logits)
        if targets is not None:
            targets = self._ensure_tensor_on_device(targets)
            # Ensure targets is long tensor for indexing
            if targets.dtype != torch.long:
                targets = targets.long()
        
        # Compute loss based on suppression mode
        if self.suppression_mode == SuppressionMode.CONFIDENCE_REDUCTION:
            loss = self._compute_confidence_reduction_loss(logits, targets)
        elif self.suppression_mode == SuppressionMode.ENTROPY_MAXIMIZATION:
            loss = self._compute_entropy_maximization_loss(logits, targets)
        elif self.suppression_mode == SuppressionMode.LOGIT_MINIMIZATION:
            loss = self._compute_logit_minimization_loss(logits, targets)
        elif self.suppression_mode == SuppressionMode.DETECTION_SUPPRESSION:
            loss = self._compute_detection_suppression_loss(logits, targets, original_outputs)
        else:
            raise ValueError(f"Unknown suppression mode: {self.suppression_mode}")
        
        # Apply gradient smoothing if enabled
        if self.gradient_smoothing:
            loss = self._apply_gradient_smoothing(loss)
        
        return loss
    
    def _compute_confidence_reduction_loss(self, 
                                         logits: torch.Tensor,
                                         targets: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute confidence reduction loss.
        
        This loss reduces the maximum confidence of predictions, making the
        model less certain about its outputs.
        
        Args:
            logits: Model logits [batch_size, num_classes]
            targets: Optional target classes to specifically suppress
            
        Returns:
            Confidence reduction loss
        """
        probs = F.softmax(logits / self.temperature, dim=-1)
        
        if self.suppress_classes is not None:
            # Only suppress specific classes
            class_mask = torch.zeros_like(logits, dtype=torch.bool)
            for class_idx in self.suppress_classes:
                if class_idx < logits.size(-1):
                    class_mask[:, class_idx] = True
            
            # Get probabilities for classes to suppress
            suppressed_probs = probs.masked_fill(~class_mask, 0.0)
            max_suppressed_probs = torch.sum(suppressed_probs, dim=-1)
            
        else:
            # Suppress maximum confidence across all classes
            max_suppressed_probs = torch.max(probs, dim=-1)[0]
        
        # Loss increases as confidence exceeds target
        confidence_excess = F.relu(max_suppressed_probs - self.confidence_target)
        
        return confidence_excess
    
    def _compute_entropy_maximization_loss(self, 
                                         logits: torch.Tensor,
                                         targets: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute entropy maximization loss.
        
        This loss maximizes the entropy of predictions, making the model
        uncertain across all classes.
        
        Args:
            logits: Model logits [batch_size, num_classes]
            targets: Optional target classes (unused for entropy maximization)
            
        Returns:
            Negative entropy loss (to maximize entropy)
        """
        probs = F.softmax(logits / self.temperature, dim=-1)
        
        # Compute entropy: -sum(p * log(p))
        log_probs = torch.log(probs + 1e-8)  # Add small epsilon for numerical stability
        entropy = -torch.sum(probs * log_probs, dim=-1)
        
        # Return negative entropy to maximize it
        return -self.entropy_weight * entropy
    
    def _compute_logit_minimization_loss(self, 
                                       logits: torch.Tensor,
                                       targets: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute logit minimization loss.
        
        This loss minimizes the logits for specific classes or all classes,
        effectively reducing the model's confidence.
        
        Args:
            logits: Model logits [batch_size, num_classes]
            targets: Optional target classes to minimize
            
        Returns:
            Logit minimization loss
        """
        if self.suppress_classes is not None:
            # Minimize logits for specific classes only
            suppression_loss = 0.0
            for class_idx in self.suppress_classes:
                if class_idx < logits.size(-1):
                    class_logits = logits[:, class_idx]
                    # Minimize logits (make them as negative as possible)
                    suppression_loss += torch.mean(F.relu(class_logits))
            
            return suppression_loss
        
        elif targets is not None:
            # Minimize logits for target classes
            batch_size = logits.size(0)
            target_logits = logits[torch.arange(batch_size), targets]
            return torch.mean(F.relu(target_logits))
        
        else:
            # Minimize all logits
            return torch.mean(F.relu(logits))
    
    def _compute_detection_suppression_loss(self, 
                                          logits: torch.Tensor,
                                          targets: Optional[torch.Tensor] = None,
                                          original_outputs: Optional[Dict[str, Any]] = None) -> torch.Tensor:
        """Compute detection suppression loss.
        
        This loss specifically targets object detection scenarios, suppressing
        the detection of objects by reducing objectness scores and class confidence.
        
        Args:
            logits: Model logits [batch_size, num_classes]
            targets: Optional target classes to suppress
            original_outputs: Optional original model outputs for comparison
            
        Returns:
            Detection suppression loss
        """
        # For detection suppression, we combine multiple strategies
        loss_components = []
        
        # Component 1: Confidence reduction
        conf_loss = self._compute_confidence_reduction_loss(logits, targets)
        loss_components.append(conf_loss)
        
        # Component 2: Entropy maximization (but with lower weight)
        entropy_loss = self._compute_entropy_maximization_loss(logits, targets)
        loss_components.append(0.5 * entropy_loss)
        
        # Component 3: If we have original outputs, minimize the difference
        # This helps maintain "natural" looking misclassifications
        if original_outputs is not None:
            original_logits = original_outputs.get("logits")
            if original_logits is not None:
                original_logits = self._ensure_tensor_on_device(original_logits)
                
                # Encourage deviation from original predictions
                # But not too much (to avoid obvious adversarial artifacts)
                deviation = torch.abs(logits - original_logits)
                optimal_deviation = 0.5  # Moderate deviation
                deviation_loss = torch.mean((deviation - optimal_deviation) ** 2)
                loss_components.append(0.3 * deviation_loss)
        
        # Combine all loss components
        total_loss = sum(loss_components)
        
        return total_loss
    
    def _apply_gradient_smoothing(self, loss: torch.Tensor) -> torch.Tensor:
        """Apply gradient smoothing to the loss.
        
        This helps stabilize training by smoothing gradients over time.
        
        Args:
            loss: Raw loss tensor
            
        Returns:
            Smoothed loss tensor
        """
        if not hasattr(self, '_prev_loss'):
            self._prev_loss = loss.detach()
            return loss
        
        # Exponential moving average of loss
        smoothed_loss = (1 - self.smoothing_factor) * loss + self.smoothing_factor * self._prev_loss
        self._prev_loss = loss.detach()
        
        return smoothed_loss
    
    def get_suppression_info(self) -> Dict[str, Any]:
        """Get information about the suppression attack configuration.
        
        Returns:
            Dictionary with attack configuration details
        """
        return {
            "suppression_mode": self.suppression_mode.value,
            "confidence_target": self.confidence_target,
            "entropy_weight": self.entropy_weight,
            "temperature": self.temperature,
            "suppress_classes": self.suppress_classes,
            "gradient_smoothing": self.gradient_smoothing,
            "smoothing_factor": self.smoothing_factor if self.gradient_smoothing else None
        }
    
    def set_suppression_mode(self, mode: SuppressionMode):
        """Change the suppression mode.
        
        Args:
            mode: New suppression mode
        """
        self.suppression_mode = mode
        logger.info(f"Changed suppression mode to: {mode.value}")
    
    def set_confidence_target(self, target: float):
        """Change the confidence target.
        
        Args:
            target: New confidence target [0, 1]
        """
        if not (0.0 <= target <= 1.0):
            raise ValueError(f"confidence_target must be in [0, 1], got {target}")
        
        self.confidence_target = target
        logger.info(f"Changed confidence target to: {target}")
    
    def set_suppress_classes(self, classes: Optional[List[int]]):
        """Change the classes to suppress.
        
        Args:
            classes: List of class indices to suppress, or None for all classes
        """
        if classes is not None:
            if not isinstance(classes, (list, tuple)):
                raise ValueError("classes must be a list or tuple of integers")
            if not all(isinstance(c, int) and c >= 0 for c in classes):
                raise ValueError("classes must contain non-negative integers")
        
        self.suppress_classes = classes
        logger.info(f"Changed suppress classes to: {classes}")
    
    def add_suppress_class(self, class_idx: int):
        """Add a class to the suppression list.
        
        Args:
            class_idx: Class index to add to suppression list
        """
        if not isinstance(class_idx, int) or class_idx < 0:
            raise ValueError("class_idx must be a non-negative integer")
        
        if self.suppress_classes is None:
            self.suppress_classes = []
        
        if class_idx not in self.suppress_classes:
            self.suppress_classes.append(class_idx)
            logger.info(f"Added class {class_idx} to suppression list")
    
    def remove_suppress_class(self, class_idx: int):
        """Remove a class from the suppression list.
        
        Args:
            class_idx: Class index to remove from suppression list
        """
        if self.suppress_classes is not None and class_idx in self.suppress_classes:
            self.suppress_classes.remove(class_idx)
            logger.info(f"Removed class {class_idx} from suppression list")


# Helper functions for common non-targeted attack patterns

def create_confidence_reduction_loss(config: Optional[LossConfig] = None,
                                    confidence_target: float = 0.1,
                                    temperature: float = 1.0) -> NonTargetedLoss:
    """Create a non-targeted loss for confidence reduction.
    
    Args:
        config: Loss configuration
        confidence_target: Target confidence level
        temperature: Temperature for softmax scaling
        
    Returns:
        Configured NonTargetedLoss instance
    """
    return NonTargetedLoss(
        config=config,
        suppression_mode=SuppressionMode.CONFIDENCE_REDUCTION,
        confidence_target=confidence_target,
        temperature=temperature
    )


def create_entropy_maximization_loss(config: Optional[LossConfig] = None,
                                   entropy_weight: float = 1.0,
                                   temperature: float = 2.0) -> NonTargetedLoss:
    """Create a non-targeted loss for entropy maximization.
    
    Args:
        config: Loss configuration
        entropy_weight: Weight for entropy maximization
        temperature: Temperature for softmax scaling (higher = more uniform)
        
    Returns:
        Configured NonTargetedLoss instance
    """
    return NonTargetedLoss(
        config=config,
        suppression_mode=SuppressionMode.ENTROPY_MAXIMIZATION,
        entropy_weight=entropy_weight,
        temperature=temperature
    )


def create_detection_suppression_loss(config: Optional[LossConfig] = None,
                                    suppress_classes: Optional[List[int]] = None,
                                    confidence_target: float = 0.05) -> NonTargetedLoss:
    """Create a non-targeted loss for object detection suppression.
    
    Args:
        config: Loss configuration
        suppress_classes: Specific classes to suppress
        confidence_target: Very low confidence target
        
    Returns:
        Configured NonTargetedLoss instance
    """
    return NonTargetedLoss(
        config=config,
        suppression_mode=SuppressionMode.DETECTION_SUPPRESSION,
        suppress_classes=suppress_classes,
        confidence_target=confidence_target
    )