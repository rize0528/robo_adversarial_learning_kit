"""Targeted attack loss functions for adversarial patch generation.

This module implements loss functions that force VLMs to produce specific outputs
or misclassifications, including confidence-based and margin-based variants for
robust adversarial attacks.
"""

import torch
import torch.nn.functional as F
from typing import Dict, Any, Optional, Union, List
from enum import Enum
import logging

from .base import LossFunction, LossConfig

logger = logging.getLogger(__name__)


class TargetMode(Enum):
    """Modes for targeted attacks."""
    CLASSIFICATION = "classification"  # Force specific class prediction
    CONFIDENCE = "confidence"         # Force specific confidence level
    MARGIN = "margin"                # Margin-based targeting
    LIKELIHOOD = "likelihood"        # Maximum likelihood targeting


class TargetedLoss(LossFunction):
    """Targeted loss function for forcing specific model outputs.
    
    This loss function implements various strategies to force VLMs to produce
    specific outputs, including classification targeting, confidence manipulation,
    and margin-based attacks.
    """
    
    def __init__(self, 
                 config: Optional[LossConfig] = None,
                 target_mode: TargetMode = TargetMode.CLASSIFICATION,
                 confidence_threshold: float = 0.9,
                 margin: float = 0.1,
                 temperature: float = 1.0,
                 gradient_smoothing: bool = True,
                 smoothing_factor: float = 0.1):
        """Initialize targeted loss function.
        
        Args:
            config: Loss function configuration
            target_mode: Mode for targeted attack
            confidence_threshold: Minimum confidence for successful attack
            margin: Margin for margin-based attacks
            temperature: Temperature for softmax scaling
            gradient_smoothing: Whether to apply gradient smoothing
            smoothing_factor: Factor for gradient smoothing (0-1)
        """
        super().__init__(config)
        self.target_mode = target_mode
        self.confidence_threshold = confidence_threshold
        self.margin = margin
        self.temperature = temperature
        self.gradient_smoothing = gradient_smoothing
        self.smoothing_factor = smoothing_factor
        
        # Validate parameters
        self._validate_parameters()
        
        logger.info(f"Initialized TargetedLoss with mode={target_mode.value}, "
                   f"threshold={confidence_threshold}, margin={margin}")
    
    def _validate_parameters(self):
        """Validate initialization parameters."""
        if not (0.0 < self.confidence_threshold <= 1.0):
            raise ValueError(f"confidence_threshold must be in (0, 1], got {self.confidence_threshold}")
        
        if self.margin <= 0:
            raise ValueError(f"margin must be positive, got {self.margin}")
        
        if self.temperature <= 0:
            raise ValueError(f"temperature must be positive, got {self.temperature}")
        
        if not (0.0 <= self.smoothing_factor <= 1.0):
            raise ValueError(f"smoothing_factor must be in [0, 1], got {self.smoothing_factor}")
    
    def compute_loss(self, 
                    model_outputs: Dict[str, Any], 
                    patch: torch.Tensor, 
                    targets: Optional[torch.Tensor] = None,
                    target_classes: Optional[torch.Tensor] = None,
                    **kwargs) -> torch.Tensor:
        """Compute targeted attack loss.
        
        Args:
            model_outputs: Dictionary containing model outputs (must include 'logits')
            patch: Adversarial patch tensor
            targets: Optional target logits/probabilities
            target_classes: Optional target class indices
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
        if target_classes is not None:
            target_classes = self._ensure_tensor_on_device(target_classes)
            # Ensure target_classes is long tensor for indexing
            if target_classes.dtype != torch.long:
                target_classes = target_classes.long()
        
        # Compute loss based on target mode
        if self.target_mode == TargetMode.CLASSIFICATION:
            loss = self._compute_classification_loss(logits, targets, target_classes)
        elif self.target_mode == TargetMode.CONFIDENCE:
            loss = self._compute_confidence_loss(logits, targets, target_classes)
        elif self.target_mode == TargetMode.MARGIN:
            loss = self._compute_margin_loss(logits, targets, target_classes)
        elif self.target_mode == TargetMode.LIKELIHOOD:
            loss = self._compute_likelihood_loss(logits, targets, target_classes)
        else:
            raise ValueError(f"Unknown target mode: {self.target_mode}")
        
        # Apply gradient smoothing if enabled
        if self.gradient_smoothing:
            loss = self._apply_gradient_smoothing(loss)
        
        return loss
    
    def _compute_classification_loss(self, 
                                   logits: torch.Tensor,
                                   targets: Optional[torch.Tensor] = None,
                                   target_classes: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute classification-based targeted loss.
        
        Args:
            logits: Model logits [batch_size, num_classes]
            targets: Target probability distribution [batch_size, num_classes]
            target_classes: Target class indices [batch_size]
            
        Returns:
            Classification loss
        """
        if targets is not None:
            # Use KL divergence with target distribution
            probs = F.softmax(logits / self.temperature, dim=-1)
            target_probs = F.softmax(targets / self.temperature, dim=-1)
            # Minimize KL divergence to match target distribution
            loss = F.kl_div(torch.log(probs + 1e-8), target_probs, reduction='none')
            return torch.sum(loss, dim=-1)
        
        elif target_classes is not None:
            # Use cross-entropy with target classes
            # Negative because we want to maximize probability of target class
            loss = -F.cross_entropy(logits, target_classes, reduction='none')
            return loss
        
        else:
            raise ValueError("Either targets or target_classes must be provided for classification mode")
    
    def _compute_confidence_loss(self, 
                               logits: torch.Tensor,
                               targets: Optional[torch.Tensor] = None,
                               target_classes: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute confidence-based targeted loss.
        
        This loss ensures the model prediction has at least the specified confidence
        for the target class.
        
        Args:
            logits: Model logits [batch_size, num_classes]
            targets: Target probability distribution [batch_size, num_classes]
            target_classes: Target class indices [batch_size]
            
        Returns:
            Confidence-based loss
        """
        probs = F.softmax(logits / self.temperature, dim=-1)
        
        if target_classes is not None:
            # Get confidence for target classes
            batch_size = logits.size(0)
            target_confidences = probs[torch.arange(batch_size), target_classes]
            
            # Loss is max(0, threshold - confidence)
            # This pushes confidence above threshold
            confidence_gaps = self.confidence_threshold - target_confidences
            loss = F.relu(confidence_gaps)
            
        elif targets is not None:
            # For target distributions, compute expected confidence
            target_probs = F.softmax(targets / self.temperature, dim=-1)
            expected_confidence = torch.sum(probs * target_probs, dim=-1)
            
            confidence_gaps = self.confidence_threshold - expected_confidence
            loss = F.relu(confidence_gaps)
            
        else:
            raise ValueError("Either targets or target_classes must be provided for confidence mode")
        
        return loss
    
    def _compute_margin_loss(self, 
                           logits: torch.Tensor,
                           targets: Optional[torch.Tensor] = None,
                           target_classes: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute margin-based targeted loss.
        
        This loss ensures the target class has a margin over other classes,
        creating more robust adversarial examples.
        
        Args:
            logits: Model logits [batch_size, num_classes]
            targets: Target probability distribution [batch_size, num_classes]
            target_classes: Target class indices [batch_size]
            
        Returns:
            Margin-based loss
        """
        if target_classes is not None:
            batch_size = logits.size(0)
            
            # Get logits for target classes
            target_logits = logits[torch.arange(batch_size), target_classes]
            
            # Create mask to exclude target classes
            mask = torch.ones_like(logits, dtype=torch.bool)
            mask[torch.arange(batch_size), target_classes] = False
            
            # Get maximum logit among non-target classes
            other_logits = logits.masked_fill(~mask, float('-inf'))
            max_other_logits = torch.max(other_logits, dim=-1)[0]
            
            # Margin loss: ensure target logit is at least margin higher than others
            margin_gaps = self.margin - (target_logits - max_other_logits)
            loss = F.relu(margin_gaps)
            
        elif targets is not None:
            # For target distributions, use expected margin
            target_probs = F.softmax(targets / self.temperature, dim=-1)
            expected_target_logit = torch.sum(logits * target_probs, dim=-1)
            
            # Approximate max non-target logit
            # Weight logits by (1 - target_probs) and find max
            non_target_weights = 1.0 - target_probs
            weighted_logits = logits * non_target_weights
            max_other_logit = torch.max(weighted_logits, dim=-1)[0]
            
            margin_gaps = self.margin - (expected_target_logit - max_other_logit)
            loss = F.relu(margin_gaps)
            
        else:
            raise ValueError("Either targets or target_classes must be provided for margin mode")
        
        return loss
    
    def _compute_likelihood_loss(self, 
                               logits: torch.Tensor,
                               targets: Optional[torch.Tensor] = None,
                               target_classes: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute likelihood-based targeted loss.
        
        This loss maximizes the likelihood of the target class or distribution.
        
        Args:
            logits: Model logits [batch_size, num_classes]
            targets: Target probability distribution [batch_size, num_classes]
            target_classes: Target class indices [batch_size]
            
        Returns:
            Negative log-likelihood loss
        """
        log_probs = F.log_softmax(logits / self.temperature, dim=-1)
        
        if target_classes is not None:
            # Negative log-likelihood for target classes
            batch_size = logits.size(0)
            target_log_probs = log_probs[torch.arange(batch_size), target_classes]
            loss = -target_log_probs
            
        elif targets is not None:
            # Expected negative log-likelihood for target distribution
            target_probs = F.softmax(targets / self.temperature, dim=-1)
            expected_log_prob = torch.sum(target_probs * log_probs, dim=-1)
            loss = -expected_log_prob
            
        else:
            raise ValueError("Either targets or target_classes must be provided for likelihood mode")
        
        return loss
    
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
    
    def get_target_info(self) -> Dict[str, Any]:
        """Get information about the targeted attack configuration.
        
        Returns:
            Dictionary with attack configuration details
        """
        return {
            "target_mode": self.target_mode.value,
            "confidence_threshold": self.confidence_threshold,
            "margin": self.margin,
            "temperature": self.temperature,
            "gradient_smoothing": self.gradient_smoothing,
            "smoothing_factor": self.smoothing_factor if self.gradient_smoothing else None
        }
    
    def set_target_mode(self, mode: TargetMode):
        """Change the target mode.
        
        Args:
            mode: New target mode
        """
        self.target_mode = mode
        logger.info(f"Changed target mode to: {mode.value}")
    
    def set_confidence_threshold(self, threshold: float):
        """Change the confidence threshold.
        
        Args:
            threshold: New confidence threshold (0, 1]
        """
        if not (0.0 < threshold <= 1.0):
            raise ValueError(f"confidence_threshold must be in (0, 1], got {threshold}")
        
        self.confidence_threshold = threshold
        logger.info(f"Changed confidence threshold to: {threshold}")
    
    def set_margin(self, margin: float):
        """Change the margin for margin-based attacks.
        
        Args:
            margin: New margin value (> 0)
        """
        if margin <= 0:
            raise ValueError(f"margin must be positive, got {margin}")
        
        self.margin = margin
        logger.info(f"Changed margin to: {margin}")


# Helper functions for common targeted attack patterns

def create_targeted_classification_loss(config: Optional[LossConfig] = None, 
                                      confidence_threshold: float = 0.9) -> TargetedLoss:
    """Create a targeted loss for classification attacks.
    
    Args:
        config: Loss configuration
        confidence_threshold: Required confidence for target class
        
    Returns:
        Configured TargetedLoss instance
    """
    return TargetedLoss(
        config=config,
        target_mode=TargetMode.CLASSIFICATION,
        confidence_threshold=confidence_threshold
    )


def create_confidence_based_loss(config: Optional[LossConfig] = None, 
                                threshold: float = 0.95,
                                temperature: float = 1.0) -> TargetedLoss:
    """Create a confidence-based targeted loss.
    
    Args:
        config: Loss configuration
        threshold: Confidence threshold to achieve
        temperature: Temperature for softmax scaling
        
    Returns:
        Configured TargetedLoss instance
    """
    return TargetedLoss(
        config=config,
        target_mode=TargetMode.CONFIDENCE,
        confidence_threshold=threshold,
        temperature=temperature
    )


def create_margin_based_loss(config: Optional[LossConfig] = None, 
                           margin: float = 0.2,
                           temperature: float = 1.0) -> TargetedLoss:
    """Create a margin-based targeted loss for robust attacks.
    
    Args:
        config: Loss configuration
        margin: Required margin between target and other classes
        temperature: Temperature for softmax scaling
        
    Returns:
        Configured TargetedLoss instance
    """
    return TargetedLoss(
        config=config,
        target_mode=TargetMode.MARGIN,
        margin=margin,
        temperature=temperature
    )