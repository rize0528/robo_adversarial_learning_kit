"""Loss functions for adversarial patch generation targeting Vision Language Models.

This package provides a comprehensive framework for implementing various loss functions
used in adversarial attacks on VLMs, including targeted and non-targeted attacks,
with support for regularization and batch processing.
"""

from .base import (
    LossFunction, 
    BatchLossFunction,
    LossConfig,
    RegularizationTerm,
    TotalVariationLoss,
    SmoothnessPenalty
)
from .factory import LossFactory, get_default_factory, create_loss_function

# Attack-specific loss functions (Stream B)
from .targeted import (
    TargetedLoss,
    TargetMode,
    create_targeted_classification_loss,
    create_confidence_based_loss,
    create_margin_based_loss
)
from .non_targeted import (
    NonTargetedLoss,
    SuppressionMode,
    create_confidence_reduction_loss,
    create_entropy_maximization_loss,
    create_detection_suppression_loss
)
from .composite import (
    CompositeLoss,
    BatchCompositeLoss,
    CompositionStrategy,
    LossComponent,
    create_balanced_attack_loss,
    create_adaptive_multi_objective_loss
)

__all__ = [
    # Base framework (Stream A)
    "LossFunction",
    "BatchLossFunction",
    "LossConfig",
    "RegularizationTerm", 
    "TotalVariationLoss",
    "SmoothnessPenalty",
    "LossFactory",
    "get_default_factory",
    "create_loss_function",
    
    # Targeted attacks (Stream B)
    "TargetedLoss",
    "TargetMode",
    "create_targeted_classification_loss",
    "create_confidence_based_loss",
    "create_margin_based_loss",
    
    # Non-targeted attacks (Stream B)
    "NonTargetedLoss",
    "SuppressionMode",
    "create_confidence_reduction_loss",
    "create_entropy_maximization_loss",
    "create_detection_suppression_loss",
    
    # Composite attacks (Stream B)
    "CompositeLoss",
    "BatchCompositeLoss",
    "CompositionStrategy",
    "LossComponent",
    "create_balanced_attack_loss",
    "create_adaptive_multi_objective_loss"
]