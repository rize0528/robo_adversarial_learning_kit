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

__all__ = [
    "LossFunction",
    "BatchLossFunction",
    "LossConfig",
    "RegularizationTerm", 
    "TotalVariationLoss",
    "SmoothnessPenalty",
    "LossFactory",
    "get_default_factory",
    "create_loss_function"
]