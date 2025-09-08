"""Loss functions for adversarial patch generation targeting Vision Language Models.

This package provides a comprehensive framework for implementing various loss functions
used in adversarial attacks on VLMs, including targeted and non-targeted attacks,
with support for regularization and batch processing.
"""

from .base import LossFunction, RegularizationTerm
from .factory import LossFactory

__all__ = [
    "LossFunction",
    "RegularizationTerm", 
    "LossFactory"
]