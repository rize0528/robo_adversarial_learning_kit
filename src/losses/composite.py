"""Composite loss functions for multi-objective adversarial attacks.

This module implements loss function composition allowing combination of multiple
attack objectives with configurable weights and strategies for complex
adversarial scenarios.
"""

import torch
import torch.nn.functional as F
from typing import Dict, Any, Optional, List, Union, Tuple, Callable
from enum import Enum
from dataclasses import dataclass
import logging

from .base import LossFunction, LossConfig, BatchLossFunction

logger = logging.getLogger(__name__)


class CompositionStrategy(Enum):
    """Strategies for combining multiple loss functions."""
    WEIGHTED_SUM = "weighted_sum"           # Simple weighted combination
    ADAPTIVE_WEIGHTS = "adaptive_weights"   # Dynamically adjust weights based on performance
    PARETO_OPTIMAL = "pareto_optimal"       # Pareto-optimal multi-objective optimization
    HIERARCHICAL = "hierarchical"           # Hierarchical objective prioritization
    ALTERNATING = "alternating"             # Alternate between objectives


@dataclass
class LossComponent:
    """Configuration for a single loss component in a composite loss."""
    loss_function: LossFunction
    weight: float = 1.0
    name: str = ""
    priority: int = 1  # For hierarchical strategy (higher = more important)
    adaptive: bool = False  # Whether weight can be adapted automatically
    
    def __post_init__(self):
        """Validate component configuration."""
        if self.weight < 0:
            raise ValueError(f"weight must be non-negative, got {self.weight}")
        if self.priority < 0:
            raise ValueError(f"priority must be non-negative, got {self.priority}")
        if not self.name:
            self.name = self.loss_function.__class__.__name__


class CompositeLoss(LossFunction):
    """Composite loss function for multi-objective adversarial attacks.
    
    This class allows combining multiple loss functions with various strategies
    for complex attack scenarios that need to balance multiple objectives
    simultaneously.
    """
    
    def __init__(self, 
                 config: Optional[LossConfig] = None,
                 components: Optional[List[LossComponent]] = None,
                 strategy: CompositionStrategy = CompositionStrategy.WEIGHTED_SUM,
                 adaptation_rate: float = 0.01,
                 convergence_threshold: float = 1e-4,
                 max_adaptation_steps: int = 1000,
                 normalize_weights: bool = True,
                 gradient_balancing: bool = False):
        """Initialize composite loss function.
        
        Args:
            config: Loss function configuration
            components: List of loss components to combine
            strategy: Strategy for combining losses
            adaptation_rate: Rate for adaptive weight adjustment
            convergence_threshold: Threshold for convergence detection
            max_adaptation_steps: Maximum steps for adaptation
            normalize_weights: Whether to normalize component weights
            gradient_balancing: Whether to apply gradient balancing across components
        """
        super().__init__(config)
        
        self.components = components or []
        self.strategy = strategy
        self.adaptation_rate = adaptation_rate
        self.convergence_threshold = convergence_threshold
        self.max_adaptation_steps = max_adaptation_steps
        self.normalize_weights = normalize_weights
        self.gradient_balancing = gradient_balancing
        
        # Internal state for adaptive strategies
        self._adaptation_step = 0
        self._loss_history = []
        self._weight_history = []
        self._alternating_index = 0
        
        # Validate configuration
        self._validate_configuration()
        
        # Initialize adaptive weights if needed
        if self.strategy == CompositionStrategy.ADAPTIVE_WEIGHTS:
            self._initialize_adaptive_weights()
        
        logger.info(f"Initialized CompositeLoss with {len(self.components)} components "
                   f"using {strategy.value} strategy")
    
    def _validate_configuration(self):
        """Validate the composite loss configuration."""
        # Note: We don't validate components here as they can be added later
        # The validation happens in compute_loss when components are actually needed
        
        if not (0.0 < self.adaptation_rate <= 1.0):
            raise ValueError(f"adaptation_rate must be in (0, 1], got {self.adaptation_rate}")
        
        if self.convergence_threshold <= 0:
            raise ValueError(f"convergence_threshold must be positive, got {self.convergence_threshold}")
        
        # Check for duplicate names
        names = [comp.name for comp in self.components]
        if len(names) != len(set(names)):
            raise ValueError("Component names must be unique")
        
        # Ensure all components use compatible devices
        for comp in self.components:
            if comp.loss_function.device != self.device:
                logger.warning(f"Component {comp.name} device mismatch. Moving to {self.device}")
                # Try to move component to correct device if possible
                try:
                    comp.loss_function.device = self.device
                except Exception as e:
                    logger.error(f"Failed to move component {comp.name} to device {self.device}: {e}")
    
    def _initialize_adaptive_weights(self):
        """Initialize adaptive weights for components."""
        for comp in self.components:
            if comp.adaptive:
                # Start with equal weights for adaptive components
                comp.weight = 1.0 / len([c for c in self.components if c.adaptive])
    
    def add_component(self, loss_function: LossFunction, weight: float = 1.0, 
                     name: str = "", priority: int = 1, adaptive: bool = False):
        """Add a loss component to the composite loss.
        
        Args:
            loss_function: Loss function to add
            weight: Weight for this component
            name: Name for the component
            priority: Priority for hierarchical strategy
            adaptive: Whether this component's weight can be adapted
        """
        if not name:
            name = f"{loss_function.__class__.__name__}_{len(self.components)}"
        
        # Check for name conflicts
        existing_names = [comp.name for comp in self.components]
        if name in existing_names:
            raise ValueError(f"Component name '{name}' already exists")
        
        component = LossComponent(
            loss_function=loss_function,
            weight=weight,
            name=name,
            priority=priority,
            adaptive=adaptive
        )
        
        self.components.append(component)
        logger.info(f"Added component: {name} (weight={weight}, priority={priority})")
    
    def remove_component(self, name: str):
        """Remove a loss component by name.
        
        Args:
            name: Name of the component to remove
        """
        self.components = [comp for comp in self.components if comp.name != name]
        logger.info(f"Removed component: {name}")
    
    def get_component(self, name: str) -> Optional[LossComponent]:
        """Get a component by name.
        
        Args:
            name: Component name
            
        Returns:
            LossComponent if found, None otherwise
        """
        for comp in self.components:
            if comp.name == name:
                return comp
        return None
    
    def set_component_weight(self, name: str, weight: float):
        """Set the weight for a specific component.
        
        Args:
            name: Component name
            weight: New weight value
        """
        comp = self.get_component(name)
        if comp is None:
            raise ValueError(f"Component '{name}' not found")
        
        if weight < 0:
            raise ValueError(f"Weight must be non-negative, got {weight}")
        
        comp.weight = weight
        logger.info(f"Set weight for {name} to {weight}")
    
    def compute_loss(self, 
                    model_outputs: Dict[str, Any], 
                    patch: torch.Tensor, 
                    targets: Optional[torch.Tensor] = None,
                    component_kwargs: Optional[Dict[str, Dict[str, Any]]] = None,
                    **kwargs) -> torch.Tensor:
        """Compute composite loss using the specified strategy.
        
        Args:
            model_outputs: Dictionary containing model outputs
            patch: Adversarial patch tensor
            targets: Optional target labels/values
            component_kwargs: Optional per-component keyword arguments
            **kwargs: Additional arguments passed to all components
            
        Returns:
            Computed composite loss tensor
        """
        if not self.components:
            raise ValueError("No loss components configured")
        
        component_kwargs = component_kwargs or {}
        
        # Compute individual component losses
        component_losses = {}
        component_results = {}
        
        for comp in self.components:
            try:
                # Get component-specific kwargs
                comp_kwargs = component_kwargs.get(comp.name, {})
                merged_kwargs = {**kwargs, **comp_kwargs}
                
                # Compute component loss
                result = comp.loss_function.forward(model_outputs, patch, targets, **merged_kwargs)
                component_losses[comp.name] = result["total_loss"]
                component_results[comp.name] = result
                
            except Exception as e:
                logger.error(f"Failed to compute loss for component {comp.name}: {e}")
                # Use zero loss for failed components to prevent crash
                component_losses[comp.name] = torch.tensor(0.0, device=self.device)
                component_results[comp.name] = {
                    "total_loss": component_losses[comp.name],
                    "main_loss": component_losses[comp.name],
                    "regularization_loss": torch.tensor(0.0, device=self.device)
                }
        
        # Store loss history for adaptive strategies
        if self.strategy in [CompositionStrategy.ADAPTIVE_WEIGHTS, CompositionStrategy.PARETO_OPTIMAL]:
            current_losses = [component_losses[comp.name].item() for comp in self.components]
            self._loss_history.append(current_losses)
            if len(self._loss_history) > 100:  # Keep only recent history
                self._loss_history.pop(0)
        
        # Apply composition strategy
        if self.strategy == CompositionStrategy.WEIGHTED_SUM:
            composite_loss = self._compute_weighted_sum(component_losses)
        elif self.strategy == CompositionStrategy.ADAPTIVE_WEIGHTS:
            composite_loss = self._compute_adaptive_weighted(component_losses)
        elif self.strategy == CompositionStrategy.PARETO_OPTIMAL:
            composite_loss = self._compute_pareto_optimal(component_losses)
        elif self.strategy == CompositionStrategy.HIERARCHICAL:
            composite_loss = self._compute_hierarchical(component_losses)
        elif self.strategy == CompositionStrategy.ALTERNATING:
            composite_loss = self._compute_alternating(component_losses)
        else:
            raise ValueError(f"Unknown composition strategy: {self.strategy}")
        
        # Apply gradient balancing if enabled
        if self.gradient_balancing:
            composite_loss = self._apply_gradient_balancing(composite_loss, component_losses)
        
        return composite_loss
    
    def _compute_weighted_sum(self, component_losses: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute weighted sum of component losses."""
        weights = [comp.weight for comp in self.components]
        
        if self.normalize_weights and sum(weights) > 0:
            total_weight = sum(weights)
            weights = [w / total_weight for w in weights]
        
        composite_loss = torch.tensor(0.0, device=self.device, dtype=self.config.dtype)
        
        for comp, weight in zip(self.components, weights):
            weighted_loss = weight * component_losses[comp.name]
            composite_loss = composite_loss + weighted_loss
        
        return composite_loss
    
    def _compute_adaptive_weighted(self, component_losses: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute adaptively weighted sum of component losses."""
        # Update weights based on loss progress
        if len(self._loss_history) > 1 and self._adaptation_step < self.max_adaptation_steps:
            self._update_adaptive_weights()
            self._adaptation_step += 1
        
        return self._compute_weighted_sum(component_losses)
    
    def _update_adaptive_weights(self):
        """Update weights for adaptive components based on loss progress."""
        if len(self._loss_history) < 2:
            return
        
        current_losses = self._loss_history[-1]
        previous_losses = self._loss_history[-2]
        
        for i, comp in enumerate(self.components):
            if not comp.adaptive:
                continue
            
            # Compute progress (reduction in loss)
            current_loss = current_losses[i]
            previous_loss = previous_losses[i]
            
            if previous_loss > 0:
                progress = (previous_loss - current_loss) / previous_loss
            else:
                progress = 0.0
            
            # Increase weight for components making poor progress
            if progress < self.convergence_threshold:
                weight_increase = self.adaptation_rate * (1 - progress)
                comp.weight = min(comp.weight + weight_increase, 2.0)  # Cap at 2.0
            else:
                # Decrease weight for components making good progress
                weight_decrease = self.adaptation_rate * progress
                comp.weight = max(comp.weight - weight_decrease, 0.1)  # Floor at 0.1
    
    def _compute_pareto_optimal(self, component_losses: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute Pareto-optimal combination of losses."""
        # Simplified Pareto optimization using weighted Chebyshev scalarization
        losses = [component_losses[comp.name] for comp in self.components]
        weights = [comp.weight for comp in self.components]
        
        if self.normalize_weights and sum(weights) > 0:
            total_weight = sum(weights)
            weights = [w / total_weight for w in weights]
        
        # Normalize losses to [0, 1] range for fair comparison
        normalized_losses = []
        for loss in losses:
            if hasattr(self, '_loss_range'):
                min_loss, max_loss = self._loss_range.get(loss, (0, 1))
                if max_loss > min_loss:
                    norm_loss = (loss - min_loss) / (max_loss - min_loss)
                else:
                    norm_loss = loss
            else:
                norm_loss = torch.sigmoid(loss)  # Fallback normalization
            normalized_losses.append(norm_loss)
        
        # Weighted Chebyshev scalarization
        weighted_normalized = [w * loss for w, loss in zip(weights, normalized_losses)]
        composite_loss = torch.max(torch.stack(weighted_normalized))
        
        return composite_loss
    
    def _compute_hierarchical(self, component_losses: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute hierarchical combination based on priorities."""
        # Sort components by priority (descending)
        sorted_components = sorted(self.components, key=lambda x: x.priority, reverse=True)
        
        composite_loss = torch.tensor(0.0, device=self.device, dtype=self.config.dtype)
        total_priority = sum(comp.priority for comp in self.components)
        
        for comp in sorted_components:
            priority_weight = comp.priority / total_priority if total_priority > 0 else 1.0
            weighted_loss = comp.weight * priority_weight * component_losses[comp.name]
            composite_loss = composite_loss + weighted_loss
        
        return composite_loss
    
    def _compute_alternating(self, component_losses: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute alternating combination (focus on one component at a time)."""
        if not self.components:
            return torch.tensor(0.0, device=self.device)
        
        # Select current component based on alternating index
        current_comp = self.components[self._alternating_index % len(self.components)]
        
        # Use only the current component's loss
        composite_loss = current_comp.weight * component_losses[current_comp.name]
        
        # Move to next component for next iteration
        self._alternating_index += 1
        
        return composite_loss
    
    def _apply_gradient_balancing(self, 
                                composite_loss: torch.Tensor,
                                component_losses: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Apply gradient balancing across components."""
        # This is a simplified gradient balancing approach
        # In practice, you might want more sophisticated methods
        
        if len(component_losses) <= 1:
            return composite_loss
        
        # Compute gradient norms for each component (approximation)
        loss_magnitudes = [torch.abs(loss).item() for loss in component_losses.values()]
        
        if max(loss_magnitudes) > 0:
            # Normalize by maximum magnitude to prevent any component from dominating
            max_magnitude = max(loss_magnitudes)
            balanced_loss = composite_loss / max_magnitude
        else:
            balanced_loss = composite_loss
        
        return balanced_loss
    
    def get_component_info(self) -> Dict[str, Any]:
        """Get information about all components.
        
        Returns:
            Dictionary with component information
        """
        component_info = []
        for comp in self.components:
            info = {
                "name": comp.name,
                "weight": comp.weight,
                "priority": comp.priority,
                "adaptive": comp.adaptive,
                "class": comp.loss_function.__class__.__name__
            }
            
            # Add component-specific info if available
            if hasattr(comp.loss_function, 'get_info'):
                info["details"] = comp.loss_function.get_info()
            
            component_info.append(info)
        
        return {
            "strategy": self.strategy.value,
            "num_components": len(self.components),
            "components": component_info,
            "adaptation_step": self._adaptation_step,
            "normalize_weights": self.normalize_weights,
            "gradient_balancing": self.gradient_balancing
        }
    
    def reset_adaptation(self):
        """Reset adaptive state for adaptive strategies."""
        self._adaptation_step = 0
        self._loss_history.clear()
        self._weight_history.clear()
        self._alternating_index = 0
        
        if self.strategy == CompositionStrategy.ADAPTIVE_WEIGHTS:
            self._initialize_adaptive_weights()
        
        logger.info("Reset adaptation state")


class BatchCompositeLoss(BatchLossFunction):
    """Batch-optimized version of CompositeLoss."""
    
    def __init__(self, 
                 config: Optional[LossConfig] = None,
                 batch_size: int = 8,
                 **composite_kwargs):
        """Initialize batch composite loss.
        
        Args:
            config: Loss configuration
            batch_size: Expected batch size
            **composite_kwargs: Arguments for CompositeLoss
        """
        super().__init__(config, batch_size)
        
        # Create internal composite loss
        self.composite_loss = CompositeLoss(config=config, **composite_kwargs)
        
        # Delegate component management to internal composite loss
        self.components = self.composite_loss.components
        self.strategy = self.composite_loss.strategy
    
    def compute_loss(self, *args, **kwargs) -> torch.Tensor:
        """Delegate to internal composite loss."""
        return self.composite_loss.compute_loss(*args, **kwargs)
    
    def add_component(self, *args, **kwargs):
        """Delegate to internal composite loss."""
        return self.composite_loss.add_component(*args, **kwargs)
    
    def remove_component(self, *args, **kwargs):
        """Delegate to internal composite loss."""
        return self.composite_loss.remove_component(*args, **kwargs)
    
    def get_component_info(self) -> Dict[str, Any]:
        """Get component info including batch information."""
        info = self.composite_loss.get_component_info()
        info["batch_size"] = self.batch_size
        return info


# Helper functions for creating common composite loss configurations

def create_balanced_attack_loss(targeted_loss: LossFunction,
                              non_targeted_loss: LossFunction,
                              config: Optional[LossConfig] = None,
                              targeted_weight: float = 0.7,
                              non_targeted_weight: float = 0.3) -> CompositeLoss:
    """Create a balanced attack combining targeted and non-targeted losses.
    
    Args:
        targeted_loss: Targeted attack loss function
        non_targeted_loss: Non-targeted attack loss function
        config: Loss configuration
        targeted_weight: Weight for targeted component
        non_targeted_weight: Weight for non-targeted component
        
    Returns:
        Configured CompositeLoss instance
    """
    composite = CompositeLoss(config=config, strategy=CompositionStrategy.WEIGHTED_SUM)
    composite.add_component(targeted_loss, targeted_weight, "targeted", priority=2)
    composite.add_component(non_targeted_loss, non_targeted_weight, "non_targeted", priority=1)
    
    return composite


def create_adaptive_multi_objective_loss(loss_functions: List[LossFunction],
                                       config: Optional[LossConfig] = None,
                                       initial_weights: Optional[List[float]] = None) -> CompositeLoss:
    """Create an adaptive multi-objective composite loss.
    
    Args:
        loss_functions: List of loss functions to combine
        config: Loss configuration
        initial_weights: Initial weights for components
        
    Returns:
        Configured CompositeLoss with adaptive weights
    """
    if initial_weights is None:
        initial_weights = [1.0] * len(loss_functions)
    
    if len(initial_weights) != len(loss_functions):
        raise ValueError("Number of weights must match number of loss functions")
    
    composite = CompositeLoss(
        config=config, 
        strategy=CompositionStrategy.ADAPTIVE_WEIGHTS,
        adaptation_rate=0.05
    )
    
    for i, (loss_fn, weight) in enumerate(zip(loss_functions, initial_weights)):
        composite.add_component(
            loss_fn, 
            weight, 
            f"objective_{i}",
            adaptive=True
        )
    
    return composite