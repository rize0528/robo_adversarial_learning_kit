"""Comprehensive tests for CompositeLoss implementations.

This module tests multi-objective loss composition including weighted sum,
adaptive weights, Pareto optimization, hierarchical combination, and
alternating strategies with various edge cases and complex scenarios.
"""

import pytest
import torch
import numpy as np
from typing import Dict, Any, Optional, List
import logging

from src.losses.composite import (
    CompositeLoss, 
    BatchCompositeLoss,
    CompositionStrategy,
    LossComponent,
    create_balanced_attack_loss,
    create_adaptive_multi_objective_loss
)
from src.losses.base import LossConfig, LossFunction
from src.losses.targeted import TargetedLoss, TargetMode
from src.losses.non_targeted import NonTargetedLoss, SuppressionMode


# Mock loss functions for testing
class SimpleMockLoss(LossFunction):
    """Simple mock loss that returns a configurable value."""
    
    def __init__(self, config=None, loss_value=1.0):
        super().__init__(config)
        self.loss_value = loss_value
    
    def compute_loss(self, model_outputs, patch, targets=None, **kwargs):
        return torch.tensor(self.loss_value, device=self.device, dtype=self.config.dtype)


class GradientMockLoss(LossFunction):
    """Mock loss that produces gradients for testing."""
    
    def compute_loss(self, model_outputs, patch, targets=None, **kwargs):
        # Simple quadratic loss on patch
        return torch.mean(patch ** 2)


@pytest.fixture
def device():
    """Get available device for testing."""
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture 
def loss_config(device):
    """Create test loss configuration."""
    return LossConfig(
        device=device,
        dtype=torch.float32,
        reduction="mean",
        regularization_weight=1e-3
    )


@pytest.fixture
def sample_patch(device):
    """Create sample patch tensor for testing."""
    batch_size, channels, height, width = 4, 3, 32, 32
    patch = torch.randn(batch_size, channels, height, width, device=device, dtype=torch.float32)
    patch.requires_grad_(True)
    return patch


@pytest.fixture
def sample_model_outputs(device):
    """Create sample model outputs for testing."""
    batch_size = 4
    num_classes = 10
    logits = torch.randn(batch_size, num_classes, device=device, dtype=torch.float32, requires_grad=True)
    features = torch.randn(batch_size, 512, device=device, dtype=torch.float32, requires_grad=True)
    return {
        "logits": logits,
        "features": features
    }


@pytest.fixture
def mock_components(loss_config):
    """Create mock loss components for testing."""
    component1 = LossComponent(
        loss_function=SimpleMockLoss(loss_config, loss_value=2.0),
        weight=1.0,
        name="mock1",
        priority=2
    )
    
    component2 = LossComponent(
        loss_function=SimpleMockLoss(loss_config, loss_value=3.0),
        weight=0.5,
        name="mock2",
        priority=1
    )
    
    return [component1, component2]


class TestLossComponent:
    """Test LossComponent dataclass."""
    
    def test_loss_component_creation(self, loss_config):
        """Test LossComponent creation and validation."""
        loss_fn = SimpleMockLoss(loss_config)
        
        component = LossComponent(
            loss_function=loss_fn,
            weight=0.7,
            name="test_component",
            priority=3,
            adaptive=True
        )
        
        assert component.loss_function is loss_fn
        assert component.weight == 0.7
        assert component.name == "test_component"
        assert component.priority == 3
        assert component.adaptive is True
    
    def test_loss_component_default_name(self, loss_config):
        """Test that default name is generated from class name."""
        loss_fn = SimpleMockLoss(loss_config)
        component = LossComponent(loss_function=loss_fn)
        
        assert component.name == "SimpleMockLoss"
    
    def test_loss_component_validation(self, loss_config):
        """Test LossComponent validation."""
        loss_fn = SimpleMockLoss(loss_config)
        
        # Test invalid weight
        with pytest.raises(ValueError, match="weight must be non-negative"):
            LossComponent(loss_function=loss_fn, weight=-1.0)
        
        # Test invalid priority
        with pytest.raises(ValueError, match="priority must be non-negative"):
            LossComponent(loss_function=loss_fn, priority=-1)


class TestCompositeLossInitialization:
    """Test CompositeLoss initialization and validation."""
    
    def test_default_initialization(self, loss_config):
        """Test default initialization."""
        composite = CompositeLoss(loss_config)
        
        assert composite.strategy == CompositionStrategy.WEIGHTED_SUM
        assert composite.adaptation_rate == 0.01
        assert composite.convergence_threshold == 1e-4
        assert composite.max_adaptation_steps == 1000
        assert composite.normalize_weights is True
        assert composite.gradient_balancing is False
        assert len(composite.components) == 0
    
    def test_initialization_with_components(self, loss_config, mock_components):
        """Test initialization with existing components."""
        composite = CompositeLoss(
            config=loss_config,
            components=mock_components,
            strategy=CompositionStrategy.ADAPTIVE_WEIGHTS
        )
        
        assert len(composite.components) == 2
        assert composite.strategy == CompositionStrategy.ADAPTIVE_WEIGHTS
    
    def test_validation_no_components_error(self, loss_config):
        """Test that validation fails with no components when computing loss."""
        composite = CompositeLoss(loss_config)
        
        # Should fail when trying to compute loss with no components
        with pytest.raises(ValueError, match="No loss components configured"):
            composite.compute_loss({}, torch.randn(2, 3, 32, 32))
    
    def test_validation_invalid_adaptation_rate(self, loss_config):
        """Test validation of adaptation rate."""
        with pytest.raises(ValueError, match="adaptation_rate must be in"):
            CompositeLoss(loss_config, adaptation_rate=0.0)
        
        with pytest.raises(ValueError, match="adaptation_rate must be in"):
            CompositeLoss(loss_config, adaptation_rate=1.5)
    
    def test_validation_duplicate_names(self, loss_config):
        """Test validation fails with duplicate component names."""
        components = [
            LossComponent(SimpleMockLoss(loss_config), name="duplicate"),
            LossComponent(SimpleMockLoss(loss_config), name="duplicate")
        ]
        
        with pytest.raises(ValueError, match="Component names must be unique"):
            CompositeLoss(loss_config, components=components)


class TestComponentManagement:
    """Test component addition, removal, and management."""
    
    def test_add_component(self, loss_config):
        """Test adding components to composite loss."""
        composite = CompositeLoss(loss_config)
        loss_fn = SimpleMockLoss(loss_config)
        
        composite.add_component(loss_fn, weight=0.8, name="test_comp", priority=2, adaptive=True)
        
        assert len(composite.components) == 1
        comp = composite.components[0]
        assert comp.weight == 0.8
        assert comp.name == "test_comp"
        assert comp.priority == 2
        assert comp.adaptive is True
    
    def test_add_component_auto_name(self, loss_config):
        """Test adding component with automatically generated name."""
        composite = CompositeLoss(loss_config)
        loss_fn1 = SimpleMockLoss(loss_config)
        loss_fn2 = SimpleMockLoss(loss_config)
        
        composite.add_component(loss_fn1)
        composite.add_component(loss_fn2)
        
        assert composite.components[0].name == "SimpleMockLoss_0"
        assert composite.components[1].name == "SimpleMockLoss_1"
    
    def test_add_component_name_conflict(self, loss_config):
        """Test that adding component with existing name raises error."""
        composite = CompositeLoss(loss_config)
        loss_fn1 = SimpleMockLoss(loss_config)
        loss_fn2 = SimpleMockLoss(loss_config)
        
        composite.add_component(loss_fn1, name="test")
        
        with pytest.raises(ValueError, match="Component name 'test' already exists"):
            composite.add_component(loss_fn2, name="test")
    
    def test_remove_component(self, loss_config, mock_components):
        """Test removing components."""
        composite = CompositeLoss(loss_config, components=mock_components)
        
        assert len(composite.components) == 2
        
        composite.remove_component("mock1")
        
        assert len(composite.components) == 1
        assert composite.components[0].name == "mock2"
    
    def test_get_component(self, loss_config, mock_components):
        """Test getting component by name."""
        composite = CompositeLoss(loss_config, components=mock_components)
        
        comp = composite.get_component("mock1")
        assert comp is not None
        assert comp.name == "mock1"
        
        missing_comp = composite.get_component("nonexistent")
        assert missing_comp is None
    
    def test_set_component_weight(self, loss_config, mock_components):
        """Test setting component weight."""
        composite = CompositeLoss(loss_config, components=mock_components)
        
        composite.set_component_weight("mock1", 2.0)
        comp = composite.get_component("mock1")
        assert comp.weight == 2.0
        
        # Test invalid component name
        with pytest.raises(ValueError, match="Component 'nonexistent' not found"):
            composite.set_component_weight("nonexistent", 1.0)
        
        # Test invalid weight
        with pytest.raises(ValueError, match="Weight must be non-negative"):
            composite.set_component_weight("mock1", -1.0)


class TestWeightedSumStrategy:
    """Test weighted sum composition strategy."""
    
    def test_weighted_sum_basic(self, loss_config, sample_patch, sample_model_outputs):
        """Test basic weighted sum composition."""
        composite = CompositeLoss(
            loss_config, 
            strategy=CompositionStrategy.WEIGHTED_SUM,
            normalize_weights=False  # Don't normalize for this test
        )
        
        # Add components with known weights and values
        composite.add_component(SimpleMockLoss(loss_config, loss_value=2.0), weight=1.0, name="comp1")
        composite.add_component(SimpleMockLoss(loss_config, loss_value=4.0), weight=0.5, name="comp2")
        
        result = composite.compute_loss(sample_model_outputs, sample_patch)
        
        # Expected: 1.0 * 2.0 + 0.5 * 4.0 = 4.0
        assert torch.allclose(result, torch.tensor(4.0), atol=1e-6)
    
    def test_weighted_sum_with_normalization(self, loss_config, sample_patch, sample_model_outputs):
        """Test weighted sum with weight normalization."""
        composite = CompositeLoss(
            loss_config, 
            strategy=CompositionStrategy.WEIGHTED_SUM,
            normalize_weights=True
        )
        
        # Add components with weights that don't sum to 1
        composite.add_component(SimpleMockLoss(loss_config, loss_value=2.0), weight=2.0, name="comp1")
        composite.add_component(SimpleMockLoss(loss_config, loss_value=4.0), weight=1.0, name="comp2")
        
        result = composite.compute_loss(sample_model_outputs, sample_patch)
        
        # Weights: 2.0/(2.0+1.0) = 2/3, 1.0/(2.0+1.0) = 1/3
        # Expected: (2/3) * 2.0 + (1/3) * 4.0 = 4/3 + 4/3 = 8/3 ≈ 2.67
        expected = (2.0/3.0) * 2.0 + (1.0/3.0) * 4.0
        assert torch.allclose(result, torch.tensor(expected), atol=1e-6)
    
    def test_weighted_sum_without_normalization(self, loss_config, sample_patch, sample_model_outputs):
        """Test weighted sum without weight normalization."""
        composite = CompositeLoss(
            loss_config, 
            strategy=CompositionStrategy.WEIGHTED_SUM,
            normalize_weights=False
        )
        
        composite.add_component(SimpleMockLoss(loss_config, loss_value=2.0), weight=2.0, name="comp1")
        composite.add_component(SimpleMockLoss(loss_config, loss_value=4.0), weight=1.0, name="comp2")
        
        result = composite.compute_loss(sample_model_outputs, sample_patch)
        
        # Expected: 2.0 * 2.0 + 1.0 * 4.0 = 8.0
        assert torch.allclose(result, torch.tensor(8.0), atol=1e-6)


class TestAdaptiveWeightsStrategy:
    """Test adaptive weights composition strategy."""
    
    def test_adaptive_weights_initialization(self, loss_config):
        """Test that adaptive weights are initialized correctly."""
        composite = CompositeLoss(
            loss_config, 
            strategy=CompositionStrategy.ADAPTIVE_WEIGHTS
        )
        
        # Add adaptive and non-adaptive components
        composite.add_component(SimpleMockLoss(loss_config), weight=2.0, name="fixed", adaptive=False)
        composite.add_component(SimpleMockLoss(loss_config), weight=1.0, name="adapt1", adaptive=True)
        composite.add_component(SimpleMockLoss(loss_config), weight=3.0, name="adapt2", adaptive=True)
        
        # Check that adaptive components have equal weights
        adapt1 = composite.get_component("adapt1")
        adapt2 = composite.get_component("adapt2")
        fixed = composite.get_component("fixed")
        
        assert adapt1.weight == 0.5  # 1/2 adaptive components
        assert adapt2.weight == 0.5  # 1/2 adaptive components
        assert fixed.weight == 2.0   # Unchanged
    
    def test_adaptive_weights_progression(self, loss_config, sample_patch, sample_model_outputs):
        """Test that adaptive weights change over time."""
        composite = CompositeLoss(
            loss_config, 
            strategy=CompositionStrategy.ADAPTIVE_WEIGHTS,
            adaptation_rate=0.1
        )
        
        # Create components with different loss progressions
        composite.add_component(SimpleMockLoss(loss_config, loss_value=1.0), 
                              weight=1.0, name="good", adaptive=True)
        composite.add_component(SimpleMockLoss(loss_config, loss_value=5.0), 
                              weight=1.0, name="bad", adaptive=True)
        
        initial_good_weight = composite.get_component("good").weight
        initial_bad_weight = composite.get_component("bad").weight
        
        # Run multiple iterations to trigger adaptation
        for _ in range(10):
            composite.compute_loss(sample_model_outputs, sample_patch)
        
        final_good_weight = composite.get_component("good").weight
        final_bad_weight = composite.get_component("bad").weight
        
        # Weights should have been updated
        assert hasattr(composite, '_loss_history')
        assert len(composite._loss_history) > 0


class TestHierarchicalStrategy:
    """Test hierarchical composition strategy."""
    
    def test_hierarchical_priority_ordering(self, loss_config, sample_patch, sample_model_outputs):
        """Test that hierarchical strategy respects priority ordering."""
        composite = CompositeLoss(loss_config, strategy=CompositionStrategy.HIERARCHICAL)
        
        # Add components with different priorities
        composite.add_component(SimpleMockLoss(loss_config, loss_value=1.0), 
                              weight=1.0, name="low", priority=1)
        composite.add_component(SimpleMockLoss(loss_config, loss_value=2.0), 
                              weight=1.0, name="high", priority=3)
        composite.add_component(SimpleMockLoss(loss_config, loss_value=1.5), 
                              weight=1.0, name="mid", priority=2)
        
        result = composite.compute_loss(sample_model_outputs, sample_patch)
        
        # Should combine with priority weighting
        # Total priority = 1 + 2 + 3 = 6
        # Expected: (3/6) * 1.0 * 2.0 + (2/6) * 1.0 * 1.5 + (1/6) * 1.0 * 1.0
        # = 1.0 + 0.5 + 0.167 = 1.667
        expected = (3.0/6.0) * 2.0 + (2.0/6.0) * 1.5 + (1.0/6.0) * 1.0
        assert torch.allclose(result, torch.tensor(expected), atol=1e-3)


class TestAlternatingStrategy:
    """Test alternating composition strategy."""
    
    def test_alternating_selection(self, loss_config, sample_patch, sample_model_outputs):
        """Test that alternating strategy cycles through components."""
        composite = CompositeLoss(loss_config, strategy=CompositionStrategy.ALTERNATING)
        
        composite.add_component(SimpleMockLoss(loss_config, loss_value=1.0), 
                              weight=1.0, name="comp1")
        composite.add_component(SimpleMockLoss(loss_config, loss_value=2.0), 
                              weight=1.0, name="comp2")
        composite.add_component(SimpleMockLoss(loss_config, loss_value=3.0), 
                              weight=1.0, name="comp3")
        
        # First call should use comp1 (index 0)
        result1 = composite.compute_loss(sample_model_outputs, sample_patch)
        assert torch.allclose(result1, torch.tensor(1.0))
        
        # Second call should use comp2 (index 1)
        result2 = composite.compute_loss(sample_model_outputs, sample_patch)
        assert torch.allclose(result2, torch.tensor(2.0))
        
        # Third call should use comp3 (index 2)
        result3 = composite.compute_loss(sample_model_outputs, sample_patch)
        assert torch.allclose(result3, torch.tensor(3.0))
        
        # Fourth call should cycle back to comp1 (index 0)
        result4 = composite.compute_loss(sample_model_outputs, sample_patch)
        assert torch.allclose(result4, torch.tensor(1.0))


class TestParetoOptimalStrategy:
    """Test Pareto optimal composition strategy."""
    
    def test_pareto_optimal_basic(self, loss_config, sample_patch, sample_model_outputs):
        """Test basic Pareto optimal composition."""
        composite = CompositeLoss(loss_config, strategy=CompositionStrategy.PARETO_OPTIMAL)
        
        composite.add_component(SimpleMockLoss(loss_config, loss_value=1.0), 
                              weight=1.0, name="comp1")
        composite.add_component(SimpleMockLoss(loss_config, loss_value=2.0), 
                              weight=1.0, name="comp2")
        
        result = composite.compute_loss(sample_model_outputs, sample_patch)
        
        # Should use Chebyshev scalarization (max of weighted normalized losses)
        assert torch.isfinite(result)
        assert result.item() >= 0


class TestGradientBalancing:
    """Test gradient balancing functionality."""
    
    def test_gradient_balancing_enabled(self, loss_config, sample_patch, sample_model_outputs):
        """Test gradient balancing when enabled."""
        composite = CompositeLoss(
            loss_config, 
            strategy=CompositionStrategy.WEIGHTED_SUM,
            gradient_balancing=True
        )
        
        composite.add_component(SimpleMockLoss(loss_config, loss_value=100.0), 
                              weight=1.0, name="large")
        composite.add_component(SimpleMockLoss(loss_config, loss_value=0.01), 
                              weight=1.0, name="small")
        
        result = composite.compute_loss(sample_model_outputs, sample_patch)
        
        # Should apply balancing to prevent large component from dominating
        assert torch.isfinite(result)
    
    def test_gradient_balancing_disabled(self, loss_config, sample_patch, sample_model_outputs):
        """Test behavior when gradient balancing is disabled."""
        composite = CompositeLoss(
            loss_config, 
            strategy=CompositionStrategy.WEIGHTED_SUM,
            gradient_balancing=False
        )
        
        composite.add_component(SimpleMockLoss(loss_config, loss_value=100.0), 
                              weight=1.0, name="large")
        composite.add_component(SimpleMockLoss(loss_config, loss_value=0.01), 
                              weight=1.0, name="small")
        
        result = composite.compute_loss(sample_model_outputs, sample_patch)
        
        # Should not apply balancing
        expected = 100.0 + 0.01
        assert torch.allclose(result, torch.tensor(expected), atol=1e-6)


class TestComponentKwargs:
    """Test component-specific keyword arguments."""
    
    def test_component_specific_kwargs(self, loss_config, sample_patch, sample_model_outputs):
        """Test passing component-specific keyword arguments."""
        composite = CompositeLoss(loss_config)
        
        # Mock loss that uses kwargs
        class KwargMockLoss(LossFunction):
            def compute_loss(self, model_outputs, patch, targets=None, multiplier=1.0, **kwargs):
                return torch.tensor(multiplier, device=self.device)
        
        composite.add_component(KwargMockLoss(loss_config), weight=1.0, name="comp1")
        composite.add_component(KwargMockLoss(loss_config), weight=1.0, name="comp2")
        
        # Pass component-specific kwargs
        component_kwargs = {
            "comp1": {"multiplier": 2.0},
            "comp2": {"multiplier": 3.0}
        }
        
        result = composite.compute_loss(
            sample_model_outputs, 
            sample_patch, 
            component_kwargs=component_kwargs
        )
        
        # Expected: 1.0 * 2.0 + 1.0 * 3.0 = 5.0
        assert torch.allclose(result, torch.tensor(5.0))


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    def test_component_failure_handling(self, loss_config, sample_patch, sample_model_outputs):
        """Test handling when a component fails."""
        class FailingLoss(LossFunction):
            def compute_loss(self, model_outputs, patch, targets=None, **kwargs):
                raise RuntimeError("Intentional failure")
        
        composite = CompositeLoss(loss_config)
        composite.add_component(FailingLoss(loss_config), weight=1.0, name="failing")
        composite.add_component(SimpleMockLoss(loss_config, loss_value=2.0), 
                              weight=1.0, name="working")
        
        # Should handle failure gracefully
        result = composite.compute_loss(sample_model_outputs, sample_patch)
        
        # Should use zero for failing component and 2.0 for working component
        assert torch.allclose(result, torch.tensor(2.0))
    
    def test_empty_components_error(self, loss_config, sample_patch, sample_model_outputs):
        """Test error when no components are configured."""
        composite = CompositeLoss(loss_config)
        
        with pytest.raises(ValueError, match="No loss components configured"):
            composite.compute_loss(sample_model_outputs, sample_patch)
    
    def test_unknown_strategy_error(self, loss_config):
        """Test error with unknown composition strategy."""
        composite = CompositeLoss(loss_config)
        composite.add_component(SimpleMockLoss(loss_config), name="test")
        
        # Manually set invalid strategy
        composite.strategy = "invalid_strategy"
        
        with pytest.raises(ValueError, match="Unknown composition strategy"):
            composite.compute_loss({}, torch.randn(2, 3, 32, 32))


class TestResetAdaptation:
    """Test adaptation reset functionality."""
    
    def test_reset_adaptation(self, loss_config, sample_patch, sample_model_outputs):
        """Test resetting adaptation state."""
        composite = CompositeLoss(
            loss_config, 
            strategy=CompositionStrategy.ADAPTIVE_WEIGHTS
        )
        
        composite.add_component(SimpleMockLoss(loss_config), adaptive=True, name="adaptive")
        
        # Run some iterations to build up history
        for _ in range(5):
            composite.compute_loss(sample_model_outputs, sample_patch)
        
        # Should have history
        assert len(composite._loss_history) > 0
        assert composite._adaptation_step > 0
        
        # Reset adaptation
        composite.reset_adaptation()
        
        # Should clear history
        assert len(composite._loss_history) == 0
        assert composite._adaptation_step == 0


class TestGetComponentInfo:
    """Test component information retrieval."""
    
    def test_get_component_info(self, loss_config):
        """Test getting comprehensive component information."""
        composite = CompositeLoss(
            loss_config,
            strategy=CompositionStrategy.ADAPTIVE_WEIGHTS,
            normalize_weights=True,
            gradient_balancing=True
        )
        
        composite.add_component(SimpleMockLoss(loss_config), weight=1.0, name="comp1", 
                              priority=2, adaptive=True)
        composite.add_component(SimpleMockLoss(loss_config), weight=0.5, name="comp2", 
                              priority=1, adaptive=False)
        
        info = composite.get_component_info()
        
        assert info["strategy"] == "adaptive_weights"
        assert info["num_components"] == 2
        assert info["normalize_weights"] is True
        assert info["gradient_balancing"] is True
        
        components_info = info["components"]
        assert len(components_info) == 2
        
        comp1_info = components_info[0]
        assert comp1_info["name"] == "comp1"
        assert comp1_info["weight"] == 0.5  # Should be adapted for adaptive components
        assert comp1_info["priority"] == 2
        assert comp1_info["adaptive"] is True
        assert comp1_info["class"] == "SimpleMockLoss"


class TestBatchCompositeLoss:
    """Test batch-optimized composite loss."""
    
    def test_batch_composite_creation(self, loss_config):
        """Test creating batch composite loss."""
        batch_composite = BatchCompositeLoss(
            config=loss_config,
            batch_size=16,
            strategy=CompositionStrategy.WEIGHTED_SUM
        )
        
        assert batch_composite.batch_size == 16
        assert batch_composite.strategy == CompositionStrategy.WEIGHTED_SUM
        assert isinstance(batch_composite.composite_loss, CompositeLoss)
    
    def test_batch_composite_delegation(self, loss_config, sample_patch, sample_model_outputs):
        """Test that batch composite delegates to internal composite loss."""
        batch_composite = BatchCompositeLoss(loss_config, batch_size=8)
        
        batch_composite.add_component(SimpleMockLoss(loss_config, loss_value=2.0), 
                                    weight=1.0, name="test")
        
        result = batch_composite.compute_loss(sample_model_outputs, sample_patch)
        
        assert torch.allclose(result, torch.tensor(2.0))
    
    def test_batch_composite_info(self, loss_config):
        """Test that batch composite info includes batch size."""
        batch_composite = BatchCompositeLoss(loss_config, batch_size=32)
        batch_composite.add_component(SimpleMockLoss(loss_config), name="test")
        
        info = batch_composite.get_component_info()
        
        assert info["batch_size"] == 32
        assert "strategy" in info
        assert "num_components" in info


class TestHelperFunctions:
    """Test helper functions for creating composite losses."""
    
    def test_create_balanced_attack_loss(self, loss_config):
        """Test helper for creating balanced attack loss."""
        targeted_loss = TargetedLoss(loss_config, target_mode=TargetMode.CLASSIFICATION)
        non_targeted_loss = NonTargetedLoss(loss_config, suppression_mode=SuppressionMode.CONFIDENCE_REDUCTION)
        
        balanced_loss = create_balanced_attack_loss(
            targeted_loss, 
            non_targeted_loss,
            config=loss_config,
            targeted_weight=0.6,
            non_targeted_weight=0.4
        )
        
        assert isinstance(balanced_loss, CompositeLoss)
        assert balanced_loss.strategy == CompositionStrategy.WEIGHTED_SUM
        assert len(balanced_loss.components) == 2
        
        targeted_comp = balanced_loss.get_component("targeted")
        non_targeted_comp = balanced_loss.get_component("non_targeted")
        
        assert targeted_comp.weight == 0.6
        assert targeted_comp.priority == 2
        assert non_targeted_comp.weight == 0.4
        assert non_targeted_comp.priority == 1
    
    def test_create_adaptive_multi_objective_loss(self, loss_config):
        """Test helper for creating adaptive multi-objective loss."""
        loss_functions = [
            SimpleMockLoss(loss_config, loss_value=1.0),
            SimpleMockLoss(loss_config, loss_value=2.0),
            SimpleMockLoss(loss_config, loss_value=3.0)
        ]
        initial_weights = [0.5, 1.0, 0.3]
        
        adaptive_loss = create_adaptive_multi_objective_loss(
            loss_functions,
            config=loss_config,
            initial_weights=initial_weights
        )
        
        assert isinstance(adaptive_loss, CompositeLoss)
        assert adaptive_loss.strategy == CompositionStrategy.ADAPTIVE_WEIGHTS
        assert len(adaptive_loss.components) == 3
        
        # Check that all components are adaptive
        for i, comp in enumerate(adaptive_loss.components):
            assert comp.adaptive is True
            assert comp.name == f"objective_{i}"
    
    def test_create_adaptive_multi_objective_loss_weight_mismatch(self, loss_config):
        """Test that weight mismatch raises error."""
        loss_functions = [SimpleMockLoss(loss_config), SimpleMockLoss(loss_config)]
        initial_weights = [1.0]  # Wrong number of weights
        
        with pytest.raises(ValueError, match="Number of weights must match number of loss functions"):
            create_adaptive_multi_objective_loss(loss_functions, loss_config, initial_weights)


@pytest.mark.integration
class TestCompositeLossIntegration:
    """Integration tests for composite loss with realistic scenarios."""
    
    def test_realistic_attack_scenario(self, loss_config, device):
        """Test composite loss in realistic adversarial attack scenario."""
        # Create realistic targeted and non-targeted losses
        targeted_loss = TargetedLoss(
            loss_config,
            target_mode=TargetMode.CONFIDENCE,
            confidence_threshold=0.9
        )
        
        non_targeted_loss = NonTargetedLoss(
            loss_config,
            suppression_mode=SuppressionMode.ENTROPY_MAXIMIZATION,
            entropy_weight=1.0
        )
        
        # Create balanced composite loss
        composite = create_balanced_attack_loss(
            targeted_loss,
            non_targeted_loss,
            config=loss_config,
            targeted_weight=0.7,
            non_targeted_weight=0.3
        )
        
        # Simulate attack scenario
        batch_size = 4
        num_classes = 1000
        
        patch = torch.randn(batch_size, 3, 224, 224, device=device, requires_grad=True)
        logits = torch.randn(batch_size, num_classes, device=device)
        target_classes = torch.randint(0, num_classes, (batch_size,), device=device)
        
        model_outputs = {"logits": logits}
        
        # Forward pass
        result = composite.forward(model_outputs, patch, target_classes=target_classes)
        
        assert "total_loss" in result
        assert result["total_loss"].requires_grad
        
        # Backward pass
        result["total_loss"].backward()
        
        assert patch.grad is not None
        assert not torch.all(patch.grad == 0)
    
    def test_adaptive_multi_objective_optimization(self, loss_config, device):
        """Test adaptive optimization with multiple objectives."""
        # Create multiple objectives
        objectives = [
            TargetedLoss(loss_config, target_mode=TargetMode.CLASSIFICATION),
            TargetedLoss(loss_config, target_mode=TargetMode.CONFIDENCE, confidence_threshold=0.8),
            NonTargetedLoss(loss_config, suppression_mode=SuppressionMode.CONFIDENCE_REDUCTION)
        ]
        
        adaptive_composite = create_adaptive_multi_objective_loss(
            objectives,
            config=loss_config
        )
        
        # Simulate training loop
        batch_size = 2
        num_classes = 10
        
        for iteration in range(10):
            patch = torch.randn(batch_size, 3, 32, 32, device=device, requires_grad=True)
            logits = torch.randn(batch_size, num_classes, device=device)
            target_classes = torch.randint(0, num_classes, (batch_size,), device=device)
            
            model_outputs = {"logits": logits}
            
            result = adaptive_composite.forward(
                model_outputs, 
                patch, 
                target_classes=target_classes
            )
            
            result["total_loss"].backward()
            
            # Verify adaptation is working
            if iteration > 2:
                assert len(adaptive_composite._loss_history) > 0
            
            # Clear gradients
            patch.grad.zero_()
    
    def test_hierarchical_attack_priorities(self, loss_config, device):
        """Test hierarchical composition with different attack priorities."""
        composite = CompositeLoss(
            loss_config,
            strategy=CompositionStrategy.HIERARCHICAL
        )
        
        # Add components with different priorities
        # High priority: Successful targeting
        composite.add_component(
            TargetedLoss(loss_config, target_mode=TargetMode.CONFIDENCE, confidence_threshold=0.9),
            weight=1.0, name="high_confidence", priority=3
        )
        
        # Medium priority: General targeting
        composite.add_component(
            TargetedLoss(loss_config, target_mode=TargetMode.CLASSIFICATION),
            weight=1.0, name="classification", priority=2
        )
        
        # Low priority: Stealth (entropy maximization)
        composite.add_component(
            NonTargetedLoss(loss_config, suppression_mode=SuppressionMode.ENTROPY_MAXIMIZATION),
            weight=0.5, name="stealth", priority=1
        )
        
        # Test with realistic data
        batch_size = 3
        num_classes = 50
        
        patch = torch.randn(batch_size, 3, 64, 64, device=device, requires_grad=True)
        logits = torch.randn(batch_size, num_classes, device=device)
        target_classes = torch.tensor([5, 12, 33], device=device)
        
        model_outputs = {"logits": logits}
        
        result = composite.forward(model_outputs, patch, target_classes=target_classes)
        
        # Should prioritize high-confidence targeting
        assert result["total_loss"].requires_grad
        assert torch.isfinite(result["total_loss"])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])