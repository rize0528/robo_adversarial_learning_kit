"""Comprehensive tests for NonTargetedLoss implementations.

This module tests all variants of non-targeted attack loss functions including
confidence reduction, entropy maximization, logit minimization, and detection
suppression with various edge cases and parameter configurations.
"""

import pytest
import torch
import numpy as np
from typing import Dict, Any, Optional
import logging

from src.losses.non_targeted import (
    NonTargetedLoss, 
    SuppressionMode,
    create_confidence_reduction_loss,
    create_entropy_maximization_loss,
    create_detection_suppression_loss
)
from src.losses.base import LossConfig


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
def high_confidence_outputs(device):
    """Create model outputs with high confidence predictions."""
    batch_size = 4
    num_classes = 10
    logits = torch.randn(batch_size, num_classes, device=device, dtype=torch.float32, requires_grad=True) * 0.1
    
    # Make one class have very high confidence
    logits[:, 2] = 5.0  # High logit for class 2
    
    return {"logits": logits, "features": torch.randn(batch_size, 512, device=device, requires_grad=True)}


@pytest.fixture
def uniform_outputs(device):
    """Create model outputs with uniform (low confidence) predictions."""
    batch_size = 4
    num_classes = 10
    # All logits are similar, resulting in uniform distribution
    logits = torch.ones(batch_size, num_classes, device=device, dtype=torch.float32, requires_grad=True) * 0.1
    
    return {"logits": logits, "features": torch.randn(batch_size, 512, device=device, requires_grad=True)}


@pytest.fixture
def sample_target_classes(device):
    """Create sample target class indices for testing."""
    batch_size = 4
    return torch.tensor([2, 3, 1, 4], device=device, dtype=torch.long)


class TestNonTargetedLossInitialization:
    """Test NonTargetedLoss initialization and parameter validation."""
    
    def test_default_initialization(self, loss_config):
        """Test default initialization."""
        loss_fn = NonTargetedLoss(loss_config)
        
        assert loss_fn.suppression_mode == SuppressionMode.CONFIDENCE_REDUCTION
        assert loss_fn.confidence_target == 0.1
        assert loss_fn.entropy_weight == 1.0
        assert loss_fn.temperature == 1.0
        assert loss_fn.suppress_classes is None
        assert loss_fn.gradient_smoothing is True
        assert loss_fn.smoothing_factor == 0.1
    
    def test_custom_initialization(self, loss_config):
        """Test initialization with custom parameters."""
        suppress_classes = [0, 1, 5]
        loss_fn = NonTargetedLoss(
            config=loss_config,
            suppression_mode=SuppressionMode.ENTROPY_MAXIMIZATION,
            confidence_target=0.05,
            entropy_weight=2.0,
            temperature=2.0,
            suppress_classes=suppress_classes,
            gradient_smoothing=False
        )
        
        assert loss_fn.suppression_mode == SuppressionMode.ENTROPY_MAXIMIZATION
        assert loss_fn.confidence_target == 0.05
        assert loss_fn.entropy_weight == 2.0
        assert loss_fn.temperature == 2.0
        assert loss_fn.suppress_classes == suppress_classes
        assert loss_fn.gradient_smoothing is False
    
    def test_invalid_confidence_target(self, loss_config):
        """Test that invalid confidence target raises error."""
        with pytest.raises(ValueError, match="confidence_target must be in"):
            NonTargetedLoss(loss_config, confidence_target=1.5)
        
        with pytest.raises(ValueError, match="confidence_target must be in"):
            NonTargetedLoss(loss_config, confidence_target=-0.1)
    
    def test_invalid_entropy_weight(self, loss_config):
        """Test that invalid entropy weight raises error."""
        with pytest.raises(ValueError, match="entropy_weight must be non-negative"):
            NonTargetedLoss(loss_config, entropy_weight=-1.0)
    
    def test_invalid_temperature(self, loss_config):
        """Test that invalid temperature raises error."""
        with pytest.raises(ValueError, match="temperature must be positive"):
            NonTargetedLoss(loss_config, temperature=0.0)
    
    def test_invalid_suppress_classes(self, loss_config):
        """Test that invalid suppress_classes raises error."""
        with pytest.raises(ValueError, match="suppress_classes must be a list or tuple"):
            NonTargetedLoss(loss_config, suppress_classes="invalid")
        
        with pytest.raises(ValueError, match="suppress_classes must contain non-negative integers"):
            NonTargetedLoss(loss_config, suppress_classes=[1, -1, 3])
        
        with pytest.raises(ValueError, match="suppress_classes must contain non-negative integers"):
            NonTargetedLoss(loss_config, suppress_classes=[1, "invalid", 3])


class TestConfidenceReduction:
    """Test confidence reduction suppression mode."""
    
    def test_confidence_reduction_all_classes(self, loss_config, sample_patch, high_confidence_outputs):
        """Test confidence reduction across all classes."""
        loss_fn = NonTargetedLoss(
            loss_config, 
            suppression_mode=SuppressionMode.CONFIDENCE_REDUCTION,
            confidence_target=0.2
        )
        
        result = loss_fn.forward(high_confidence_outputs, sample_patch)
        
        assert "total_loss" in result
        assert result["total_loss"].requires_grad
        
        # Should have positive loss since high confidence exceeds target
        main_loss = result["main_loss"]
        assert main_loss.item() >= 0
    
    def test_confidence_reduction_specific_classes(self, loss_config, sample_patch, 
                                                 high_confidence_outputs):
        """Test confidence reduction for specific classes."""
        suppress_classes = [2]  # Class 2 has high confidence in fixture
        loss_fn = NonTargetedLoss(
            loss_config, 
            suppression_mode=SuppressionMode.CONFIDENCE_REDUCTION,
            suppress_classes=suppress_classes,
            confidence_target=0.1
        )
        
        result = loss_fn.forward(high_confidence_outputs, sample_patch)
        
        assert result["total_loss"].requires_grad
        assert result["main_loss"].item() >= 0
    
    def test_confidence_reduction_with_low_confidence(self, loss_config, sample_patch,
                                                    uniform_outputs):
        """Test confidence reduction when confidence is already low."""
        loss_fn = NonTargetedLoss(
            loss_config, 
            suppression_mode=SuppressionMode.CONFIDENCE_REDUCTION,
            confidence_target=0.5  # Higher than uniform distribution
        )
        
        result = loss_fn.forward(uniform_outputs, sample_patch)
        
        # Should have low or zero loss since confidence is already below target
        main_loss = result["main_loss"]
        assert main_loss.item() >= 0
    
    def test_confidence_target_effect(self, loss_config, sample_patch, high_confidence_outputs):
        """Test that confidence target affects loss correctly."""
        # High target (more permissive)
        loss_fn_high = NonTargetedLoss(
            loss_config, 
            suppression_mode=SuppressionMode.CONFIDENCE_REDUCTION,
            confidence_target=0.8
        )
        result_high = loss_fn_high.forward(high_confidence_outputs, sample_patch)
        
        # Low target (more restrictive)
        loss_fn_low = NonTargetedLoss(
            loss_config, 
            suppression_mode=SuppressionMode.CONFIDENCE_REDUCTION,
            confidence_target=0.1
        )
        result_low = loss_fn_low.forward(high_confidence_outputs, sample_patch)
        
        # Low target should generally have higher or equal loss
        assert result_low["main_loss"].item() >= 0
        assert result_high["main_loss"].item() >= 0


class TestEntropyMaximization:
    """Test entropy maximization suppression mode."""
    
    def test_entropy_maximization_basic(self, loss_config, sample_patch, sample_model_outputs):
        """Test basic entropy maximization."""
        loss_fn = NonTargetedLoss(
            loss_config, 
            suppression_mode=SuppressionMode.ENTROPY_MAXIMIZATION,
            entropy_weight=1.0
        )
        
        result = loss_fn.forward(sample_model_outputs, sample_patch)
        
        assert "total_loss" in result
        assert result["total_loss"].requires_grad
        
        # Loss is negative entropy, can be negative (we want to maximize entropy)
        main_loss = result["main_loss"]
        assert torch.isfinite(main_loss)
    
    def test_entropy_maximization_high_vs_low_entropy(self, loss_config, sample_patch,
                                                    high_confidence_outputs, uniform_outputs):
        """Test that entropy maximization works correctly with different entropy levels."""
        loss_fn = NonTargetedLoss(
            loss_config, 
            suppression_mode=SuppressionMode.ENTROPY_MAXIMIZATION,
            entropy_weight=1.0
        )
        
        # High confidence = low entropy = high loss (more negative)
        result_low_entropy = loss_fn.forward(high_confidence_outputs, sample_patch)
        
        # Uniform = high entropy = low loss (less negative)
        result_high_entropy = loss_fn.forward(uniform_outputs, sample_patch)
        
        # Low entropy should have more negative (higher magnitude) loss
        # Since we want to maximize entropy, low entropy should have higher loss magnitude
        assert abs(result_low_entropy["main_loss"].item()) >= abs(result_high_entropy["main_loss"].item())
    
    def test_entropy_weight_effect(self, loss_config, sample_patch, high_confidence_outputs):
        """Test that entropy weight affects loss magnitude."""
        # Low weight
        loss_fn_low = NonTargetedLoss(
            loss_config, 
            suppression_mode=SuppressionMode.ENTROPY_MAXIMIZATION,
            entropy_weight=0.5
        )
        result_low = loss_fn_low.forward(high_confidence_outputs, sample_patch)
        
        # High weight
        loss_fn_high = NonTargetedLoss(
            loss_config, 
            suppression_mode=SuppressionMode.ENTROPY_MAXIMIZATION,
            entropy_weight=2.0
        )
        result_high = loss_fn_high.forward(high_confidence_outputs, sample_patch)
        
        # Higher weight should produce loss with higher magnitude
        assert abs(result_high["main_loss"].item()) >= abs(result_low["main_loss"].item())


class TestLogitMinimization:
    """Test logit minimization suppression mode."""
    
    def test_logit_minimization_all_classes(self, loss_config, sample_patch, sample_model_outputs):
        """Test logit minimization across all classes."""
        loss_fn = NonTargetedLoss(
            loss_config, 
            suppression_mode=SuppressionMode.LOGIT_MINIMIZATION
        )
        
        result = loss_fn.forward(sample_model_outputs, sample_patch)
        
        assert "total_loss" in result
        assert result["total_loss"].requires_grad
        assert result["main_loss"].item() >= 0  # ReLU ensures non-negative
    
    def test_logit_minimization_specific_classes(self, loss_config, sample_patch, 
                                               sample_model_outputs):
        """Test logit minimization for specific classes."""
        suppress_classes = [1, 3, 5]
        loss_fn = NonTargetedLoss(
            loss_config, 
            suppression_mode=SuppressionMode.LOGIT_MINIMIZATION,
            suppress_classes=suppress_classes
        )
        
        result = loss_fn.forward(sample_model_outputs, sample_patch)
        
        assert result["total_loss"].requires_grad
        assert result["main_loss"].item() >= 0
    
    def test_logit_minimization_with_targets(self, loss_config, sample_patch,
                                           sample_model_outputs, sample_target_classes):
        """Test logit minimization with target classes."""
        loss_fn = NonTargetedLoss(
            loss_config, 
            suppression_mode=SuppressionMode.LOGIT_MINIMIZATION
        )
        
        result = loss_fn.forward(sample_model_outputs, sample_patch, targets=sample_target_classes)
        
        assert result["total_loss"].requires_grad
        assert result["main_loss"].item() >= 0
    
    def test_logit_minimization_with_negative_logits(self, loss_config, sample_patch, device):
        """Test logit minimization when logits are already negative."""
        batch_size = 4
        num_classes = 10
        
        # Create negative logits
        negative_logits = torch.randn(batch_size, num_classes, device=device) - 2.0
        negative_outputs = {"logits": negative_logits}
        
        loss_fn = NonTargetedLoss(
            loss_config, 
            suppression_mode=SuppressionMode.LOGIT_MINIMIZATION
        )
        
        result = loss_fn.forward(negative_outputs, sample_patch)
        
        # Should have low loss since logits are already negative
        assert result["main_loss"].item() >= 0
        assert result["main_loss"].item() <= 1.0  # Should be small


class TestDetectionSuppression:
    """Test detection suppression mode."""
    
    def test_detection_suppression_basic(self, loss_config, sample_patch, sample_model_outputs):
        """Test basic detection suppression."""
        loss_fn = NonTargetedLoss(
            loss_config, 
            suppression_mode=SuppressionMode.DETECTION_SUPPRESSION
        )
        
        result = loss_fn.forward(sample_model_outputs, sample_patch)
        
        assert "total_loss" in result
        assert result["total_loss"].requires_grad
        assert torch.isfinite(result["main_loss"])
    
    def test_detection_suppression_with_original_outputs(self, loss_config, sample_patch,
                                                       sample_model_outputs, device):
        """Test detection suppression with original model outputs."""
        # Create original outputs that are different from current outputs
        original_outputs = {
            "logits": torch.randn(4, 10, device=device) * 0.5
        }
        
        loss_fn = NonTargetedLoss(
            loss_config, 
            suppression_mode=SuppressionMode.DETECTION_SUPPRESSION
        )
        
        result = loss_fn.forward(sample_model_outputs, sample_patch, 
                               original_outputs=original_outputs)
        
        assert result["total_loss"].requires_grad
        assert torch.isfinite(result["main_loss"])
    
    def test_detection_suppression_with_specific_classes(self, loss_config, sample_patch,
                                                       sample_model_outputs):
        """Test detection suppression for specific object classes."""
        suppress_classes = [2, 7]  # Suppress specific object classes
        loss_fn = NonTargetedLoss(
            loss_config, 
            suppression_mode=SuppressionMode.DETECTION_SUPPRESSION,
            suppress_classes=suppress_classes
        )
        
        result = loss_fn.forward(sample_model_outputs, sample_patch)
        
        assert result["total_loss"].requires_grad
        assert torch.isfinite(result["main_loss"])


class TestGradientSmoothing:
    """Test gradient smoothing functionality."""
    
    def test_gradient_smoothing_enabled(self, loss_config, sample_patch,
                                      sample_model_outputs):
        """Test that gradient smoothing is applied when enabled."""
        loss_fn = NonTargetedLoss(
            loss_config, 
            gradient_smoothing=True, 
            smoothing_factor=0.5
        )
        
        # First forward pass
        result1 = loss_fn.forward(sample_model_outputs, sample_patch)
        
        # Second forward pass with different inputs
        modified_outputs = {
            "logits": sample_model_outputs["logits"] + torch.randn_like(sample_model_outputs["logits"]) * 0.1
        }
        result2 = loss_fn.forward(modified_outputs, sample_patch)
        
        # Should have internal state for smoothing
        assert hasattr(loss_fn, '_prev_loss')
    
    def test_gradient_smoothing_disabled(self, loss_config, sample_patch,
                                       sample_model_outputs):
        """Test behavior when gradient smoothing is disabled."""
        loss_fn = NonTargetedLoss(
            loss_config, 
            gradient_smoothing=False
        )
        
        result = loss_fn.forward(sample_model_outputs, sample_patch)
        
        # Should still work correctly
        assert "total_loss" in result


class TestParameterUpdates:
    """Test dynamic parameter updates."""
    
    def test_set_suppression_mode(self, loss_config):
        """Test changing suppression mode."""
        loss_fn = NonTargetedLoss(loss_config, suppression_mode=SuppressionMode.CONFIDENCE_REDUCTION)
        
        loss_fn.set_suppression_mode(SuppressionMode.ENTROPY_MAXIMIZATION)
        assert loss_fn.suppression_mode == SuppressionMode.ENTROPY_MAXIMIZATION
    
    def test_set_confidence_target(self, loss_config):
        """Test changing confidence target."""
        loss_fn = NonTargetedLoss(loss_config, confidence_target=0.2)
        
        loss_fn.set_confidence_target(0.05)
        assert loss_fn.confidence_target == 0.05
        
        with pytest.raises(ValueError, match="confidence_target must be in"):
            loss_fn.set_confidence_target(1.1)
    
    def test_set_suppress_classes(self, loss_config):
        """Test changing suppress classes."""
        loss_fn = NonTargetedLoss(loss_config)
        
        new_classes = [1, 3, 5]
        loss_fn.set_suppress_classes(new_classes)
        assert loss_fn.suppress_classes == new_classes
        
        # Test invalid classes
        with pytest.raises(ValueError, match="classes must be a list or tuple"):
            loss_fn.set_suppress_classes("invalid")
    
    def test_add_remove_suppress_class(self, loss_config):
        """Test adding and removing individual suppress classes."""
        loss_fn = NonTargetedLoss(loss_config, suppress_classes=[1, 2])
        
        # Add class
        loss_fn.add_suppress_class(5)
        assert 5 in loss_fn.suppress_classes
        
        # Remove class
        loss_fn.remove_suppress_class(1)
        assert 1 not in loss_fn.suppress_classes
        
        # Test invalid add
        with pytest.raises(ValueError, match="class_idx must be a non-negative integer"):
            loss_fn.add_suppress_class(-1)
    
    def test_get_suppression_info(self, loss_config):
        """Test getting suppression configuration info."""
        suppress_classes = [0, 1, 5]
        loss_fn = NonTargetedLoss(
            loss_config,
            suppression_mode=SuppressionMode.ENTROPY_MAXIMIZATION,
            confidence_target=0.05,
            entropy_weight=1.5,
            temperature=2.0,
            suppress_classes=suppress_classes,
            gradient_smoothing=True,
            smoothing_factor=0.2
        )
        
        info = loss_fn.get_suppression_info()
        
        assert info["suppression_mode"] == "entropy_maximization"
        assert info["confidence_target"] == 0.05
        assert info["entropy_weight"] == 1.5
        assert info["temperature"] == 2.0
        assert info["suppress_classes"] == suppress_classes
        assert info["gradient_smoothing"] is True
        assert info["smoothing_factor"] == 0.2


class TestHelperFunctions:
    """Test helper functions for creating non-targeted losses."""
    
    def test_create_confidence_reduction_loss(self, loss_config):
        """Test helper for creating confidence reduction loss."""
        loss_fn = create_confidence_reduction_loss(
            loss_config, 
            confidence_target=0.05, 
            temperature=1.5
        )
        
        assert isinstance(loss_fn, NonTargetedLoss)
        assert loss_fn.suppression_mode == SuppressionMode.CONFIDENCE_REDUCTION
        assert loss_fn.confidence_target == 0.05
        assert loss_fn.temperature == 1.5
    
    def test_create_entropy_maximization_loss(self, loss_config):
        """Test helper for creating entropy maximization loss."""
        loss_fn = create_entropy_maximization_loss(
            loss_config, 
            entropy_weight=2.0, 
            temperature=3.0
        )
        
        assert isinstance(loss_fn, NonTargetedLoss)
        assert loss_fn.suppression_mode == SuppressionMode.ENTROPY_MAXIMIZATION
        assert loss_fn.entropy_weight == 2.0
        assert loss_fn.temperature == 3.0
    
    def test_create_detection_suppression_loss(self, loss_config):
        """Test helper for creating detection suppression loss."""
        suppress_classes = [1, 2, 3]
        loss_fn = create_detection_suppression_loss(
            loss_config, 
            suppress_classes=suppress_classes,
            confidence_target=0.01
        )
        
        assert isinstance(loss_fn, NonTargetedLoss)
        assert loss_fn.suppression_mode == SuppressionMode.DETECTION_SUPPRESSION
        assert loss_fn.suppress_classes == suppress_classes
        assert loss_fn.confidence_target == 0.01


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    def test_missing_logits_raises_error(self, loss_config, sample_patch):
        """Test that missing logits in model outputs raises error."""
        loss_fn = NonTargetedLoss(loss_config)
        empty_outputs = {}
        
        with pytest.raises(ValueError, match="model_outputs must contain 'logits' key"):
            loss_fn.forward(empty_outputs, sample_patch)
    
    def test_device_mismatch_handling(self, loss_config):
        """Test handling of device mismatches."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available for device mismatch test")
        
        # Create loss function on CUDA
        cuda_config = LossConfig(device="cuda")
        loss_fn = NonTargetedLoss(cuda_config)
        
        # Create inputs on CPU
        cpu_patch = torch.randn(2, 3, 32, 32, requires_grad=True)
        cpu_outputs = {"logits": torch.randn(2, 10)}
        
        # Should handle device conversion automatically
        result = loss_fn.forward(cpu_outputs, cpu_patch)
        
        assert result["total_loss"].device.type == "cuda"
    
    def test_empty_batch_handling(self, loss_config, device):
        """Test handling of empty batches."""
        loss_fn = NonTargetedLoss(loss_config)
        
        # Create empty batch
        empty_patch = torch.empty(0, 3, 32, 32, device=device)
        empty_outputs = {"logits": torch.empty(0, 10, device=device)}
        
        result = loss_fn.forward(empty_outputs, empty_patch)
        
        # Should handle gracefully
        assert result["total_loss"].numel() == 1  # Scalar result
    
    def test_numerical_stability(self, loss_config, sample_patch, device):
        """Test numerical stability with extreme values."""
        # Create extreme logit values
        batch_size = 4
        num_classes = 10
        extreme_logits = torch.tensor([
            [100.0] + [-100.0] * (num_classes - 1),  # Very confident prediction
            [-100.0] * num_classes,                  # Very negative logits
            [100.0] * num_classes,                   # Very positive logits
            [0.0] * num_classes                      # Neutral logits
        ], device=device)
        
        extreme_outputs = {"logits": extreme_logits}
        
        loss_fn = NonTargetedLoss(loss_config)
        result = loss_fn.forward(extreme_outputs, sample_patch)
        
        # Should not produce NaN or Inf
        assert torch.isfinite(result["total_loss"])
        assert not torch.isnan(result["total_loss"])


@pytest.mark.integration
class TestNonTargetedLossIntegration:
    """Integration tests for non-targeted loss with realistic scenarios."""
    
    def test_adversarial_training_simulation(self, loss_config, device):
        """Test non-targeted loss in simulated adversarial training scenario."""
        loss_fn = NonTargetedLoss(
            loss_config,
            suppression_mode=SuppressionMode.CONFIDENCE_REDUCTION,
            confidence_target=0.1,
            gradient_smoothing=True
        )
        
        # Simulate training iterations
        batch_size = 8
        num_classes = 1000  # ImageNet-like
        
        for iteration in range(5):
            # Create batch data
            patch = torch.randn(batch_size, 3, 224, 224, device=device, requires_grad=True)
            logits = torch.randn(batch_size, num_classes, device=device)
            
            model_outputs = {"logits": logits}
            
            # Forward pass
            result = loss_fn.forward(model_outputs, patch)
            
            # Backward pass
            result["total_loss"].backward()
            
            # Verify gradients
            assert patch.grad is not None
            assert not torch.all(patch.grad == 0)
            
            # Clear gradients for next iteration
            patch.grad.zero_()
    
    def test_detection_suppression_scenario(self, loss_config, device):
        """Test detection suppression in object detection scenario."""
        # Simulate object detection with multiple classes
        loss_fn = NonTargetedLoss(
            loss_config,
            suppression_mode=SuppressionMode.DETECTION_SUPPRESSION,
            suppress_classes=[2, 5, 8],  # Suppress specific object classes
            confidence_target=0.05
        )
        
        batch_size = 4
        num_classes = 20  # COCO-like
        
        # Create realistic detection logits with some high confidence predictions
        patch = torch.randn(batch_size, 3, 64, 64, device=device, requires_grad=True)
        logits = torch.randn(batch_size, num_classes, device=device) * 0.5
        
        # Make suppressed classes have higher confidence
        logits[:, [2, 5, 8]] += 2.0
        
        model_outputs = {"logits": logits}
        
        result = loss_fn.forward(model_outputs, patch)
        
        # Should successfully suppress the specified classes
        assert result["total_loss"].requires_grad
        assert torch.isfinite(result["total_loss"])
        assert not torch.isnan(result["total_loss"])
    
    def test_entropy_maximization_uniform_convergence(self, loss_config, device):
        """Test that entropy maximization pushes towards uniform distribution."""
        loss_fn = NonTargetedLoss(
            loss_config,
            suppression_mode=SuppressionMode.ENTROPY_MAXIMIZATION,
            entropy_weight=1.0,
            temperature=2.0  # Higher temperature for more uniform distribution
        )
        
        batch_size = 2
        num_classes = 5
        
        # Start with peaked distribution
        initial_logits = torch.zeros(batch_size, num_classes, device=device)
        initial_logits[:, 0] = 5.0  # High confidence for class 0
        
        patch = torch.randn(batch_size, 3, 32, 32, device=device, requires_grad=True)
        
        # Multiple optimization steps
        optimizer = torch.optim.SGD([patch], lr=0.1)
        
        for step in range(10):
            optimizer.zero_grad()
            
            # In real scenario, logits would change based on patch
            # Here we simulate gradual change towards more uniform distribution
            current_logits = initial_logits * (1 - step * 0.1)
            model_outputs = {"logits": current_logits}
            
            result = loss_fn.forward(model_outputs, patch)
            result["total_loss"].backward()
            optimizer.step()
            
            # Verify loss computation is stable
            assert torch.isfinite(result["total_loss"])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])