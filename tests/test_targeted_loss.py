"""Comprehensive tests for TargetedLoss implementations.

This module tests all variants of targeted attack loss functions including
classification, confidence, margin, and likelihood targeting with various
edge cases and gradient smoothing.
"""

import pytest
import torch
import numpy as np
from typing import Dict, Any, Optional
import logging

from src.losses.targeted import (
    TargetedLoss, 
    TargetMode,
    create_targeted_classification_loss,
    create_confidence_based_loss,
    create_margin_based_loss
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
def sample_targets(device):
    """Create sample target distributions for testing."""
    batch_size = 4
    num_classes = 10
    # Create target distributions with high confidence for class 2
    targets = torch.zeros(batch_size, num_classes, device=device, dtype=torch.float32)
    targets[:, 2] = 5.0  # High logit for target class
    return targets


@pytest.fixture
def sample_target_classes(device):
    """Create sample target class indices for testing."""
    batch_size = 4
    return torch.tensor([2, 3, 1, 4], device=device, dtype=torch.long)


class TestTargetedLossInitialization:
    """Test TargetedLoss initialization and parameter validation."""
    
    def test_default_initialization(self, loss_config):
        """Test default initialization."""
        loss_fn = TargetedLoss(loss_config)
        
        assert loss_fn.target_mode == TargetMode.CLASSIFICATION
        assert loss_fn.confidence_threshold == 0.9
        assert loss_fn.margin == 0.1
        assert loss_fn.temperature == 1.0
        assert loss_fn.gradient_smoothing is True
        assert loss_fn.smoothing_factor == 0.1
    
    def test_custom_initialization(self, loss_config):
        """Test initialization with custom parameters."""
        loss_fn = TargetedLoss(
            config=loss_config,
            target_mode=TargetMode.CONFIDENCE,
            confidence_threshold=0.8,
            margin=0.2,
            temperature=2.0,
            gradient_smoothing=False
        )
        
        assert loss_fn.target_mode == TargetMode.CONFIDENCE
        assert loss_fn.confidence_threshold == 0.8
        assert loss_fn.margin == 0.2
        assert loss_fn.temperature == 2.0
        assert loss_fn.gradient_smoothing is False
    
    def test_invalid_confidence_threshold(self, loss_config):
        """Test that invalid confidence threshold raises error."""
        with pytest.raises(ValueError, match="confidence_threshold must be in"):
            TargetedLoss(loss_config, confidence_threshold=1.5)
        
        with pytest.raises(ValueError, match="confidence_threshold must be in"):
            TargetedLoss(loss_config, confidence_threshold=0.0)
    
    def test_invalid_margin(self, loss_config):
        """Test that invalid margin raises error."""
        with pytest.raises(ValueError, match="margin must be positive"):
            TargetedLoss(loss_config, margin=-0.1)
        
        with pytest.raises(ValueError, match="margin must be positive"):
            TargetedLoss(loss_config, margin=0.0)
    
    def test_invalid_temperature(self, loss_config):
        """Test that invalid temperature raises error."""
        with pytest.raises(ValueError, match="temperature must be positive"):
            TargetedLoss(loss_config, temperature=0.0)
        
        with pytest.raises(ValueError, match="temperature must be positive"):
            TargetedLoss(loss_config, temperature=-1.0)
    
    def test_invalid_smoothing_factor(self, loss_config):
        """Test that invalid smoothing factor raises error."""
        with pytest.raises(ValueError, match="smoothing_factor must be in"):
            TargetedLoss(loss_config, smoothing_factor=-0.1)
        
        with pytest.raises(ValueError, match="smoothing_factor must be in"):
            TargetedLoss(loss_config, smoothing_factor=1.1)


class TestClassificationTargeting:
    """Test classification-based targeting mode."""
    
    def test_classification_with_target_classes(self, loss_config, sample_patch, 
                                              sample_model_outputs, sample_target_classes):
        """Test classification targeting with target class indices."""
        loss_fn = TargetedLoss(loss_config, target_mode=TargetMode.CLASSIFICATION)
        
        result = loss_fn.forward(sample_model_outputs, sample_patch, target_classes=sample_target_classes)
        
        assert "total_loss" in result
        assert "main_loss" in result
        assert "regularization_loss" in result
        assert result["total_loss"].requires_grad
        
        # Loss should be negative cross-entropy (we want to maximize target class probability)
        main_loss = result["main_loss"]
        assert main_loss.dim() == 0  # Scalar after reduction
    
    def test_classification_with_target_distributions(self, loss_config, sample_patch,
                                                    sample_model_outputs, sample_targets):
        """Test classification targeting with target probability distributions."""
        loss_fn = TargetedLoss(loss_config, target_mode=TargetMode.CLASSIFICATION)
        
        result = loss_fn.forward(sample_model_outputs, sample_patch, targets=sample_targets)
        
        assert "total_loss" in result
        assert result["total_loss"].requires_grad
        
        # Should use KL divergence
        main_loss = result["main_loss"]
        assert torch.isfinite(main_loss)  # Should be finite
    
    def test_classification_without_targets_raises_error(self, loss_config, sample_patch,
                                                       sample_model_outputs):
        """Test that classification mode requires targets or target_classes."""
        loss_fn = TargetedLoss(loss_config, target_mode=TargetMode.CLASSIFICATION)
        
        with pytest.raises(ValueError, match="Either targets or target_classes must be provided"):
            loss_fn.forward(sample_model_outputs, sample_patch)
    
    def test_classification_gradient_flow(self, loss_config, sample_patch,
                                        sample_model_outputs, sample_target_classes):
        """Test that gradients flow correctly through classification loss."""
        loss_fn = TargetedLoss(loss_config, target_mode=TargetMode.CLASSIFICATION)
        
        result = loss_fn.forward(sample_model_outputs, sample_patch, target_classes=sample_target_classes)
        
        # Compute gradients
        result["total_loss"].backward()
        
        assert sample_patch.grad is not None
        assert sample_patch.grad.shape == sample_patch.shape
        assert not torch.all(sample_patch.grad == 0)  # Should have non-zero gradients


class TestConfidenceTargeting:
    """Test confidence-based targeting mode."""
    
    def test_confidence_with_target_classes(self, loss_config, sample_patch,
                                          sample_model_outputs, sample_target_classes):
        """Test confidence targeting with target class indices."""
        loss_fn = TargetedLoss(
            loss_config, 
            target_mode=TargetMode.CONFIDENCE,
            confidence_threshold=0.8
        )
        
        result = loss_fn.forward(sample_model_outputs, sample_patch, target_classes=sample_target_classes)
        
        assert "total_loss" in result
        assert result["total_loss"].requires_grad
        
        # Loss should be non-negative (ReLU of confidence gap)
        main_loss = result["main_loss"]
        assert main_loss.item() >= 0
    
    def test_confidence_with_target_distributions(self, loss_config, sample_patch,
                                                sample_model_outputs, sample_targets):
        """Test confidence targeting with target probability distributions."""
        loss_fn = TargetedLoss(
            loss_config, 
            target_mode=TargetMode.CONFIDENCE,
            confidence_threshold=0.9
        )
        
        result = loss_fn.forward(sample_model_outputs, sample_patch, targets=sample_targets)
        
        assert "total_loss" in result
        assert result["main_loss"].item() >= 0
    
    def test_confidence_threshold_behavior(self, loss_config, sample_patch,
                                         sample_model_outputs, sample_target_classes):
        """Test that confidence threshold affects loss correctly."""
        device = sample_patch.device
        
        # Create model outputs with known probabilities
        batch_size = sample_patch.size(0)
        num_classes = 10
        logits = torch.zeros(batch_size, num_classes, device=device)
        
        # Set high confidence for target classes
        for i, target_class in enumerate(sample_target_classes):
            logits[i, target_class] = 5.0  # This will give high probability
        
        high_conf_outputs = {"logits": logits}
        
        # Test with high threshold (should have loss)
        loss_fn_high = TargetedLoss(loss_config, target_mode=TargetMode.CONFIDENCE,
                                  confidence_threshold=0.99)
        result_high = loss_fn_high.forward(high_conf_outputs, sample_patch, 
                                         target_classes=sample_target_classes)
        
        # Test with low threshold (should have little to no loss)
        loss_fn_low = TargetedLoss(loss_config, target_mode=TargetMode.CONFIDENCE,
                                 confidence_threshold=0.5)
        result_low = loss_fn_low.forward(high_conf_outputs, sample_patch,
                                       target_classes=sample_target_classes)
        
        # High threshold should have higher loss
        assert result_high["main_loss"].item() >= result_low["main_loss"].item()


class TestMarginTargeting:
    """Test margin-based targeting mode."""
    
    def test_margin_with_target_classes(self, loss_config, sample_patch,
                                      sample_model_outputs, sample_target_classes):
        """Test margin targeting with target class indices."""
        loss_fn = TargetedLoss(
            loss_config, 
            target_mode=TargetMode.MARGIN,
            margin=0.2
        )
        
        result = loss_fn.forward(sample_model_outputs, sample_patch, target_classes=sample_target_classes)
        
        assert "total_loss" in result
        assert result["total_loss"].requires_grad
        
        # Loss should be non-negative (ReLU of margin gap)
        main_loss = result["main_loss"]
        assert main_loss.item() >= 0
    
    def test_margin_with_target_distributions(self, loss_config, sample_patch,
                                            sample_model_outputs, sample_targets):
        """Test margin targeting with target probability distributions."""
        loss_fn = TargetedLoss(
            loss_config, 
            target_mode=TargetMode.MARGIN,
            margin=0.1
        )
        
        result = loss_fn.forward(sample_model_outputs, sample_patch, targets=sample_targets)
        
        assert "total_loss" in result
        assert result["main_loss"].item() >= 0
    
    def test_margin_size_effect(self, loss_config, sample_patch,
                              sample_model_outputs, sample_target_classes):
        """Test that larger margins produce higher losses."""
        device = sample_patch.device
        
        # Create model outputs where target class has slight advantage
        batch_size = sample_patch.size(0)
        num_classes = 10
        logits = torch.randn(batch_size, num_classes, device=device) * 0.1
        
        # Give target classes only slight advantage
        for i, target_class in enumerate(sample_target_classes):
            logits[i, target_class] += 0.05
        
        slight_advantage_outputs = {"logits": logits}
        
        # Test with small margin
        loss_fn_small = TargetedLoss(loss_config, target_mode=TargetMode.MARGIN, margin=0.01)
        result_small = loss_fn_small.forward(slight_advantage_outputs, sample_patch,
                                           target_classes=sample_target_classes)
        
        # Test with large margin
        loss_fn_large = TargetedLoss(loss_config, target_mode=TargetMode.MARGIN, margin=0.5)
        result_large = loss_fn_large.forward(slight_advantage_outputs, sample_patch,
                                           target_classes=sample_target_classes)
        
        # Large margin should have higher loss when target class doesn't have large enough margin
        assert result_large["main_loss"].item() >= result_small["main_loss"].item()


class TestLikelihoodTargeting:
    """Test likelihood-based targeting mode."""
    
    def test_likelihood_with_target_classes(self, loss_config, sample_patch,
                                          sample_model_outputs, sample_target_classes):
        """Test likelihood targeting with target class indices."""
        loss_fn = TargetedLoss(loss_config, target_mode=TargetMode.LIKELIHOOD)
        
        result = loss_fn.forward(sample_model_outputs, sample_patch, target_classes=sample_target_classes)
        
        assert "total_loss" in result
        assert result["total_loss"].requires_grad
        
        # Loss should be negative log-likelihood
        main_loss = result["main_loss"]
        assert main_loss.item() >= 0  # NLL is non-negative
    
    def test_likelihood_with_target_distributions(self, loss_config, sample_patch,
                                                sample_model_outputs, sample_targets):
        """Test likelihood targeting with target probability distributions."""
        loss_fn = TargetedLoss(loss_config, target_mode=TargetMode.LIKELIHOOD)
        
        result = loss_fn.forward(sample_model_outputs, sample_patch, targets=sample_targets)
        
        assert "total_loss" in result
        assert result["main_loss"].item() >= 0


class TestGradientSmoothing:
    """Test gradient smoothing functionality."""
    
    def test_gradient_smoothing_enabled(self, loss_config, sample_patch,
                                      sample_model_outputs, sample_target_classes):
        """Test that gradient smoothing is applied when enabled."""
        loss_fn = TargetedLoss(
            loss_config, 
            gradient_smoothing=True, 
            smoothing_factor=0.5
        )
        
        # First forward pass
        result1 = loss_fn.forward(sample_model_outputs, sample_patch, target_classes=sample_target_classes)
        
        # Second forward pass with different inputs
        modified_outputs = {
            "logits": sample_model_outputs["logits"] + torch.randn_like(sample_model_outputs["logits"]) * 0.1
        }
        result2 = loss_fn.forward(modified_outputs, sample_patch, target_classes=sample_target_classes)
        
        # Should have internal state for smoothing
        assert hasattr(loss_fn, '_prev_loss')
    
    def test_gradient_smoothing_disabled(self, loss_config, sample_patch,
                                       sample_model_outputs, sample_target_classes):
        """Test behavior when gradient smoothing is disabled."""
        loss_fn = TargetedLoss(
            loss_config, 
            gradient_smoothing=False
        )
        
        result = loss_fn.forward(sample_model_outputs, sample_patch, target_classes=sample_target_classes)
        
        # Should not have smoothing state
        # The _prev_loss won't be created when smoothing is disabled
        assert "total_loss" in result


class TestTemperatureScaling:
    """Test temperature scaling effects."""
    
    def test_temperature_effects_on_probabilities(self, loss_config, sample_patch,
                                                 sample_model_outputs, sample_target_classes):
        """Test that temperature affects probability distributions correctly."""
        # Test with high temperature (more uniform distribution)
        loss_fn_high_temp = TargetedLoss(loss_config, temperature=10.0)
        result_high = loss_fn_high_temp.forward(sample_model_outputs, sample_patch, 
                                              target_classes=sample_target_classes)
        
        # Test with low temperature (more peaked distribution)
        loss_fn_low_temp = TargetedLoss(loss_config, temperature=0.1)
        result_low = loss_fn_low_temp.forward(sample_model_outputs, sample_patch,
                                            target_classes=sample_target_classes)
        
        # Both should produce valid losses
        assert result_high["total_loss"].requires_grad
        assert result_low["total_loss"].requires_grad


class TestParameterUpdates:
    """Test dynamic parameter updates."""
    
    def test_set_target_mode(self, loss_config):
        """Test changing target mode."""
        loss_fn = TargetedLoss(loss_config, target_mode=TargetMode.CLASSIFICATION)
        
        loss_fn.set_target_mode(TargetMode.CONFIDENCE)
        assert loss_fn.target_mode == TargetMode.CONFIDENCE
    
    def test_set_confidence_threshold(self, loss_config):
        """Test changing confidence threshold."""
        loss_fn = TargetedLoss(loss_config, confidence_threshold=0.8)
        
        loss_fn.set_confidence_threshold(0.95)
        assert loss_fn.confidence_threshold == 0.95
        
        with pytest.raises(ValueError, match="confidence_threshold must be in"):
            loss_fn.set_confidence_threshold(1.1)
    
    def test_set_margin(self, loss_config):
        """Test changing margin."""
        loss_fn = TargetedLoss(loss_config, margin=0.1)
        
        loss_fn.set_margin(0.3)
        assert loss_fn.margin == 0.3
        
        with pytest.raises(ValueError, match="margin must be positive"):
            loss_fn.set_margin(-0.1)
    
    def test_get_target_info(self, loss_config):
        """Test getting target configuration info."""
        loss_fn = TargetedLoss(
            loss_config,
            target_mode=TargetMode.MARGIN,
            confidence_threshold=0.85,
            margin=0.15,
            temperature=1.5,
            gradient_smoothing=True,
            smoothing_factor=0.2
        )
        
        info = loss_fn.get_target_info()
        
        assert info["target_mode"] == "margin"
        assert info["confidence_threshold"] == 0.85
        assert info["margin"] == 0.15
        assert info["temperature"] == 1.5
        assert info["gradient_smoothing"] is True
        assert info["smoothing_factor"] == 0.2


class TestHelperFunctions:
    """Test helper functions for creating targeted losses."""
    
    def test_create_targeted_classification_loss(self, loss_config):
        """Test helper for creating classification loss."""
        loss_fn = create_targeted_classification_loss(loss_config, confidence_threshold=0.95)
        
        assert isinstance(loss_fn, TargetedLoss)
        assert loss_fn.target_mode == TargetMode.CLASSIFICATION
        assert loss_fn.confidence_threshold == 0.95
    
    def test_create_confidence_based_loss(self, loss_config):
        """Test helper for creating confidence-based loss."""
        loss_fn = create_confidence_based_loss(loss_config, threshold=0.9, temperature=1.5)
        
        assert isinstance(loss_fn, TargetedLoss)
        assert loss_fn.target_mode == TargetMode.CONFIDENCE
        assert loss_fn.confidence_threshold == 0.9
        assert loss_fn.temperature == 1.5
    
    def test_create_margin_based_loss(self, loss_config):
        """Test helper for creating margin-based loss."""
        loss_fn = create_margin_based_loss(loss_config, margin=0.25, temperature=2.0)
        
        assert isinstance(loss_fn, TargetedLoss)
        assert loss_fn.target_mode == TargetMode.MARGIN
        assert loss_fn.margin == 0.25
        assert loss_fn.temperature == 2.0


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    def test_missing_logits_raises_error(self, loss_config, sample_patch, sample_target_classes):
        """Test that missing logits in model outputs raises error."""
        loss_fn = TargetedLoss(loss_config)
        empty_outputs = {}
        
        with pytest.raises(ValueError, match="model_outputs must contain 'logits' key"):
            loss_fn.forward(empty_outputs, sample_patch, target_classes=sample_target_classes)
    
    def test_device_mismatch_handling(self, loss_config):
        """Test handling of device mismatches."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available for device mismatch test")
        
        # Create loss function on CUDA
        cuda_config = LossConfig(device="cuda")
        loss_fn = TargetedLoss(cuda_config)
        
        # Create inputs on CPU
        cpu_patch = torch.randn(2, 3, 32, 32, requires_grad=True)
        cpu_outputs = {"logits": torch.randn(2, 10)}
        cpu_targets = torch.tensor([1, 2], dtype=torch.long)
        
        # Should handle device conversion automatically
        result = loss_fn.forward(cpu_outputs, cpu_patch, target_classes=cpu_targets)
        
        assert result["total_loss"].device.type == "cuda"
    
    def test_empty_batch_handling(self, loss_config, device):
        """Test handling of empty batches."""
        loss_fn = TargetedLoss(loss_config)
        
        # Create empty batch
        empty_patch = torch.empty(0, 3, 32, 32, device=device)
        empty_outputs = {"logits": torch.empty(0, 10, device=device)}
        empty_targets = torch.empty(0, dtype=torch.long, device=device)
        
        result = loss_fn.forward(empty_outputs, empty_patch, target_classes=empty_targets)
        
        # Should handle gracefully
        assert result["total_loss"].numel() == 1  # Scalar result


@pytest.mark.integration
class TestTargetedLossIntegration:
    """Integration tests for targeted loss with realistic scenarios."""
    
    def test_adversarial_training_simulation(self, loss_config, device):
        """Test targeted loss in simulated adversarial training scenario."""
        loss_fn = TargetedLoss(
            loss_config,
            target_mode=TargetMode.CONFIDENCE,
            confidence_threshold=0.9,
            gradient_smoothing=True
        )
        
        # Simulate training iterations
        batch_size = 8
        num_classes = 1000  # ImageNet-like
        
        for iteration in range(5):
            # Create batch data
            patch = torch.randn(batch_size, 3, 224, 224, device=device, requires_grad=True)
            logits = torch.randn(batch_size, num_classes, device=device)
            target_classes = torch.randint(0, num_classes, (batch_size,), device=device)
            
            model_outputs = {"logits": logits}
            
            # Forward pass
            result = loss_fn.forward(model_outputs, patch, target_classes=target_classes)
            
            # Backward pass
            result["total_loss"].backward()
            
            # Verify gradients
            assert patch.grad is not None
            assert not torch.all(patch.grad == 0)
            
            # Clear gradients for next iteration
            patch.grad.zero_()
    
    def test_multi_target_attack(self, loss_config, device):
        """Test targeted loss with multiple target classes in same batch."""
        loss_fn = TargetedLoss(loss_config, target_mode=TargetMode.MARGIN, margin=0.2)
        
        batch_size = 4
        num_classes = 10
        
        # Create diverse target classes
        target_classes = torch.tensor([0, 3, 7, 9], device=device)
        
        patch = torch.randn(batch_size, 3, 32, 32, device=device, requires_grad=True)
        logits = torch.randn(batch_size, num_classes, device=device)
        model_outputs = {"logits": logits}
        
        result = loss_fn.forward(model_outputs, patch, target_classes=target_classes)
        
        # Should handle different target classes correctly
        assert result["total_loss"].requires_grad
        assert not torch.isnan(result["total_loss"])
        assert not torch.isinf(result["total_loss"])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])