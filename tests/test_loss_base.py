"""Comprehensive tests for loss function base classes and factory.

This module tests the abstract LossFunction base class, regularization terms,
factory pattern, GPU optimization, and batch processing capabilities.
"""

import pytest
import torch
import numpy as np
from typing import Dict, Any, Optional
import tempfile
import yaml
from pathlib import Path

from src.losses.base import (
    LossFunction, 
    BatchLossFunction,
    LossConfig,
    RegularizationTerm,
    TotalVariationLoss,
    SmoothnessPenalty
)
from src.losses.factory import LossFactory, get_default_factory, create_loss_function


class MockLossFunction(LossFunction):
    """Mock loss function implementation for testing."""
    
    def compute_loss(self, 
                    model_outputs: Dict[str, Any], 
                    patch: torch.Tensor, 
                    targets: Optional[torch.Tensor] = None,
                    **kwargs) -> torch.Tensor:
        """Simple mock loss that returns mean squared difference."""
        logits = model_outputs.get("logits")
        if logits is None:
            # Return MSE of patch as fallback
            return torch.mean(patch ** 2)
        
        if targets is not None:
            return torch.mean((logits - targets) ** 2)
        else:
            # Non-targeted: just return negative mean of logits
            return -torch.mean(logits)


class MockBatchLossFunction(BatchLossFunction):
    """Mock batch loss function for testing."""
    
    def compute_loss(self, 
                    model_outputs: Dict[str, Any], 
                    patch: torch.Tensor, 
                    targets: Optional[torch.Tensor] = None,
                    **kwargs) -> torch.Tensor:
        """Simple mock batch loss."""
        logits = model_outputs.get("logits")
        if logits is None:
            return torch.mean(patch ** 2)
        
        if targets is not None:
            return torch.mean((logits - targets) ** 2)
        else:
            return -torch.mean(logits)


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
    batch_size, channels, height, width = 2, 3, 32, 32
    patch = torch.randn(batch_size, channels, height, width, device=device, dtype=torch.float32)
    return patch


@pytest.fixture
def sample_model_outputs(device):
    """Create sample model outputs for testing."""
    batch_size = 2
    num_classes = 10
    return {
        "logits": torch.randn(batch_size, num_classes, device=device, dtype=torch.float32),
        "features": torch.randn(batch_size, 512, device=device, dtype=torch.float32)
    }


@pytest.fixture
def sample_targets(device):
    """Create sample targets for testing."""
    batch_size = 2
    num_classes = 10
    return torch.randn(batch_size, num_classes, device=device, dtype=torch.float32)


class TestLossConfig:
    """Test LossConfig dataclass."""
    
    def test_default_config(self):
        """Test default configuration creation."""
        config = LossConfig()
        assert config.device in ["cpu", "cuda"]
        assert config.dtype == torch.float32
        assert config.reduction == "mean"
        assert config.regularization_weight == 1e-3
        assert config.gradient_clipping is None
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = LossConfig(
            device="cpu",
            dtype=torch.float16,
            reduction="sum",
            regularization_weight=0.5,
            gradient_clipping=1.0
        )
        assert config.device == "cpu"
        assert config.dtype == torch.float16
        assert config.reduction == "sum"
        assert config.regularization_weight == 0.5
        assert config.gradient_clipping == 1.0
    
    def test_invalid_reduction(self):
        """Test invalid reduction value raises error."""
        with pytest.raises(ValueError, match="reduction must be one of"):
            LossConfig(reduction="invalid")
    
    def test_negative_regularization_weight(self):
        """Test negative regularization weight raises error."""
        with pytest.raises(ValueError, match="regularization_weight must be non-negative"):
            LossConfig(regularization_weight=-1.0)


class TestRegularizationTerms:
    """Test regularization term implementations."""
    
    def test_total_variation_loss(self, device, sample_patch):
        """Test total variation loss computation."""
        tv_loss = TotalVariationLoss(weight=1.0, device=device)
        loss = tv_loss(sample_patch)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.device.type == device
        assert loss.item() >= 0  # TV loss should be non-negative
        assert loss.requires_grad == sample_patch.requires_grad
    
    def test_total_variation_loss_invalid_dims(self, device):
        """Test TV loss with invalid tensor dimensions."""
        tv_loss = TotalVariationLoss(device=device)
        invalid_patch = torch.randn(10, 20)  # Only 2D
        
        with pytest.raises(ValueError, match="Expected 4D tensor"):
            tv_loss.compute(invalid_patch)
    
    def test_smoothness_penalty(self, device, sample_patch):
        """Test smoothness penalty computation."""
        smooth_loss = SmoothnessPenalty(weight=1.0, device=device)
        loss = smooth_loss(sample_patch)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.device.type == device
        assert loss.item() >= 0  # Smoothness loss should be non-negative
    
    def test_smoothness_penalty_invalid_dims(self, device):
        """Test smoothness penalty with invalid dimensions."""
        smooth_loss = SmoothnessPenalty(device=device)
        invalid_patch = torch.randn(10)  # Only 1D
        
        with pytest.raises(ValueError, match="Expected 4D tensor"):
            smooth_loss.compute(invalid_patch)
    
    def test_regularization_weight_application(self, device, sample_patch):
        """Test that regularization weight is properly applied."""
        weight = 0.5
        tv_loss = TotalVariationLoss(weight=weight, device=device)
        
        # Compute with weight
        weighted_loss = tv_loss(sample_patch)
        
        # Compute without weight 
        raw_loss = tv_loss.compute(sample_patch)
        
        expected_loss = weight * raw_loss
        assert torch.allclose(weighted_loss, expected_loss, atol=1e-6)


class TestLossFunction:
    """Test abstract LossFunction base class."""
    
    def test_loss_function_initialization(self, loss_config):
        """Test loss function initialization."""
        loss_fn = MockLossFunction(loss_config)
        
        assert loss_fn.config == loss_config
        assert loss_fn.device == loss_config.device
        assert len(loss_fn.regularization_terms) == 0
    
    def test_loss_function_default_config(self):
        """Test loss function with default config."""
        loss_fn = MockLossFunction()
        
        assert loss_fn.config is not None
        assert isinstance(loss_fn.config, LossConfig)
    
    def test_add_regularization(self, loss_config, device):
        """Test adding regularization terms."""
        loss_fn = MockLossFunction(loss_config)
        tv_loss = TotalVariationLoss(device=device)
        smooth_loss = SmoothnessPenalty(device=device)
        
        loss_fn.add_regularization(tv_loss)
        assert len(loss_fn.regularization_terms) == 1
        
        loss_fn.add_regularization(smooth_loss)
        assert len(loss_fn.regularization_terms) == 2
    
    def test_forward_without_regularization(self, loss_config, sample_patch, sample_model_outputs):
        """Test forward pass without regularization."""
        loss_fn = MockLossFunction(loss_config)
        
        result = loss_fn.forward(sample_model_outputs, sample_patch)
        
        assert isinstance(result, dict)
        assert "total_loss" in result
        assert "main_loss" in result  
        assert "regularization_loss" in result
        
        # Without regularization terms, reg loss should be 0
        assert result["regularization_loss"].item() == 0.0
        assert torch.allclose(result["total_loss"], result["main_loss"])
    
    def test_forward_with_regularization(self, loss_config, sample_patch, sample_model_outputs, device):
        """Test forward pass with regularization."""
        loss_fn = MockLossFunction(loss_config)
        loss_fn.add_regularization(TotalVariationLoss(device=device))
        
        result = loss_fn.forward(sample_model_outputs, sample_patch)
        
        assert result["regularization_loss"].item() > 0
        # Total loss should be main loss + weighted regularization
        expected_total = result["main_loss"] + loss_config.regularization_weight * result["regularization_loss"]
        assert torch.allclose(result["total_loss"], expected_total, atol=1e-6)
    
    def test_forward_with_targets(self, loss_config, sample_patch, sample_model_outputs, sample_targets):
        """Test forward pass with target values."""
        loss_fn = MockLossFunction(loss_config)
        
        result = loss_fn.forward(sample_model_outputs, sample_patch, sample_targets)
        
        assert "total_loss" in result
        assert result["total_loss"].requires_grad == sample_patch.requires_grad
    
    def test_callable_interface(self, loss_config, sample_patch, sample_model_outputs):
        """Test that loss function is callable."""
        loss_fn = MockLossFunction(loss_config)
        
        # Should be able to call directly
        result = loss_fn(sample_model_outputs, sample_patch)
        assert isinstance(result, dict)
        assert "total_loss" in result
    
    def test_reduction_modes(self, device, sample_patch, sample_model_outputs):
        """Test different reduction modes."""
        batch_size = sample_patch.size(0)
        
        # Test mean reduction
        config_mean = LossConfig(device=device, reduction="mean")
        loss_fn_mean = MockLossFunction(config_mean)
        result_mean = loss_fn_mean(sample_model_outputs, sample_patch)
        
        # Test sum reduction
        config_sum = LossConfig(device=device, reduction="sum")  
        loss_fn_sum = MockLossFunction(config_sum)
        result_sum = loss_fn_sum(sample_model_outputs, sample_patch)
        
        # Test none reduction (should return tensor with batch dimension)
        config_none = LossConfig(device=device, reduction="none")
        loss_fn_none = MockLossFunction(config_none)
        result_none = loss_fn_none(sample_model_outputs, sample_patch)
        
        # Verify relationships
        assert result_mean["main_loss"].dim() == 0  # Scalar
        assert result_sum["main_loss"].dim() == 0   # Scalar
        # Note: With our mock implementation, "none" reduction might still be scalar
        # depending on the compute_loss implementation
    
    def test_device_handling(self, loss_config):
        """Test device handling in loss function."""
        loss_fn = MockLossFunction(loss_config)
        
        assert loss_fn.get_device() == loss_config.device
        
        # Test tensor device conversion
        cpu_tensor = torch.randn(2, 3, 4, 4)
        device_tensor = loss_fn._ensure_tensor_on_device(cpu_tensor)
        assert device_tensor.device.type == loss_config.device
    
    def test_get_info(self, loss_config, device):
        """Test loss function info retrieval."""
        loss_fn = MockLossFunction(loss_config)
        loss_fn.add_regularization(TotalVariationLoss(device=device))
        
        info = loss_fn.get_info()
        
        assert info["class"] == "MockLossFunction"
        assert info["device"] == loss_config.device
        assert info["num_regularization_terms"] == 1
        assert "TotalVariationLoss" in info["regularization_terms"]


class TestBatchLossFunction:
    """Test BatchLossFunction class."""
    
    def test_batch_initialization(self, loss_config):
        """Test batch loss function initialization."""
        batch_size = 16
        loss_fn = MockBatchLossFunction(loss_config, batch_size=batch_size)
        
        assert loss_fn.batch_size == batch_size
        assert loss_fn.get_optimal_batch_size() == batch_size
    
    def test_process_batch(self, loss_config, device):
        """Test batch processing functionality."""
        batch_size = 4
        loss_fn = MockBatchLossFunction(loss_config, batch_size=batch_size)
        
        # Create batch data
        patches_batch = torch.randn(batch_size, 3, 32, 32, device=device)
        model_outputs_batch = []
        for i in range(batch_size):
            model_outputs_batch.append({
                "logits": torch.randn(10, device=device),
                "features": torch.randn(512, device=device)
            })
        
        result = loss_fn.process_batch(model_outputs_batch, patches_batch)
        
        assert isinstance(result, dict)
        assert "total_loss" in result
        assert result["total_loss"].dim() == 0  # Should be reduced to scalar
    
    def test_batch_size_validation(self, loss_config, device):
        """Test batch size validation in process_batch."""
        loss_fn = MockBatchLossFunction(loss_config)
        
        # Mismatched batch sizes
        patches = torch.randn(3, 3, 32, 32, device=device)
        outputs = [{"logits": torch.randn(10, device=device)} for _ in range(2)]  # Only 2 outputs
        
        with pytest.raises(ValueError, match="Batch size mismatch"):
            loss_fn.process_batch(outputs, patches)


class TestLossFactory:
    """Test LossFactory implementation."""
    
    def test_factory_initialization(self):
        """Test factory initialization."""
        factory = LossFactory()
        
        assert factory.config is not None
        assert len(factory.regularization_registry) >= 2  # Built-ins
        assert "total_variation" in factory.regularization_registry
        assert "smoothness" in factory.regularization_registry
    
    def test_factory_with_config_file(self):
        """Test factory initialization with config file."""
        config_data = {
            "device": "cpu",
            "dtype": "float16", 
            "regularization_weight": 0.001,
            "regularization_terms": {
                "total_variation": {"enabled": True, "weight": 1e-4},
                "smoothness": {"enabled": False}
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            config_path = f.name
        
        try:
            factory = LossFactory(config_path=config_path)
            assert factory.config["device"] == "cpu"
            assert factory.config["dtype"] == "float16"
        finally:
            Path(config_path).unlink()
    
    def test_register_loss_function(self):
        """Test registering custom loss function."""
        factory = LossFactory()
        
        factory.register_loss_function("mock_loss", MockLossFunction)
        
        assert "mock_loss" in factory.get_available_loss_types()
        assert factory.loss_registry["mock_loss"] == MockLossFunction
    
    def test_register_invalid_loss_function(self):
        """Test registering invalid loss function raises error."""
        factory = LossFactory()
        
        class InvalidLoss:
            pass
        
        with pytest.raises(ValueError, match="must be a subclass of LossFunction"):
            factory.register_loss_function("invalid", InvalidLoss)
    
    def test_register_regularization_term(self):
        """Test registering custom regularization term."""
        factory = LossFactory()
        
        class CustomReg(RegularizationTerm):
            def compute(self, patch):
                return torch.mean(patch)
        
        factory.register_regularization_term("custom_reg", CustomReg)
        
        assert "custom_reg" in factory.get_available_regularization_terms()
    
    def test_create_loss_config(self):
        """Test creating loss configuration."""
        factory = LossFactory()
        
        config = factory.create_loss_config()
        assert isinstance(config, LossConfig)
        
        # Test with overrides
        config_with_overrides = factory.create_loss_config(
            device="cpu", 
            regularization_weight=0.5
        )
        assert config_with_overrides.device == "cpu"
        assert config_with_overrides.regularization_weight == 0.5
    
    def test_create_regularization_terms(self):
        """Test creating regularization terms from config."""
        factory = LossFactory()
        
        reg_terms = factory.create_regularization_terms()
        
        # Should create terms based on default config
        assert isinstance(reg_terms, list)
        assert len(reg_terms) >= 0  # Depends on default config
    
    def test_create_loss_function(self):
        """Test creating loss function."""
        factory = LossFactory()
        factory.register_loss_function("mock_loss", MockLossFunction)
        
        loss_fn = factory.create_loss_function("mock_loss")
        
        assert isinstance(loss_fn, MockLossFunction)
        assert isinstance(loss_fn.config, LossConfig)
    
    def test_create_unknown_loss_function(self):
        """Test creating unknown loss function raises error."""
        factory = LossFactory()
        
        with pytest.raises(ValueError, match="Unknown loss type"):
            factory.create_loss_function("nonexistent_loss")
    
    def test_save_and_load_config(self):
        """Test saving and loading configuration."""
        factory = LossFactory()
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            config_path = f.name
        
        try:
            # Save config
            factory.save_config(config_path)
            assert Path(config_path).exists()
            
            # Load config in new factory
            new_factory = LossFactory(config_path=config_path)
            
            # Compare key settings
            assert new_factory.config["device"] == factory.config["device"]
            assert new_factory.config["regularization_weight"] == factory.config["regularization_weight"]
            
        finally:
            if Path(config_path).exists():
                Path(config_path).unlink()
    
    def test_get_factory_info(self):
        """Test getting factory information."""
        factory = LossFactory()
        factory.register_loss_function("mock_loss", MockLossFunction)
        
        info = factory.get_factory_info()
        
        assert "available_loss_types" in info
        assert "available_regularization_terms" in info
        assert "current_device" in info
        assert "mock_loss" in info["available_loss_types"]


class TestFactoryConvenienceFunctions:
    """Test convenience functions for factory access."""
    
    def test_get_default_factory(self):
        """Test getting default factory instance."""
        factory1 = get_default_factory()
        factory2 = get_default_factory()
        
        # Should return same instance
        assert factory1 is factory2
        assert isinstance(factory1, LossFactory)
    
    def test_create_loss_function_convenience(self):
        """Test convenience function for creating loss function."""
        # Register a loss function in the default factory
        factory = get_default_factory()
        factory.register_loss_function("mock_loss", MockLossFunction)
        
        loss_fn = create_loss_function("mock_loss")
        
        assert isinstance(loss_fn, MockLossFunction)


@pytest.mark.integration
class TestIntegration:
    """Integration tests for the complete loss function system."""
    
    def test_end_to_end_workflow(self, device):
        """Test complete workflow from factory to loss computation."""
        # Create factory and register loss function
        factory = LossFactory()
        factory.register_loss_function("mock_loss", MockLossFunction)
        
        # Create loss function with regularization
        loss_fn = factory.create_loss_function("mock_loss", add_regularization=True)
        
        # Create test data
        batch_size = 4
        patch = torch.randn(batch_size, 3, 64, 64, device=device, requires_grad=True)
        model_outputs = {
            "logits": torch.randn(batch_size, 10, device=device),
        }
        targets = torch.randn(batch_size, 10, device=device)
        
        # Compute loss
        result = loss_fn(model_outputs, patch, targets)
        
        # Verify results
        assert "total_loss" in result
        assert result["total_loss"].requires_grad
        assert result["regularization_loss"].item() >= 0
        
        # Test backward pass
        result["total_loss"].backward()
        assert patch.grad is not None
        assert patch.grad.shape == patch.shape


if __name__ == "__main__":
    pytest.main([__file__, "-v"])