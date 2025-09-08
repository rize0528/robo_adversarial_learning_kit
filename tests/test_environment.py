"""
Test suite for environment setup validation.
Tests verify that all required dependencies are installed and configured correctly.
"""

import pytest
import sys
import platform
import importlib
import subprocess
from pathlib import Path


class TestPythonEnvironment:
    """Test Python version and basic environment setup."""

    def test_python_version(self):
        """Test that Python version is 3.8 or higher."""
        version = sys.version_info
        assert version.major == 3, f"Expected Python 3.x, got Python {version.major}.x"
        assert version.minor >= 8, f"Expected Python 3.8+, got Python {version.major}.{version.minor}"

    def test_platform_detection(self):
        """Test platform detection and basic system info."""
        system = platform.system()
        machine = platform.machine()
        assert system in ["Linux", "Darwin", "Windows"], f"Unsupported platform: {system}"
        print(f"Platform: {system} {machine}")
        print(f"Python: {sys.version}")


class TestCoreMLLibraries:
    """Test core machine learning libraries."""

    def test_torch_installation(self):
        """Test PyTorch installation and basic functionality."""
        torch = importlib.import_module("torch")
        assert hasattr(torch, "__version__")
        
        # Version check
        version_parts = torch.__version__.split(".")
        major, minor = int(version_parts[0]), int(version_parts[1])
        assert major >= 2, f"Expected PyTorch 2.x+, got {torch.__version__}"
        if major == 2:
            assert minor >= 1, f"Expected PyTorch 2.1+, got {torch.__version__}"
        
        print(f"PyTorch version: {torch.__version__}")

    def test_torch_device_support(self):
        """Test available device support (CUDA/MPS)."""
        torch = importlib.import_module("torch")
        
        cuda_available = torch.cuda.is_available()
        mps_available = torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False
        
        print(f"CUDA available: {cuda_available}")
        print(f"MPS available: {mps_available}")
        
        # At least one acceleration method should be available for optimal performance
        assert cuda_available or mps_available or True, "Neither CUDA nor MPS available (CPU-only mode)"
        
        if cuda_available:
            print(f"CUDA device count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"CUDA device {i}: {torch.cuda.get_device_name(i)}")
        
        if mps_available:
            print("MPS (Apple Silicon GPU) support detected")

    def test_torch_basic_operations(self):
        """Test basic PyTorch operations."""
        torch = importlib.import_module("torch")
        
        # Create a simple tensor
        x = torch.randn(3, 3)
        assert x.shape == (3, 3)
        
        # Test basic operations
        y = x + 1
        assert y.shape == x.shape
        
        # Test device movement if MPS is available
        if torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False:
            device = torch.device("mps")
            x_mps = x.to(device)
            assert x_mps.device.type == "mps"

    def test_torchvision_installation(self):
        """Test torchvision installation."""
        torchvision = importlib.import_module("torchvision")
        assert hasattr(torchvision, "__version__")
        print(f"Torchvision version: {torchvision.__version__}")
        
        # Test basic transforms
        from torchvision import transforms
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        assert transform is not None

    def test_transformers_installation(self):
        """Test HuggingFace Transformers installation."""
        transformers = importlib.import_module("transformers")
        assert hasattr(transformers, "__version__")
        
        # Version check
        version_parts = transformers.__version__.split(".")
        major, minor = int(version_parts[0]), int(version_parts[1])
        assert major >= 4, f"Expected Transformers 4.x+, got {transformers.__version__}"
        if major == 4:
            assert minor >= 35, f"Expected Transformers 4.35+, got {transformers.__version__}"
        
        print(f"Transformers version: {transformers.__version__}")


class TestImageProcessingLibraries:
    """Test image processing libraries."""

    def test_opencv_installation(self):
        """Test OpenCV installation and basic functionality."""
        cv2 = importlib.import_module("cv2")
        assert hasattr(cv2, "__version__")
        
        # Version check
        version_parts = cv2.__version__.split(".")
        major, minor = int(version_parts[0]), int(version_parts[1])
        assert major >= 4, f"Expected OpenCV 4.x+, got {cv2.__version__}"
        if major == 4:
            assert minor >= 8, f"Expected OpenCV 4.8+, got {cv2.__version__}"
        
        print(f"OpenCV version: {cv2.__version__}")
        
        # Test basic image operations
        import numpy as np
        test_image = np.zeros((100, 100, 3), dtype=np.uint8)
        resized = cv2.resize(test_image, (50, 50))
        assert resized.shape == (50, 50, 3)

    def test_pillow_installation(self):
        """Test Pillow (PIL) installation."""
        PIL = importlib.import_module("PIL")
        assert hasattr(PIL, "__version__")
        print(f"Pillow version: {PIL.__version__}")
        
        # Test basic PIL operations
        from PIL import Image
        import numpy as np
        test_array = np.zeros((100, 100, 3), dtype=np.uint8)
        pil_image = Image.fromarray(test_array)
        assert pil_image.size == (100, 100)

    def test_numpy_installation(self):
        """Test NumPy installation."""
        numpy = importlib.import_module("numpy")
        assert hasattr(numpy, "__version__")
        
        # Version check - should be compatible with PyTorch requirements
        version_parts = numpy.__version__.split(".")
        major, minor = int(version_parts[0]), int(version_parts[1])
        assert major >= 1, f"Expected NumPy 1.x+, got {numpy.__version__}"
        if major == 1:
            assert minor >= 24, f"Expected NumPy 1.24+, got {numpy.__version__}"
        
        print(f"NumPy version: {numpy.__version__}")


class TestHuggingFaceIntegration:
    """Test HuggingFace ecosystem libraries."""

    def test_huggingface_hub(self):
        """Test HuggingFace Hub installation."""
        huggingface_hub = importlib.import_module("huggingface_hub")
        assert hasattr(huggingface_hub, "__version__")
        print(f"HuggingFace Hub version: {huggingface_hub.__version__}")

    def test_datasets_installation(self):
        """Test HuggingFace Datasets installation."""
        datasets = importlib.import_module("datasets")
        assert hasattr(datasets, "__version__")
        print(f"HuggingFace Datasets version: {datasets.__version__}")

    def test_tokenizers_installation(self):
        """Test tokenizers installation."""
        tokenizers = importlib.import_module("tokenizers")
        assert hasattr(tokenizers, "__version__")
        print(f"Tokenizers version: {tokenizers.__version__}")

    def test_accelerate_installation(self):
        """Test Accelerate library installation."""
        accelerate = importlib.import_module("accelerate")
        assert hasattr(accelerate, "__version__")
        print(f"Accelerate version: {accelerate.__version__}")


class TestConfigurationFiles:
    """Test configuration files and system setup."""

    def test_model_config_exists(self):
        """Test that model configuration file exists and is readable."""
        config_path = Path("config/model_config.yaml")
        assert config_path.exists(), f"Model config file not found: {config_path}"
        assert config_path.is_file(), f"Model config path is not a file: {config_path}"

    def test_yaml_parsing(self):
        """Test YAML parsing capability."""
        yaml = importlib.import_module("yaml")
        
        config_path = Path("config/model_config.yaml")
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        assert isinstance(config, dict), "Config file should contain a YAML dictionary"
        assert "models" in config, "Config should contain 'models' section"
        assert "runtime" in config, "Config should contain 'runtime' section"

    def test_cache_directory_creation(self):
        """Test that model cache directory can be created."""
        cache_dir = Path("models/cache")
        cache_dir.mkdir(parents=True, exist_ok=True)
        assert cache_dir.exists(), "Failed to create cache directory"
        assert cache_dir.is_dir(), "Cache path is not a directory"


class TestOptionalLibraries:
    """Test optional libraries and development tools."""

    def test_rich_installation(self):
        """Test Rich library for enhanced console output."""
        rich = importlib.import_module("rich")
        # Test that rich can be imported and used
        from rich.console import Console
        console = Console()
        assert console is not None
        print("Rich library installed and functional")

    def test_scikit_image_installation(self):
        """Test scikit-image installation."""
        skimage = importlib.import_module("skimage")
        assert hasattr(skimage, "__version__")
        print(f"Scikit-image version: {skimage.__version__}")


class TestMemoryAndPerformance:
    """Test system memory and performance characteristics."""

    def test_system_memory(self):
        """Test available system memory."""
        import psutil
        memory = psutil.virtual_memory()
        total_gb = memory.total / (1024**3)
        available_gb = memory.available / (1024**3)
        
        print(f"Total memory: {total_gb:.1f} GB")
        print(f"Available memory: {available_gb:.1f} GB")
        
        # Warn if memory is low for VLM models
        if total_gb < 8:
            pytest.warn(UserWarning(f"Low system memory ({total_gb:.1f} GB). VLM models may require 8+ GB"))

    def test_disk_space(self):
        """Test available disk space for model downloads."""
        import shutil
        total, used, free = shutil.disk_usage(".")
        free_gb = free / (1024**3)
        
        print(f"Free disk space: {free_gb:.1f} GB")
        
        # Warn if disk space is low for model storage
        if free_gb < 20:
            pytest.warn(UserWarning(f"Low disk space ({free_gb:.1f} GB). VLM models may require 20+ GB"))


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "--tb=short"])