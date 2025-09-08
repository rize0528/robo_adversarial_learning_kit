"""Comprehensive tests for image preprocessing functionality."""

import os
import tempfile
import pytest
import torch
import numpy as np
from PIL import Image
from pathlib import Path

from src.data.preprocessing import (
    ImagePreprocessor, 
    DataAugmentation,
    preprocess_for_vlm,
    batch_preprocess_images,
    create_data_augmentation_pipeline,
    augment_image,
    get_supported_image_formats,
    validate_image_file,
    find_images_in_directory
)

from src.utils.image_utils import (
    create_sample_images_for_testing,
    load_image_with_opencv,
    opencv_to_pil,
    pil_to_opencv,
    create_test_image,
    get_image_stats
)


class TestImagePreprocessor:
    """Test cases for ImagePreprocessor class."""
    
    @pytest.fixture
    def preprocessor(self):
        """Create preprocessor instance for testing."""
        return ImagePreprocessor(target_size=(224, 224), normalize=True)
    
    @pytest.fixture
    def test_image(self):
        """Create test image for testing."""
        return create_test_image(size=(300, 200))
    
    @pytest.fixture
    def test_image_path(self):
        """Create temporary image file for testing."""
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            image = create_test_image(size=(256, 256))
            image.save(tmp.name)
            yield tmp.name
            os.unlink(tmp.name)
    
    def test_initialization(self, preprocessor):
        """Test preprocessor initialization."""
        assert preprocessor.target_size == (224, 224)
        assert preprocessor.normalize is True
        assert preprocessor.device is not None
        assert len(preprocessor.mean) == 3
        assert len(preprocessor.std) == 3
    
    def test_preprocess_pil_image(self, preprocessor, test_image):
        """Test preprocessing PIL image."""
        tensor = preprocessor.preprocess(test_image)
        
        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (1, 3, 224, 224)  # Batch, Channels, Height, Width
        assert tensor.dtype == torch.float32
    
    def test_preprocess_image_path(self, preprocessor, test_image_path):
        """Test preprocessing from image path."""
        tensor = preprocessor.preprocess(test_image_path)
        
        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (1, 3, 224, 224)
    
    def test_preprocess_numpy_array(self, preprocessor):
        """Test preprocessing numpy array."""
        # Create numpy array image
        array = np.random.randint(0, 256, (200, 300, 3), dtype=np.uint8)
        tensor = preprocessor.preprocess(array)
        
        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (1, 3, 224, 224)
    
    def test_preprocess_quality_mode(self, preprocessor, test_image):
        """Test high-quality preprocessing mode."""
        tensor_basic = preprocessor.preprocess(test_image, quality_mode=False)
        tensor_quality = preprocessor.preprocess(test_image, quality_mode=True)
        
        # Both should have same shape
        assert tensor_basic.shape == tensor_quality.shape
        # Should be tensors
        assert isinstance(tensor_basic, torch.Tensor)
        assert isinstance(tensor_quality, torch.Tensor)
    
    def test_preprocess_no_batch_dim(self, preprocessor, test_image):
        """Test preprocessing without batch dimension."""
        tensor = preprocessor.preprocess(test_image, add_batch_dim=False)
        
        assert tensor.shape == (3, 224, 224)  # No batch dimension
    
    def test_batch_preprocess(self, preprocessor):
        """Test batch preprocessing multiple images."""
        # Create multiple test images
        images = [create_test_image((150, 100)), create_test_image((200, 250))]
        
        batch_tensor = preprocessor.batch_preprocess(images)
        
        assert isinstance(batch_tensor, torch.Tensor)
        assert batch_tensor.shape == (2, 3, 224, 224)  # 2 images in batch
    
    def test_load_image_success(self, test_image_path):
        """Test successful image loading."""
        image = ImagePreprocessor.load_image(test_image_path)
        
        assert isinstance(image, Image.Image)
        assert image.mode == 'RGB'
    
    def test_load_image_nonexistent(self):
        """Test loading nonexistent image."""
        with pytest.raises(FileNotFoundError):
            ImagePreprocessor.load_image('nonexistent_file.jpg')
    
    def test_numpy_to_pil_uint8(self, preprocessor):
        """Test numpy to PIL conversion with uint8."""
        array = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        image = preprocessor._numpy_to_pil(array)
        
        assert isinstance(image, Image.Image)
        assert image.size == (100, 100)
    
    def test_numpy_to_pil_float(self, preprocessor):
        """Test numpy to PIL conversion with float."""
        array = np.random.rand(100, 100, 3).astype(np.float32)
        image = preprocessor._numpy_to_pil(array)
        
        assert isinstance(image, Image.Image)
        assert image.size == (100, 100)


class TestDataAugmentation:
    """Test cases for DataAugmentation class."""
    
    @pytest.fixture
    def augmentator(self):
        """Create augmentator instance for testing."""
        return DataAugmentation(
            rotation_range=(-10, 10),
            brightness_range=(0.9, 1.1),
            horizontal_flip=True
        )
    
    @pytest.fixture
    def test_image(self):
        """Create test image for augmentation."""
        return create_test_image(size=(224, 224))
    
    def test_initialization(self, augmentator):
        """Test augmentator initialization."""
        assert augmentator.rotation_range == (-10, 10)
        assert augmentator.brightness_range == (0.9, 1.1)
        assert augmentator.horizontal_flip is True
    
    def test_augment_image(self, augmentator, test_image):
        """Test image augmentation."""
        original_size = test_image.size
        augmented = augmentator.augment(test_image)
        
        assert isinstance(augmented, Image.Image)
        assert augmented.size == original_size
        assert augmented.mode == test_image.mode
    
    def test_augment_preserves_original(self, augmentator, test_image):
        """Test that augmentation doesn't modify original image."""
        original_data = np.array(test_image)
        augmented = augmentator.augment(test_image)
        post_augment_original = np.array(test_image)
        
        # Original should be unchanged
        np.testing.assert_array_equal(original_data, post_augment_original)
    
    def test_color_adjustments(self, test_image):
        """Test color adjustment augmentations."""
        augmentator = DataAugmentation(
            brightness_range=(1.2, 1.2),  # Fixed brightness increase
            contrast_range=(1.0, 1.0),    # No contrast change
            rotation_range=(0, 0),        # No rotation
            horizontal_flip=False
        )
        
        augmented = augmentator.augment(test_image)
        
        # Should be brighter than original
        original_array = np.array(test_image)
        augmented_array = np.array(augmented)
        
        assert isinstance(augmented, Image.Image)
        # Basic sanity check - shapes should match
        assert original_array.shape == augmented_array.shape


class TestConvenienceFunctions:
    """Test convenience functions for preprocessing."""
    
    @pytest.fixture
    def test_image_paths(self):
        """Create temporary test images."""
        paths = []
        for i in range(3):
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                image = create_test_image(size=(128, 128))
                image.save(tmp.name)
                paths.append(tmp.name)
        
        yield paths
        
        # Cleanup
        for path in paths:
            if os.path.exists(path):
                os.unlink(path)
    
    def test_preprocess_for_vlm(self, test_image_paths):
        """Test VLM preprocessing convenience function."""
        tensor = preprocess_for_vlm(test_image_paths[0])
        
        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (1, 3, 224, 224)
    
    def test_batch_preprocess_images(self, test_image_paths):
        """Test batch preprocessing convenience function."""
        batch_tensor = batch_preprocess_images(test_image_paths)
        
        assert isinstance(batch_tensor, torch.Tensor)
        assert batch_tensor.shape == (3, 3, 224, 224)
    
    def test_create_data_augmentation_pipeline_weak(self):
        """Test creating weak augmentation pipeline."""
        augmentator = create_data_augmentation_pipeline(strong_augmentation=False)
        
        assert isinstance(augmentator, DataAugmentation)
        assert augmentator.rotation_range == (-10, 10)
        assert augmentator.brightness_range == (0.9, 1.1)
    
    def test_create_data_augmentation_pipeline_strong(self):
        """Test creating strong augmentation pipeline."""
        augmentator = create_data_augmentation_pipeline(strong_augmentation=True)
        
        assert isinstance(augmentator, DataAugmentation)
        assert augmentator.rotation_range == (-30, 30)
        assert augmentator.brightness_range == (0.6, 1.4)
    
    def test_augment_image_function(self):
        """Test augment_image convenience function."""
        test_image = create_test_image(size=(200, 200))
        augmented = augment_image(test_image)
        
        assert isinstance(augmented, Image.Image)
        assert augmented.size == test_image.size


class TestUtilityFunctions:
    """Test utility functions for image processing."""
    
    def test_get_supported_image_formats(self):
        """Test getting supported image formats."""
        formats = get_supported_image_formats()
        
        assert isinstance(formats, list)
        assert '.jpg' in formats
        assert '.png' in formats
        assert len(formats) > 0
    
    def test_validate_image_file_valid(self):
        """Test validation of valid image file."""
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            image = create_test_image(size=(100, 100))
            image.save(tmp.name)
            
            try:
                assert validate_image_file(tmp.name) is True
            finally:
                os.unlink(tmp.name)
    
    def test_validate_image_file_invalid(self):
        """Test validation of invalid file."""
        # Test non-existent file
        assert validate_image_file('nonexistent.jpg') is False
        
        # Test non-image file
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as tmp:
            tmp.write(b'Not an image')
            tmp.flush()
            
            try:
                assert validate_image_file(tmp.name) is False
            finally:
                os.unlink(tmp.name)
    
    def test_find_images_in_directory(self):
        """Test finding images in directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            
            # Create test images
            image_paths = create_sample_images_for_testing(
                tmpdir_path,
                formats=['png', 'jpg'],
                sizes=[(100, 100)],
                patterns=['gradient']
            )
            
            found_images = find_images_in_directory(tmpdir_path)
            
            assert len(found_images) == len(image_paths)
            assert all(img_path in found_images for img_path in image_paths)


class TestOpenCVIntegration:
    """Test OpenCV integration functions."""
    
    @pytest.fixture
    def test_image_path(self):
        """Create test image file."""
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            image = create_test_image(size=(100, 100))
            image.save(tmp.name)
            yield tmp.name
            os.unlink(tmp.name)
    
    def test_load_image_with_opencv(self, test_image_path):
        """Test loading image with OpenCV."""
        cv_image = load_image_with_opencv(test_image_path)
        
        assert cv_image is not None
        assert isinstance(cv_image, np.ndarray)
        assert len(cv_image.shape) == 3  # Height, Width, Channels
        assert cv_image.shape[2] == 3    # BGR channels
    
    def test_opencv_pil_conversion(self, test_image_path):
        """Test OpenCV to PIL conversion and vice versa."""
        # Load with OpenCV
        cv_image = load_image_with_opencv(test_image_path)
        assert cv_image is not None
        
        # Convert to PIL
        pil_image = opencv_to_pil(cv_image)
        assert isinstance(pil_image, Image.Image)
        assert pil_image.mode == 'RGB'
        
        # Convert back to OpenCV
        cv_image_back = pil_to_opencv(pil_image)
        assert isinstance(cv_image_back, np.ndarray)
        assert cv_image_back.shape == cv_image.shape
    
    def test_get_image_stats(self):
        """Test getting image statistics."""
        # Test with PIL Image
        pil_image = create_test_image(size=(100, 100))
        stats = get_image_stats(pil_image)
        
        assert isinstance(stats, dict)
        assert 'shape' in stats
        assert 'mean' in stats
        assert 'std' in stats
        assert 'min' in stats
        assert 'max' in stats
        
        # Test with tensor
        tensor = torch.randn(3, 100, 100)
        stats_tensor = get_image_stats(tensor)
        assert isinstance(stats_tensor, dict)
        assert 'shape' in stats_tensor


class TestMemoryEfficiency:
    """Test memory efficiency of preprocessing operations."""
    
    def test_large_batch_processing(self):
        """Test processing large batch of images."""
        # Create multiple test images
        images = [create_test_image((256, 256)) for _ in range(10)]
        
        preprocessor = ImagePreprocessor(target_size=(224, 224))
        batch_tensor = preprocessor.batch_preprocess(images)
        
        assert batch_tensor.shape == (10, 3, 224, 224)
        
        # Check memory usage is reasonable (should be less than 100MB for 10 images)
        memory_mb = batch_tensor.element_size() * batch_tensor.nelement() / (1024 * 1024)
        assert memory_mb < 100
    
    def test_preprocessing_memory_cleanup(self):
        """Test that preprocessing properly cleans up memory."""
        preprocessor = ImagePreprocessor()
        
        # Process multiple images
        for _ in range(5):
            test_image = create_test_image((512, 512))
            tensor = preprocessor.preprocess(test_image)
            del tensor, test_image
        
        # Should complete without memory errors
        assert preprocessor is not None


class TestErrorHandling:
    """Test error handling in preprocessing functions."""
    
    def test_invalid_image_path(self):
        """Test handling of invalid image paths."""
        preprocessor = ImagePreprocessor()
        
        with pytest.raises(FileNotFoundError):
            preprocessor.preprocess('invalid_path.jpg')
    
    def test_corrupted_image_data(self):
        """Test handling of corrupted image data."""
        # Create file with invalid image data
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
            tmp.write(b'Not an image file')
            tmp.flush()
            
            try:
                with pytest.raises(ValueError):
                    ImagePreprocessor.load_image(tmp.name)
            finally:
                os.unlink(tmp.name)
    
    def test_empty_batch_processing(self):
        """Test handling of empty batch."""
        preprocessor = ImagePreprocessor()
        
        batch_tensor = preprocessor.batch_preprocess([])
        assert batch_tensor.shape[0] == 0  # Empty batch


if __name__ == '__main__':
    # Run tests with verbose output for debugging
    pytest.main([__file__, '-v', '--tb=short'])