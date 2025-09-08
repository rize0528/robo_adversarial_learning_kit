"""Comprehensive image preprocessing pipeline for VLM adversarial training."""

import logging
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image, ImageEnhance, ImageFilter

logger = logging.getLogger(__name__)


class ImagePreprocessor:
    """Main image preprocessing class with VLM-specific optimizations."""
    
    def __init__(
        self,
        target_size: Tuple[int, int] = (224, 224),
        normalize: bool = True,
        device: Optional[torch.device] = None
    ):
        """Initialize image preprocessor.
        
        Args:
            target_size: Target size for image resizing (height, width)
            normalize: Whether to apply ImageNet normalization
            device: Target device for tensor operations
        """
        self.target_size = target_size
        self.normalize = normalize
        self.device = device or torch.device('cpu')
        
        # ImageNet normalization parameters
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        
        self._create_transforms()
        
    def _create_transforms(self):
        """Create transformation pipelines for different use cases."""
        # Basic preprocessing pipeline
        basic_transforms = [
            transforms.Resize(self.target_size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor()
        ]
        
        if self.normalize:
            basic_transforms.append(
                transforms.Normalize(mean=self.mean, std=self.std)
            )
        
        self.basic_transform = transforms.Compose(basic_transforms)
        
        # High-quality preprocessing for evaluation
        quality_transforms = [
            transforms.Resize(self.target_size, interpolation=transforms.InterpolationMode.LANCZOS),
            transforms.ToTensor()
        ]
        
        if self.normalize:
            quality_transforms.append(
                transforms.Normalize(mean=self.mean, std=self.std)
            )
        
        self.quality_transform = transforms.Compose(quality_transforms)
    
    def preprocess(
        self, 
        image: Union[Image.Image, np.ndarray, str, Path], 
        quality_mode: bool = False,
        add_batch_dim: bool = True
    ) -> torch.Tensor:
        """Preprocess a single image for VLM input.
        
        Args:
            image: Input image (PIL, numpy, or file path)
            quality_mode: Use higher quality but slower preprocessing
            add_batch_dim: Add batch dimension to output tensor
            
        Returns:
            Preprocessed image tensor
        """
        # Load image if path provided
        if isinstance(image, (str, Path)):
            image = self.load_image(image)
        
        # Convert numpy to PIL if needed
        if isinstance(image, np.ndarray):
            image = self._numpy_to_pil(image)
        
        # Ensure RGB format
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Apply transforms
        transform = self.quality_transform if quality_mode else self.basic_transform
        tensor = transform(image)
        
        # Add batch dimension if requested
        if add_batch_dim and tensor.dim() == 3:
            tensor = tensor.unsqueeze(0)
        
        # Move to target device
        tensor = tensor.to(self.device)
        
        logger.debug(f"Preprocessed image to tensor shape: {tensor.shape}")
        return tensor
    
    def batch_preprocess(
        self, 
        images: List[Union[Image.Image, np.ndarray, str, Path]],
        quality_mode: bool = False
    ) -> torch.Tensor:
        """Preprocess multiple images into a batch tensor.
        
        Args:
            images: List of input images
            quality_mode: Use higher quality preprocessing
            
        Returns:
            Batch tensor of preprocessed images
        """
        if not images:
            # Return empty tensor with correct shape for empty batch
            empty_tensor = torch.empty((0, 3, self.target_size[0], self.target_size[1]))
            logger.debug("Created empty batch tensor")
            return empty_tensor.to(self.device)
        
        processed_images = []
        
        for image in images:
            tensor = self.preprocess(image, quality_mode=quality_mode, add_batch_dim=False)
            processed_images.append(tensor)
        
        # Stack into batch
        batch_tensor = torch.stack(processed_images, dim=0)
        logger.debug(f"Created batch tensor with shape: {batch_tensor.shape}")
        
        return batch_tensor
    
    @staticmethod
    def load_image(image_path: Union[str, Path]) -> Image.Image:
        """Load image from file path with error handling.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Loaded PIL Image
            
        Raises:
            FileNotFoundError: If image file doesn't exist
            ValueError: If image format not supported
        """
        image_path = Path(image_path)
        
        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        try:
            image = Image.open(image_path)
            # Verify image can be loaded
            image.verify()
            # Reopen after verify (verify closes the image)
            image = Image.open(image_path)
            logger.debug(f"Loaded image: {image_path} ({image.size}, {image.mode})")
            return image
            
        except Exception as e:
            raise ValueError(f"Failed to load image {image_path}: {e}")
    
    @staticmethod
    def _numpy_to_pil(array: np.ndarray) -> Image.Image:
        """Convert numpy array to PIL Image.
        
        Args:
            array: Input numpy array
            
        Returns:
            PIL Image
        """
        if array.dtype == np.uint8:
            return Image.fromarray(array)
        elif array.dtype in [np.float32, np.float64]:
            # Assume values in [0,1] range
            array_uint8 = (np.clip(array, 0, 1) * 255).astype(np.uint8)
            return Image.fromarray(array_uint8)
        else:
            # Try to normalize to [0,1] first
            array_norm = (array - array.min()) / (array.max() - array.min())
            array_uint8 = (array_norm * 255).astype(np.uint8)
            return Image.fromarray(array_uint8)


class DataAugmentation:
    """Data augmentation pipeline for training robustness."""
    
    def __init__(
        self,
        rotation_range: Tuple[float, float] = (-15, 15),
        brightness_range: Tuple[float, float] = (0.8, 1.2),
        contrast_range: Tuple[float, float] = (0.8, 1.2),
        saturation_range: Tuple[float, float] = (0.8, 1.2),
        hue_range: Tuple[float, float] = (-0.1, 0.1),
        blur_probability: float = 0.1,
        noise_probability: float = 0.1,
        horizontal_flip: bool = True
    ):
        """Initialize data augmentation parameters.
        
        Args:
            rotation_range: Random rotation angle range in degrees
            brightness_range: Brightness adjustment factor range
            contrast_range: Contrast adjustment factor range
            saturation_range: Saturation adjustment factor range
            hue_range: Hue adjustment range
            blur_probability: Probability of applying gaussian blur
            noise_probability: Probability of adding gaussian noise
            horizontal_flip: Whether to randomly flip horizontally
        """
        self.rotation_range = rotation_range
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.saturation_range = saturation_range
        self.hue_range = hue_range
        self.blur_probability = blur_probability
        self.noise_probability = noise_probability
        self.horizontal_flip = horizontal_flip
        
    def augment(self, image: Image.Image) -> Image.Image:
        """Apply random augmentations to image.
        
        Args:
            image: Input PIL Image
            
        Returns:
            Augmented PIL Image
        """
        # Make a copy to avoid modifying original
        augmented = image.copy()
        
        # Random horizontal flip
        if self.horizontal_flip and random.random() > 0.5:
            augmented = augmented.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        
        # Random rotation
        if self.rotation_range != (0, 0):
            angle = random.uniform(*self.rotation_range)
            augmented = augmented.rotate(
                angle, 
                resample=Image.Resampling.BILINEAR,
                fillcolor=(128, 128, 128)  # Gray fill for rotated areas
            )
        
        # Color adjustments
        augmented = self._adjust_colors(augmented)
        
        # Blur
        if random.random() < self.blur_probability:
            blur_radius = random.uniform(0.1, 1.0)
            augmented = augmented.filter(ImageFilter.GaussianBlur(radius=blur_radius))
        
        # Noise
        if random.random() < self.noise_probability:
            augmented = self._add_noise(augmented)
            
        return augmented
    
    def _adjust_colors(self, image: Image.Image) -> Image.Image:
        """Apply random color adjustments."""
        # Brightness
        if self.brightness_range != (1.0, 1.0):
            factor = random.uniform(*self.brightness_range)
            enhancer = ImageEnhance.Brightness(image)
            image = enhancer.enhance(factor)
        
        # Contrast
        if self.contrast_range != (1.0, 1.0):
            factor = random.uniform(*self.contrast_range)
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(factor)
        
        # Saturation (Color)
        if self.saturation_range != (1.0, 1.0):
            factor = random.uniform(*self.saturation_range)
            enhancer = ImageEnhance.Color(image)
            image = enhancer.enhance(factor)
        
        return image
    
    def _add_noise(self, image: Image.Image) -> Image.Image:
        """Add gaussian noise to image."""
        # Convert to numpy for noise addition
        array = np.array(image)
        
        # Add gaussian noise
        noise_std = random.uniform(5, 15)  # Noise standard deviation
        noise = np.random.normal(0, noise_std, array.shape).astype(np.float32)
        
        # Add noise and clip to valid range
        noisy_array = np.clip(array.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        
        return Image.fromarray(noisy_array)


def preprocess_for_vlm(
    image_path: Union[str, Path],
    target_size: Tuple[int, int] = (224, 224),
    normalize: bool = True,
    device: Optional[torch.device] = None
) -> torch.Tensor:
    """Convenience function for VLM preprocessing.
    
    Args:
        image_path: Path to image file
        target_size: Target size (height, width)
        normalize: Apply ImageNet normalization
        device: Target device
        
    Returns:
        Preprocessed tensor ready for VLM input
    """
    preprocessor = ImagePreprocessor(target_size, normalize, device)
    return preprocessor.preprocess(image_path)


def batch_preprocess_images(
    image_paths: List[Union[str, Path]],
    target_size: Tuple[int, int] = (224, 224),
    normalize: bool = True,
    device: Optional[torch.device] = None,
    quality_mode: bool = False
) -> torch.Tensor:
    """Batch preprocess multiple images.
    
    Args:
        image_paths: List of image file paths
        target_size: Target size for all images
        normalize: Apply normalization
        device: Target device
        quality_mode: Use higher quality preprocessing
        
    Returns:
        Batch tensor of preprocessed images
    """
    preprocessor = ImagePreprocessor(target_size, normalize, device)
    return preprocessor.batch_preprocess(image_paths, quality_mode)


def create_data_augmentation_pipeline(
    strong_augmentation: bool = False
) -> DataAugmentation:
    """Create data augmentation pipeline with preset configurations.
    
    Args:
        strong_augmentation: Use stronger augmentation settings
        
    Returns:
        Configured DataAugmentation instance
    """
    if strong_augmentation:
        return DataAugmentation(
            rotation_range=(-30, 30),
            brightness_range=(0.6, 1.4),
            contrast_range=(0.6, 1.4),
            saturation_range=(0.6, 1.4),
            hue_range=(-0.2, 0.2),
            blur_probability=0.2,
            noise_probability=0.2,
            horizontal_flip=True
        )
    else:
        return DataAugmentation(
            rotation_range=(-10, 10),
            brightness_range=(0.9, 1.1),
            contrast_range=(0.9, 1.1),
            saturation_range=(0.9, 1.1),
            hue_range=(-0.05, 0.05),
            blur_probability=0.05,
            noise_probability=0.05,
            horizontal_flip=True
        )


def augment_image(
    image: Union[Image.Image, str, Path],
    strong_augmentation: bool = False
) -> Image.Image:
    """Convenience function to augment a single image.
    
    Args:
        image: Input image or path
        strong_augmentation: Use strong augmentation settings
        
    Returns:
        Augmented PIL Image
    """
    if isinstance(image, (str, Path)):
        image = ImagePreprocessor.load_image(image)
    
    augmentator = create_data_augmentation_pipeline(strong_augmentation)
    return augmentator.augment(image)


def get_supported_image_formats() -> List[str]:
    """Get list of supported image formats.
    
    Returns:
        List of supported file extensions
    """
    return ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp']


def validate_image_file(file_path: Union[str, Path]) -> bool:
    """Validate if file is a supported image format.
    
    Args:
        file_path: Path to file
        
    Returns:
        True if file is a valid image format
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        return False
    
    # Check extension
    if file_path.suffix.lower() not in get_supported_image_formats():
        return False
    
    # Try to open and verify
    try:
        with Image.open(file_path) as img:
            img.verify()
        return True
    except Exception:
        return False


def find_images_in_directory(
    directory: Union[str, Path],
    recursive: bool = True
) -> List[Path]:
    """Find all supported image files in directory.
    
    Args:
        directory: Directory to search
        recursive: Search subdirectories recursively
        
    Returns:
        List of image file paths
    """
    directory = Path(directory)
    supported_formats = get_supported_image_formats()
    
    if recursive:
        pattern = "**/*"
    else:
        pattern = "*"
    
    image_files = []
    for file_path in directory.glob(pattern):
        if file_path.is_file() and file_path.suffix.lower() in supported_formats:
            if validate_image_file(file_path):
                image_files.append(file_path)
    
    return sorted(image_files)