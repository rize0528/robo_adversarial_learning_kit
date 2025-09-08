"""Image processing utilities for VLM adversarial testing."""

import numpy as np
from PIL import Image
from typing import Union, Tuple, Optional
import torch
import torchvision.transforms as transforms
import logging

logger = logging.getLogger(__name__)


def load_test_image(image_path: str) -> Optional[Image.Image]:
    """Load a test image from file path.
    
    Args:
        image_path: Path to image file
        
    Returns:
        PIL Image object or None if loading fails
    """
    try:
        image = Image.open(image_path)
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        logger.info(f"Loaded image: {image_path} ({image.size})")
        return image
    except Exception as e:
        logger.error(f"Failed to load image {image_path}: {e}")
        return None


def preprocess_image(
    image: Union[Image.Image, np.ndarray], 
    target_size: Tuple[int, int] = (224, 224),
    normalize: bool = True
) -> torch.Tensor:
    """Preprocess image for VLM input.
    
    Args:
        image: Input image (PIL Image or numpy array)
        target_size: Target size (height, width)
        normalize: Whether to normalize pixel values
        
    Returns:
        Preprocessed image tensor
    """
    try:
        # Convert numpy array to PIL if needed
        if isinstance(image, np.ndarray):
            if image.dtype == np.uint8:
                image = Image.fromarray(image)
            else:
                # Assume float values in [0,1], convert to uint8
                image = Image.fromarray((image * 255).astype(np.uint8))
        
        # Create preprocessing pipeline
        transforms_list = [
            transforms.Resize(target_size),
            transforms.ToTensor()
        ]
        
        if normalize:
            # Standard ImageNet normalization
            transforms_list.append(
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            )
        
        transform = transforms.Compose(transforms_list)
        
        # Apply preprocessing
        tensor = transform(image)
        
        # Add batch dimension
        if tensor.dim() == 3:
            tensor = tensor.unsqueeze(0)
            
        logger.debug(f"Preprocessed image to tensor shape: {tensor.shape}")
        return tensor
        
    except Exception as e:
        logger.error(f"Failed to preprocess image: {e}")
        raise


def create_test_image(size: Tuple[int, int] = (224, 224)) -> Image.Image:
    """Create a simple test image for debugging.
    
    Args:
        size: Image size (width, height)
        
    Returns:
        PIL Image with test pattern
    """
    # Create a simple gradient pattern
    width, height = size
    array = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Create a colorful test pattern
    for i in range(height):
        for j in range(width):
            array[i, j, 0] = int((i / height) * 255)  # Red gradient
            array[i, j, 1] = int((j / width) * 255)   # Green gradient  
            array[i, j, 2] = 128                       # Constant blue
    
    image = Image.fromarray(array)
    logger.debug(f"Created test image: {size}")
    return image


def save_tensor_as_image(tensor: torch.Tensor, output_path: str, denormalize: bool = True):
    """Save a tensor as an image file.
    
    Args:
        tensor: Image tensor to save
        output_path: Output file path
        denormalize: Whether to reverse ImageNet normalization
    """
    try:
        # Remove batch dimension if present
        if tensor.dim() == 4:
            tensor = tensor.squeeze(0)
        
        # Denormalize if needed
        if denormalize:
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            tensor = tensor * std + mean
        
        # Clamp values to valid range
        tensor = torch.clamp(tensor, 0, 1)
        
        # Convert to PIL Image
        transform = transforms.ToPILImage()
        image = transform(tensor)
        
        # Save image
        image.save(output_path)
        logger.info(f"Saved tensor as image: {output_path}")
        
    except Exception as e:
        logger.error(f"Failed to save tensor as image: {e}")
        raise