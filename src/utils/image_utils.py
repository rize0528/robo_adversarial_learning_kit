"""Image processing utilities for VLM adversarial testing."""

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from typing import Union, Tuple, Optional, List, Dict
import torch
import torchvision.transforms as transforms
import logging
from pathlib import Path
import tempfile
import os

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


def load_image_with_opencv(image_path: Union[str, Path]) -> Optional[np.ndarray]:
    """Load image using OpenCV with robust error handling.
    
    Args:
        image_path: Path to image file
        
    Returns:
        Image as numpy array (BGR format) or None if loading fails
    """
    try:
        image_path = str(image_path)
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        
        if image is None:
            logger.error(f"OpenCV failed to load image: {image_path}")
            return None
            
        logger.debug(f"Loaded image with OpenCV: {image_path} ({image.shape})")
        return image
        
    except Exception as e:
        logger.error(f"Failed to load image with OpenCV {image_path}: {e}")
        return None


def opencv_to_pil(opencv_image: np.ndarray) -> Image.Image:
    """Convert OpenCV image (BGR) to PIL Image (RGB).
    
    Args:
        opencv_image: OpenCV image in BGR format
        
    Returns:
        PIL Image in RGB format
    """
    # Convert BGR to RGB
    rgb_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb_image)


def pil_to_opencv(pil_image: Image.Image) -> np.ndarray:
    """Convert PIL Image to OpenCV format.
    
    Args:
        pil_image: PIL Image
        
    Returns:
        OpenCV image in BGR format
    """
    # Ensure RGB format
    if pil_image.mode != 'RGB':
        pil_image = pil_image.convert('RGB')
    
    # Convert to numpy array and BGR
    rgb_array = np.array(pil_image)
    bgr_array = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)
    return bgr_array


def resize_image_maintaining_aspect_ratio(
    image: Union[Image.Image, np.ndarray],
    target_size: Tuple[int, int],
    fill_color: Tuple[int, int, int] = (128, 128, 128)
) -> Union[Image.Image, np.ndarray]:
    """Resize image while maintaining aspect ratio with padding.
    
    Args:
        image: Input image (PIL Image or numpy array)
        target_size: Target size (width, height)
        fill_color: Color for padding areas
        
    Returns:
        Resized image with padding
    """
    is_opencv = isinstance(image, np.ndarray)
    
    if is_opencv:
        h, w = image.shape[:2]
        pil_image = opencv_to_pil(image)
    else:
        w, h = image.size
        pil_image = image
    
    target_w, target_h = target_size
    
    # Calculate scaling factor
    scale = min(target_w / w, target_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    # Resize image
    resized = pil_image.resize((new_w, new_h), Image.Resampling.LANCZOS)
    
    # Create new image with target size and fill color
    result = Image.new('RGB', target_size, fill_color)
    
    # Paste resized image in center
    x_offset = (target_w - new_w) // 2
    y_offset = (target_h - new_h) // 2
    result.paste(resized, (x_offset, y_offset))
    
    if is_opencv:
        return pil_to_opencv(result)
    else:
        return result


def create_image_grid(
    images: List[Union[Image.Image, np.ndarray]],
    grid_size: Optional[Tuple[int, int]] = None,
    image_size: Tuple[int, int] = (224, 224),
    padding: int = 2
) -> Image.Image:
    """Create a grid of images for visualization.
    
    Args:
        images: List of images to arrange in grid
        grid_size: Grid dimensions (cols, rows). If None, auto-calculate square grid
        image_size: Size to resize each image to
        padding: Padding between images
        
    Returns:
        Grid image as PIL Image
    """
    if not images:
        raise ValueError("No images provided")
    
    num_images = len(images)
    
    if grid_size is None:
        # Calculate square grid
        cols = int(np.ceil(np.sqrt(num_images)))
        rows = int(np.ceil(num_images / cols))
        grid_size = (cols, rows)
    
    cols, rows = grid_size
    
    # Calculate grid dimensions
    grid_width = cols * image_size[0] + (cols - 1) * padding
    grid_height = rows * image_size[1] + (rows - 1) * padding
    
    # Create grid image
    grid = Image.new('RGB', (grid_width, grid_height), (255, 255, 255))
    
    for idx, img in enumerate(images):
        if idx >= cols * rows:
            break
            
        # Convert to PIL if needed
        if isinstance(img, np.ndarray):
            img = opencv_to_pil(img)
        
        # Resize image
        img = img.resize(image_size, Image.Resampling.LANCZOS)
        
        # Calculate position
        col = idx % cols
        row = idx // cols
        x = col * (image_size[0] + padding)
        y = row * (image_size[1] + padding)
        
        # Paste image
        grid.paste(img, (x, y))
    
    return grid


def add_text_to_image(
    image: Image.Image,
    text: str,
    position: Tuple[int, int] = (10, 10),
    font_size: int = 20,
    text_color: Tuple[int, int, int] = (255, 255, 255),
    background_color: Optional[Tuple[int, int, int]] = (0, 0, 0)
) -> Image.Image:
    """Add text overlay to image.
    
    Args:
        image: Input PIL Image
        text: Text to add
        position: Text position (x, y)
        font_size: Font size
        text_color: Text color (R, G, B)
        background_color: Background color for text. None for transparent
        
    Returns:
        Image with text overlay
    """
    img_copy = image.copy()
    draw = ImageDraw.Draw(img_copy)
    
    try:
        # Try to use a better font if available
        font = ImageFont.truetype("arial.ttf", font_size)
    except (OSError, IOError):
        # Fallback to default font
        font = ImageFont.load_default()
    
    # Get text dimensions
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    
    # Add background rectangle if specified
    if background_color is not None:
        padding = 5
        draw.rectangle([
            position[0] - padding,
            position[1] - padding,
            position[0] + text_width + padding,
            position[1] + text_height + padding
        ], fill=background_color)
    
    # Add text
    draw.text(position, text, fill=text_color, font=font)
    
    return img_copy


def create_comparison_image(
    images: List[Image.Image],
    labels: List[str],
    title: str = "Comparison"
) -> Image.Image:
    """Create comparison image with labels.
    
    Args:
        images: List of images to compare
        labels: Labels for each image
        title: Overall title
        
    Returns:
        Comparison image
    """
    if len(images) != len(labels):
        raise ValueError("Number of images and labels must match")
    
    # Add labels to images
    labeled_images = []
    for img, label in zip(images, labels):
        labeled_img = add_text_to_image(img, label, position=(10, 10))
        labeled_images.append(labeled_img)
    
    # Create grid
    grid = create_image_grid(labeled_images)
    
    # Add title
    if title:
        grid = add_text_to_image(
            grid, 
            title, 
            position=(10, grid.height - 40),
            font_size=24,
            text_color=(0, 0, 0),
            background_color=(255, 255, 255)
        )
    
    return grid


def save_tensor_batch_as_images(
    tensor_batch: torch.Tensor,
    output_dir: Union[str, Path],
    prefix: str = "image",
    denormalize: bool = True
) -> List[Path]:
    """Save a batch of tensors as individual image files.
    
    Args:
        tensor_batch: Batch of image tensors
        output_dir: Output directory
        prefix: Filename prefix
        denormalize: Whether to reverse ImageNet normalization
        
    Returns:
        List of saved file paths
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    saved_paths = []
    
    for i, tensor in enumerate(tensor_batch):
        output_path = output_dir / f"{prefix}_{i:04d}.png"
        save_tensor_as_image(tensor, str(output_path), denormalize)
        saved_paths.append(output_path)
        
    logger.info(f"Saved {len(saved_paths)} images to {output_dir}")
    return saved_paths


def create_sample_images_for_testing(
    output_dir: Union[str, Path],
    formats: List[str] = ['jpg', 'png'],
    sizes: List[Tuple[int, int]] = [(224, 224), (512, 512)],
    patterns: List[str] = ['gradient', 'checkerboard', 'noise']
) -> List[Path]:
    """Create sample images for testing in various formats and patterns.
    
    Args:
        output_dir: Directory to save test images
        formats: Image formats to create
        sizes: Image sizes to create
        patterns: Pattern types to create
        
    Returns:
        List of created image paths
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    created_images = []
    
    for pattern in patterns:
        for size in sizes:
            # Create pattern image
            if pattern == 'gradient':
                image = _create_gradient_image(size)
            elif pattern == 'checkerboard':
                image = _create_checkerboard_image(size)
            elif pattern == 'noise':
                image = _create_noise_image(size)
            else:
                image = create_test_image(size)
            
            # Save in different formats
            for fmt in formats:
                filename = f"{pattern}_{size[0]}x{size[1]}.{fmt}"
                filepath = output_dir / filename
                
                if fmt.lower() == 'jpg':
                    image.save(filepath, 'JPEG', quality=95)
                else:
                    image.save(filepath)
                
                created_images.append(filepath)
                
    logger.info(f"Created {len(created_images)} test images in {output_dir}")
    return created_images


def _create_gradient_image(size: Tuple[int, int]) -> Image.Image:
    """Create gradient test image."""
    width, height = size
    array = np.zeros((height, width, 3), dtype=np.uint8)
    
    for i in range(height):
        for j in range(width):
            array[i, j, 0] = int((i / height) * 255)  # Red gradient
            array[i, j, 1] = int((j / width) * 255)   # Green gradient
            array[i, j, 2] = int(((i + j) / (height + width)) * 255)  # Blue gradient
    
    return Image.fromarray(array)


def _create_checkerboard_image(size: Tuple[int, int], square_size: int = 32) -> Image.Image:
    """Create checkerboard test image."""
    width, height = size
    array = np.zeros((height, width, 3), dtype=np.uint8)
    
    for i in range(height):
        for j in range(width):
            # Determine if square should be white or black
            square_i = i // square_size
            square_j = j // square_size
            
            if (square_i + square_j) % 2 == 0:
                array[i, j] = [255, 255, 255]  # White
            else:
                array[i, j] = [0, 0, 0]  # Black
    
    return Image.fromarray(array)


def _create_noise_image(size: Tuple[int, int]) -> Image.Image:
    """Create random noise test image."""
    width, height = size
    array = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
    return Image.fromarray(array)


def get_image_stats(image: Union[Image.Image, np.ndarray, torch.Tensor]) -> Dict[str, float]:
    """Get statistical information about an image.
    
    Args:
        image: Input image
        
    Returns:
        Dictionary with image statistics
    """
    # Convert to numpy array
    if isinstance(image, Image.Image):
        array = np.array(image)
    elif isinstance(image, torch.Tensor):
        array = image.cpu().numpy()
        if array.ndim == 4:  # Batch dimension
            array = array[0]
        if array.ndim == 3 and array.shape[0] in [1, 3]:  # Channel first
            array = np.transpose(array, (1, 2, 0))
    else:
        array = image
    
    # Ensure uint8 range for meaningful stats
    if array.dtype != np.uint8:
        array = (array * 255).astype(np.uint8)
    
    stats = {
        'shape': array.shape,
        'dtype': str(array.dtype),
        'min': float(array.min()),
        'max': float(array.max()),
        'mean': float(array.mean()),
        'std': float(array.std()),
        'size_mb': array.nbytes / (1024 * 1024)
    }
    
    # Channel-wise stats if color image
    if array.ndim == 3 and array.shape[2] == 3:
        channel_names = ['red', 'green', 'blue']
        for i, name in enumerate(channel_names):
            channel = array[:, :, i]
            stats[f'{name}_mean'] = float(channel.mean())
            stats[f'{name}_std'] = float(channel.std())
    
    return stats