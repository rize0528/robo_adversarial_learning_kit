"""PyTorch Dataset and DataLoader implementations for adversarial training data."""

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any, Callable

import torch
import torch.utils.data
from torch.utils.data import Dataset, DataLoader
from PIL import Image

from .preprocessing import (
    ImagePreprocessor,
    DataAugmentation,
    create_data_augmentation_pipeline,
    find_images_in_directory,
    validate_image_file
)

logger = logging.getLogger(__name__)


class AdversarialDataset(Dataset):
    """PyTorch Dataset for adversarial training data with integrated preprocessing."""
    
    def __init__(
        self,
        data_sources: Union[List[str], str, List[Path], Path],
        targets: Optional[Union[List[Any], Dict[str, Any]]] = None,
        preprocessor: Optional[ImagePreprocessor] = None,
        augmentator: Optional[DataAugmentation] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        cache_preprocessed: bool = False,
        max_cache_size: Optional[int] = None,
        recursive_search: bool = True,
        validate_images: bool = True
    ):
        """Initialize adversarial training dataset.
        
        Args:
            data_sources: Image files/directories or list of image paths
            targets: Optional targets/labels for supervised learning
            preprocessor: ImagePreprocessor instance (creates default if None)
            augmentator: DataAugmentation instance for training augmentation
            transform: Additional transform to apply after preprocessing
            target_transform: Transform to apply to targets
            cache_preprocessed: Cache preprocessed tensors in memory
            max_cache_size: Maximum number of cached items (None = unlimited)
            recursive_search: Search directories recursively for images
            validate_images: Validate image files before including them
        """
        self.targets = targets
        self.transform = transform
        self.target_transform = target_transform
        self.cache_preprocessed = cache_preprocessed
        self.max_cache_size = max_cache_size
        self.recursive_search = recursive_search
        self.validate_images = validate_images
        
        # Initialize preprocessor
        self.preprocessor = preprocessor or ImagePreprocessor()
        self.augmentator = augmentator
        
        # Cache for preprocessed images
        self._cache: Dict[int, torch.Tensor] = {}
        self._cache_order: List[int] = []  # For LRU eviction
        
        # Process data sources and build image list
        self.image_paths = self._process_data_sources(data_sources)
        
        # Validate targets if provided
        if self.targets is not None:
            self._validate_targets()
        
        logger.info(f"Initialized AdversarialDataset with {len(self.image_paths)} images")
    
    def _process_data_sources(
        self, 
        data_sources: Union[List[str], str, List[Path], Path]
    ) -> List[Path]:
        """Process data sources and build list of image paths.
        
        Args:
            data_sources: Various input formats for image data
            
        Returns:
            List of validated image paths
        """
        image_paths = []
        
        # Normalize to list of Path objects
        if isinstance(data_sources, (str, Path)):
            data_sources = [Path(data_sources)]
        else:
            data_sources = [Path(source) for source in data_sources]
        
        for source in data_sources:
            if source.is_file():
                # Single image file
                if self.validate_images:
                    if validate_image_file(source):
                        image_paths.append(source)
                    else:
                        logger.warning(f"Skipping invalid image file: {source}")
                else:
                    image_paths.append(source)
                    
            elif source.is_dir():
                # Directory of images
                found_images = find_images_in_directory(source, self.recursive_search)
                if self.validate_images:
                    # Images are already validated by find_images_in_directory
                    image_paths.extend(found_images)
                else:
                    # Re-find without validation for speed
                    from .preprocessing import get_supported_image_formats
                    supported_formats = get_supported_image_formats()
                    pattern = "**/*" if self.recursive_search else "*"
                    
                    for file_path in source.glob(pattern):
                        if (file_path.is_file() and 
                            file_path.suffix.lower() in supported_formats):
                            image_paths.append(file_path)
            else:
                if not self.validate_images:
                    # Include non-existent files when validation is disabled
                    # They will be handled gracefully during loading
                    image_paths.append(source)
                else:
                    logger.warning(f"Skipping non-existent path: {source}")
        
        if not image_paths:
            raise ValueError("No valid images found in provided data sources")
        
        return sorted(image_paths)  # Sort for consistent ordering
    
    def _validate_targets(self):
        """Validate that targets match the number of images."""
        if isinstance(self.targets, list):
            if len(self.targets) != len(self.image_paths):
                raise ValueError(
                    f"Number of targets ({len(self.targets)}) does not match "
                    f"number of images ({len(self.image_paths)})"
                )
        elif isinstance(self.targets, dict):
            # Map target keys to image paths
            missing_targets = []
            for img_path in self.image_paths:
                # Try various key formats
                key_candidates = [
                    str(img_path),
                    img_path.name,
                    img_path.stem,
                    str(img_path.relative_to(img_path.parent.parent))
                ]
                
                if not any(key in self.targets for key in key_candidates):
                    missing_targets.append(img_path)
            
            if missing_targets:
                logger.warning(
                    f"No targets found for {len(missing_targets)} images. "
                    f"Using None for missing targets."
                )
    
    def __len__(self) -> int:
        """Return number of images in dataset."""
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Any]:
        """Get preprocessed image and target for given index.
        
        Args:
            idx: Index of item to retrieve
            
        Returns:
            Tuple of (preprocessed_image_tensor, target)
        """
        # Handle negative indexing
        if idx < 0:
            idx = len(self.image_paths) + idx
            
        if idx < 0 or idx >= len(self.image_paths):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self)}")
        
        # Get image tensor (with caching if enabled)
        image_tensor = self._get_image_tensor(idx)
        
        # Get target
        target = self._get_target(idx)
        
        # Apply additional transforms if provided
        if self.transform is not None:
            image_tensor = self.transform(image_tensor)
        
        if self.target_transform is not None and target is not None:
            target = self.target_transform(target)
        
        return image_tensor, target
    
    def _get_image_tensor(self, idx: int) -> torch.Tensor:
        """Get preprocessed image tensor for given index with caching support.
        
        Args:
            idx: Index of image
            
        Returns:
            Preprocessed image tensor
        """
        # Check cache first
        if self.cache_preprocessed and idx in self._cache:
            # Move to end of LRU order
            self._cache_order.remove(idx)
            self._cache_order.append(idx)
            return self._cache[idx].clone()  # Clone to avoid accidental modification
        
        # Load and preprocess image
        image_path = self.image_paths[idx]
        
        try:
            # Load image
            image = Image.open(image_path)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Apply augmentation if provided (only during training)
            if self.augmentator is not None:
                image = self.augmentator.augment(image)
            
            # Preprocess image (don't add batch dim here, DataLoader will handle batching)
            image_tensor = self.preprocessor.preprocess(
                image, 
                add_batch_dim=False
            )
            
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {e}")
            # Return a black image tensor of the correct size
            image_tensor = torch.zeros(
                (3, self.preprocessor.target_size[0], self.preprocessor.target_size[1]),
                device=self.preprocessor.device
            )
        
        # Cache if enabled
        if self.cache_preprocessed:
            self._add_to_cache(idx, image_tensor.clone())
        
        return image_tensor
    
    def _get_target(self, idx: int) -> Any:
        """Get target for given index.
        
        Args:
            idx: Index of item
            
        Returns:
            Target value or None
        """
        if self.targets is None:
            return None
        
        if isinstance(self.targets, list):
            return self.targets[idx]
        
        elif isinstance(self.targets, dict):
            image_path = self.image_paths[idx]
            
            # Try various key formats
            key_candidates = [
                str(image_path),
                image_path.name,
                image_path.stem,
                str(image_path.relative_to(image_path.parent.parent))
            ]
            
            for key in key_candidates:
                if key in self.targets:
                    return self.targets[key]
            
            # No target found
            return None
        
        return None
    
    def _add_to_cache(self, idx: int, tensor: torch.Tensor):
        """Add tensor to cache with LRU eviction if needed.
        
        Args:
            idx: Index to cache
            tensor: Tensor to cache
        """
        # Remove if already cached (update case)
        if idx in self._cache:
            self._cache_order.remove(idx)
        
        # Add to cache
        self._cache[idx] = tensor
        self._cache_order.append(idx)
        
        # Evict if over max size
        if (self.max_cache_size is not None and 
            len(self._cache) > self.max_cache_size):
            oldest_idx = self._cache_order.pop(0)
            del self._cache[oldest_idx]
    
    def clear_cache(self):
        """Clear the preprocessed image cache."""
        self._cache.clear()
        self._cache_order.clear()
        logger.info("Cleared dataset cache")
    
    def get_cache_info(self) -> Dict[str, Any]:
        """Get information about the current cache state.
        
        Returns:
            Dictionary with cache statistics
        """
        return {
            'cache_enabled': self.cache_preprocessed,
            'cached_items': len(self._cache),
            'max_cache_size': self.max_cache_size,
            'hit_ratio': len(self._cache) / len(self.image_paths) if self.image_paths else 0
        }
    
    def get_image_path(self, idx: int) -> Path:
        """Get image path for given index.
        
        Args:
            idx: Index of image
            
        Returns:
            Path to image file
        """
        return self.image_paths[idx]
    
    def get_dataset_info(self) -> Dict[str, Any]:
        """Get comprehensive information about the dataset.
        
        Returns:
            Dictionary with dataset statistics
        """
        return {
            'num_images': len(self.image_paths),
            'has_targets': self.targets is not None,
            'target_type': type(self.targets).__name__ if self.targets else None,
            'preprocessing_target_size': self.preprocessor.target_size,
            'augmentation_enabled': self.augmentator is not None,
            'cache_info': self.get_cache_info(),
            'device': str(self.preprocessor.device)
        }


def collate_fn_with_none(batch):
    """Custom collate function that handles None targets properly."""
    images = []
    targets = []
    
    for image, target in batch:
        images.append(image)
        targets.append(target)
    
    # Stack images into batch tensor
    batch_images = torch.stack(images, dim=0)
    
    # Handle targets - if all None, keep as list of None
    # If mixed or all non-None, convert appropriately
    if all(t is None for t in targets):
        batch_targets = targets  # Keep as list of None
    else:
        # Convert to tensor if all are numbers, otherwise keep as list
        try:
            if all(isinstance(t, (int, float)) or t is None for t in targets):
                # Convert None to -1 or handle appropriately
                processed_targets = []
                for t in targets:
                    if t is None:
                        processed_targets.append(-1)  # Use -1 for None targets
                    else:
                        processed_targets.append(t)
                batch_targets = torch.tensor(processed_targets)
            else:
                batch_targets = targets  # Keep as list for complex targets
        except:
            batch_targets = targets  # Fallback to list
    
    return batch_images, batch_targets


def create_adversarial_dataloader(
    dataset: AdversarialDataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = None,
    drop_last: bool = False,
    **kwargs
) -> DataLoader:
    """Create DataLoader for adversarial training dataset.
    
    Args:
        dataset: AdversarialDataset instance
        batch_size: Batch size for training
        shuffle: Whether to shuffle data
        num_workers: Number of multiprocessing workers
        pin_memory: Pin memory for faster GPU transfer (auto-detect if None)
        drop_last: Drop last incomplete batch
        **kwargs: Additional DataLoader arguments
        
    Returns:
        Configured PyTorch DataLoader
    """
    # Auto-detect pin_memory based on device
    if pin_memory is None:
        pin_memory = str(dataset.preprocessor.device) != 'cpu'
    
    # Set reasonable defaults for num_workers if not specified
    if num_workers == 0 and len(dataset) > batch_size * 4:
        # Use multiple workers for larger datasets
        num_workers = min(4, os.cpu_count() or 1)
        logger.info(f"Auto-setting num_workers to {num_workers} for dataset of size {len(dataset)}")
    
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        collate_fn=collate_fn_with_none,
        **kwargs
    )
    
    logger.info(
        f"Created DataLoader: batch_size={batch_size}, shuffle={shuffle}, "
        f"num_workers={num_workers}, pin_memory={pin_memory}"
    )
    
    return dataloader


def create_training_dataloader(
    data_sources: Union[List[str], str, List[Path], Path],
    targets: Optional[Union[List[Any], Dict[str, Any]]] = None,
    batch_size: int = 32,
    target_size: Tuple[int, int] = (224, 224),
    augmentation: str = 'weak',  # 'none', 'weak', 'strong'
    num_workers: int = 0,
    cache_preprocessed: bool = False,
    device: Optional[torch.device] = None,
    shuffle: bool = True,
    validate_images: bool = True,
    **dataloader_kwargs
) -> DataLoader:
    """Create a complete training DataLoader with preprocessing and augmentation.
    
    Args:
        data_sources: Image files/directories
        targets: Optional targets for supervised learning
        batch_size: Training batch size
        target_size: Image preprocessing target size
        augmentation: Augmentation level ('none', 'weak', 'strong')
        num_workers: Number of DataLoader workers
        cache_preprocessed: Enable image caching
        device: Target device for tensors
        shuffle: Shuffle training data
        **dataloader_kwargs: Additional DataLoader arguments
        
    Returns:
        Ready-to-use training DataLoader
    """
    # Create preprocessor
    preprocessor = ImagePreprocessor(
        target_size=target_size,
        normalize=True,
        device=device
    )
    
    # Create augmentator if requested
    augmentator = None
    if augmentation != 'none':
        strong_aug = (augmentation == 'strong')
        augmentator = create_data_augmentation_pipeline(strong_augmentation=strong_aug)
        logger.info(f"Created {augmentation} augmentation pipeline")
    
    # Create dataset
    dataset = AdversarialDataset(
        data_sources=data_sources,
        targets=targets,
        preprocessor=preprocessor,
        augmentator=augmentator,
        cache_preprocessed=cache_preprocessed,
        validate_images=validate_images
    )
    
    # Create dataloader
    return create_adversarial_dataloader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        **dataloader_kwargs
    )


def create_validation_dataloader(
    data_sources: Union[List[str], str, List[Path], Path],
    targets: Optional[Union[List[Any], Dict[str, Any]]] = None,
    batch_size: int = 32,
    target_size: Tuple[int, int] = (224, 224),
    num_workers: int = 0,
    device: Optional[torch.device] = None,
    **dataloader_kwargs
) -> DataLoader:
    """Create a validation DataLoader without augmentation.
    
    Args:
        data_sources: Image files/directories
        targets: Optional targets for validation
        batch_size: Validation batch size
        target_size: Image preprocessing target size
        num_workers: Number of DataLoader workers
        device: Target device for tensors
        **dataloader_kwargs: Additional DataLoader arguments
        
    Returns:
        Ready-to-use validation DataLoader
    """
    # Create preprocessor (high quality for validation)
    preprocessor = ImagePreprocessor(
        target_size=target_size,
        normalize=True,
        device=device
    )
    
    # Create dataset (no augmentation, cache enabled for faster validation)
    dataset = AdversarialDataset(
        data_sources=data_sources,
        targets=targets,
        preprocessor=preprocessor,
        augmentator=None,  # No augmentation for validation
        cache_preprocessed=True  # Cache for faster repeated validation
    )
    
    # Create dataloader (no shuffling for validation)
    return create_adversarial_dataloader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        **dataloader_kwargs
    )


def create_single_image_dataloader(
    image_path: Union[str, Path],
    target_size: Tuple[int, int] = (224, 224),
    device: Optional[torch.device] = None
) -> DataLoader:
    """Create DataLoader for single image processing.
    
    Args:
        image_path: Path to single image
        target_size: Image preprocessing target size
        device: Target device for tensors
        
    Returns:
        DataLoader with single image (batch size 1)
    """
    # Create preprocessor
    preprocessor = ImagePreprocessor(
        target_size=target_size,
        normalize=True,
        device=device
    )
    
    # Create dataset with single image
    dataset = AdversarialDataset(
        data_sources=[image_path],
        preprocessor=preprocessor,
        cache_preprocessed=True  # Cache single image
    )
    
    # Create dataloader with batch size 1, no shuffling, no workers
    return create_adversarial_dataloader(
        dataset=dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        drop_last=False
    )


def get_dataset_statistics(dataloader: DataLoader) -> Dict[str, Any]:
    """Analyze dataset and return comprehensive statistics.
    
    Args:
        dataloader: DataLoader to analyze
        
    Returns:
        Dictionary with dataset statistics
    """
    dataset = dataloader.dataset
    
    if not isinstance(dataset, AdversarialDataset):
        raise ValueError("DataLoader must use AdversarialDataset")
    
    stats = dataset.get_dataset_info()
    
    # Add DataLoader-specific info
    # Determine shuffle status from sampler type
    shuffle_status = isinstance(dataloader.sampler, torch.utils.data.RandomSampler)
    
    stats.update({
        'batch_size': dataloader.batch_size,
        'num_workers': dataloader.num_workers,
        'pin_memory': dataloader.pin_memory,
        'shuffle': shuffle_status,
        'drop_last': dataloader.drop_last,
        'num_batches': len(dataloader)
    })
    
    return stats