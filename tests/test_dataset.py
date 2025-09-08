"""Comprehensive tests for AdversarialDataset and DataLoader functionality."""

import os
import tempfile
import shutil
from pathlib import Path
from typing import List, Dict, Any
import unittest

import torch
import torch.utils.data
import numpy as np
from PIL import Image

from src.data.dataset import (
    AdversarialDataset,
    create_adversarial_dataloader,
    create_training_dataloader,
    create_validation_dataloader,
    create_single_image_dataloader,
    get_dataset_statistics
)
from src.data.preprocessing import ImagePreprocessor, create_data_augmentation_pipeline


class TestAdversarialDataset(unittest.TestCase):
    """Test cases for AdversarialDataset class."""
    
    def setUp(self):
        """Set up test environment with temporary directory and test images."""
        # Create temporary directory
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
        
        # Create test images
        self.test_images = []
        self.image_paths = []
        
        # Create various test images
        for i in range(5):
            # Create RGB image
            img = Image.new('RGB', (100, 100), color=(i*50, 100, 150))
            
            # Add some pattern to make images distinct
            img_array = np.array(img)
            img_array[i*10:(i+1)*10, i*10:(i+1)*10] = [255, 0, 0]  # Red square
            img = Image.fromarray(img_array)
            
            # Save image
            img_path = self.temp_path / f"test_image_{i}.png"
            img.save(img_path)
            
            self.test_images.append(img)
            self.image_paths.append(img_path)
        
        # Create subdirectory with more images
        self.subdir = self.temp_path / "subdir"
        self.subdir.mkdir()
        
        for i in range(3):
            img = Image.new('RGB', (80, 80), color=(200, i*80, 100))
            img_path = self.subdir / f"sub_image_{i}.jpg"
            img.save(img_path)
            self.image_paths.append(img_path)
        
        # Create targets for supervised learning tests
        self.list_targets = list(range(len(self.image_paths)))
        self.dict_targets = {str(path): i for i, path in enumerate(self.image_paths)}
        
    def tearDown(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.temp_dir)
    
    def test_dataset_initialization_from_file_list(self):
        """Test dataset initialization with list of image files."""
        dataset = AdversarialDataset(data_sources=self.image_paths[:3])
        
        self.assertEqual(len(dataset), 3)
        self.assertEqual(len(dataset.image_paths), 3)
        self.assertIsInstance(dataset.preprocessor, ImagePreprocessor)
        
        # Test dataset info
        info = dataset.get_dataset_info()
        self.assertEqual(info['num_images'], 3)
        self.assertFalse(info['has_targets'])
        self.assertFalse(info['augmentation_enabled'])
    
    def test_dataset_initialization_from_directory(self):
        """Test dataset initialization with directory path."""
        # Test single directory
        dataset = AdversarialDataset(
            data_sources=str(self.temp_path),
            recursive_search=False
        )
        
        # Should find 5 images in root directory (not subdirectory)
        self.assertEqual(len(dataset), 5)
        
        # Test recursive directory search
        dataset_recursive = AdversarialDataset(
            data_sources=str(self.temp_path),
            recursive_search=True
        )
        
        # Should find all 8 images (5 in root + 3 in subdir)
        self.assertEqual(len(dataset_recursive), 8)
    
    def test_dataset_with_list_targets(self):
        """Test dataset with list-based targets."""
        dataset = AdversarialDataset(
            data_sources=self.image_paths[:5],
            targets=self.list_targets[:5]
        )
        
        self.assertEqual(len(dataset), 5)
        self.assertTrue(dataset.get_dataset_info()['has_targets'])
        
        # Test getting items with targets
        for i in range(5):
            image_tensor, target = dataset[i]
            self.assertIsInstance(image_tensor, torch.Tensor)
            self.assertEqual(target, i)
            self.assertEqual(image_tensor.shape, (3, 224, 224))  # Default preprocessing size
    
    def test_dataset_with_dict_targets(self):
        """Test dataset with dictionary-based targets."""
        # Create dict targets using image names as keys
        dict_targets = {path.name: i for i, path in enumerate(self.image_paths[:5])}
        
        dataset = AdversarialDataset(
            data_sources=self.image_paths[:5],
            targets=dict_targets
        )
        
        self.assertEqual(len(dataset), 5)
        
        # Test getting items with dict targets
        for i in range(5):
            image_tensor, target = dataset[i]
            self.assertIsInstance(image_tensor, torch.Tensor)
            self.assertEqual(target, i)
    
    def test_dataset_preprocessing_integration(self):
        """Test integration with ImagePreprocessor."""
        # Custom preprocessor
        preprocessor = ImagePreprocessor(
            target_size=(128, 128),
            normalize=False,
            device=torch.device('cpu')
        )
        
        dataset = AdversarialDataset(
            data_sources=self.image_paths[:3],
            preprocessor=preprocessor
        )
        
        # Test preprocessed output
        image_tensor, _ = dataset[0]
        self.assertEqual(image_tensor.shape, (3, 128, 128))
        
        # Test without normalization (values should be in [0, 1] range)
        self.assertGreaterEqual(image_tensor.min().item(), 0.0)
        self.assertLessEqual(image_tensor.max().item(), 1.0)
    
    def test_dataset_augmentation_integration(self):
        """Test integration with data augmentation."""
        augmentator = create_data_augmentation_pipeline(strong_augmentation=False)
        
        dataset = AdversarialDataset(
            data_sources=self.image_paths[:3],
            augmentator=augmentator
        )
        
        # Get same image multiple times - should be different due to augmentation
        img1, _ = dataset[0]
        img2, _ = dataset[0]
        
        # Images should be different due to random augmentation
        # (This test might occasionally fail due to randomness, but very unlikely)
        self.assertFalse(torch.equal(img1, img2))
    
    def test_dataset_caching(self):
        """Test image caching functionality."""
        dataset = AdversarialDataset(
            data_sources=self.image_paths[:3],
            cache_preprocessed=True,
            max_cache_size=2
        )
        
        # Initially no cache
        cache_info = dataset.get_cache_info()
        self.assertEqual(cache_info['cached_items'], 0)
        self.assertTrue(cache_info['cache_enabled'])
        
        # Load images - should populate cache
        _ = dataset[0]
        _ = dataset[1]
        
        cache_info = dataset.get_cache_info()
        self.assertEqual(cache_info['cached_items'], 2)
        
        # Load third image - should evict first due to max_cache_size=2
        _ = dataset[2]
        
        cache_info = dataset.get_cache_info()
        self.assertEqual(cache_info['cached_items'], 2)  # Still 2 due to LRU eviction
        
        # Clear cache
        dataset.clear_cache()
        cache_info = dataset.get_cache_info()
        self.assertEqual(cache_info['cached_items'], 0)
    
    def test_dataset_error_handling(self):
        """Test dataset error handling for corrupted/missing images."""
        # Create dataset with non-existent image
        invalid_paths = [self.image_paths[0], Path(self.temp_dir) / "nonexistent.jpg"]
        
        # Should skip invalid image during initialization with validation
        dataset = AdversarialDataset(
            data_sources=invalid_paths,
            validate_images=True
        )
        self.assertEqual(len(dataset), 1)  # Only valid image
        
        # Without validation, should include invalid path but handle gracefully during loading
        dataset_no_validation = AdversarialDataset(
            data_sources=invalid_paths,
            validate_images=False
        )
        self.assertEqual(len(dataset_no_validation), 2)
        
        # Should return zero tensor for invalid image
        # Check both images to find which one is the invalid one (paths get sorted)
        found_zero_tensor = False
        for i in range(len(dataset_no_validation)):
            image_tensor, _ = dataset_no_validation[i]
            self.assertEqual(image_tensor.shape, (3, 224, 224))
            if torch.equal(image_tensor, torch.zeros_like(image_tensor)):
                found_zero_tensor = True
                break
        
        self.assertTrue(found_zero_tensor, "Should find at least one zero tensor for invalid image")
    
    def test_dataset_target_validation(self):
        """Test target validation functionality."""
        # Test mismatched list targets
        with self.assertRaises(ValueError):
            AdversarialDataset(
                data_sources=self.image_paths[:3],
                targets=[1, 2]  # Only 2 targets for 3 images
            )
        
        # Test dict targets with missing keys (should work with warning)
        incomplete_dict = {"missing_key": 1}
        dataset = AdversarialDataset(
            data_sources=self.image_paths[:2],
            targets=incomplete_dict
        )
        
        # Should return None for missing targets
        _, target = dataset[0]
        self.assertIsNone(target)
    
    def test_dataset_get_image_path(self):
        """Test getting image paths by index."""
        dataset = AdversarialDataset(data_sources=self.image_paths[:3])
        
        for i in range(3):
            path = dataset.get_image_path(i)
            self.assertEqual(path, self.image_paths[i])
    
    def test_dataset_indexing_bounds(self):
        """Test dataset indexing boundary conditions."""
        dataset = AdversarialDataset(data_sources=self.image_paths[:3])
        
        # Valid indices
        for i in range(3):
            image_tensor, _ = dataset[i]
            self.assertIsInstance(image_tensor, torch.Tensor)
        
        # Invalid indices
        with self.assertRaises(IndexError):
            _ = dataset[3]
        
        # Test negative indexing (should work now)
        image_tensor, _ = dataset[-1]  # Should get last image
        self.assertIsInstance(image_tensor, torch.Tensor)
        
        # Test out of bounds negative index
        with self.assertRaises(IndexError):
            _ = dataset[-4]  # Too negative for 3 images


class TestDataLoaderCreation(unittest.TestCase):
    """Test cases for DataLoader creation functions."""
    
    def setUp(self):
        """Set up test environment."""
        # Create temporary directory and test images
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
        
        # Create test images
        self.image_paths = []
        for i in range(10):
            img = Image.new('RGB', (100, 100), color=(i*25, 100, 150))
            img_path = self.temp_path / f"test_{i:02d}.png"
            img.save(img_path)
            self.image_paths.append(img_path)
    
    def tearDown(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.temp_dir)
    
    def test_create_adversarial_dataloader(self):
        """Test basic adversarial dataloader creation."""
        dataset = AdversarialDataset(data_sources=self.image_paths[:5])
        dataloader = create_adversarial_dataloader(
            dataset=dataset,
            batch_size=2,
            shuffle=True,
            num_workers=0
        )
        
        self.assertEqual(dataloader.batch_size, 2)
        # DataLoader shuffle is determined by sampler type
        self.assertIsInstance(dataloader.sampler, torch.utils.data.RandomSampler)
        self.assertEqual(len(dataloader), 3)  # 5 images / 2 batch_size = 3 batches (rounded up)
        
        # Test iterating through dataloader
        total_samples = 0
        for batch_images, batch_targets in dataloader:
            self.assertIsInstance(batch_images, torch.Tensor)
            self.assertEqual(len(batch_images.shape), 4)  # (batch, channels, height, width)
            self.assertEqual(batch_images.shape[1:], (3, 224, 224))  # Default preprocessing
            total_samples += batch_images.shape[0]
        
        self.assertEqual(total_samples, 5)  # All images processed
    
    def test_create_training_dataloader(self):
        """Test training dataloader creation with augmentation."""
        # Test with different augmentation levels
        for augmentation in ['none', 'weak', 'strong']:
            dataloader = create_training_dataloader(
                data_sources=str(self.temp_path),
                batch_size=3,
                target_size=(128, 128),
                augmentation=augmentation,
                num_workers=0,
                shuffle=True
            )
            
            self.assertEqual(dataloader.batch_size, 3)
            # DataLoader shuffle is determined by sampler type
            self.assertIsInstance(dataloader.sampler, torch.utils.data.RandomSampler)
            
            # Test batch processing
            batch_images, batch_targets = next(iter(dataloader))
            self.assertEqual(batch_images.shape, (3, 3, 128, 128))
            self.assertTrue(all(t is None for t in batch_targets))  # No targets provided
    
    def test_create_training_dataloader_with_targets(self):
        """Test training dataloader with targets."""
        targets = list(range(len(self.image_paths)))
        
        dataloader = create_training_dataloader(
            data_sources=self.image_paths,
            targets=targets,
            batch_size=4,
            augmentation='weak',
            num_workers=0
        )
        
        # Test that targets are properly batched
        batch_images, batch_targets = next(iter(dataloader))
        self.assertEqual(len(batch_targets), 4)
        # With our custom collate function, integer targets become tensor
        if isinstance(batch_targets, torch.Tensor):
            self.assertTrue(all(isinstance(t.item(), int) for t in batch_targets))
        else:
            self.assertTrue(all(isinstance(t, int) for t in batch_targets))
    
    def test_create_validation_dataloader(self):
        """Test validation dataloader creation."""
        dataloader = create_validation_dataloader(
            data_sources=self.image_paths[:6],
            batch_size=2,
            target_size=(256, 256),
            num_workers=0
        )
        
        self.assertEqual(dataloader.batch_size, 2)
        # DataLoader with no shuffle should use SequentialSampler
        self.assertIsInstance(dataloader.sampler, torch.utils.data.SequentialSampler)  # Validation shouldn't shuffle
        
        # Test batch processing
        batch_images, _ = next(iter(dataloader))
        self.assertEqual(batch_images.shape, (2, 3, 256, 256))
        
        # Check that caching is enabled for validation dataset
        dataset = dataloader.dataset
        self.assertTrue(dataset.cache_preprocessed)
        self.assertIsNone(dataset.augmentator)  # No augmentation for validation
    
    def test_create_single_image_dataloader(self):
        """Test single image dataloader creation."""
        dataloader = create_single_image_dataloader(
            image_path=self.image_paths[0],
            target_size=(128, 128)
        )
        
        self.assertEqual(dataloader.batch_size, 1)
        # DataLoader with no shuffle should use SequentialSampler
        self.assertIsInstance(dataloader.sampler, torch.utils.data.SequentialSampler)
        self.assertEqual(len(dataloader), 1)
        
        # Test single batch
        batch_images, batch_targets = next(iter(dataloader))
        self.assertEqual(batch_images.shape, (1, 3, 128, 128))
        self.assertTrue(all(t is None for t in batch_targets))
    
    def test_dataloader_auto_num_workers(self):
        """Test automatic num_workers detection."""
        dataset = AdversarialDataset(data_sources=self.image_paths)  # 10 images
        
        # For datasets larger than batch_size * 4, should auto-set num_workers
        dataloader = create_adversarial_dataloader(
            dataset=dataset,
            batch_size=2,  # 10 > 2*4, so should set num_workers > 0
            num_workers=0  # Start with 0, should be auto-adjusted
        )
        
        # The actual num_workers might be auto-set, but we can't easily test this
        # without modifying the function or mocking os.cpu_count()
        self.assertIsInstance(dataloader, torch.utils.data.DataLoader)
    
    def test_get_dataset_statistics(self):
        """Test dataset statistics extraction."""
        targets = list(range(5))
        dataset = AdversarialDataset(
            data_sources=self.image_paths[:5],
            targets=targets,
            cache_preprocessed=True
        )
        
        dataloader = create_adversarial_dataloader(
            dataset=dataset,
            batch_size=2,
            shuffle=True,
            num_workers=0
        )
        
        stats = get_dataset_statistics(dataloader)
        
        # Test dataset statistics
        self.assertEqual(stats['num_images'], 5)
        self.assertTrue(stats['has_targets'])
        self.assertEqual(stats['target_type'], 'list')
        self.assertEqual(stats['batch_size'], 2)
        self.assertTrue(stats['shuffle'])
        self.assertEqual(stats['num_workers'], 0)
        self.assertEqual(stats['num_batches'], 3)  # 5 images / 2 batch_size = 3 batches
        
        # Test cache info
        cache_info = stats['cache_info']
        self.assertTrue(cache_info['cache_enabled'])
        self.assertIsNone(cache_info['max_cache_size'])


class TestDataLoaderIntegration(unittest.TestCase):
    """Integration tests for complete data pipeline."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
        
        # Create more diverse test images
        self.image_paths = []
        
        # Different sizes and formats
        sizes = [(100, 100), (150, 200), (300, 150)]
        formats = ['PNG', 'JPEG']
        
        for i, (size, fmt) in enumerate(zip(sizes, formats)):
            img = Image.new('RGB', size, color=(i*80, 100+i*50, 150-i*30))
            
            # Add some patterns
            img_array = np.array(img)
            if i % 2 == 0:
                img_array[::10, ::10] = [255, 255, 255]  # White dots
            else:
                img_array[size[1]//2:, :] = [0, 0, 255]  # Blue bottom half
            
            img = Image.fromarray(img_array)
            
            ext = '.png' if fmt == 'PNG' else '.jpg'
            img_path = self.temp_path / f"diverse_{i}{ext}"
            img.save(img_path, format=fmt)
            self.image_paths.append(img_path)
        
        # Add more standard images
        for i in range(3, 12):
            img = Image.new('RGB', (224, 224), color=(i*20, 255-i*20, 128))
            img_path = self.temp_path / f"standard_{i}.png"
            img.save(img_path)
            self.image_paths.append(img_path)
    
    def tearDown(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.temp_dir)
    
    def test_full_training_pipeline(self):
        """Test complete training pipeline with real data loading."""
        targets = [i % 3 for i in range(len(self.image_paths))]  # 3 classes
        
        # Create training dataloader
        train_loader = create_training_dataloader(
            data_sources=self.image_paths,
            targets=targets,
            batch_size=4,
            augmentation='weak',
            num_workers=0,
            shuffle=True
        )
        
        # Simulate training loop
        total_batches = 0
        total_samples = 0
        
        for batch_idx, (images, batch_targets) in enumerate(train_loader):
            # Verify batch shapes
            self.assertEqual(len(images.shape), 4)  # (batch, channels, height, width)
            self.assertEqual(images.shape[1:], (3, 224, 224))  # Default size
            self.assertEqual(len(batch_targets), images.shape[0])
            
            # Verify target values
            for target in batch_targets:
                self.assertIn(target, [0, 1, 2])
            
            # Verify image data is properly normalized
            self.assertTrue(images.min() >= -3.0)  # Roughly normalized range
            self.assertTrue(images.max() <= 3.0)
            
            total_batches += 1
            total_samples += images.shape[0]
            
            # Test only first few batches for speed
            if batch_idx >= 2:
                break
        
        self.assertGreater(total_batches, 0)
        self.assertGreater(total_samples, 0)
    
    def test_validation_pipeline_reproducibility(self):
        """Test that validation pipeline is reproducible (no augmentation)."""
        val_loader = create_validation_dataloader(
            data_sources=self.image_paths[:5],
            batch_size=2,
            num_workers=0
        )
        
        # Get first batch twice
        iter1 = iter(val_loader)
        batch1_images, _ = next(iter1)
        
        iter2 = iter(val_loader)
        batch2_images, _ = next(iter2)
        
        # Should be identical (no augmentation, no shuffling)
        self.assertTrue(torch.allclose(batch1_images, batch2_images, atol=1e-6))
    
    def test_mixed_format_loading(self):
        """Test loading images of different formats and sizes."""
        # Use the diverse images created in setUp
        diverse_paths = self.image_paths[:3]  # Different sizes and formats
        
        dataloader = create_training_dataloader(
            data_sources=diverse_paths,
            batch_size=3,
            augmentation='none',  # No augmentation to test pure preprocessing
            num_workers=0
        )
        
        # All images should be preprocessed to same size
        batch_images, _ = next(iter(dataloader))
        self.assertEqual(batch_images.shape, (3, 3, 224, 224))
        
        # All images should have similar value ranges after normalization
        for i in range(3):
            img = batch_images[i]
            self.assertTrue(img.min() >= -3.0)
            self.assertTrue(img.max() <= 3.0)
    
    def test_memory_efficiency_large_batch(self):
        """Test memory efficiency with larger batches."""
        # Create a larger number of images for memory testing
        large_image_paths = self.image_paths * 3  # 36 images total
        
        # Test with caching disabled (memory efficient)
        dataset_no_cache = AdversarialDataset(
            data_sources=large_image_paths,
            cache_preprocessed=False
        )
        
        dataloader_no_cache = create_adversarial_dataloader(
            dataset=dataset_no_cache,
            batch_size=8,
            num_workers=0
        )
        
        # Should be able to iterate through without memory issues
        batch_count = 0
        for images, _ in dataloader_no_cache:
            self.assertEqual(images.shape[0], min(8, len(large_image_paths) - batch_count * 8))
            batch_count += 1
            
            # Test only first few batches
            if batch_count >= 3:
                break
        
        self.assertGreater(batch_count, 0)
    
    def test_error_recovery_in_batch(self):
        """Test that dataloader handles individual image errors gracefully."""
        # Mix valid and invalid paths
        mixed_paths = self.image_paths[:3] + [Path(self.temp_dir) / "nonexistent.jpg"]
        
        # Without validation, should include all paths
        dataset = AdversarialDataset(
            data_sources=mixed_paths,
            validate_images=False
        )
        
        dataloader = create_adversarial_dataloader(dataset, batch_size=4, num_workers=0)
        
        # Should still create batches, with zero tensor for invalid image
        batch_images, _ = next(iter(dataloader))
        self.assertEqual(batch_images.shape, (4, 3, 224, 224))
        
        # Last image should be zeros (invalid image)
        last_image = batch_images[3]
        self.assertTrue(torch.equal(last_image, torch.zeros_like(last_image)))


if __name__ == '__main__':
    # Set up logging for test debugging
    logging.basicConfig(level=logging.INFO)
    
    # Run all tests
    unittest.main(verbosity=2)