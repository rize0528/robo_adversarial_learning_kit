"""Integration tests for complete data pipeline functionality."""

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
    create_training_dataloader,
    create_validation_dataloader,
    create_single_image_dataloader,
    get_dataset_statistics
)
from src.data.preprocessing import (
    ImagePreprocessor,
    create_data_augmentation_pipeline,
    preprocess_for_vlm,
    batch_preprocess_images
)


class TestDataPipelineIntegration(unittest.TestCase):
    """End-to-end tests for complete data pipeline integration."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment with comprehensive test data."""
        cls.temp_dir = tempfile.mkdtemp()
        cls.temp_path = Path(cls.temp_dir)
        
        # Create comprehensive test dataset structure
        cls._create_test_dataset_structure()
        
        print(f"Created test dataset at: {cls.temp_path}")
        print(f"Total images created: {len(cls.all_image_paths)}")
    
    @classmethod
    def tearDownClass(cls):
        """Clean up test environment."""
        shutil.rmtree(cls.temp_dir)
    
    @classmethod
    def _create_test_dataset_structure(cls):
        """Create a comprehensive test dataset with various scenarios."""
        cls.all_image_paths = []
        cls.train_paths = []
        cls.val_paths = []
        
        # Create train directory
        train_dir = cls.temp_path / "train"
        train_dir.mkdir()
        
        # Create class subdirectories in train
        for class_idx in range(3):
            class_dir = train_dir / f"class_{class_idx}"
            class_dir.mkdir()
            
            # Create 10 images per class
            for img_idx in range(10):
                img = cls._create_test_image(
                    size=(150 + class_idx * 50, 150 + class_idx * 50),
                    base_color=(class_idx * 80, 100, 150 - class_idx * 30),
                    pattern_offset=img_idx
                )
                
                img_path = class_dir / f"img_{img_idx:03d}.png"
                img.save(img_path)
                
                cls.all_image_paths.append(img_path)
                cls.train_paths.append(img_path)
        
        # Create validation directory
        val_dir = cls.temp_path / "val"
        val_dir.mkdir()
        
        for class_idx in range(3):
            class_dir = val_dir / f"class_{class_idx}"
            class_dir.mkdir()
            
            # Create 5 validation images per class
            for img_idx in range(5):
                img = cls._create_test_image(
                    size=(200, 200),
                    base_color=(class_idx * 60, 150, 200 - class_idx * 40),
                    pattern_offset=img_idx + 100  # Different patterns from training
                )
                
                img_path = class_dir / f"val_{img_idx:03d}.jpg"
                img.save(img_path, 'JPEG')
                
                cls.all_image_paths.append(img_path)
                cls.val_paths.append(img_path)
        
        # Create single test images for various tests
        cls.single_test_dir = cls.temp_path / "single_tests"
        cls.single_test_dir.mkdir()
        
        # Different formats
        formats = [('PNG', '.png'), ('JPEG', '.jpg'), ('BMP', '.bmp')]
        cls.format_test_paths = []
        
        for i, (fmt, ext) in enumerate(formats):
            img = cls._create_test_image(
                size=(100, 100),
                base_color=(i * 100, 50, 200),
                pattern_offset=i
            )
            img_path = cls.single_test_dir / f"format_test_{i}{ext}"
            img.save(img_path, fmt)
            cls.format_test_paths.append(img_path)
        
        # Large images for memory testing
        cls.large_img_paths = []
        for i in range(3):
            img = cls._create_test_image(
                size=(800, 600),
                base_color=(i * 50, 100, 255 - i * 50),
                pattern_offset=i
            )
            img_path = cls.single_test_dir / f"large_{i}.png"
            img.save(img_path)
            cls.large_img_paths.append(img_path)
        
        # Create targets
        cls.train_targets = []
        for class_idx in range(3):
            cls.train_targets.extend([class_idx] * 10)  # 10 images per class
        
        cls.val_targets = []
        for class_idx in range(3):
            cls.val_targets.extend([class_idx] * 5)  # 5 images per class
    
    @classmethod
    def _create_test_image(cls, size: tuple, base_color: tuple, pattern_offset: int = 0):
        """Create a test image with distinctive patterns."""
        img = Image.new('RGB', size, color=base_color)
        img_array = np.array(img)
        
        # Add patterns based on offset
        h, w = size[1], size[0]  # PIL uses (width, height)
        
        # Diagonal stripes
        for i in range(0, h, 20):
            for j in range(0, w, 20):
                if (i + j + pattern_offset) % 40 < 20:
                    img_array[i:min(i+10, h), j:min(j+10, w)] = [255, 255, 255]
        
        # Corner markers
        corner_color = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)][pattern_offset % 4]
        corner_size = min(20, h//4, w//4)
        img_array[:corner_size, :corner_size] = corner_color
        
        return Image.fromarray(img_array)
    
    def test_end_to_end_training_pipeline(self):
        """Test complete training pipeline from data loading to batch processing."""
        print("\nTesting end-to-end training pipeline...")
        
        # Create training dataloader with all features
        train_loader = create_training_dataloader(
            data_sources=str(self.temp_path / "train"),
            targets=self.train_targets,
            batch_size=8,
            target_size=(224, 224),
            augmentation='weak',
            num_workers=0,
            shuffle=True,
            cache_preprocessed=False  # Test without caching
        )
        
        # Verify dataloader properties
        self.assertEqual(len(train_loader.dataset), 30)  # 3 classes * 10 images
        self.assertEqual(train_loader.batch_size, 8)
        # Check shuffle via sampler type
        self.assertIsInstance(train_loader.sampler, torch.utils.data.RandomSampler)
        
        # Process all batches and verify properties
        total_samples = 0
        class_counts = {0: 0, 1: 0, 2: 0}
        
        for batch_idx, (images, targets) in enumerate(train_loader):
            # Verify batch shapes
            self.assertLessEqual(images.shape[0], 8)  # Batch size constraint
            self.assertEqual(images.shape[1:], (3, 224, 224))  # Image dimensions
            self.assertEqual(len(targets), images.shape[0])
            
            # Verify normalized data range (approximately)
            self.assertGreater(images.min(), -4.0)
            self.assertLess(images.max(), 4.0)
            
            # Count samples and classes
            total_samples += images.shape[0]
            for target in targets:
                # Handle both tensor and regular targets
                if isinstance(target, torch.Tensor):
                    target_val = target.item()
                else:
                    target_val = target
                class_counts[target_val] += 1
        
        # Verify all samples processed
        self.assertEqual(total_samples, 30)
        
        # Verify all classes present
        for class_idx in range(3):
            self.assertEqual(class_counts[class_idx], 10)
        
        print(f"Processed {total_samples} training samples in {batch_idx + 1} batches")
    
    def test_end_to_end_validation_pipeline(self):
        """Test validation pipeline with reproducibility."""
        print("\nTesting validation pipeline...")
        
        # Create validation dataloader
        val_loader = create_validation_dataloader(
            data_sources=str(self.temp_path / "val"),
            targets=self.val_targets,
            batch_size=5,
            target_size=(256, 256),
            num_workers=0
        )
        
        # Verify dataloader properties
        self.assertEqual(len(val_loader.dataset), 15)  # 3 classes * 5 images
        self.assertEqual(val_loader.batch_size, 5)
        # Verify no shuffling for validation
        self.assertIsInstance(val_loader.sampler, torch.utils.data.SequentialSampler)
        
        # Test reproducibility - run twice and compare
        results1 = []
        for images, targets in val_loader:
            results1.append((images.clone(), targets))
        
        results2 = []
        for images, targets in val_loader:
            results2.append((images.clone(), targets))
        
        # Results should be identical (no augmentation, no shuffling)
        self.assertEqual(len(results1), len(results2))
        for (img1, tgt1), (img2, tgt2) in zip(results1, results2):
            self.assertTrue(torch.allclose(img1, img2, atol=1e-6))
            # Handle tensor target comparison
            if isinstance(tgt1, torch.Tensor) and isinstance(tgt2, torch.Tensor):
                self.assertTrue(torch.equal(tgt1, tgt2))
            else:
                self.assertEqual(tgt1, tgt2)
        
        print(f"Validated reproducibility across {len(results1)} batches")
    
    def test_preprocessing_integration_consistency(self):
        """Test consistency between standalone preprocessing and dataset preprocessing."""
        print("\nTesting preprocessing consistency...")
        
        test_image_path = self.format_test_paths[0]
        target_size = (128, 128)
        
        # Method 1: Standalone preprocessing
        standalone_tensor = preprocess_for_vlm(
            test_image_path,
            target_size=target_size,
            normalize=True
        )
        
        # Method 2: Through dataset
        dataset = AdversarialDataset(
            data_sources=[test_image_path],
            preprocessor=ImagePreprocessor(target_size=target_size, normalize=True)
        )
        dataset_tensor, _ = dataset[0]
        dataset_tensor = dataset_tensor.unsqueeze(0)  # Add batch dim for comparison
        
        # Method 3: Batch preprocessing
        batch_tensor = batch_preprocess_images(
            [test_image_path],
            target_size=target_size,
            normalize=True
        )
        
        # All methods should produce identical results
        self.assertTrue(torch.allclose(standalone_tensor, dataset_tensor, atol=1e-6))
        self.assertTrue(torch.allclose(standalone_tensor, batch_tensor, atol=1e-6))
        
        print("Preprocessing consistency verified across all methods")
    
    def test_augmentation_integration(self):
        """Test data augmentation integration in training pipeline."""
        print("\nTesting augmentation integration...")
        
        # Create dataset with augmentation
        augmentator = create_data_augmentation_pipeline(strong_augmentation=True)
        dataset = AdversarialDataset(
            data_sources=self.format_test_paths[:2],
            augmentator=augmentator
        )
        
        # Get same image multiple times - should be different due to augmentation
        img1, _ = dataset[0]
        img2, _ = dataset[0]
        img3, _ = dataset[0]
        
        # Images should be different (very high probability with strong augmentation)
        differences = [
            not torch.allclose(img1, img2, atol=1e-3),
            not torch.allclose(img2, img3, atol=1e-3),
            not torch.allclose(img1, img3, atol=1e-3)
        ]
        
        # At least 2 out of 3 should be different (accounting for randomness)
        self.assertGreater(sum(differences), 1)
        
        # All images should still have valid shapes and ranges
        for img in [img1, img2, img3]:
            self.assertEqual(img.shape, (3, 224, 224))
            self.assertGreater(img.min(), -4.0)
            self.assertLess(img.max(), 4.0)
        
        print("Augmentation produces varied outputs with valid ranges")
    
    def test_memory_efficiency_large_images(self):
        """Test memory efficiency with large images."""
        print("\nTesting memory efficiency with large images...")
        
        # Create dataset with large images
        dataset = AdversarialDataset(
            data_sources=self.large_img_paths,
            cache_preprocessed=False,  # Disable caching for memory efficiency
            validate_images=True
        )
        
        dataloader = create_training_dataloader(
            data_sources=self.large_img_paths,
            batch_size=2,
            cache_preprocessed=False,
            num_workers=0,
            augmentation='none'  # No augmentation for cleaner memory testing
        )
        
        # Process batches and verify memory handling
        processed_batches = 0
        for images, _ in dataloader:
            self.assertEqual(images.shape, (2, 3, 224, 224))
            
            # Verify images are properly preprocessed from large originals
            self.assertGreater(images.min(), -4.0)
            self.assertLess(images.max(), 4.0)
            
            processed_batches += 1
        
        # Should process all large images without memory issues
        expected_batches = len(self.large_img_paths) // 2  # 3 images / 2 batch_size = 1 full batch
        self.assertEqual(processed_batches, expected_batches + (1 if len(self.large_img_paths) % 2 else 0))
        
        print(f"Successfully processed {len(self.large_img_paths)} large images in {processed_batches} batches")
    
    def test_mixed_format_handling(self):
        """Test handling of mixed image formats."""
        print("\nTesting mixed format handling...")
        
        # Use images with different formats
        mixed_dataset = AdversarialDataset(
            data_sources=self.format_test_paths,  # PNG, JPG, BMP
            validate_images=True
        )
        
        # All images should load successfully
        self.assertEqual(len(mixed_dataset), len(self.format_test_paths))
        
        # Process all images and verify consistency
        processed_images = []
        for i in range(len(mixed_dataset)):
            img, _ = mixed_dataset[i]
            processed_images.append(img)
            
            # Each image should have consistent shape and range
            self.assertEqual(img.shape, (3, 224, 224))
            self.assertGreater(img.min(), -4.0)
            self.assertLess(img.max(), 4.0)
        
        # Images should be different (different source images)
        for i in range(len(processed_images) - 1):
            self.assertFalse(torch.allclose(processed_images[i], processed_images[i+1], atol=1e-2))
        
        print(f"Successfully processed {len(self.format_test_paths)} images of different formats")
    
    def test_single_image_processing(self):
        """Test single image processing pipeline."""
        print("\nTesting single image processing...")
        
        test_image = self.format_test_paths[0]
        
        # Create single image dataloader
        single_loader = create_single_image_dataloader(
            image_path=test_image,
            target_size=(192, 192)
        )
        
        # Should have exactly one batch with one image
        self.assertEqual(len(single_loader), 1)
        self.assertEqual(single_loader.batch_size, 1)
        
        # Process the single batch
        batch_images, batch_targets = next(iter(single_loader))
        
        # Verify shapes and properties
        self.assertEqual(batch_images.shape, (1, 3, 192, 192))
        self.assertIsNone(batch_targets[0])  # No target for single image
        
        # Verify image is properly processed
        self.assertGreater(batch_images.min(), -4.0)
        self.assertLess(batch_images.max(), 4.0)
        
        print("Single image processing pipeline working correctly")
    
    def test_comprehensive_dataset_statistics(self):
        """Test comprehensive dataset statistics extraction."""
        print("\nTesting dataset statistics extraction...")
        
        # Create dataset with all features enabled
        dataset = AdversarialDataset(
            data_sources=str(self.temp_path / "train"),
            targets=self.train_targets,
            cache_preprocessed=True,
            max_cache_size=10,
            augmentator=create_data_augmentation_pipeline(strong_augmentation=False)
        )
        
        dataloader = create_training_dataloader(
            data_sources=str(self.temp_path / "train"),
            targets=self.train_targets,
            batch_size=6,
            augmentation='weak',
            num_workers=0,
            shuffle=True,
            cache_preprocessed=True
        )
        
        # Get comprehensive statistics
        stats = get_dataset_statistics(dataloader)
        
        # Verify dataset statistics
        expected_stats = {
            'num_images': 30,
            'has_targets': True,
            'target_type': 'list',
            'preprocessing_target_size': (224, 224),
            'augmentation_enabled': True,
            'batch_size': 6,
            'shuffle': True,
            'num_workers': 0,
            'num_batches': 5  # 30 images / 6 batch_size = 5 batches
        }
        
        for key, expected_value in expected_stats.items():
            self.assertEqual(stats[key], expected_value, f"Mismatch for {key}")
        
        # Verify cache info structure
        cache_info = stats['cache_info']
        self.assertIn('cache_enabled', cache_info)
        self.assertIn('cached_items', cache_info)
        self.assertIn('max_cache_size', cache_info)
        self.assertIn('hit_ratio', cache_info)
        
        print("Dataset statistics extraction working correctly")
        print(f"Key statistics: {stats['num_images']} images, {stats['num_batches']} batches")
    
    def test_error_handling_resilience(self):
        """Test pipeline resilience to various error conditions."""
        print("\nTesting error handling resilience...")
        
        # Create mixed valid/invalid dataset
        mixed_paths = (
            self.format_test_paths[:2] +  # Valid images
            [Path(self.temp_dir) / "nonexistent.jpg"] +  # Non-existent file
            [Path(self.temp_dir) / "corrupted.png"]  # We'll create an invalid file
        )
        
        # Create invalid file
        invalid_file = Path(self.temp_dir) / "corrupted.png"
        invalid_file.write_text("This is not a valid image file")
        
        # Create dataset with validation disabled (to include invalid files)
        resilient_dataset = AdversarialDataset(
            data_sources=mixed_paths,
            validate_images=False  # Include invalid files for error testing
        )
        
        # Should include all paths
        self.assertEqual(len(resilient_dataset), 4)
        
        # Create dataloader
        resilient_loader = create_training_dataloader(
            data_sources=mixed_paths,
            batch_size=4,
            num_workers=0,
            augmentation='none',
            validate_images=False
        )
        
        # Process batch - should handle errors gracefully
        batch_images, batch_targets = next(iter(resilient_loader))
        
        # Should still create a batch of correct size
        self.assertEqual(batch_images.shape, (4, 3, 224, 224))
        
        # Valid images should have normal range, invalid ones should be zeros
        valid_image_count = 0
        zero_image_count = 0
        
        for i in range(4):
            img = batch_images[i]
            if torch.equal(img, torch.zeros_like(img)):
                zero_image_count += 1
            else:
                valid_image_count += 1
                # Valid images should have reasonable range
                self.assertGreater(img.abs().sum(), 10.0)
        
        # Should have 2 valid images and 2 zero (error) images
        self.assertEqual(valid_image_count, 2)
        self.assertEqual(zero_image_count, 2)
        
        print(f"Pipeline handled {zero_image_count} invalid images gracefully")
    
    def test_performance_characteristics(self):
        """Test performance characteristics of the data pipeline."""
        print("\nTesting performance characteristics...")
        
        import time
        
        # Test 1: Caching performance
        start_time = time.time()
        
        cached_dataset = AdversarialDataset(
            data_sources=self.format_test_paths * 5,  # Repeat paths for cache testing
            cache_preprocessed=True
        )
        
        # First pass - populate cache
        for i in range(len(cached_dataset)):
            _ = cached_dataset[i]
        
        first_pass_time = time.time() - start_time
        
        # Second pass - should be faster due to caching
        start_time = time.time()
        for i in range(len(cached_dataset)):
            _ = cached_dataset[i]
        
        second_pass_time = time.time() - start_time
        
        # Cache should provide speedup (though this might be system-dependent)
        print(f"First pass: {first_pass_time:.3f}s, Second pass: {second_pass_time:.3f}s")
        
        # Test 2: Memory efficiency comparison
        # (This is more of a smoke test - actual memory testing would require more infrastructure)
        
        # Large dataset without caching
        large_dataset = AdversarialDataset(
            data_sources=self.all_image_paths,
            cache_preprocessed=False
        )
        
        # Sample processing without memory accumulation
        sample_indices = [0, 10, 20, 30, 40] if len(self.all_image_paths) > 40 else [0, 1, 2]
        
        start_time = time.time()
        for idx in sample_indices:
            _ = large_dataset[idx]
        
        processing_time = time.time() - start_time
        
        print(f"Processed {len(sample_indices)} samples in {processing_time:.3f}s")
        print("Performance testing completed successfully")


class TestDataPipelineEdgeCases(unittest.TestCase):
    """Test edge cases and boundary conditions in the data pipeline."""
    
    def setUp(self):
        """Set up minimal test environment for edge case testing."""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)
    
    def test_empty_dataset_handling(self):
        """Test handling of empty datasets."""
        # Create empty directory
        empty_dir = self.temp_path / "empty"
        empty_dir.mkdir()
        
        # Should raise error for empty dataset
        with self.assertRaises(ValueError):
            AdversarialDataset(data_sources=str(empty_dir))
    
    def test_single_image_dataset(self):
        """Test dataset with single image."""
        # Create single image
        img = Image.new('RGB', (100, 100), color=(128, 128, 128))
        img_path = self.temp_path / "single.png"
        img.save(img_path)
        
        # Create dataset
        dataset = AdversarialDataset(data_sources=[img_path])
        
        self.assertEqual(len(dataset), 1)
        
        # Test retrieval
        image_tensor, target = dataset[0]
        self.assertEqual(image_tensor.shape, (3, 224, 224))
        self.assertIsNone(target)
    
    def test_very_small_images(self):
        """Test preprocessing of very small images."""
        # Create tiny image
        tiny_img = Image.new('RGB', (10, 10), color=(255, 0, 0))
        img_path = self.temp_path / "tiny.png"
        tiny_img.save(img_path)
        
        dataset = AdversarialDataset(
            data_sources=[img_path],
            preprocessor=ImagePreprocessor(target_size=(224, 224))
        )
        
        # Should upscale correctly
        image_tensor, _ = dataset[0]
        self.assertEqual(image_tensor.shape, (3, 224, 224))
    
    def test_very_large_target_size(self):
        """Test preprocessing with very large target size."""
        # Create normal image
        img = Image.new('RGB', (100, 100), color=(0, 255, 0))
        img_path = self.temp_path / "normal.png"
        img.save(img_path)
        
        # Use large target size
        dataset = AdversarialDataset(
            data_sources=[img_path],
            preprocessor=ImagePreprocessor(target_size=(1024, 1024))
        )
        
        # Should handle large sizes
        image_tensor, _ = dataset[0]
        self.assertEqual(image_tensor.shape, (3, 1024, 1024))
    
    def test_batch_size_edge_cases(self):
        """Test dataloader with edge case batch sizes."""
        # Create test images
        test_images = []
        for i in range(7):  # Odd number for batch size testing
            img = Image.new('RGB', (50, 50), color=(i*30, 100, 200))
            img_path = self.temp_path / f"test_{i}.png"
            img.save(img_path)
            test_images.append(img_path)
        
        dataset = AdversarialDataset(data_sources=test_images)
        
        # Test batch size 1
        loader_1 = create_training_dataloader(
            data_sources=test_images,
            batch_size=1,
            num_workers=0,
            shuffle=False
        )
        
        batches_1 = list(loader_1)
        self.assertEqual(len(batches_1), 7)  # 7 batches of size 1
        
        # Test batch size larger than dataset
        loader_large = create_training_dataloader(
            data_sources=test_images,
            batch_size=10,  # Larger than 7 images
            num_workers=0,
            shuffle=False,
            drop_last=False
        )
        
        batches_large = list(loader_large)
        self.assertEqual(len(batches_large), 1)  # 1 batch with all images
        self.assertEqual(batches_large[0][0].shape[0], 7)  # Batch contains all 7 images


if __name__ == '__main__':
    # Set up logging for debugging
    import logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
    
    # Run all tests with high verbosity
    unittest.main(verbosity=2)