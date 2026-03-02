"""
ZOIDBERG2.0 - Data Validation Utilities
Medical Imaging / Computer Aided Diagnosis

This module provides validation utilities for data integrity checks.
"""

from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
from PIL import Image
from tqdm import tqdm


class DataValidator:
    """
    Validates data integrity and quality for medical imaging datasets.
    """
    
    def __init__(self):
        self.supported_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.dcm'}
    
    def validate_image(self, image_path: Path) -> Dict:
        """
        Validate a single image file.
        
        Args:
            image_path (Path): Path to image file
        
        Returns:
            Dict: Validation results
        """
        result = {
            'path': str(image_path),
            'valid': False,
            'readable': False,
            'dimensions': None,
            'mode': None,
            'error': None
        }
        
        try:
            # Check file exists
            if not image_path.exists():
                result['error'] = "File not found"
                return result
            
            # Check file extension
            if image_path.suffix.lower() not in self.supported_extensions:
                result['error'] = f"Unsupported format: {image_path.suffix}"
                return result
            
            # Try to open and read image
            with Image.open(image_path) as img:
                result['readable'] = True
                result['dimensions'] = img.size
                result['mode'] = img.mode
                
                # Verify image data can be loaded
                _ = np.array(img)
                
                result['valid'] = True
        
        except Exception as e:
            result['error'] = str(e)
        
        return result
    
    def validate_dataset(
        self, 
        dataset_path: Path,
        sample_size: Optional[int] = None
    ) -> Dict:
        """
        Validate an entire dataset.
        
        Args:
            dataset_path (Path): Path to dataset directory
            sample_size (int, optional): Number of images to sample for validation
        
        Returns:
            Dict: Validation report
        """
        report = {
            'dataset_path': str(dataset_path),
            'total_images': 0,
            'sampled_images': 0,
            'valid_images': 0,
            'corrupted_images': 0,
            'errors': [],
            'dimensions': [],
            'color_modes': {}
        }
        
        if not dataset_path.exists():
            report['errors'].append(f"Dataset path does not exist: {dataset_path}")
            return report
        
        # Collect all image paths
        all_images = [f for f in dataset_path.rglob('*') 
                     if f.is_file() and f.suffix.lower() in self.supported_extensions]
        
        report['total_images'] = len(all_images)
        
        # Sample if needed
        if sample_size and sample_size < len(all_images):
            sample_images = np.random.choice(all_images, sample_size, replace=False)
        else:
            sample_images = all_images
        
        report['sampled_images'] = len(sample_images)
        
        # Validate each image
        print(f"Validating {len(sample_images)} images...")
        
        for img_path in tqdm(sample_images):
            validation = self.validate_image(img_path)
            
            if validation['valid']:
                report['valid_images'] += 1
                report['dimensions'].append(validation['dimensions'])
                
                mode = validation['mode']
                report['color_modes'][mode] = report['color_modes'].get(mode, 0) + 1
            else:
                report['corrupted_images'] += 1
                report['errors'].append({
                    'file': validation['path'],
                    'error': validation['error']
                })
        
        return report
    
    def check_class_balance(self, labels: List[str]) -> Dict:
        """
        Check class balance in labels.
        
        Args:
            labels (List[str]): List of class labels
        
        Returns:
            Dict: Balance statistics
        """
        unique, counts = np.unique(labels, return_counts=True)
        class_counts = dict(zip(unique, counts))
        
        total = len(labels)
        
        balance_stats = {
            'class_counts': class_counts,
            'class_percentages': {cls: (count/total)*100 
                                 for cls, count in class_counts.items()},
            'imbalance_ratio': max(counts) / min(counts) if len(counts) > 1 else 1.0,
            'is_balanced': max(counts) / min(counts) < 2.0 if len(counts) > 1 else True
        }
        
        return balance_stats
    
    def check_image_quality(
        self,
        image: np.ndarray,
        min_resolution: Tuple[int, int] = (64, 64),
        check_contrast: bool = True
    ) -> Dict:
        """
        Check quality metrics for a single image.
        
        Args:
            image (np.ndarray): Image array
            min_resolution (Tuple[int, int]): Minimum acceptable resolution
            check_contrast (bool): Whether to check contrast
        
        Returns:
            Dict: Quality metrics
        """
        metrics = {
            'resolution': image.shape[:2],
            'meets_min_resolution': all(d >= m for d, m in zip(image.shape[:2], min_resolution)),
            'mean_intensity': float(np.mean(image)),
            'std_intensity': float(np.std(image)),
            'min_intensity': float(np.min(image)),
            'max_intensity': float(np.max(image)),
            'contrast_ratio': None,
            'sufficient_contrast': None
        }
        
        if check_contrast:
            # Simple contrast measure
            contrast = metrics['std_intensity'] / (metrics['mean_intensity'] + 1e-7)
            metrics['contrast_ratio'] = float(contrast)
            metrics['sufficient_contrast'] = contrast > 0.1  # Threshold can be adjusted
        
        return metrics
