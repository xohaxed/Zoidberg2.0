"""
ZOIDBERG2.0 - Data Loading Utilities
Medical Imaging / Computer Aided Diagnosis

This module provides utilities for loading and managing the 3 datasets.
"""

import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
from PIL import Image
import cv2
from tqdm import tqdm
import yaml


class DatasetLoader:
    """
    Handles loading and management of pneumonia X-ray datasets.
    
    Attributes:
        config (dict): Configuration dictionary
        dataset_paths (dict): Paths to the three datasets
    """
    
    def __init__(self, config_path: str):
        """
        Initialize the DatasetLoader.
        
        Args:
            config_path (str): Path to configuration YAML file
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        project_root = Path(config_path).parent.parent
        self.dataset_paths = {
            'dataset1': project_root / self.config['paths']['dataset1'],
            'dataset2': project_root / self.config['paths']['dataset2'],
            'dataset3_tuning': project_root / self.config['paths']['dataset3_tuning']
        }
    
    def get_image_paths_and_labels(
        self, 
        dataset_name: str,
        multiclass: bool = False
    ) -> Tuple[List[Path], List[str]]:
        """
        Get all image paths and their corresponding labels from a dataset.
        
        Args:
            dataset_name (str): Name of dataset ('dataset1', 'dataset2', 'dataset3_tuning')
            multiclass (bool): If True, use multiclass labels (NORMAL, VIRUS, BACTERIA)
                              If False, use binary labels (NORMAL, PNEUMONIA)
        
        Returns:
            Tuple[List[Path], List[str]]: Image paths and corresponding labels
        """
        dataset_path = self.dataset_paths.get(dataset_name)
        
        if not dataset_path or not dataset_path.exists():
            raise ValueError(f"Dataset {dataset_name} not found at {dataset_path}")
        
        image_paths = []
        labels = []
        
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        
        # Get class directories
        class_dirs = [d for d in dataset_path.iterdir() if d.is_dir()]
        
        for class_dir in class_dirs:
            class_name = class_dir.name.upper()
            
            # Get all images in this class
            images = [f for f in class_dir.rglob('*') 
                     if f.is_file() and f.suffix.lower() in image_extensions]
            
            for img_path in images:
                image_paths.append(img_path)
                
                # Assign label based on classification task
                if multiclass:
                    # For 3-class: NORMAL, VIRUS, BACTERIA
                    if 'VIRUS' in class_name or 'VIRAL' in class_name:
                        labels.append('VIRUS')
                    elif 'BACTERIA' in class_name or 'BACTERIAL' in class_name:
                        labels.append('BACTERIA')
                    else:
                        labels.append('NORMAL')
                else:
                    # For binary: NORMAL vs PNEUMONIA
                    if 'NORMAL' in class_name:
                        labels.append('NORMAL')
                    else:
                        labels.append('PNEUMONIA')
        
        return image_paths, labels
    
    def load_image(
        self, 
        image_path: Path, 
        target_size: Optional[Tuple[int, int]] = None,
        color_mode: str = 'grayscale'
    ) -> np.ndarray:
        """
        Load and preprocess a single image.
        
        Args:
            image_path (Path): Path to image file
            target_size (Tuple[int, int], optional): Target size (width, height)
            color_mode (str): 'grayscale' or 'rgb'
        
        Returns:
            np.ndarray: Preprocessed image array
        """
        try:
            # Load image
            img = Image.open(image_path)
            
            # Convert to appropriate color mode
            if color_mode == 'grayscale':
                img = img.convert('L')
            elif color_mode == 'rgb':
                img = img.convert('RGB')
            
            # Resize if target size specified
            if target_size:
                img = img.resize(target_size, Image.LANCZOS)
            
            # Convert to numpy array
            img_array = np.array(img)
            
            return img_array
        
        except Exception as e:
            raise IOError(f"Error loading image {image_path}: {str(e)}")
    
    def load_dataset(
        self,
        dataset_name: str,
        target_size: Optional[Tuple[int, int]] = None,
        color_mode: str = 'grayscale',
        multiclass: bool = False,
        max_samples: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Load an entire dataset into memory.
        
        Args:
            dataset_name (str): Name of dataset
            target_size (Tuple[int, int], optional): Target size for images
            color_mode (str): 'grayscale' or 'rgb'
            multiclass (bool): Use multiclass labels
            max_samples (int, optional): Maximum number of samples to load
        
        Returns:
            Tuple[np.ndarray, np.ndarray, List[str]]: Images, labels, and label names
        """
        # Get image paths and labels
        image_paths, labels = self.get_image_paths_and_labels(dataset_name, multiclass)
        
        # Limit samples if specified
        if max_samples and max_samples < len(image_paths):
            indices = np.random.choice(len(image_paths), max_samples, replace=False)
            image_paths = [image_paths[i] for i in indices]
            labels = [labels[i] for i in indices]
        
        # Load images
        images = []
        valid_labels = []
        
        print(f"Loading {len(image_paths)} images from {dataset_name}...")
        
        for img_path, label in tqdm(zip(image_paths, labels), total=len(image_paths)):
            try:
                img = self.load_image(img_path, target_size, color_mode)
                images.append(img)
                valid_labels.append(label)
            except Exception as e:
                print(f"Warning: Skipping {img_path.name}: {e}")
                continue
        
        # Convert to numpy arrays
        images = np.array(images)
        
        # Expand dims if grayscale for consistency (batch, height, width, channels)
        if color_mode == 'grayscale' and len(images.shape) == 3:
            images = np.expand_dims(images, axis=-1)
        
        print(f"✓ Loaded {len(images)} images with shape {images.shape}")
        
        return images, np.array(valid_labels), sorted(list(set(valid_labels)))
    
    def get_dataset_statistics(self, dataset_name: str) -> Dict:
        """
        Get statistics about a dataset without loading all images.
        
        Args:
            dataset_name (str): Name of dataset
        
        Returns:
            Dict: Statistics dictionary
        """
        image_paths, labels = self.get_image_paths_and_labels(dataset_name)
        
        stats = {
            'total_images': len(image_paths),
            'class_distribution': pd.Series(labels).value_counts().to_dict(),
            'class_balance_ratio': None
        }
        
        # Calculate class balance ratio
        counts = list(stats['class_distribution'].values())
        if len(counts) > 1:
            stats['class_balance_ratio'] = max(counts) / min(counts)
        
        return stats


def load_configuration(config_path: str = None) -> Dict:
    """
    Load project configuration from YAML file.
    
    Args:
        config_path (str, optional): Path to config file
    
    Returns:
        Dict: Configuration dictionary
    """
    if config_path is None:
        # Try to find config in standard location
        project_root = Path(__file__).parent.parent
        config_path = project_root / 'config' / 'config.yaml'
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config
