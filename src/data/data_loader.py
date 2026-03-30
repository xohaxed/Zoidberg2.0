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
from tqdm import tqdm
import yaml


class DatasetLoader:
    """
    Handles loading and management of pneumonia X-ray datasets.
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
        Get all image paths and their corresponding labels.

        Args:
            dataset_name (str): dataset1 / dataset2 / dataset3_tuning
            multiclass (bool):
                True  -> NORMAL / BACTERIA / VIRUS
                False -> NORMAL / PNEUMONIA

        Returns:
            (image_paths, labels)
        """

        dataset_path = self.dataset_paths.get(dataset_name)

        if not dataset_path or not dataset_path.exists():
            raise ValueError(f"Dataset {dataset_name} not found at {dataset_path}")

        image_paths = []
        labels = []

        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}

        class_dirs = [d for d in dataset_path.iterdir() if d.is_dir()]

        for class_dir in class_dirs:
            class_name = class_dir.name.upper()

            images = [
                f for f in class_dir.rglob('*')
                if f.is_file() and f.suffix.lower() in image_extensions
            ]

            for img_path in images:
                filename = img_path.name.upper()

                # MULTICLASS MODE
                if multiclass:
                    if 'NORMAL' in class_name:
                        label = 'NORMAL'

                    elif 'BACTERIA' in filename or 'BACTERIAL' in filename:
                        label = 'BACTERIA'

                    elif 'VIRUS' in filename or 'VIRAL' in filename:
                        label = 'VIRUS'

                    else:
                        # Ignore ambiguous samples
                        continue

                # BINARY MODE
                else:
                    if 'NORMAL' in class_name:
                        label = 'NORMAL'
                    else:
                        label = 'PNEUMONIA'

                image_paths.append(img_path)
                labels.append(label)

        return image_paths, labels

    def load_image(
        self,
        image_path: Path,
        target_size: Optional[Tuple[int, int]] = None,
        color_mode: str = 'grayscale'
    ) -> np.ndarray:
        """
        Load a single image.

        Args:
            image_path (Path)
            target_size (tuple)
            color_mode (str)

        Returns:
            np.ndarray
        """
        try:
            img = Image.open(image_path)

            if color_mode == 'grayscale':
                img = img.convert('L')
            elif color_mode == 'rgb':
                img = img.convert('RGB')

            if target_size:
                img = img.resize(target_size, Image.LANCZOS)

            return np.array(img)

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
        Load full dataset into memory.
        """

        image_paths, labels = self.get_image_paths_and_labels(dataset_name, multiclass)

        if max_samples and max_samples < len(image_paths):
            indices = np.random.choice(len(image_paths), max_samples, replace=False)
            image_paths = [image_paths[i] for i in indices]
            labels = [labels[i] for i in indices]

        images = []
        valid_labels = []

        print(f"Loading {len(image_paths)} images from {dataset_name}...")

        for img_path, label in tqdm(zip(image_paths, labels), total=len(image_paths)):
            try:
                img = self.load_image(img_path, target_size, color_mode)
                images.append(img)
                valid_labels.append(label)
            except Exception as e:
                print(f"Skipping {img_path.name}: {e}")

        images = np.array(images)

        if color_mode == 'grayscale' and len(images.shape) == 3:
            images = np.expand_dims(images, axis=-1)

        print(f"✓ Loaded {len(images)} images with shape {images.shape}")

        return images, np.array(valid_labels), sorted(list(set(valid_labels)))

    def get_dataset_statistics(self, dataset_name: str) -> Dict:
        """
        Get dataset statistics.
        """

        image_paths, labels = self.get_image_paths_and_labels(dataset_name)

        stats = {
            'total_images': len(image_paths),
            'class_distribution': pd.Series(labels).value_counts().to_dict(),
            'class_balance_ratio': None
        }

        counts = list(stats['class_distribution'].values())

        if len(counts) > 1:
            stats['class_balance_ratio'] = max(counts) / min(counts)

        return stats


def load_configuration(config_path: str = None) -> Dict:
    """
    Load YAML config.
    """

    if config_path is None:
        project_root = Path(__file__).parent.parent
        config_path = project_root / 'config' / 'config.yaml'

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    return config