"""
ZOIDBERG2.0 - Utility Functions
Medical Imaging / Computer Aided Diagnosis

General utility functions for the project.
"""

import os
import random
import numpy as np
import joblib
from pathlib import Path
from typing import Any, Dict
import json
import yaml


def set_seed(seed: int = 42):
    """
    Set random seeds for reproducibility.
    
    Args:
        seed (int): Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # Set seeds for deep learning frameworks if available
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass
    
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
    
    print(f"✓ Random seed set to {seed}")


def save_model(model: Any, filepath: Path, metadata: Dict = None):
    """
    Save a trained model to disk.
    
    Args:
        model: Model object to save
        filepath (Path): Path to save the model
        metadata (Dict, optional): Additional metadata to save
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    # Save model
    joblib.dump(model, filepath)
    
    # Save metadata if provided
    if metadata:
        metadata_path = filepath.with_suffix('.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)
    
    print(f"✓ Model saved to {filepath}")


def load_model(filepath: Path) -> Any:
    """
    Load a trained model from disk.
    
    Args:
        filepath (Path): Path to the saved model
    
    Returns:
        Any: Loaded model object
    """
    if not Path(filepath).exists():
        raise FileNotFoundError(f"Model file not found: {filepath}")
    
    model = joblib.load(filepath)
    print(f"✓ Model loaded from {filepath}")
    
    # Try to load metadata
    metadata_path = Path(filepath).with_suffix('.json')
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        print(f"✓ Metadata loaded")
        return model, metadata
    
    return model


def load_config(config_path: str = None) -> Dict:
    """
    Load configuration from YAML file.
    
    Args:
        config_path (str, optional): Path to config file
    
    Returns:
        Dict: Configuration dictionary
    """
    if config_path is None:
        project_root = Path(__file__).parent.parent.parent
        config_path = project_root / 'config' / 'config.yaml'
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def ensure_dir(directory: Path):
    """
    Ensure a directory exists, create if it doesn't.
    
    Args:
        directory (Path): Directory path
    """
    Path(directory).mkdir(parents=True, exist_ok=True)


def get_project_root() -> Path:
    """
    Get the project root directory.
    
    Returns:
        Path: Project root path
    """
    # Assume this file is in src/utils/
    return Path(__file__).parent.parent.parent


class Logger:
    """
    Simple logger for tracking pipeline progress.
    """
    
    def __init__(self, log_file: Path = None):
        """
        Initialize logger.
        
        Args:
            log_file (Path, optional): Path to log file
        """
        self.log_file = log_file
        if self.log_file:
            ensure_dir(self.log_file.parent)
    
    def log(self, message: str, level: str = "INFO"):
        """
        Log a message.
        
        Args:
            message (str): Message to log
            level (str): Log level (INFO, WARNING, ERROR)
        """
        log_message = f"[{level}] {message}"
        print(log_message)
        
        if self.log_file:
            with open(self.log_file, 'a') as f:
                f.write(log_message + '\n')
    
    def info(self, message: str):
        """Log info message."""
        self.log(message, "INFO")
    
    def warning(self, message: str):
        """Log warning message."""
        self.log(message, "WARNING")
    
    def error(self, message: str):
        """Log error message."""
        self.log(message, "ERROR")
