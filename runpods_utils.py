"""
Utility functions for RunPods operations.
These are common functions used across data gathering and preprocessing modules.
"""

import os
import logging
import json
from typing import Optional, Any
from PIL import Image
import numpy as np

logger = logging.getLogger(__name__)

def ensure_dir_exists(directory: str) -> None:
    """
    Ensure that a directory exists, creating it if necessary.
    
    Args:
        directory: Directory path to create
    """
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
        logger.info(f"Created directory: {directory}")

def load_image(image_path: str) -> Optional[Image.Image]:
    """
    Load an image from a file path.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        PIL Image object or None if loading fails
    """
    try:
        img = Image.open(image_path)
        return img
    except Exception as e:
        logger.error(f"Error loading image {image_path}: {str(e)}")
        return None

def save_image(img: Image.Image, output_path: str) -> bool:
    """
    Save a PIL Image to a file path.
    
    Args:
        img: PIL Image to save
        output_path: Path to save the image to
        
    Returns:
        True if saving was successful, False otherwise
    """
    try:
        # Ensure the directory exists
        output_dir = os.path.dirname(output_path)
        ensure_dir_exists(output_dir)
        
        # Save the image
        img.save(output_path, format="JPEG", quality=95)
        return True
    except Exception as e:
        logger.error(f"Error saving image to {output_path}: {str(e)}")
        return False

def save_json(data: Any, file_path: str, indent: int = 2) -> bool:
    """
    Save data to a JSON file.
    
    Args:
        data: Data to save (must be JSON serializable)
        file_path: Path to save the JSON file
        indent: Number of spaces for indentation (default: 2)
        
    Returns:
        True if saving was successful, False otherwise
    """
    try:
        # Ensure the directory exists
        output_dir = os.path.dirname(file_path)
        ensure_dir_exists(output_dir)
        
        # Save the JSON data
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=indent)
        logger.info(f"Saved JSON data to {file_path}")
        return True
    except Exception as e:
        logger.error(f"Error saving JSON to {file_path}: {str(e)}")
        return False

def load_json(file_path: str) -> Optional[Any]:
    """
    Load data from a JSON file.
    
    Args:
        file_path: Path to the JSON file
        
    Returns:
        Loaded data or None if loading fails
    """
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        logger.info(f"Loaded JSON data from {file_path}")
        return data
    except Exception as e:
        logger.error(f"Error loading JSON from {file_path}: {str(e)}")
        return None 