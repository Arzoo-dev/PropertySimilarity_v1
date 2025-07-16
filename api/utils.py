"""
Utility functions for property comparison API
"""

import os
import logging
import json
# import aiohttp
import httpx
import asyncio
import tempfile
from typing import List, Dict, Tuple, Any, Optional
import torch
from PIL import Image
import io
import time
import shutil
from pathlib import Path
import numpy as np
import torchvision.transforms as transforms
import sys

# Google Cloud imports
try:
    from google.cloud import storage
    from google.cloud.logging import Client as LoggingClient
    CLOUD_IMPORTS_AVAILABLE = True
except ImportError:
    CLOUD_IMPORTS_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Image processing constants
IMG_SIZE = (518, 518)  # DINOv2 input size
IMG_MEAN = [0.5, 0.5, 0.5]      # DINOv2 normalization
IMG_STD = [0.5, 0.5, 0.5]       # DINOv2 normalization

# Cache for downloaded images
IMAGE_CACHE = {}
MAX_CACHE_SIZE = 200  # Maximum number of images to keep in cache

# Transformation pipeline for images
transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMG_MEAN, std=IMG_STD)
])


def setup_json_logging():
    """
    Configure logging to output in JSON format for Google Cloud Logging
    """
    if os.getenv("ENABLE_CLOUD_LOGGING", "").lower() == "true" and CLOUD_IMPORTS_AVAILABLE:
        try:
            # Initialize Cloud Logging
            logging_client = LoggingClient()
            handler = logging_client.get_default_handler()
            
            # Configure the root logger
            cloud_logger = logging.getLogger()
            cloud_logger.setLevel(logging.INFO)
            
            # Remove existing handlers
            for handler in cloud_logger.handlers:
                cloud_logger.removeHandler(handler)
                
            # Add Google Cloud handler
            cloud_logger.addHandler(handler)
            
            logger.info("Google Cloud Logging configured successfully")
        except Exception as e:
            logger.error(f"Failed to configure Google Cloud Logging: {str(e)}")
            _setup_json_console_logging()
    else:
        _setup_json_console_logging()


def _setup_json_console_logging():
    """
    Configure JSON-structured logging to stdout for Cloud Logging compatibility
    """
    # Create a handler that outputs to stdout
    handler = logging.StreamHandler(sys.stdout)
    
    # Define a formatter that outputs logs as JSON
    class JsonFormatter(logging.Formatter):
        def format(self, record):
            log_record = {
                "severity": record.levelname,
                "message": record.getMessage(),
                "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(record.created)),
                "logger": record.name
            }
            
            # Add extra fields
            if hasattr(record, '__dict__'):
                # Add any extra fields that aren't standard LogRecord attributes
                standard_attrs = {'name', 'msg', 'args', 'levelname', 'levelno', 'pathname', 'filename', 
                                 'module', 'exc_info', 'exc_text', 'stack_info', 'lineno', 'funcName', 
                                 'created', 'msecs', 'relativeCreated', 'thread', 'threadName', 
                                 'processName', 'process', 'getMessage', 'message'}
                extra_fields = {k: v for k, v in record.__dict__.items() if k not in standard_attrs}
                if extra_fields:
                    log_record.update(extra_fields)
                
            # Add exception info if available
            if record.exc_info:
                log_record["exception"] = self.formatException(record.exc_info)
                
            return json.dumps(log_record)
    
    # Set the formatter for the handler
    handler.setFormatter(JsonFormatter())
    
    # Configure the root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    # Remove existing handlers
    for old_handler in root_logger.handlers:
        root_logger.removeHandler(old_handler)
        
    # Add the JSON handler
    root_logger.addHandler(handler)
    
    logger.info("JSON console logging configured for Cloud Logging compatibility")


async def download_image(session: httpx.AsyncClient, url: str, 
                         max_retries: int = 3, timeout: int = 30) -> Optional[bytes]:
    """
    Download an image from a URL with retry logic
    
    Args:
        session: httpx async client
        url: URL of the image to download
        max_retries: Maximum number of retry attempts
        timeout: Timeout in seconds
        
    Returns:
        Image bytes or None if download failed
    """
    retry_count = 0
    
    # Explicitly log the type and value of url BEFORE the try block
    logger.info(f"Attempting download. URL type: {type(url)}, value: {repr(url)}")
    
    # Ensure url is a string
    if not isinstance(url, str):
        logger.error(f"CRITICAL: URL is NOT a string before try block! Type: {type(url)}")
        try:
            url = str(url)
            logger.warning(f"Converted URL to string: {url}")
        except:
            logger.error("Failed to convert URL to string")
            return None

    while retry_count < max_retries:
        try:
            # Log again right before the call
            logger.debug(f"Calling session.get with URL (type {type(url)}): {repr(url)}")
            
            response = await session.get(url, timeout=timeout)
            if response.status_code == 200:
                return response.content
            else:
                logger.warning(f"Failed to download image: {url}, status: {response.status_code}")
                    
        except asyncio.TimeoutError:
            logger.warning(f"Timeout downloading image: {url}")
        except Exception as e:
            # Log the type of url *inside* the except block too, just in case
            logger.error(f"Error downloading image (URL type {type(url)}): {url}, error: {str(e)}")
            
        retry_count += 1
        # Add a small delay before retrying
        await asyncio.sleep(0.5 * (retry_count))  # Exponential backoff
        
    logger.error(f"Download failed for {url} after {max_retries} retries.")
    return None


async def download_property_images(session: httpx.AsyncClient, photo_urls: List[str], 
                                 max_images: int = 10) -> Dict[str, Image.Image]:
    """
    Download images for a property
    
    Args:
        session: httpx async client
        photo_urls: List of photo URLs
        max_images: Maximum number of images to download
        
    Returns:
        Dictionary mapping URLs to PIL Images
    """
    # Limit number of images
    limited_urls = photo_urls[:max_images]
    
    # Check cache first
    cached_images = {url: IMAGE_CACHE[url] for url in limited_urls if url in IMAGE_CACHE}
    urls_to_download = [url for url in limited_urls if url not in IMAGE_CACHE]
    
    logger.info(f"Using {len(cached_images)} cached images, downloading {len(urls_to_download)} new images")
    
    # Download missing images
    tasks = [download_image(session, url) for url in urls_to_download]
    results = await asyncio.gather(*tasks)
    
    # Process downloaded images
    images = {}
    for url, img_bytes in zip(urls_to_download, results):
        if img_bytes:
            try:
                img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
                images[url] = img
                
                # Cache the image if cache isn't too big
                if len(IMAGE_CACHE) < MAX_CACHE_SIZE:
                    IMAGE_CACHE[url] = img
            except Exception as e:
                logger.error(f"Error processing image: {url}, error: {str(e)}")
    
    # Combine with cached images
    images.update(cached_images)
    
    return images


def preprocess_images(images: Dict[str, Image.Image]) -> torch.Tensor:
    """
    Preprocess images for model input
    
    Args:
        images: Dictionary of PIL images
        
    Returns:
        Tensor of preprocessed images [N, C, H, W]
    """
    if not images:
        logger.warning("No images to preprocess")
        return torch.zeros((0, 3, IMG_SIZE[0], IMG_SIZE[1]))
    
    processed_images = []
    
    for img in images.values():
        try:
            processed = transform(img)
            processed_images.append(processed)
        except Exception as e:
            logger.error(f"Error preprocessing image: {str(e)}")
    
    if not processed_images:
        logger.warning("No images successfully preprocessed")
        return torch.zeros((0, 3, IMG_SIZE[0], IMG_SIZE[1]))
    
    return torch.stack(processed_images)


def compute_accuracy_metrics(true_labels: List[bool], predicted_labels: List[bool]) -> Dict[str, Any]:
    """
    Compute accuracy metrics for property comparisons
    
    Args:
        true_labels: List of true labels (True for similar, False for dissimilar)
        predicted_labels: List of predicted labels
        
    Returns:
        Dictionary of metrics
    """
    # Count TP, FP, TN, FN
    tp = sum(1 for t, p in zip(true_labels, predicted_labels) if t and p)
    fp = sum(1 for t, p in zip(true_labels, predicted_labels) if not t and p)
    tn = sum(1 for t, p in zip(true_labels, predicted_labels) if not t and not p)
    fn = sum(1 for t, p in zip(true_labels, predicted_labels) if t and not p)
    
    # Calculate metrics
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        "num_test_pairs": len(true_labels),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1_score),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn)
    }


async def download_blob_to_temp(bucket_name: str, blob_path: str) -> Optional[str]:
    """
    Download a blob from GCS to a temporary file
    
    Args:
        bucket_name: GCS bucket name
        blob_path: Path to blob within bucket
        
    Returns:
        Path to temporary file or None if download failed
    """
    if not CLOUD_IMPORTS_AVAILABLE:
        logger.error("Google Cloud Storage imports not available")
        return None
        
    try:
        # Create a temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(blob_path)[1])
        temp_path = temp_file.name
        temp_file.close()
        
        # Download the blob
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_path)
        blob.download_to_filename(temp_path)
        
        logger.info(f"Downloaded GCS blob {bucket_name}/{blob_path} to {temp_path}")
        return temp_path
        
    except Exception as e:
        logger.error(f"Error downloading from GCS: {str(e)}")
        # Clean up the temporary file if it exists
        if 'temp_path' in locals():
            try:
                os.unlink(temp_path)
            except:
                pass
        return None


def clear_cache():
    """Clear the image cache"""
    global IMAGE_CACHE
    IMAGE_CACHE.clear()
    logger.info("Image cache cleared") 