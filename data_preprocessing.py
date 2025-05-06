"""
Data preprocessing utilities for Siamese network training on RunPods.
This module contains functions to generate triplets for training with data augmentation.
"""

import os
import random
import numpy as np
from PIL import Image
from typing import List, Dict, Tuple, Optional, Any, Generator, Callable
import logging
from tqdm import tqdm
import torch
import albumentations as A
import json
import gc
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import time
import shutil
import cv2
import uuid

# Import local file utilities
from runpods_utils import (
    load_image,
    save_image,
    ensure_dir_exists
)

# Configure logging - modify existing logging settings to show debug messages
logging.basicConfig(
    level=logging.DEBUG,  # Changed from INFO to DEBUG
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # Output to console
        logging.FileHandler("augmentation_logs.txt")  # Also save logs to file
    ]
)
logger = logging.getLogger(__name__)

# Cache for preprocessed images to avoid loading multiple times
image_cache = {}
MAX_CACHE_SIZE = 2000  # Maximum number of images to cache in memory

# Helper function for multiprocessing - must be at module level
def process_batch_helper(batch_args):
    """Helper function to unpack arguments for process_batch."""
    batch_idx, batch_size = batch_args
    result = generate_triplet_batch(
        batch_idx, 
        batch_size,
        image_paths_global,
        valid_property_types_global,
        property_types_global,
        target_size_global,
        augment_probability_global,
        output_dir_global,
        prefix_global
    )
    
    # Handle different return types - sometimes generate_triplet_batch returns a tuple
    if isinstance(result, tuple):
        triplets, successful = result
        return successful
    else:
        return result

# Global variables for multiprocessing
image_paths_global = None
valid_property_types_global = None
property_types_global = None
target_size_global = None
augment_probability_global = None
output_dir_global = None
prefix_global = None

# Create data augmentation pipeline
def create_augmentation_pipeline():
    """
    Create an enhanced data augmentation pipeline using Albumentations.
    Includes more aggressive transformations for better model generalization.
    
    Returns:
        Albumentations transformation pipeline
    """
    # Create individual transforms with names for logging
    transforms = [
        # Color augmentations (enhanced)
        ("RandomBrightnessContrast", A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.8)),
        ("HueSaturationValue", A.HueSaturationValue(hue_shift_limit=30, sat_shift_limit=50, val_shift_limit=30, p=0.7)),
        ("ColorJitter", A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.7)),
        ("CLAHE", A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.5)),
        
        # Geometric augmentations (enhanced)
        ("Rotate", A.Rotate(limit=30, border_mode=cv2.BORDER_REFLECT, p=0.8)),
        ("RandomResizedCrop", A.RandomResizedCrop(size=(224, 224), scale=(0.7, 1.0), ratio=(0.8, 1.2), p=0.8)),
        ("ShiftScaleRotate", A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=25, border_mode=cv2.BORDER_REFLECT, p=0.7)),
        ("HorizontalFlip", A.HorizontalFlip(p=0.5)),
        ("VerticalFlip", A.VerticalFlip(p=0.3)),
        ("Perspective", A.Perspective(scale=(0.05, 0.1), p=0.4)),
        
        # Advanced augmentations (updated parameters)
        ("GaussNoise", A.GaussNoise(var_limit=(10, 70), p=0.4)),
        ("GaussianBlur", A.GaussianBlur(blur_limit=(3, 9), p=0.3)),
        ("MotionBlur", A.MotionBlur(blur_limit=(5, 15), p=0.3)),
        ("GridDistortion", A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.3)),
        ("ElasticTransform", A.ElasticTransform(alpha=1, sigma=50, p=0.2)),
        
        # Weather and lighting effects (updated parameters)
        ("RandomRain", A.RandomRain(drop_length=20, drop_width=1, drop_color=(200, 200, 200), p=0.2)),
        ("RandomShadow", A.RandomShadow(shadow_dimension=5, p=0.2)),
        ("RandomSunFlare", A.RandomSunFlare(flare_roi=(0, 0, 1, 0.5), p=0.1)),
        
        # Compression and quality degradation
        ("ImageCompression", A.ImageCompression(quality_lower=60, p=0.3)),
        ("ToGray", A.ToGray(p=0.1)),
        
        # Random cropping and masking (updated parameters)
        ("CoarseDropout", A.CoarseDropout(max_holes=8, max_height=32, max_width=32, p=0.3)),
    ]
    
    # Create a logging class that wraps the Compose
    class LoggingCompose(A.Compose):
        def __call__(self, *args, **kwargs):
            result = super().__call__(*args, **kwargs)
            
            # Log all the applied transformations
            applied = []
            for transform in self.transforms:
                if hasattr(transform, 'applied') and transform.applied:
                    transform_name = transform.__class__.__name__
                    applied.append(transform_name)
            
            if applied:
                logger.debug(f"Applied augmentations: {', '.join(applied)}")
            
            return result
    
    # Extract just the transforms without names
    transform_list = [t[1] for t in transforms]
    
    return LoggingCompose(transform_list, p=1.0)

def apply_augmentation(image: np.ndarray, transform_pipeline: A.Compose) -> np.ndarray:
    """
    Apply augmentation to a single image.
    
    Args:
        image: Image as numpy array (height, width, channels)
        transform_pipeline: Albumentations transformation pipeline
        
    Returns:
        Augmented image as numpy array
    """
    # Force higher logging level to ensure augmentation logs are captured
    augmentation_logger = logging.getLogger("augmentation")
    file_handler = logging.FileHandler("augmentation_detailed_logs.txt")
    file_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    augmentation_logger.addHandler(file_handler)
    augmentation_logger.setLevel(logging.DEBUG)
    
    # Calculate image stats before augmentation for comparison
    before_mean = np.mean(image)
    before_std = np.std(image)
    
    # Apply the transformations with forced probability
    forced_pipeline = A.Compose([
        # Color transformations (always apply at least one)
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=1.0),
            A.HueSaturationValue(hue_shift_limit=30, sat_shift_limit=50, val_shift_limit=30, p=1.0),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=1.0),
        ], p=1.0),
        
        # Geometric transformations (always apply at least one)
        A.OneOf([
            A.Rotate(limit=30, border_mode=cv2.BORDER_REFLECT, p=1.0),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=25, p=1.0),
            A.HorizontalFlip(p=1.0),
        ], p=1.0),
        
        # Add some quality degradation (sometimes)
        A.OneOf([
            A.GaussNoise(var_limit=(10, 70), p=1.0),
            A.GaussianBlur(blur_limit=(3, 9), p=1.0),
            A.ImageCompression(quality_lower=60, p=1.0),
        ], p=0.5),
    ])
    
    # Log that we're applying augmentation
    augmentation_logger.debug(f"Applying augmentation to image with shape {image.shape}")
    
    # Apply transformations
    result = forced_pipeline(image=image)
    augmented_image = result["image"]
    
    # Calculate image stats after augmentation
    after_mean = np.mean(augmented_image)
    after_std = np.std(augmented_image)
    
    # Verify augmentation made a difference
    mean_diff = abs(after_mean - before_mean)
    std_diff = abs(after_std - before_std)
    
    augmentation_logger.debug(f"Augmentation stats: Before [mean={before_mean:.3f}, std={before_std:.3f}], After [mean={after_mean:.3f}, std={after_std:.3f}]")
    augmentation_logger.debug(f"Difference: mean_diff={mean_diff:.3f}, std_diff={std_diff:.3f}")
    
    # Ensure augmentation had an effect (add a small random change if not)
    if mean_diff < 0.01 and std_diff < 0.01:
        augmentation_logger.warning("Augmentation had minimal effect, adding additional random changes")
        # Add some random noise to ensure difference
        noise = np.random.normal(0, 0.05, augmented_image.shape)
        augmented_image = np.clip(augmented_image + noise, 0, 1)
    
    # Print to console as well for visibility during execution
    print(f"Applied augmentation: Before [mean={before_mean:.3f}, std={before_std:.3f}], After [mean={after_mean:.3f}, std={after_std:.3f}]")
    
    return augmented_image

def load_and_preprocess_image(image_path: str, target_size: Tuple[int, int] = (224, 224)) -> Optional[np.ndarray]:
    """
    Load and preprocess an image from a local file.
    
    Args:
        image_path: Path to the image file
        target_size: Target size for resizing
        
    Returns:
        Preprocessed image as numpy array or None if loading fails
    """
    # Check if image is in cache
    if image_path in image_cache:
        return image_cache[image_path]
    
    try:
        # Load image
        img = load_image(image_path)
        if img is None:
            return None
        
        # Convert to RGB (in case of grayscale or RGBA)
        img = img.convert('RGB')
        
        # Resize to target size
        img = img.resize(target_size, Image.LANCZOS)
        
        # Convert to numpy array and normalize to [0,1]
        img_array = np.array(img).astype(np.float32) / 255.0
        
        # Add to cache if it's not too big
        if len(image_cache) < MAX_CACHE_SIZE:
            image_cache[image_path] = img_array
            
        return img_array
    except Exception as e:
        logger.error(f"Error preprocessing image {image_path}: {str(e)}")
        return None

def clear_image_cache():
    """Clear the image cache to free memory."""
    global image_cache
    image_cache.clear()
    gc.collect()
    logger.info("Image cache cleared")

def load_and_preprocess_property_images(
    image_paths: List[str],
    target_size: Tuple[int, int] = (224, 224),
    max_images: int = 10,
    augment_probability: float = 0.0,
    augment_pipeline = None
) -> List[np.ndarray]:
    """
    Load and preprocess a set of property images, optionally applying augmentation.
    
    Args:
        image_paths: List of image paths for a property
        target_size: Target size for resizing images
        max_images: Maximum number of images to use per property (randomly samples if more)
        augment_probability: Probability of applying augmentation
        augment_pipeline: Optional pre-created augmentation pipeline
        
    Returns:
        List of preprocessed images as numpy arrays
    """
    if not image_paths:
        logger.warning("No image paths provided")
        return []
    
    # Randomly sample if we have more images than the maximum allowed
    if len(image_paths) > max_images:
        image_paths = random.sample(image_paths, max_images)
    
    # Create augmentation pipeline if needed and not provided
    if augment_probability > 0 and augment_pipeline is None:
        augment_pipeline = create_augmentation_pipeline()
    
    processed_images = []
    
    for path in image_paths:
        try:
            # Load and preprocess the image
            img_array = load_and_preprocess_image(path, target_size)
            
            if img_array is None:
                continue
                
            # Apply augmentation with specified probability
            if augment_probability > 0 and random.random() < augment_probability:
                augmented = augment_pipeline(image=img_array)
                img_array = augmented["image"]
            
            processed_images.append(img_array)
            
        except Exception as e:
            logger.error(f"Error processing image {path}: {str(e)}")
    
    if not processed_images:
        logger.warning(f"No valid images processed from {len(image_paths)} paths")
    
    return processed_images

def generate_property_pairs(
    property_data: Dict[str, List[str]],
    ratings_data: List[Dict],
    num_pairs: int = 100,
    target_size: Tuple[int, int] = (224, 224),
    max_images_per_property: int = 10,
    augment_probability: float = 0.3,
    output_dir: Optional[str] = None
) -> List[Dict]:
    """
    Generate property pairs with expert ratings for training.
    
    Args:
        property_data: Dictionary mapping property UIDs to lists of image paths
        ratings_data: List of dictionaries with subject_uid, comparable_uid, and similarity_score
        num_pairs: Number of pairs to generate (will be capped by available ratings)
        target_size: Target size for image preprocessing
        max_images_per_property: Maximum number of images to use per property
        augment_probability: Probability of applying augmentation to each image
        output_dir: Optional directory to save processed property pairs
        
    Returns:
        List of property pair dictionaries with processed images
    """
    logger.info(f"Generating {num_pairs} property pairs for training")
    
    # Create augmentation pipeline
    augment_pipeline = create_augmentation_pipeline()
    
    # Shuffle and limit ratings
    random.shuffle(ratings_data)
    ratings_data = ratings_data[:num_pairs]
    
    property_pairs = []
    valid_pairs = 0
    
    for idx, rating in enumerate(tqdm(ratings_data, desc="Processing property pairs")):
        try:
            subject_uid = rating.get("subject_uid")
            comparable_uid = rating.get("comparable_uid")
            similarity_score = rating.get("similarity_score")
            
            if not subject_uid or not comparable_uid or similarity_score is None:
                logger.warning(f"Skipping invalid rating: {rating}")
                continue
            
            # Get image paths for both properties
            subject_images = property_data.get(subject_uid, [])
            comparable_images = property_data.get(comparable_uid, [])
            
            if not subject_images or not comparable_images:
                logger.warning(f"Skipping pair with missing images for {subject_uid} or {comparable_uid}")
                continue
            
            # Process images for both properties
            subject_processed = load_and_preprocess_property_images(
                subject_images,
                target_size=target_size,
                max_images=max_images_per_property,
                augment_probability=augment_probability,
                augment_pipeline=augment_pipeline
            )
            
            comparable_processed = load_and_preprocess_property_images(
                comparable_images,
                target_size=target_size,
                max_images=max_images_per_property,
                augment_probability=augment_probability,
                augment_pipeline=augment_pipeline
            )
            
            # Skip if we didn't get any valid images
            if not subject_processed or not comparable_processed:
                logger.warning(f"Skipping pair with no valid processed images")
                continue
            
            pair_data = {
                "subject_uid": subject_uid,
                "comparable_uid": comparable_uid,
                "similarity_score": float(similarity_score),
                "subject_processed": subject_processed,
                "comparable_processed": comparable_processed
            }
            
            # Save to disk if output_dir is provided
            if output_dir:
                pair_dir = os.path.join(output_dir, f"pair_{idx:05d}")
                os.makedirs(pair_dir, exist_ok=True)
                
                # Save subject images
                subject_dir = os.path.join(pair_dir, "subject")
                os.makedirs(subject_dir, exist_ok=True)
                for i, img_array in enumerate(subject_processed):
                    img_path = os.path.join(subject_dir, f"image_{i:03d}.jpg")
                    img_uint8 = (img_array * 255).astype(np.uint8)
                    Image.fromarray(img_uint8).save(img_path, format='JPEG', quality=95)
                
                # Save comparable images
                comparable_dir = os.path.join(pair_dir, "comparable")
                os.makedirs(comparable_dir, exist_ok=True)
                for i, img_array in enumerate(comparable_processed):
                    img_path = os.path.join(comparable_dir, f"image_{i:03d}.jpg")
                    img_uint8 = (img_array * 255).astype(np.uint8)
                    Image.fromarray(img_uint8).save(img_path, format='JPEG', quality=95)
                
                # Save metadata
                meta_path = os.path.join(pair_dir, "metadata.json")
                with open(meta_path, 'w') as f:
                    json.dump({
                        "subject_uid": subject_uid,
                        "comparable_uid": comparable_uid,
                        "similarity_score": similarity_score,
                        "subject_count": len(subject_processed),
                        "comparable_count": len(comparable_processed)
                    }, f, indent=2)
                
                # In this case, don't keep the processed images in memory
                pair_data.pop("subject_processed", None)
                pair_data.pop("comparable_processed", None)
                pair_data["pair_dir"] = pair_dir
            
            property_pairs.append(pair_data)
            valid_pairs += 1
            
            # Yield to other processes occasionally to reduce CPU pressure
            if idx % 20 == 0:
                time.sleep(0.01)
                
        except Exception as e:
            logger.error(f"Error processing property pair: {str(e)}")
    
    logger.info(f"Generated {valid_pairs} valid property pairs out of {len(ratings_data)} ratings")
    
    # Save a manifest if output_dir is provided
    if output_dir and property_pairs:
        manifest_path = os.path.join(output_dir, "property_pairs_manifest.json")
        with open(manifest_path, 'w') as f:
            json.dump({
                "property_pairs": property_pairs,
                "total_pairs": len(property_pairs),
                "creation_date": time.strftime("%Y-%m-%d %H:%M:%S")
            }, f, indent=2)
        logger.info(f"Saved property pairs manifest to {manifest_path}")
    
    return property_pairs

def load_property_pairs_from_disk(
    pairs_dir: str, 
    max_pairs: Optional[int] = None
) -> List[Dict]:
    """
    Load property pairs that were previously saved to disk.
    
    Args:
        pairs_dir: Directory containing property pair subdirectories
        max_pairs: Maximum number of pairs to load
        
    Returns:
        List of property pair dictionaries
    """
    logger.info(f"Loading property pairs from {pairs_dir}")
    
    # Try to load from manifest first
    manifest_path = os.path.join(pairs_dir, "property_pairs_manifest.json")
    if os.path.exists(manifest_path):
        try:
            with open(manifest_path, 'r') as f:
                manifest = json.load(f)
            
            pairs = manifest.get("property_pairs", [])
            logger.info(f"Loaded {len(pairs)} property pairs from manifest")
            
            if max_pairs and len(pairs) > max_pairs:
                pairs = pairs[:max_pairs]
                logger.info(f"Limited to {max_pairs} pairs")
            
            return pairs
        except Exception as e:
            logger.warning(f"Error loading property pairs manifest: {str(e)}")
    
    # Fallback to scanning directories
    logger.warning("No manifest found, scanning directories for property pairs")
    pairs = []
    
    for item in os.listdir(pairs_dir):
        if not item.startswith("pair_"):
            continue
        
        pair_dir = os.path.join(pairs_dir, item)
        if not os.path.isdir(pair_dir):
            continue
        
        # Look for metadata.json
        meta_path = os.path.join(pair_dir, "metadata.json")
        if not os.path.exists(meta_path):
            continue
        
        try:
            with open(meta_path, 'r') as f:
                metadata = json.load(f)
            
            # Add pair directory to metadata
            metadata["pair_dir"] = pair_dir
            pairs.append(metadata)
            
            if max_pairs and len(pairs) >= max_pairs:
                break
                
        except Exception as e:
            logger.warning(f"Error loading metadata from {meta_path}: {str(e)}")
    
    logger.info(f"Loaded {len(pairs)} property pairs by scanning directories")
    return pairs

def load_ratings_from_json(json_path: str) -> List[Dict]:
    """
    Load expert ratings from a JSON file.
    
    Expected format:
    {
        "property_pairs": [
            {
                "subject_uid": "prop123",
                "comparable_uid": "prop456",
                "similarity_score": 0.85
            },
            ...
        ]
    }
    
    Args:
        json_path: Path to the JSON file
        
    Returns:
        List of rating dictionaries
    """
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        ratings = data.get("property_pairs", [])
        logger.info(f"Loaded {len(ratings)} expert ratings from {json_path}")
        return ratings
    except Exception as e:
        logger.error(f"Error loading ratings from {json_path}: {str(e)}")
        return []

def generate_triplet_batch(
    batch_index: int,
    batch_size: int,
    image_paths: Dict[str, List[str]],
    valid_property_types: List[str],
    property_types: List[str],
    target_size: Tuple[int, int] = (224, 224),
    augment_probability: float = 0.5,
    output_dir: str = None,
    prefix: str = "triplet"
) -> int:
    """
    Generate and save a batch of triplets.
    
    Args:
        batch_index: Batch index for naming
        batch_size: Number of triplets to generate in this batch
        image_paths: Dictionary mapping property types to lists of image file paths
        valid_property_types: List of property types with at least 2 images
        property_types: List of all property types
        target_size: Target size for image preprocessing
        augment_probability: Probability of applying augmentation to each image
        output_dir: Directory to save triplets (None to not save)
        prefix: Prefix for saved triplet directories
        
    Returns:
        Number of successful triplets generated
    """
    augmentation_pipeline = create_augmentation_pipeline()
    start_idx = batch_index * batch_size
    batch_triplets = []
    successful = 0
    
    logger.info(f"Generating batch {batch_index} with {batch_size} triplets (augment_prob={augment_probability:.2f})")
    
    # Create a simple check for image quality
    def is_valid_image(img):
        if img is None:
            return False
        # Check if image has enough variation (not solid color)
        std = np.std(img)
        return std > 0.01  # Minimum variation threshold
    
    # Try to generate triplets until we have enough or hit the limit
    max_attempts = batch_size * 3  # Allow for some failures
    attempts = 0
    
    while successful < batch_size and attempts < max_attempts:
        attempts += 1
        
        try:
            # Select random property type for anchor/positive
            anchor_type = random.choice(valid_property_types)
            
            # Select two different images from the same type
            anchor_path, positive_path = random.sample(image_paths[anchor_type], 2)
            
            # Select a different property type for negative
            negative_types = [t for t in property_types if t != anchor_type]
            negative_type = random.choice(negative_types)
            negative_path = random.choice(image_paths[negative_type])
            
            # Log the image paths
            logger.debug(f"Triplet #{start_idx + successful}: anchor_type={anchor_type}, negative_type={negative_type}")
            logger.debug(f"  - Anchor: {os.path.basename(anchor_path)}")
            logger.debug(f"  - Positive: {os.path.basename(positive_path)}")
            logger.debug(f"  - Negative: {os.path.basename(negative_path)}")
            
            # Load and preprocess images
            anchor_img = load_and_preprocess_image(anchor_path, target_size)
            positive_img = load_and_preprocess_image(positive_path, target_size)
            negative_img = load_and_preprocess_image(negative_path, target_size)
            
            # Skip if any image is invalid or low quality
            if not all(map(is_valid_image, [anchor_img, positive_img, negative_img])):
                logger.debug(f"Skipping triplet - invalid or low quality images detected")
                continue
            
            # Apply augmentation with a certain probability
            if random.random() < augment_probability:
                logger.debug(f"Augmenting anchor image from {os.path.basename(anchor_path)}")
                anchor_img = apply_augmentation(anchor_img, augmentation_pipeline)
                
            if random.random() < augment_probability:
                logger.debug(f"Augmenting positive image from {os.path.basename(positive_path)}")
                positive_img = apply_augmentation(positive_img, augmentation_pipeline)
                
            if random.random() < augment_probability:
                logger.debug(f"Augmenting negative image from {os.path.basename(negative_path)}")
                negative_img = apply_augmentation(negative_img, augmentation_pipeline)
            
            # Skip if augmentation produced invalid images
            if not all(map(is_valid_image, [anchor_img, positive_img, negative_img])):
                logger.debug(f"Skipping triplet - augmentation produced invalid images")
                continue
                
            # If output_dir is provided, save directly
            if output_dir:
                triplet_idx = start_idx + successful
                triplet_name = f"{prefix}_{triplet_idx:05d}"
                logger.debug(f"Saving triplet as {triplet_name}")
                save_individual_triplet(
                    (anchor_img, positive_img, negative_img),
                    output_dir,
                    triplet_name
                )
            else:
                batch_triplets.append((anchor_img, positive_img, negative_img))
            successful += 1
            
        except Exception as e:
            logger.warning(f"Error generating triplet: {str(e)}")
            continue
            
        # Yield to other processes occasionally to reduce CPU pressure
        if attempts % 20 == 0:
            time.sleep(0.01)  # Small sleep to let other processes run
    
    logger.info(f"Generated {successful}/{batch_size} triplets in batch {batch_index} after {attempts} attempts")
    
    # If we're not saving directly and have triplets, return them
    if not output_dir and batch_triplets:
        return batch_triplets, successful
    return successful

def save_individual_triplet(triplet: Tuple[np.ndarray, np.ndarray, np.ndarray], 
                          output_dir: str, 
                          triplet_name: str) -> bool:
    """
    Save a single triplet to disk with proper error handling.
    
    Args:
        triplet: Tuple of (anchor, positive, negative) images as numpy arrays
        output_dir: Base directory for saving triplets
        triplet_name: Name/ID for this triplet
        
    Returns:
        True if saving was successful, False otherwise
    """
    try:
        triplet_dir = os.path.join(output_dir, triplet_name)
        os.makedirs(triplet_dir, exist_ok=True)
        
        # Convert and save each image
        for img_array, img_name in zip(triplet, ['anchor.jpg', 'positive.jpg', 'negative.jpg']):
            # Convert from float [0,1] to uint8 [0,255]
            img_uint8 = (img_array * 255).astype(np.uint8)
            img = Image.fromarray(img_uint8)
            
            # Save with high quality
            img_path = os.path.join(triplet_dir, img_name)
            img.save(img_path, format='JPEG', quality=95)
        
        return True
    except Exception as e:
        logger.error(f"Error saving triplet {triplet_name}: {str(e)}")
        # Clean up failed triplet directory
        if os.path.exists(triplet_dir):
            shutil.rmtree(triplet_dir)
        return False

def save_triplet_info(
    num_triplets: int,
    output_dir: str,
    prefix: str = "triplet"
) -> None:
    """
    Save triplet information to a JSON manifest file.
    
    Args:
        num_triplets: Number of triplets generated
        output_dir: Directory where triplets are saved
        prefix: Prefix used for triplet directories
    """
    triplet_manifest = {
        "num_triplets": num_triplets,
        "triplet_prefix": prefix,
        "creation_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "triplets": [f"{prefix}_{i:05d}" for i in range(num_triplets)]
    }
    
    manifest_path = os.path.join(output_dir, "triplet_manifest.json")
    with open(manifest_path, 'w') as f:
        json.dump(triplet_manifest, f, indent=2)
    
    logger.info(f"Saved triplet manifest to {manifest_path}")

def generate_triplets_with_augmentation(
    image_paths: Dict[str, List[str]],
    num_triplets: int = 10000,
    batch_size: int = 100,
    target_size: Tuple[int, int] = (224, 224),
    augment_probability: float = 0.7,
    output_dir: str = None,
    prefix: str = "triplet",
    max_workers: int = None,
    save_triplets: bool = False
) -> Optional[List[Tuple[np.ndarray, np.ndarray, np.ndarray]]]:
    """
    Generate triplets with data augmentation, optionally saving directly to disk.
    Now automatically uses property-level structure with UID folder names.
    
    Args:
        image_paths: Dictionary mapping property types to lists of image paths
        num_triplets: Number of triplets to generate
        batch_size: Size of batches for parallel processing
        target_size: Target size for image preprocessing
        augment_probability: Probability of applying augmentation
        output_dir: Directory to save triplets (None to return in memory)
        prefix: Prefix for saved triplet directories
        max_workers: Maximum number of worker processes
        save_triplets: Whether to save generated triplets to disk
        
    Returns:
        List of triplets if output_dir is None, else None or the number of successful triplets
    """
    logger.info("⚠️ Using property-level triplet generation with UID-based folder naming ⚠️")
    print("⚠️ NOTICE: Using property-level triplet generation with UID-based folder naming ⚠️")
    print("⚠️ Each property will maintain ALL its images and have multiple augmentations per image ⚠️")
    print("⚠️ Using low-memory mode to prevent memory exhaustion ⚠️")
    
    # Switch to calling our property-level implementation with low_memory_mode enabled
    return generate_property_triplets(
        property_data=image_paths,
        num_triplets=num_triplets,
        target_size=target_size,
        augment_probability=augment_probability,
        max_images_per_property=999,  # Very high limit to ensure we use all images
        output_dir=output_dir,
        prefix=prefix,
        augmentations_per_image=3,  # Create 3 augmented versions per original image
        low_memory_mode=True,        # Enable memory-saving features 
        memory_limit_percentage=70.0  # Conservative memory limit
    )

def create_property_type_mapping(image_paths: Dict[str, List[str]]) -> Dict[str, int]:
    """
    Create a mapping from property type names to integer labels.
    
    Args:
        image_paths: Dictionary mapping property types to lists of image paths
        
    Returns:
        Dictionary mapping property type names to integer labels
    """
    property_types = sorted(list(image_paths.keys()))
    return {property_type: i for i, property_type in enumerate(property_types)}

def save_triplets(triplets: List[Tuple[np.ndarray, np.ndarray, np.ndarray]], 
                 output_dir: str, 
                 prefix: str = "triplet") -> None:
    """
    Save triplets to disk with proper directory creation and manifest generation.
    
    Args:
        triplets: List of (anchor, positive, negative) triplets as numpy arrays
        output_dir: Directory to save triplets
        prefix: Prefix for triplet directories
    """
    try:
        # Ensure output directory exists and is empty
        output_dir = os.path.abspath(output_dir)
        ensure_dir_exists(output_dir)
        
        # Generate property UIDs for each triplet (since we don't have real ones)
        # This simulates using property UIDs instead of sequential numbers
        property_uids = [f"property_{str(uuid.uuid4())[:8]}" for _ in range(len(triplets))]
        
        # Initialize manifest
        manifest = {
            "triplets": [],
            "total_triplets": len(triplets),
            "creation_date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "prefix": prefix,
            "base_dir": output_dir
        }
        
        # Save triplets with progress bar
        successful = 0
        for idx, (triplet, uid) in enumerate(zip(triplets, property_uids)):
            # Use property UID for the folder name instead of sequential index
            property_dir = os.path.join(output_dir, uid)
            
            # Create subdirectories for anchor, positive, negative
            anchor_dir = os.path.join(property_dir, "anchor")
            positive_dir = os.path.join(property_dir, "positive")
            negative_dir = os.path.join(property_dir, "negative")
            
            os.makedirs(anchor_dir, exist_ok=True)
            os.makedirs(positive_dir, exist_ok=True)
            os.makedirs(negative_dir, exist_ok=True)
            
            # Save images with proper property-level structure
            anchor_img, positive_img, negative_img = triplet
            
            # Save anchor image in anchor folder
            anchor_path = os.path.join(anchor_dir, "image_001.jpg")
            anchor_uint8 = (anchor_img * 255).astype(np.uint8)
            Image.fromarray(anchor_uint8).save(anchor_path, format='JPEG', quality=95)
            
            # Save positive image in positive folder
            positive_path = os.path.join(positive_dir, "image_001.jpg")
            positive_uint8 = (positive_img * 255).astype(np.uint8)
            Image.fromarray(positive_uint8).save(positive_path, format='JPEG', quality=95)
            
            # Save negative image in negative folder
            negative_path = os.path.join(negative_dir, "image_001.jpg")
            negative_uint8 = (negative_img * 255).astype(np.uint8)
            Image.fromarray(negative_uint8).save(negative_path, format='JPEG', quality=95)
            
            # Add to manifest with relative paths
            manifest["triplets"].append({
                "id": uid,
                "anchor_dir": anchor_dir,
                "positive_dir": positive_dir,
                "negative_dir": negative_dir,
                "creation_date": time.strftime("%Y-%m-%d %H:%M:%S")
            })
            
            successful += 1
            
            # Save manifest periodically
            if (idx + 1) % 100 == 0:
                manifest_path = os.path.join(output_dir, "triplet_manifest.json")
                with open(manifest_path, 'w') as f:
                    json.dump(manifest, f, indent=2)
                logger.debug(f"Saved intermediate manifest with {successful} triplets")
        
        # Update final statistics
        manifest["successful_triplets"] = successful
        manifest["failed_triplets"] = len(triplets) - successful
        
        # Save final manifest
        manifest_path = os.path.join(output_dir, "triplet_manifest.json")
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        # Verify manifest was saved
        if not os.path.exists(manifest_path):
            raise IOError(f"Failed to save manifest to {manifest_path}")
            
        logger.info(f"Successfully saved {successful} triplets out of {len(triplets)}")
        logger.info(f"Saved triplet manifest to {manifest_path}")
        
        # Debug: Print manifest contents
        logger.debug(f"Manifest contents: {json.dumps(manifest, indent=2)}")
        
    except Exception as e:
        logger.error(f"Error saving triplets: {str(e)}")
        raise

def ensure_multiple_property_types(image_paths: Dict[str, List[str]], min_types: int = 2) -> Dict[str, List[str]]:
    """
    Ensure we have at least min_types different property types with at least 2 images each.
    If not, we'll duplicate some property types to ensure we have enough.
    
    Args:
        image_paths: Dictionary mapping property types to lists of image paths
        min_types: Minimum number of property types needed
        
    Returns:
        Dictionary with at least min_types different property types
    """
    # Find valid property types (with at least 2 images)
    valid_types = {pt: images for pt, images in image_paths.items() if len(images) >= 2}
    
    if len(valid_types) >= min_types:
        return image_paths
    
    # Not enough valid types, duplicate some
    logger.warning(f"Only {len(valid_types)} valid property types found, need at least {min_types}")
    
    if len(valid_types) == 0:
        logger.error("No valid property types with 2+ images found")
        # Try to salvage what we can - group images together
        all_images = []
        for images in image_paths.values():
            all_images.extend(images)
        
        if len(all_images) < 3:
            logger.error("Not enough images to continue")
            return {}
        
        # Split images into artificial groups
        artificial_types = {}
        n_groups = min_types
        images_per_group = len(all_images) // n_groups
        
        for i in range(n_groups):
            start_idx = i * images_per_group
            end_idx = start_idx + images_per_group if i < n_groups - 1 else len(all_images)
            artificial_types[f"group_{i+1}"] = all_images[start_idx:end_idx]
        
        logger.info(f"Created {n_groups} artificial property type groups")
        return artificial_types
    
    # We have at least 1 valid type, duplicate it
    result = valid_types.copy()
    valid_types_list = list(valid_types.keys())
    
    for i in range(min_types - len(valid_types)):
        original_type = valid_types_list[i % len(valid_types_list)]
        new_type = f"{original_type}_duplicate_{i+1}"
        result[new_type] = valid_types[original_type].copy()
        logger.info(f"Duplicated property type {original_type} as {new_type}")
    
    return result

def create_torch_dataset_from_triplets(triplets_source):
    """
    Create a TripletDataset instance.
    This function is kept for backward compatibility.
    """
    return TripletDataset(triplets_source)

class TripletDataset(torch.utils.data.Dataset):
    """Dataset for loading triplets from a directory."""
    
    def __init__(self, 
                 root_dir: str,
                 is_directory: bool = True,
                 max_triplets: Optional[int] = None,
                 transform: Optional[Callable] = None):
        """
        Initialize the dataset.
        
        Args:
            root_dir: Directory containing triplets or path to manifest file
            is_directory: If True, root_dir is a directory containing triplets
            max_triplets: Maximum number of triplets to load
            transform: Optional transform to apply to images
        """
        self.root_dir = os.path.abspath(root_dir)
        self.transform = transform or self._get_default_transform()
        
        # First try to load from manifest
        manifest_loaded = False
        if is_directory:
            # Look for manifest in the directory
            manifest_path = os.path.join(self.root_dir, "triplet_manifest.json")
            if os.path.exists(manifest_path):
                try:
                    with open(manifest_path, 'r') as f:
                        self.manifest = json.load(f)
                    self.triplets = self.manifest.get("triplets", [])
                    manifest_loaded = True
                    logger.info(f"Loaded {len(self.triplets)} triplets from manifest at {manifest_path}")
                except Exception as e:
                    logger.warning(f"Error loading manifest from {manifest_path}: {str(e)}")
        else:
            # Use root_dir as manifest path
            manifest_path = self.root_dir
            if os.path.exists(manifest_path):
                try:
                    with open(manifest_path, 'r') as f:
                        self.manifest = json.load(f)
                    self.triplets = self.manifest.get("triplets", [])
                    manifest_loaded = True
                    logger.info(f"Loaded {len(self.triplets)} triplets from manifest file at {manifest_path}")
                except Exception as e:
                    logger.warning(f"Error loading manifest from {manifest_path}: {str(e)}")
        
        # If manifest not loaded, fallback to scanning directories
        if not manifest_loaded:
            logger.warning(f"No valid manifest found. Falling back to scanning directories in {self.root_dir}")
            self.triplets = []
            
            # Look for directories named triplet_*
            for item in os.listdir(self.root_dir):
                item_path = os.path.join(self.root_dir, item)
                if os.path.isdir(item_path) and item.startswith("triplet_"):
                    # Check if this directory has the required files
                    anchor_path = os.path.join(item_path, "anchor.jpg")
                    positive_path = os.path.join(item_path, "positive.jpg")
                    negative_path = os.path.join(item_path, "negative.jpg")
                    
                    if all(os.path.exists(p) for p in [anchor_path, positive_path, negative_path]):
                        self.triplets.append({
                            "id": item,
                            "anchor": f"{item}/anchor.jpg",
                            "positive": f"{item}/positive.jpg",
                            "negative": f"{item}/negative.jpg"
                        })
            
            # If we still have no triplets, create dummy triplets for testing
            if not self.triplets and max_triplets:
                logger.warning(f"No triplet directories found. Creating {max_triplets} dummy triplets for testing.")
                for i in range(max_triplets):
                    dummy_id = f"triplet_{i:05d}"
                    self.triplets.append({
                        "id": dummy_id,
                        "anchor": f"{dummy_id}/anchor.jpg",
                        "positive": f"{dummy_id}/positive.jpg",
                        "negative": f"{dummy_id}/negative.jpg"
                    })
            
            logger.info(f"Found {len(self.triplets)} triplets by scanning directories")
            
            # Create and save a manifest for future use
            try:
                self.manifest = {
                    "triplets": self.triplets,
                    "total_triplets": len(self.triplets),
                    "creation_date": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "prefix": "triplet",
                    "note": "Created by directory scanning fallback"
                }
                
                # Save manifest for future use
                if is_directory:
                    new_manifest_path = os.path.join(self.root_dir, "triplet_manifest.json")
                    with open(new_manifest_path, 'w') as f:
                        json.dump(self.manifest, f, indent=2)
                    logger.info(f"Created new manifest at {new_manifest_path}")
            except Exception as e:
                logger.warning(f"Error creating new manifest: {str(e)}")
        
        # Apply max_triplets limit
        if max_triplets and len(self.triplets) > max_triplets:
            self.triplets = self.triplets[:max_triplets]
            logger.info(f"Limited to {max_triplets} triplets")
        
        # If no triplets found, raise error
        if not self.triplets:
            raise ValueError(f"No triplets found in {self.root_dir}")
        
        logger.info(f"TripletDataset initialized with {len(self.triplets)} triplets")
    
    def _get_default_transform(self) -> Callable:
        """Get default transform for images."""
        return A.Compose([
            A.Resize(224, 224),
            A.Normalize(mean=[0.485, 0.456, 0.406], 
                       std=[0.229, 0.224, 0.225])
        ])
    
    def __len__(self) -> int:
        return len(self.triplets)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        triplet = self.triplets[idx]
        
        # Load images using absolute paths
        anchor_path = os.path.join(self.root_dir, triplet["anchor"])
        positive_path = os.path.join(self.root_dir, triplet["positive"])
        negative_path = os.path.join(self.root_dir, triplet["negative"])
        
        try:
            # Load and transform images
            anchor = self._load_and_transform(anchor_path)
            positive = self._load_and_transform(positive_path)
            negative = self._load_and_transform(negative_path)
            
            return anchor, positive, negative
        except Exception as e:
            logger.error(f"Error loading triplet {idx} ({triplet['id']}): {str(e)}")
            # Return a simple fallback triplet with random tensors
            shape = (3, 224, 224)  # RGB image with default size
            return torch.rand(shape), torch.rand(shape), torch.rand(shape)
    
    def _load_and_transform(self, image_path: str) -> torch.Tensor:
        """Load and transform a single image."""
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Failed to load image: {image_path}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Apply transform
            transformed = self.transform(image=image)
            image_tensor = torch.from_numpy(transformed["image"]).float()
            image_tensor = image_tensor.permute(2, 0, 1)  # HWC to CHW
            
            return image_tensor
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {str(e)}")
            raise

def load_property_directory(property_dir: str, target_size: Tuple[int, int] = (224, 224)) -> List[np.ndarray]:
    """
    Load all images from a property directory and preprocess them.
    
    Args:
        property_dir: Directory containing property images
        target_size: Target size for resizing
        
    Returns:
        List of preprocessed images as numpy arrays
    """
    if not os.path.isdir(property_dir):
        logger.warning(f"Property directory does not exist: {property_dir}")
        return []
    
    image_extensions = ['.jpg', '.jpeg', '.png']
    images = []
    
    # Get all image paths
    image_paths = []
    for filename in os.listdir(property_dir):
        if any(filename.lower().endswith(ext) for ext in image_extensions):
            image_paths.append(os.path.join(property_dir, filename))
    
    # Load and preprocess each image
    for image_path in image_paths:
        img_array = load_and_preprocess_image(image_path, target_size)
        if img_array is not None:
            images.append(img_array)
    
    if not images:
        logger.warning(f"No valid images found in directory: {property_dir}")
    
    return images

def generate_property_triplets(
    property_data: Dict[str, List[str]],
    num_triplets: int = 1000,
    target_size: Tuple[int, int] = (224, 224),
    augment_probability: float = 0.5,
    max_images_per_property: int = 100,  # Increased default
    output_dir: Optional[str] = None,
    prefix: str = "property",
    augmentations_per_image: int = 3,  # New parameter for multiple augmentations
    low_memory_mode: bool = True,  # Enable memory-saving features
    memory_limit_percentage: float = 80.0  # Percentage of system memory to use before saving
) -> List[Dict]:
    """
    Generate triplets at the property level for Siamese network training with memory optimization.
    Each triplet consists of:
        - Anchor: A property's original images (ALL original images)
        - Positive: The same property's images with multiple augmentations per original
        - Negative: A different property's images (ALL images from negative property)
    
    Args:
        property_data: Dictionary mapping property UIDs to lists of image paths
        num_triplets: Number of triplets to generate
        target_size: Target size for resizing images
        augment_probability: Probability of applying augmentation to negative examples
        max_images_per_property: Maximum images (only used for memory constraints)
        output_dir: Directory to save triplets
        prefix: Prefix for additional properties in manifest
        augmentations_per_image: Number of augmented versions to create for each image
        low_memory_mode: Process one image at a time to minimize memory usage
        memory_limit_percentage: Percentage of system memory to use before clearing cache
        
    Returns:
        List of property triplet dictionaries
    """
    if len(property_data) < 2:
        logger.error("Need at least 2 different properties to generate triplets")
        return []
    
    # Import psutil for memory monitoring if available
    try:
        import psutil
        have_psutil = True
        process = psutil.Process(os.getpid())
        logger.info(f"Memory monitoring enabled, limit set to {memory_limit_percentage}% of system memory")
    except ImportError:
        have_psutil = False
        logger.warning("psutil not available, memory monitoring disabled")
    
    # Create augmentation pipeline with different transformations
    augmentation_types = [
        # Color transforms
        A.Compose([
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=1.0),
            A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.7),
        ], p=1.0),
        
        # Geometric transforms
        A.Compose([
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=25, p=1.0),
            A.HorizontalFlip(p=0.5),
        ], p=1.0),
        
        # Quality/noise transforms
        A.Compose([
            A.GaussNoise(var_limit=(10, 50), p=0.7),
            A.GaussianBlur(blur_limit=(3, 7), p=0.7),
        ], p=1.0),
        
        # Combined transforms
        A.Compose([
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.7),
            A.HorizontalFlip(p=0.5),
            A.MotionBlur(blur_limit=7, p=0.5),
        ], p=1.0)
    ]
    
    # Default augmentation pipeline for any additional augmentations needed
    default_augment_pipeline = create_augmentation_pipeline()
    
    # Ensure output directory exists if specified
    if output_dir:
        ensure_dir_exists(output_dir)
        # Create temp dir for low memory mode
        if low_memory_mode:
            temp_dir = os.path.join(output_dir, "_temp_images")
            ensure_dir_exists(temp_dir)
    
    property_triplets = []
    property_uids = list(property_data.keys())
    
    # Progress bar
    pbar = tqdm(total=num_triplets, desc="Generating property triplets")
    
    def check_memory_usage():
        """Check memory usage and clear cache if needed"""
        if have_psutil:
            # Get memory info
            mem_info = psutil.virtual_memory()
            process_memory = process.memory_info().rss / (1024 * 1024)  # MB
            
            # Log memory usage
            logger.debug(f"Memory usage: Process {process_memory:.1f}MB, System {mem_info.percent:.1f}%")
            
            # Clear cache if memory usage is too high
            if mem_info.percent > memory_limit_percentage:
                logger.warning(f"Memory usage high ({mem_info.percent:.1f}%), clearing cache")
                clear_image_cache()
                gc.collect()
                return True
        return False
    
    for i in range(min(num_triplets, len(property_uids))):
        try:
            # Monitor memory regularly
            check_memory_usage()
            
            # Select anchor property
            anchor_uid = property_uids[i % len(property_uids)]
            
            # Select a different property for the negative
            negative_candidates = [uid for uid in property_uids if uid != anchor_uid]
            if not negative_candidates:
                logger.warning(f"No different properties available for negative samples")
                continue
            
            negative_uid = random.choice(negative_candidates)
            
            # Get ALL image paths (no sampling/limiting)
            anchor_paths = property_data[anchor_uid]
            negative_paths = property_data[negative_uid]
            
            # Skip if either property doesn't have enough images
            if len(anchor_paths) < 1 or len(negative_paths) < 1:
                logger.warning(f"Skipping triplet due to insufficient images for properties {anchor_uid} or {negative_uid}")
                continue
            
            logger.info(f"Processing property {anchor_uid} with {len(anchor_paths)} images and negative {negative_uid} with {len(negative_paths)} images")
            
            # Create triplet property folder
            if output_dir:
                property_dir = os.path.join(output_dir, anchor_uid)
                anchor_dir = os.path.join(property_dir, "anchor")
                positive_dir = os.path.join(property_dir, "positive")
                negative_dir = os.path.join(property_dir, "negative")
                
                # Create the directories
                os.makedirs(anchor_dir, exist_ok=True)
                os.makedirs(positive_dir, exist_ok=True)
                os.makedirs(negative_dir, exist_ok=True)
            
            # In low memory mode, we process and save each image one at a time
            if low_memory_mode and output_dir:
                # Process anchor images one by one
                anchor_count = 0
                for path_idx, path in enumerate(anchor_paths):
                    img = load_and_preprocess_image(path, target_size)
                    if img is None:
                        continue
                    
                    # Save anchor image directly to disk
                    img_path = os.path.join(anchor_dir, f"image_{path_idx:03d}.jpg")
                    img_uint8 = (img * 255).astype(np.uint8)
                    Image.fromarray(img_uint8).save(img_path, format='JPEG', quality=95)
                    anchor_count += 1
                    
                    # Create augmented versions and save directly
                    for aug_idx, augmentation in enumerate(augmentation_types):
                        try:
                            img_copy = img.copy()
                            result = augmentation(image=img_copy)
                            augmented = result["image"]
                            
                            # Save directly to positive folder
                            aug_path = os.path.join(positive_dir, f"image_{path_idx:03d}_aug_{aug_idx:02d}.jpg")
                            aug_uint8 = (augmented * 255).astype(np.uint8)
                            Image.fromarray(aug_uint8).save(aug_path, format='JPEG', quality=95)
                        except Exception as e:
                            logger.error(f"Error applying augmentation {aug_idx} to image {path_idx}: {e}")
                    
                    # Add any additional augmentations if needed
                    remaining = augmentations_per_image - len(augmentation_types)
                    for aug_idx in range(remaining):
                        try:
                            img_copy = img.copy()
                            augmented = apply_augmentation(img_copy, default_augment_pipeline)
                            
                            # Save directly
                            aug_path = os.path.join(positive_dir, f"image_{path_idx:03d}_aug_extra_{aug_idx:02d}.jpg")
                            aug_uint8 = (augmented * 255).astype(np.uint8)
                            Image.fromarray(aug_uint8).save(aug_path, format='JPEG', quality=95)
                        except Exception as e:
                            logger.error(f"Error applying extra augmentation to image {path_idx}: {e}")
                    
                    # Clear memory frequently
                    del img
                    if path_idx % 5 == 0:
                        check_memory_usage()
                
                # Process negative images one by one
                negative_count = 0
                for path_idx, path in enumerate(negative_paths):
                    img = load_and_preprocess_image(path, target_size)
                    if img is None:
                        continue
                    
                    # Optionally augment negative
                    if random.random() < augment_probability:
                        img = apply_augmentation(img, default_augment_pipeline)
                    
                    # Save negative image directly to disk
                    img_path = os.path.join(negative_dir, f"image_{path_idx:03d}.jpg")
                    img_uint8 = (img * 255).astype(np.uint8)
                    Image.fromarray(img_uint8).save(img_path, format='JPEG', quality=95)
                    negative_count += 1
                    
                    # Clear memory frequently
                    del img
                    if path_idx % 5 == 0:
                        check_memory_usage()
                
                # Create metadata
                meta_path = os.path.join(property_dir, "metadata.json")
                with open(meta_path, 'w') as f:
                    json.dump({
                        "anchor_uid": anchor_uid,
                        "negative_uid": negative_uid,
                        "anchor_count": anchor_count,
                        "positive_count": anchor_count * min(augmentations_per_image, len(augmentation_types) + remaining),
                        "negative_count": negative_count,
                        "creation_date": time.strftime("%Y-%m-%d %H:%M:%S")
                    }, f, indent=2)
                
                # Add to triplets list (with minimal info)
                property_triplets.append({
                    "anchor_uid": anchor_uid,
                    "negative_uid": negative_uid,
                    "property_dir": property_dir,
                    "anchor_dir": anchor_dir,
                    "positive_dir": positive_dir,
                    "negative_dir": negative_dir
                })
                
            else:
                # Regular memory mode - process all images in memory
                # Process the original images for the anchor
                anchor_images = []
                for path in anchor_paths:
                    img = load_and_preprocess_image(path, target_size)
                    if img is not None:
                        anchor_images.append(img)
                
                # Skip if no valid anchor images
                if not anchor_images:
                    logger.warning(f"No valid anchor images for property {anchor_uid}")
                    continue
                
                # Create multiple augmented versions of EACH anchor image
                positive_images = []
                
                # Apply each augmentation type to each image
                for img in anchor_images:
                    # Apply each specific augmentation type
                    for aug_idx, augmentation in enumerate(augmentation_types):
                        try:
                            # Make a copy of the image to avoid modifying the original
                            img_copy = img.copy()
                            
                            # Apply this specific augmentation
                            result = augmentation(image=img_copy)
                            augmented = result["image"]
                            
                            # Log the augmentation
                            logger.debug(f"Applied augmentation type {aug_idx+1} to image")
                            
                            # Add to positive images
                            positive_images.append(augmented)
                        except Exception as e:
                            logger.error(f"Error applying augmentation type {aug_idx+1}: {str(e)}")
                    
                    # Add additional augmentations if needed
                    remaining = max(0, augmentations_per_image - len(augmentation_types))
                    for _ in range(remaining):
                        try:
                            img_copy = img.copy()
                            augmented = apply_augmentation(img_copy, default_augment_pipeline)
                            positive_images.append(augmented)
                        except Exception as e:
                            logger.error(f"Error applying additional augmentation: {str(e)}")
                
                # Process ALL images for the negative property
                negative_images = []
                for path in negative_paths:
                    img = load_and_preprocess_image(path, target_size)
                    if img is not None:
                        # Optionally apply augmentation to negative samples
                        if random.random() < augment_probability:
                            img = apply_augmentation(img, default_augment_pipeline)
                        negative_images.append(img)
                
                # Skip if no valid negative images
                if not negative_images:
                    logger.warning(f"No valid negative images for property {negative_uid}")
                    continue
                
                # Log summary of images
                logger.info(f"Created triplet with {len(anchor_images)} anchor images, " +
                           f"{len(positive_images)} positive augmentations, and " +
                           f"{len(negative_images)} negative images")
                
                # Create the triplet
                triplet = {
                    "anchor_uid": anchor_uid,
                    "negative_uid": negative_uid,
                    "anchor_images": anchor_images,
                    "positive_images": positive_images,
                    "negative_images": negative_images
                }
                
                # Save the triplet if an output directory is specified
                if output_dir:
                    save_property_triplet(triplet, output_dir)
                
                property_triplets.append(triplet)
            
            pbar.update(1)
            
            # Occasionally clear the cache to prevent memory issues
            if (i + 1) % 5 == 0:  # More frequent cache clearing (every 5 instead of 10)
                clear_image_cache()
                gc.collect()
                
        except Exception as e:
            logger.error(f"Error generating property triplet: {str(e)}")
            continue
    
    pbar.close()
    
    logger.info(f"Generated {len(property_triplets)} property triplets")
    
    # Save triplet info
    if output_dir:
        triplet_info = {
            "num_triplets": len(property_triplets),
            "property_uids": list(set([t["anchor_uid"] for t in property_triplets])),
            "properties_used": len(set([t["anchor_uid"] for t in property_triplets])),
            "target_size": target_size,
            "augment_probability": augment_probability,
            "augmentations_per_image": augmentations_per_image,
            "max_images_per_property": max_images_per_property,
            "augmentation_types": len(augmentation_types),
            "low_memory_mode": low_memory_mode
        }
        
        with open(os.path.join(output_dir, f"{prefix}_triplets_info.json"), "w") as f:
            json.dump(triplet_info, f, indent=2)
        
        # Clean up temp directory if it exists
        if low_memory_mode:
            temp_dir = os.path.join(output_dir, "_temp_images")
            if os.path.exists(temp_dir):
                try:
                    shutil.rmtree(temp_dir)
                except Exception as e:
                    logger.error(f"Error removing temp directory: {str(e)}")
    
    return property_triplets

def save_property_triplet(triplet: Dict, output_dir: str) -> bool:
    """
    Save a property triplet to disk with structure:
    output_dir/
        {anchor_uid}/
            anchor/
                image_001.jpg
                image_002.jpg
                ...
            positive/
                image_001.jpg
                image_002.jpg
                ...
            negative/
                image_001.jpg
                image_002.jpg
                ...
            metadata.json
    
    Args:
        triplet: Property triplet dictionary
        output_dir: Base directory for saving triplets
        
    Returns:
        True if saving was successful, False otherwise
    """
    try:
        anchor_uid = triplet["anchor_uid"]
        property_dir = os.path.join(output_dir, anchor_uid)
        
        # Create subdirectories
        anchor_dir = os.path.join(property_dir, "anchor")
        positive_dir = os.path.join(property_dir, "positive")
        negative_dir = os.path.join(property_dir, "negative")
        
        os.makedirs(anchor_dir, exist_ok=True)
        os.makedirs(positive_dir, exist_ok=True)
        os.makedirs(negative_dir, exist_ok=True)
        
        # Save anchor images
        for i, img_array in enumerate(triplet["anchor_images"]):
            img_path = os.path.join(anchor_dir, f"image_{i:03d}.jpg")
            img_uint8 = (img_array * 255).astype(np.uint8)
            Image.fromarray(img_uint8).save(img_path, format='JPEG', quality=95)
        
        # Save positive images (augmented versions of anchor)
        for i, img_array in enumerate(triplet["positive_images"]):
            img_path = os.path.join(positive_dir, f"image_{i:03d}.jpg")
            img_uint8 = (img_array * 255).astype(np.uint8)
            Image.fromarray(img_uint8).save(img_path, format='JPEG', quality=95)
        
        # Save negative images
        for i, img_array in enumerate(triplet["negative_images"]):
            img_path = os.path.join(negative_dir, f"image_{i:03d}.jpg")
            img_uint8 = (img_array * 255).astype(np.uint8)
            Image.fromarray(img_uint8).save(img_path, format='JPEG', quality=95)
        
        # Save metadata
        meta_path = os.path.join(property_dir, "metadata.json")
        with open(meta_path, 'w') as f:
            json.dump({
                "anchor_uid": triplet["anchor_uid"],
                "negative_uid": triplet["negative_uid"],
                "anchor_count": len(triplet["anchor_images"]),
                "positive_count": len(triplet["positive_images"]),
                "negative_count": len(triplet["negative_images"]),
                "creation_date": time.strftime("%Y-%m-%d %H:%M:%S")
            }, f, indent=2)
        
        return True
    except Exception as e:
        logger.error(f"Error saving property triplet for {triplet.get('anchor_uid', 'unknown')}: {str(e)}")
        # Clean up failed directory
        if os.path.exists(property_dir):
            try:
                shutil.rmtree(property_dir)
            except Exception:
                pass
        return False

def load_property_triplets_from_directory(triplets_dir: str, max_triplets: Optional[int] = None) -> List[Dict]:
    """
    Load property triplets from a directory with structure:
    triplets_dir/
        {property_uid}/
            anchor/
                image_001.jpg
                ...
            positive/
                image_001.jpg
                ...
            negative/
                image_001.jpg
                ...
            metadata.json
    
    Args:
        triplets_dir: Directory containing property triplets
        max_triplets: Maximum number of triplets to load
        
    Returns:
        List of property triplet dictionaries
    """
    if not os.path.isdir(triplets_dir):
        logger.error(f"Triplets directory does not exist: {triplets_dir}")
        return []
    
    # First check for a manifest
    manifest_path = os.path.join(triplets_dir, "property_triplets_manifest.json")
    if os.path.exists(manifest_path):
        try:
            with open(manifest_path, 'r') as f:
                manifest = json.load(f)
            
            triplets = manifest.get("triplets", [])
            logger.info(f"Loaded {len(triplets)} property triplets from manifest")
            
            if max_triplets and len(triplets) > max_triplets:
                triplets = triplets[:max_triplets]
                logger.info(f"Limited to {max_triplets} triplets")
            
            return triplets
        except Exception as e:
            logger.warning(f"Error loading triplets manifest: {str(e)}")
    
    # If no manifest, scan the directory
    property_uids = []
    for item in os.listdir(triplets_dir):
        item_path = os.path.join(triplets_dir, item)
        if os.path.isdir(item_path):
            # Check if this directory has the required subdirectories and metadata
            has_anchor = os.path.isdir(os.path.join(item_path, "anchor"))
            has_positive = os.path.isdir(os.path.join(item_path, "positive"))
            has_negative = os.path.isdir(os.path.join(item_path, "negative"))
            has_metadata = os.path.exists(os.path.join(item_path, "metadata.json"))
            
            if has_anchor and has_positive and has_negative and has_metadata:
                property_uids.append(item)
    
    logger.info(f"Found {len(property_uids)} property triplets in directory")
    
    if max_triplets and len(property_uids) > max_triplets:
        property_uids = property_uids[:max_triplets]
    
    # Load each property triplet
    triplets = []
    for uid in tqdm(property_uids, desc="Loading property triplets"):
        try:
            property_dir = os.path.join(triplets_dir, uid)
            
            # Load metadata
            meta_path = os.path.join(property_dir, "metadata.json")
            with open(meta_path, 'r') as f:
                metadata = json.load(f)
            
            # Add directory info
            metadata["property_dir"] = property_dir
            metadata["anchor_dir"] = os.path.join(property_dir, "anchor")
            metadata["positive_dir"] = os.path.join(property_dir, "positive")
            metadata["negative_dir"] = os.path.join(property_dir, "negative")
            
            triplets.append(metadata)
        except Exception as e:
            logger.warning(f"Error loading property triplet {uid}: {str(e)}")
    
    # Create and save a manifest for future use
    if triplets:
        try:
            manifest = {
                "triplets": triplets,
                "total_triplets": len(triplets),
                "creation_date": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            
            with open(manifest_path, 'w') as f:
                json.dump(manifest, f, indent=2)
            
            logger.info(f"Created property triplets manifest with {len(triplets)} entries")
        except Exception as e:
            logger.warning(f"Error creating triplets manifest: {str(e)}")
    
    return triplets

class PropertyTripletDataset(torch.utils.data.Dataset):
    """
    Dataset for property-level triplets.
    Each property consists of multiple images organized in anchor, positive, and negative sets.
    """
    
    def __init__(self, 
                 triplets_dir: str,
                 transform=None,
                 max_triplets: Optional[int] = None,
                 target_size: Tuple[int, int] = (224, 224)):
        """
        Initialize the PropertyTripletDataset.
        
        Args:
            triplets_dir: Directory containing property triplets
            transform: Optional transform to be applied to images
            max_triplets: Maximum number of triplets to load
            target_size: Target size for resizing images
        """
        self.triplets_dir = triplets_dir
        self.transform = transform if transform else self._get_default_transform()
        self.target_size = target_size
        
        # Load triplets from directory
        self.triplets = load_property_triplets_from_directory(triplets_dir, max_triplets)
        
        if not self.triplets:
            raise ValueError(f"No valid property triplets found in {triplets_dir}")
        
        logger.info(f"Initialized PropertyTripletDataset with {len(self.triplets)} triplets")
    
    def _get_default_transform(self):
        """Get default transform"""
        return transforms.Compose([
            transforms.Resize(self.target_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def __len__(self):
        return len(self.triplets)
    
    def __getitem__(self, idx):
        """
        Get a property triplet.
        
        Returns:
            Tuple containing:
                - List of anchor property images (as tensors)
                - List of positive property images (as tensors)
                - List of negative property images (as tensors)
                - Anchor property UID (string)
                - Negative property UID (string)
        """
        triplet = self.triplets[idx]
        
        # Load images from each directory
        anchor_dir = triplet["anchor_dir"]
        positive_dir = triplet["positive_dir"]
        negative_dir = triplet["negative_dir"]
        
        # Get image paths
        anchor_paths = [os.path.join(anchor_dir, f) for f in os.listdir(anchor_dir) 
                       if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        positive_paths = [os.path.join(positive_dir, f) for f in os.listdir(positive_dir) 
                         if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        negative_paths = [os.path.join(negative_dir, f) for f in os.listdir(negative_dir) 
                         if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        # Load and process images
        anchor_tensors = self._load_and_process_images(anchor_paths)
        positive_tensors = self._load_and_process_images(positive_paths)
        negative_tensors = self._load_and_process_images(negative_paths)
        
        return (
            anchor_tensors,
            positive_tensors,
            negative_tensors,
            triplet["anchor_uid"],
            triplet["negative_uid"]
        )
    
    def _load_and_process_images(self, image_paths):
        """Load and process images, returning a stacked tensor"""
        tensors = []
        
        for path in image_paths:
            try:
                img = Image.open(path).convert('RGB')
                if self.transform:
                    img = self.transform(img)
                tensors.append(img)
            except Exception as e:
                logger.warning(f"Error loading image {path}: {str(e)}")
        
        # If no images loaded, create a dummy
        if not tensors:
            logger.warning(f"No valid images loaded, creating dummy")
            tensors.append(torch.zeros((3, self.target_size[0], self.target_size[1])))
        
        return torch.stack(tensors)

def create_property_triplet_dataloader(
    triplets_source,
    batch_size: int = 8,
    is_directory: bool = False,
    max_triplets: Optional[int] = None,
    num_workers: int = 4,
    shuffle: bool = True
) -> torch.utils.data.DataLoader:
    """
    Create a dataloader for property-level triplets.
    
    Args:
        triplets_source: Either a list of property triplets or a directory containing saved triplets
        batch_size: Batch size for the dataloader
        is_directory: Whether triplets_source is a directory
        max_triplets: Maximum number of triplets to load (None for all)
        num_workers: Number of worker processes for loading data
        shuffle: Whether to shuffle the data
        
    Returns:
        DataLoader for property triplets
    """
    dataset = PropertyTripletDataset(
        triplets_dir=triplets_source,
        is_directory=is_directory,
        max_triplets=max_triplets
    )
    
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )

def generate_property_level_triplets():
    """
    Direct command-line script to generate property-level triplets with UID folder names.
    This bypasses the old triplet generation completely.
    """
    print("\n===== PROPERTY-LEVEL TRIPLET GENERATION =====")
    print("This script will generate property-level triplets with folders named after property UIDs.")
    print("Each property folder will have three subfolders:")
    print("  - anchor/: Original property images")
    print("  - positive/: Augmented versions of the same property")
    print("  - negative/: Images from a different property")
    
    # Get input directory
    default_input_dir = "/workspace/data/triplets"
    input_dir = input(f"Enter directory containing data_paths.json [default: {default_input_dir}]: ").strip() or default_input_dir
    
    data_paths_file = os.path.join(input_dir, "data_paths.json")
    
    if not os.path.exists(data_paths_file):
        print(f"Data paths file not found: {data_paths_file}")
        alt_file = os.path.join(input_dir, "property_data.json")
        if os.path.exists(alt_file):
            print(f"Found property_data.json instead, using that")
            data_paths_file = alt_file
        else:
            print("Please run data_gathering.py first or provide the correct directory.")
            return
    
    # Load the data paths
    with open(data_paths_file, 'r') as f:
        data_paths = json.load(f)
    
    # Support both old and new format
    if "train_images" in data_paths and "train_properties" not in data_paths:
        train_images = data_paths["train_images"]
        test_images = data_paths["test_images"]
    else:
        train_images = data_paths["train_properties"]
        test_images = data_paths["test_properties"]
    
    print(f"Loaded data with {sum(len(v) for v in train_images.values())} training images across {len(train_images)} properties")
    print(f"and {sum(len(v) for v in test_images.values())} testing images across {len(test_images)} properties.")
    
    # Ensure we have multiple property types
    train_images = ensure_multiple_property_types(train_images)
    
    # Set up triplet parameters
    num_triplets_str = input("How many property triplets to generate? [default: 100]: ").strip()
    num_triplets = int(num_triplets_str) if num_triplets_str.isdigit() else 100
    
    max_images_str = input("Maximum images per property? [default: 10]: ").strip()
    max_images = int(max_images_str) if max_images_str.isdigit() else 10
    
    augment_probability_str = input("Augmentation probability for positive examples? [default: 0.7]: ").strip()
    augment_probability = float(augment_probability_str) if augment_probability_str else 0.7
    
    # Create output directory
    output_dir = os.path.join(input_dir, "property_triplets")
    ensure_dir_exists(output_dir)
    
    # Clear existing triplets if any
    clear_existing = input("Clear existing triplets in the output directory? (y/n) [default: y]: ").strip().lower() != 'n'
    if clear_existing:
        print(f"Clearing existing triplets in {output_dir}...")
        for item in os.listdir(output_dir):
            item_path = os.path.join(output_dir, item)
            if os.path.isdir(item_path):
                try:
                    shutil.rmtree(item_path)
                except Exception as e:
                    print(f"Failed to remove {item_path}: {e}")
    
    # Generate property triplets
    print(f"\nGenerating property triplets, this may take a while...")
    property_triplets = generate_property_triplets(
        property_data=train_images,
        num_triplets=num_triplets,
        max_images_per_property=max_images,
        augment_probability=augment_probability,
        output_dir=output_dir
    )
    
    print(f"\nGenerated {len(property_triplets)} property triplets in {output_dir}")
    print("Each property folder contains:")
    print("  - anchor/ folder: Original property images")
    print("  - positive/ folder: Augmented versions of the same property")
    print("  - negative/ folder: Different property images")
    
    # Save test images paths for later evaluation
    test_info = {
        "test_images": test_images,
        "num_test_images": sum(len(v) for v in test_images.values())
    }
    
    test_info_path = os.path.join(output_dir, "test_images_info.json")
    with open(test_info_path, 'w') as f:
        json.dump(test_info, f, indent=2)
    
    print(f"Saved test images info to {test_info_path}")
    print("\nProperty triplet generation complete!")
    print("You can now use these property triplets for training.")

if __name__ == "__main__":
    import sys
    
    # Check for direct command to generate property triplets
    if len(sys.argv) > 1 and sys.argv[1] == "--property-triplets":
        generate_property_level_triplets()
        sys.exit(0)
    
    # Regular execution
    # Check if we have data paths from the data gathering step
    default_input_dir = "/workspace/data/triplets"
    input_dir = input(f"Enter directory containing data_paths.json [default: {default_input_dir}]: ").strip() or default_input_dir
    
    data_paths_file = os.path.join(input_dir, "data_paths.json")
    
    if not os.path.exists(data_paths_file):
        print(f"Data paths file not found: {data_paths_file}")
        print("Please run data_gathering.py first or provide the correct directory.")
        exit(1)
    
    # Load the data paths
    with open(data_paths_file, 'r') as f:
        data_paths = json.load(f)
    
    train_images = data_paths["train_images"]
    test_images = data_paths["test_images"]
    num_triplets = data_paths.get("num_triplets", 10000)
    
    print(f"Loaded data with {sum(len(v) for v in train_images.values())} training images and "
          f"{sum(len(v) for v in test_images.values())} testing images.")
    
    # Ensure we have multiple property types for triplet generation
    train_images = ensure_multiple_property_types(train_images)
    
    # Ask user which type of triplet generation to use
    print("\nWhich type of triplet generation would you like to use?")
    print("[1] Property-level triplets (recommended - entire property sets with UID folder names)")
    print("[2] Legacy image-level triplets (individual images)")
    
    triplet_type = ''
    while triplet_type not in ['1', '2']:
        triplet_type = input("Enter your choice (1 or 2) [default: 1]: ").strip() or '1'
    
    # Get common settings
    max_images_str = input("Maximum images per property? [default: 10]: ").strip()
    max_images_per_property = int(max_images_str) if max_images_str.isdigit() else 10
    
    if triplet_type == '1':
        # Property-level triplets (new approach)
        print("\n=== Generating Property-Level Triplets ===")
        
        # Create output directory
        property_triplets_dir = os.path.join(input_dir, "property_triplets")
        ensure_dir_exists(property_triplets_dir)
        
        # Generate property triplets
        print(f"\nGenerating property triplets from {len(train_images)} property types...")
        property_triplets = generate_property_triplets(
            property_data=train_images,
            num_triplets=num_triplets,
            max_images_per_property=max_images_per_property,
            augment_probability=0.7,
            output_dir=property_triplets_dir
        )
        
        print(f"\nGenerated {len(property_triplets)} property triplets in {property_triplets_dir}")
        print("Each property folder contains:")
        print("  - anchor/ folder: Original property images")
        print("  - positive/ folder: Augmented versions of the same property")
        print("  - negative/ folder: Different property images")
        print("\nYou can now use these with PropertyTripletDataset for training.")
        
        # Save test images paths for later evaluation
        test_info = {
            "test_images": test_images,
            "num_test_images": sum(len(v) for v in test_images.values())
        }
        
        test_info_path = os.path.join(property_triplets_dir, "test_images_info.json")
        with open(test_info_path, 'w') as f:
            json.dump(test_info, f, indent=2)
        
        print(f"Saved test images info to {test_info_path}")
        print(f"\nData preprocessing complete!")
    else:
        # Legacy image-level triplets (old approach)
        print("\n=== Generating Legacy Image-Level Triplets ===")
        print("Warning: This approach uses individual images rather than property sets.")
        
        # Generate triplets from training data
        print("\nGenerating training triplets...")
        triplets = generate_triplets_with_augmentation(
            train_images, 
            num_triplets=num_triplets,
            augment_probability=0.7
        )
        
        # Save triplets locally
        if triplets:
            print(f"\nSaving {len(triplets)} triplets to {input_dir}...")
            save_triplets(triplets, input_dir)
            
            # Create PyTorch dataset
            dataset = TripletDataset(input_dir)
            print(f"Created PyTorch dataset with {len(dataset)} triplets")
            
            # Save test images paths for later evaluation
            test_info = {
                "test_images": test_images,
                "num_test_images": sum(len(v) for v in test_images.values())
            }
            
            test_info_path = os.path.join(input_dir, "test_images_info.json")
            with open(test_info_path, 'w') as f:
                json.dump(test_info, f, indent=2)
            
            print(f"Saved test images info to {test_info_path}")
            print(f"\nData preprocessing complete! You can now train your model using the generated triplets.")
        else:
            print("Failed to generate triplets. Check your input data.") 