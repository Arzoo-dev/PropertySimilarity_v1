"""
Training script for the Siamese network on RunPods.
This script handles data preparation, model training, and evaluation.
Enhanced for property-level similarity learning.
"""

import os
import sys
import argparse
import logging
import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import time
import json
import matplotlib.pyplot as plt
from tqdm import tqdm
import traceback
import gc
from datetime import datetime
from PIL import Image
import random
import itertools
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
import shutil
from pathlib import Path
import subprocess

# Import custom modules
from data_gathering import (
    fetch_training_and_testing_data,
    fetch_from_premier_brokerage_api,
    fetch_local_images
)
from data_preprocessing import (
    generate_triplets_with_augmentation,
    generate_property_triplets,
    TripletDataset,
    save_triplets
)
from siamese_network import SiameseNetwork
from runpods_utils import ensure_dir_exists, save_json, load_json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("training.log")
    ]
)
logger = logging.getLogger(__name__)

# Add setup_logging to configure logs based on output directory
import os  # ensure os is available

def setup_logging(args):
    """Configure logging to output to console and a file in the output directory."""
    output_dir = args.output_dir or args.model_dir
    ensure_dir_exists(output_dir)
    log_file = os.path.join(output_dir, 'training.log')
    # Remove existing handlers
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    # Create new handlers
    handlers = [logging.StreamHandler(), logging.FileHandler(log_file)]
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train siamese network for property matching')
    
    # Data sources
    data_group = parser.add_argument_group('Data Sources')
    data_group.add_argument('--train_data_dir', type=str, default=None,
                          help='Path to training data directory')
    data_group.add_argument('--api_endpoint', type=str, default=None,
                          help='API endpoint for fetching training data')
    data_group.add_argument('--triplet_dir', type=str, default=None,
                          help='Directory containing pre-structured triplets (anchor/positive/negative)')
    data_group.add_argument('--api_token', type=str, default=None,
                          help='Authentication token for API')
    data_group.add_argument('--property_types', type=str, nargs='+', default=None,
                          help='Property types to fetch from API')
    data_group.add_argument('--images_per_type', type=int, default=100,
                          help='Number of images to fetch per property type')
    data_group.add_argument('--api_cache_dir', type=str, default='/workspace/data/api_cache',
                          help='Directory to cache API results')
    data_group.add_argument('--api_workers', type=int, default=8,
                          help='Number of workers for API fetching')
    
    # Validation data
    val_group = parser.add_argument_group('Validation Data')
    val_group.add_argument('--val_data_dir', type=str, default=None,
                         help='Path to validation data directory')
    val_group.add_argument('--api_val_property_types', type=str, nargs='+', default=None,
                         help='Property types to fetch from API for validation')
    
    # Output settings
    output_group = parser.add_argument_group('Output Settings')
    output_group.add_argument('--model_dir', type=str, default='/workspace/',
                            help='Directory to save the trained model')
    output_group.add_argument('--output_dir', type=str, default=None,
                            help='Directory to save output files (defaults to model_dir)')
    
    # Training parameters
    train_group = parser.add_argument_group('Training Parameters')
    train_group.add_argument('--num_triplets', type=int, default=10000,
                           help='Number of triplets to generate for training')
    train_group.add_argument('--val_triplets', type=int, default=1000,
                           help='Number of triplets to generate for validation')
    train_group.add_argument('--batch_size', type=int, default=32,
                           help='Training batch size')
    train_group.add_argument('--epochs', type=int, default=50,
                           help='Number of training epochs')
    train_group.add_argument('--learning_rate', type=float, default=0.001,
                           help='Learning rate')
    train_group.add_argument('--scheduler', type=str, default='one_cycle',
                           choices=['step', 'plateau', 'cosine', 'one_cycle'],
                           help='Learning rate scheduler type')
    train_group.add_argument('--embedding_dim', type=int, default=256,
                           help='Embedding dimension')
    train_group.add_argument('--margin', type=float, default=0.2,
                           help='Margin for triplet loss')
    train_group.add_argument('--augment_prob', type=float, default=0.7,
                           help='Probability of applying augmentation')
    train_group.add_argument('--backbone', type=str, default='resnet50',
                           choices=['efficientnet', 'resnet50'],
                           help='Backbone network architecture')
    train_group.add_argument('--property_level', action='store_true', default=True,
                           help='Use property-level matching with multiple images per property')
    train_group.add_argument('--mixed_precision', action='store_true', default=True,
                           help='Use mixed precision training (faster, less memory)')
    train_group.add_argument('--seed', type=int, default=42,
                           help='Random seed for reproducibility')
    
    # Triplet generation
    triplet_group = parser.add_argument_group('Triplet Generation')
    triplet_group.add_argument('--save_triplets', action='store_true',
                             help='Save generated triplets to disk')
    triplet_group.add_argument('--triplets_output_dir', type=str, default=None,
                             help='Directory to save generated triplets')
    
    # Evaluation
    eval_group = parser.add_argument_group('Evaluation')
    eval_group.add_argument('--eval_model', action='store_true',
                          help='Evaluate model on test property pairs')
    eval_group.add_argument('--eval_pairs', type=int, default=500,
                          help='Number of property pairs to evaluate')
    
    # Checkpointing
    checkpoint_group = parser.add_argument_group('Checkpointing')
    checkpoint_group.add_argument('--checkpoint_freq', type=int, default=5,
                                help='Checkpoint frequency (in epochs)')
    checkpoint_group.add_argument('--load_checkpoint', type=str, default=None,
                                help='Load model from checkpoint')
    
    # User interaction
    ui_group = parser.add_argument_group('User Interaction')
    ui_group.add_argument('--no_prompt', action='store_true',
                        help='Do not prompt for user input (use defaults)')
    
    # Advanced settings
    advanced_group = parser.add_argument_group('Advanced Settings')
    advanced_group.add_argument('--val_split', type=float, default=0.2,
                              help='Validation split ratio')
    advanced_group.add_argument('--weight_decay', type=float, default=1e-5,
                              help='Weight decay for optimizer')
    
    # Parse the arguments
    args = parser.parse_args()
    
    # Check for incompatible arguments
    if sum(x is not None for x in [args.train_data_dir, args.api_endpoint, args.triplet_dir]) > 1:
        parser.error("Only one of --train_data_dir, --api_endpoint, or --triplet_dir should be specified")
    
    # Auto-detect device capabilities for mixed precision
    if args.mixed_precision:
        if not torch.cuda.is_available() or not hasattr(torch.cuda, 'amp'):
            logging.warning("Mixed precision requested but not supported; disabling")
            args.mixed_precision = False
        else:
            # For RTX A5000, mixed precision is highly recommended
            gpu_name = torch.cuda.get_device_name(0)
            if 'RTX' in gpu_name or 'A5000' in gpu_name:
                logging.info(f"Detected {gpu_name}: mixed precision enabled for optimal performance")
    
    # Auto-adjust batch size for better performance
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        
        # Default batch size optimization for RTX A5000 (24GB)
        if 'A5000' in gpu_name and args.mixed_precision and args.batch_size == 32:
            # RTX A5000 with mixed precision can handle larger batches
            args.batch_size = 64
            logging.info(f"Optimized batch size for {gpu_name} with mixed precision: {args.batch_size}")
    
    return args

def set_seed(seed):
    """Set random seed for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def clear_memory():
    """Force aggressive memory cleanup"""
    # Clear any Python caches
    gc.collect()
    
    # Clear PyTorch cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    logger.info("Performed memory cleanup")

def plot_training_history(history, save_path):
    """Plot training history and save figure."""
    plt.figure(figsize=(15, 10))
    
    # Plot losses
    plt.subplot(2, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    if 'val_loss' in history and history['val_loss']:
        plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Loss During Training')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot metrics if available
    if 'train_metrics' in history and history['train_metrics']:
        # Plot accuracy
        plt.subplot(2, 2, 2)
        train_acc = [m['accuracy'] for m in history['train_metrics']]
        plt.plot(train_acc, label='Train Accuracy')
        if 'val_metrics' in history and history['val_metrics']:
            val_acc = [m['accuracy'] for m in history['val_metrics']]
            plt.plot(val_acc, label='Val Accuracy')
        plt.title('Accuracy During Training')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
        
        # Plot F1 Score
        plt.subplot(2, 2, 3)
        train_f1 = [m['f1'] for m in history['train_metrics']]
        plt.plot(train_f1, label='Train F1')
        if 'val_metrics' in history and history['val_metrics']:
            val_f1 = [m['f1'] for m in history['val_metrics']]
            plt.plot(val_f1, label='Val F1')
        plt.title('F1 Score During Training')
        plt.xlabel('Epoch')
        plt.ylabel('F1 Score')
        plt.legend()
        plt.grid(True)
        
        # Plot distance gap (how well the model separates similar/dissimilar pairs)
        if history['train_metrics'][0].get('distance_gap') is not None:
            plt.subplot(2, 2, 4)
            train_gap = [m['distance_gap'] for m in history['train_metrics']]
            plt.plot(train_gap, label='Train Distance Gap')
            if 'val_metrics' in history and history['val_metrics']:
                val_gap = [m['distance_gap'] for m in history['val_metrics']]
                plt.plot(val_gap, label='Val Distance Gap')
            plt.title('Distance Gap (Neg - Pos)')
            plt.xlabel('Epoch')
            plt.ylabel('Distance Gap')
            plt.legend()
            plt.grid(True)
        else:
            # Plot learning rate if distance gap not available
            plt.subplot(2, 2, 4)
    plt.plot(history['learning_rate'], label='Learning Rate')
    plt.title('Learning Rate During Training')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    logger.info(f"Training history plot saved to {save_path}")

    # Plot additional metrics in a separate figure
    plt.figure(figsize=(15, 10))
    
    # Plot positive and negative distances
    if 'train_metrics' in history and history['train_metrics']:
        if history['train_metrics'][0].get('avg_pos_distance') is not None:
            plt.subplot(2, 2, 1)
            train_pos_dist = [m['avg_pos_distance'] for m in history['train_metrics']]
            train_neg_dist = [m['avg_neg_distance'] for m in history['train_metrics']]
            plt.plot(train_pos_dist, 'g-', label='Train Pos Distance')
            plt.plot(train_neg_dist, 'r-', label='Train Neg Distance')
            if 'val_metrics' in history and history['val_metrics']:
                val_pos_dist = [m['avg_pos_distance'] for m in history['val_metrics']]
                val_neg_dist = [m['avg_neg_distance'] for m in history['val_metrics']]
                plt.plot(val_pos_dist, 'g--', label='Val Pos Distance')
                plt.plot(val_neg_dist, 'r--', label='Val Neg Distance')
            plt.title('Average Distances')
            plt.xlabel('Epoch')
            plt.ylabel('Distance')
            plt.legend()
            plt.grid(True)
        
        # Plot precision/recall
        plt.subplot(2, 2, 2)
        train_prec = [m['precision'] for m in history['train_metrics']]
        train_rec = [m['recall'] for m in history['train_metrics']]
        plt.plot(train_prec, 'b-', label='Train Precision')
        plt.plot(train_rec, 'g-', label='Train Recall')
        if 'val_metrics' in history and history['val_metrics']:
            val_prec = [m['precision'] for m in history['val_metrics']]
            val_rec = [m['recall'] for m in history['val_metrics']]
            plt.plot(val_prec, 'b--', label='Val Precision')
            plt.plot(val_rec, 'g--', label='Val Recall')
        plt.title('Precision & Recall')
        plt.xlabel('Epoch')
        plt.ylabel('Score')
        plt.legend()
        plt.grid(True)
        
        # Plot threshold
        if history['train_metrics'][0].get('threshold') is not None:
            plt.subplot(2, 2, 3)
            train_threshold = [m['threshold'] for m in history['train_metrics']]
            plt.plot(train_threshold, label='Train Threshold')
            if 'val_metrics' in history and history['val_metrics']:
                val_threshold = [m['threshold'] for m in history['val_metrics']]
                plt.plot(val_threshold, label='Val Threshold')
            plt.title('Optimal Distance Threshold')
            plt.xlabel('Epoch')
            plt.ylabel('Threshold')
            plt.legend()
            plt.grid(True)
        
        # Plot learning rate
        plt.subplot(2, 2, 4)
        plt.plot(history['learning_rate'], label='Learning Rate')
        plt.title('Learning Rate Schedule')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.yscale('log')  # Log scale to better visualize changes
        plt.grid(True)
    
    plt.tight_layout()
    metrics_path = os.path.splitext(save_path)[0] + "_detailed.png"
    plt.savefig(metrics_path)
    logger.info(f"Detailed metrics plot saved to {metrics_path}")
    
    # Close figures to free memory
    plt.close('all')

def save_training_metadata(training_result, metadata_path, args):
    """Save training metadata to a JSON file."""
    # Get final metrics
    final_train_metrics = training_result['train_metrics'][-1] if training_result.get('train_metrics') else {}
    final_val_metrics = training_result['val_metrics'][-1] if training_result.get('val_metrics') else {}
    
    # Calculate training duration
    training_time = time.time() - training_result.get('start_time', time.time())
    
    metadata = {
        'training_date': time.strftime("%Y-%m-%d %H:%M:%S"),
        'final_train_loss': float(training_result['train_loss'][-1]) if training_result['train_loss'] else None,
        'final_val_loss': float(training_result['val_loss'][-1]) if training_result.get('val_loss') else None,
        'num_epochs': len(training_result['train_loss']),
        'early_stopping_triggered': len(training_result['train_loss']) < args.epochs,
        'final_learning_rate': float(training_result['learning_rate'][-1]) if training_result['learning_rate'] else None,
        'final_train_metrics': final_train_metrics,
        'final_val_metrics': final_val_metrics,
        'training_time_seconds': training_time,
        'training_time_formatted': f"{training_time//3600:.0f}h {(training_time%3600)//60:.0f}m {training_time%60:.0f}s",
        
        # Training configuration
        'config': {
            'embedding_dim': args.embedding_dim,
            'backbone': args.backbone,
            'property_level': args.property_level,
            'margin': args.margin,
            'learning_rate': args.learning_rate,
            'scheduler': args.scheduler,
            'batch_size': args.batch_size,
            'epochs': args.epochs,
            'mixed_precision': args.mixed_precision,
            'weight_decay': args.weight_decay,
            'seed': args.seed
        }
    }
    
    save_json(metadata, metadata_path)
    logger.info(f"Training metadata saved to {metadata_path}")

def evaluate_on_test_pairs(model, test_data_dir, num_pairs=500):
    """
    Evaluate model on pairs of images from test directory.
    """
    # Collect property images
    property_images = collect_property_images(test_data_dir)
    if not property_images:
        logger.error(f"No property images found in {test_data_dir}")
        return None
    
    # Create similar and dissimilar pairs
    similar_pairs = []
    dissimilar_pairs = []
    
    property_types = list(property_images.keys())
    
    # Create similar pairs (same property type)
    for prop_type in property_types:
        if len(property_images[prop_type]) >= 2:
            for _ in range(num_pairs // len(property_types) + 1):
                if len(similar_pairs) >= num_pairs // 2:
                    break
                img1, img2 = np.random.choice(property_images[prop_type], 2, replace=False)
                similar_pairs.append((img1, img2, 1))  # 1 = similar
    
    # Create dissimilar pairs (different property types)
    for _ in range(num_pairs // 2):
        type1, type2 = np.random.choice(property_types, 2, replace=False)
        img1 = np.random.choice(property_images[type1])
        img2 = np.random.choice(property_images[type2])
        dissimilar_pairs.append((img1, img2, 0))  # 0 = dissimilar
    
    # Combine and shuffle
    all_pairs = similar_pairs + dissimilar_pairs
    np.random.shuffle(all_pairs)
    
    # Evaluate each pair
    results = []
    
    for img1_path, img2_path, true_label in tqdm(all_pairs, desc="Evaluating pairs"):
        # Load and preprocess images
        from data_preparation import load_and_preprocess_image
        img1 = load_and_preprocess_image(img1_path)
        img2 = load_and_preprocess_image(img2_path)
        
        if img1 is None or img2 is None:
            continue
        
        # Compute similarity score
        similarity_score = model.compute_similarity(img1, img2)
        
        # Predict label (1 if score > 5, else 0)
        predicted_label = 1 if similarity_score > 5.0 else 0
        
        results.append({
            'image1': img1_path,
            'image2': img2_path,
            'true_label': true_label,
            'predicted_label': predicted_label,
            'similarity_score': float(similarity_score)
        })
    
    # Calculate metrics
    true_labels = [r['true_label'] for r in results]
    pred_labels = [r['predicted_label'] for r in results]
    
    # Accuracy
    accuracy = sum(t == p for t, p in zip(true_labels, pred_labels)) / len(results)
    
    # Calculate precision, recall, f1-score for similar class (label 1)
    tp = sum(1 for t, p in zip(true_labels, pred_labels) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(true_labels, pred_labels) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(true_labels, pred_labels) if t == 1 and p == 0)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    evaluation_results = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'num_pairs_evaluated': len(results),
        'threshold': 5.0,
        'detailed_results': results[:10]  # Include just a few for the report
    }
    
    return evaluation_results

def collect_property_images(data_dir: str) -> Dict[str, List[str]]:
    """
    Collect property images from the data directory.
    Expects either:
    1. A directory structure where each subdirectory is a property type
    2. A triplet structure where subdirectories contain anchor/positive/negative folders
    3. A triplet_triplets_info.json file with triplet information
    
    Args:
        data_dir: Path to the directory containing property images or triplets
        
    Returns:
        Dictionary mapping property types to lists of image paths or 'triplets' to triplet info
    """
    property_images = {}
    
    # Check if directory exists
    if not os.path.exists(data_dir):
        logger.error(f"Data directory does not exist: {data_dir}")
        return {}
        
    # First, check if triplet_triplets_info.json exists (highest priority)
    triplet_info_path = os.path.join(data_dir, "triplet_triplets_info.json")
    if os.path.exists(triplet_info_path):
        try:
            logger.info(f"Found triplet info file at {triplet_info_path}")
            triplet_info = load_json(triplet_info_path)
            # Create a list of triplet objects based on the property UIDs
            triplets = []
            property_uids = triplet_info.get("property_uids", [])
            for uid in property_uids:
                triplet_dir = os.path.join(data_dir, uid)
                if os.path.exists(triplet_dir):
                    # Verify this is a triplet directory with anchor/positive/negative folders
                    has_anchor = os.path.isdir(os.path.join(triplet_dir, "anchor"))
                    has_positive = os.path.isdir(os.path.join(triplet_dir, "positive"))
                    has_negative = os.path.isdir(os.path.join(triplet_dir, "negative"))
                    
                    if has_anchor and has_positive and has_negative:
                        # This is a valid triplet directory
                        triplet_info = {
                            "id": uid,
                            "anchor_uid": uid,
                            "negative_uid": f"{uid}_neg",  # Just a placeholder
                            "anchor_dir": os.path.join(triplet_dir, "anchor"),
                            "positive_dir": os.path.join(triplet_dir, "positive"),
                            "negative_dir": os.path.join(triplet_dir, "negative")
                        }
                        triplets.append(triplet_info)
            
            if triplets:
                logger.info(f"Found {len(triplets)} valid triplets based on triplet_triplets_info.json")
                return {'triplets': triplets}
        except Exception as e:
            logger.warning(f"Error processing triplet info file: {str(e)}")
    
    # Second, check for triplet manifest (standard JSON manifest)
    manifest_path = os.path.join(data_dir, "triplet_manifest.json")
    if os.path.exists(manifest_path):
        try:
            logger.info(f"Found triplet manifest at {manifest_path}")
            manifest = load_json(manifest_path)
            if manifest and 'triplets' in manifest:
                return {'triplets': manifest['triplets']}
        except Exception as e:
            logger.warning(f"Error reading triplet manifest: {str(e)}")
    
    # Third, scan for triplet directory structure (each folder has anchor/positive/negative)
    triplets = []
    found_triplet_structure = False
    
    # Check for triplet structure (folders with anchor/positive/negative subfolders)
    for item in os.listdir(data_dir):
        item_path = os.path.join(data_dir, item)
        if os.path.isdir(item_path):
            # Check if this directory has anchor/positive/negative subdirectories
            has_anchor = os.path.isdir(os.path.join(item_path, "anchor"))
            has_positive = os.path.isdir(os.path.join(item_path, "positive"))
            has_negative = os.path.isdir(os.path.join(item_path, "negative"))
            
            if has_anchor and has_positive and has_negative:
                found_triplet_structure = True
                # This is a triplet directory
                triplet = {
                    "id": item,
                    "anchor_uid": item,
                    "negative_uid": f"{item}_neg",  # Just a placeholder
                    "anchor_dir": os.path.join(item_path, "anchor"),
                    "positive_dir": os.path.join(item_path, "positive"),
                    "negative_dir": os.path.join(item_path, "negative")
                }
                triplets.append(triplet)
    
    if found_triplet_structure:
        logger.info(f"Found {len(triplets)} triplet directories with anchor/positive/negative subdirectories")
        return {'triplets': triplets}
    
    # Finally, check for the conventional structure (property type directories with images)
    for item in os.listdir(data_dir):
        item_path = os.path.join(data_dir, item)
        if os.path.isdir(item_path):
            images = []
            for img_file in os.listdir(item_path):
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(item_path, img_file)
                    images.append(img_path)
            if images:
                property_images[item] = images
                logger.info(f"Found {len(images)} images for property type: {item}")
    
    return property_images

def get_training_data(training_dir: str) -> Dict[str, Dict[str, List[str]]]:
    """
    Get training data from the specified directory
    
    Args:
        training_dir: Path to the training directory containing triplets
        
    Returns:
        Dictionary of training data by property UID
    """
    logging.info(f"Loading training data from {training_dir}")
    
    # Handle relative paths
    if not os.path.isabs(training_dir):
        training_dir = os.path.join(os.getcwd(), training_dir)
    
    # Check if training directory exists
    if not os.path.exists(training_dir):
        logging.error(f"Training directory {training_dir} does not exist")
        logging.error(f"Current working directory: {os.getcwd()}")
        logging.error(f"Directory contents: {os.listdir(os.getcwd())}")
        raise FileNotFoundError(f"Training directory {training_dir} does not exist")
    
    # Check for triplet structure
    manifest_path = os.path.join(training_dir, "triplet_triplets_info.json")
    
    if not os.path.exists(manifest_path):
        # Look for triplets without manifest
        logging.warning(f"Triplet manifest not found at {manifest_path}")
        property_uids = [d for d in os.listdir(training_dir) 
                         if os.path.isdir(os.path.join(training_dir, d))]
    else:
        # Load property UIDs from manifest
        try:
            triplet_info = load_json(manifest_path)
            property_uids = list(triplet_info.keys())
            logging.info(f"Found {len(property_uids)} property UIDs in manifest")
        except Exception as e:
            logging.error(f"Error loading triplet manifest: {e}")
            property_uids = [d for d in os.listdir(training_dir) 
                             if os.path.isdir(os.path.join(training_dir, d))]
            logging.info(f"Falling back to directory scan, found {len(property_uids)} potential property folders")
    
    if not property_uids:
        logging.error(f"No property UIDs found in {training_dir}")
        logging.error(f"Directory contents: {os.listdir(training_dir)}")
        raise ValueError(f"No property UIDs found in {training_dir}")
    
    # Initialize training data dictionary
    training_data = {}
    
    # Check directory structure for each property UID
    for property_uid in property_uids:
        property_dir = os.path.join(training_dir, property_uid)
        
        if not os.path.isdir(property_dir):
            continue
            
        # Check for required subdirectories
        required_subdirs = ["anchor", "positive", "negative"]
        missing_subdirs = [subdir for subdir in required_subdirs 
                         if not os.path.exists(os.path.join(property_dir, subdir))]
        
        if missing_subdirs:
            logging.warning(f"Property {property_uid} is missing subdirectories: {missing_subdirs}")
            continue
        
        # Collect images for each subdirectory
        property_images = {}
        for subdir in required_subdirs:
            subdir_path = os.path.join(property_dir, subdir)
            image_files = [os.path.join(subdir_path, f) for f in os.listdir(subdir_path) 
                         if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            property_images[subdir] = image_files
            
        # Check if we have images in all subdirectories
        if all(property_images.values()):
            training_data[property_uid] = property_images
        else:
            logging.warning(f"Property {property_uid} has empty subdirectories: "
                           f"{[k for k, v in property_images.items() if not v]}")
    
    if not training_data:
        logging.error(f"No valid training data found in {training_dir}")
        raise ValueError(f"No valid training data found in {training_dir}")
    
    logging.info(f"Loaded {len(training_data)} properties with triplet data")
    return training_data

def get_validation_data(args, source_type):
    """Get validation data either from local directory or API."""
    val_property_images = None
    
    if args.val_data_dir:
        # Get validation data from local directory
        logger.info(f"Collecting validation images from local directory: {args.val_data_dir}")
        val_property_images = collect_property_images(args.val_data_dir)
    elif args.api_val_property_types and args.api_endpoint:
        # Get validation data from API
        logger.info(f"Fetching validation images from API: {args.api_endpoint}")
        val_property_images = fetch_training_data_from_api(
            api_url=args.api_endpoint,
            auth_token=args.api_token,
            property_types=args.api_val_property_types,
            images_per_type=args.images_per_type // 2,  # Fewer images for validation
            cache_dir=os.path.join(args.api_cache_dir, "validation"),
            max_workers=args.api_workers
        )
    elif source_type == "api" and args.api_endpoint:
        # No specific validation data source, use subset of training data
        logger.info("No specific validation data source provided, will use train-validation split")
        return None
    
    if val_property_images:
        logger.info(f"Found {len(val_property_images)} property types with {sum(len(images) for images in val_property_images.values())} total images for validation")
    
    return val_property_images

def fetch_training_data_from_api(api_url, auth_token, property_types, images_per_type, cache_dir, max_workers):
    """
    Fetch training data from API for specific property types.
    
    Args:
        api_url: API endpoint URL
        auth_token: Authentication token
        property_types: List of property types to fetch
        images_per_type: Number of images to fetch per property type
        cache_dir: Directory to cache images
        max_workers: Maximum number of concurrent download threads
        
    Returns:
        Dictionary mapping property types to lists of image paths
    """
    logger.info(f"Fetching {len(property_types)} property types with {images_per_type} images each from API")
    
    # Create request for each property type
    requests = []
    for prop_type in property_types:
        requests.append({
            "property_type": prop_type,
            "count": images_per_type
        })
    
    # Fetch images for each property type
    all_images = {}
    try:
        for prop_type in property_types:
            # Use existing function to fetch from API
            images = fetch_from_premier_brokerage_api(
                api_url=api_url,
                auth_token=auth_token,
                property_type=prop_type,
                count=images_per_type,
                cache_dir=os.path.join(cache_dir, prop_type),
                max_workers=max_workers
            )
            
            if images:
                all_images[prop_type] = images
                logger.info(f"Fetched {len(images)} images for property type {prop_type}")
            else:
                logger.warning(f"No images fetched for property type {prop_type}")
    except Exception as e:
        logger.error(f"Error fetching images from API: {str(e)}")
    
    return all_images

class PropertyTripletDataset(torch.utils.data.Dataset):
    """
    Dataset for property-level triplets where each property has multiple images.
    Enables efficient training with variable numbers of images per property.
    """
    def __init__(self, property_triplets, max_images_per_property=10, transform=None):
        """
        Initialize the dataset.
        
        Args:
            property_triplets: List of dictionaries with keys 'anchor_images', 'positive_images', 'negative_images'
            max_images_per_property: Maximum number of images to use per property
            transform: Optional transform to apply to images
        """
        self.property_triplets = property_triplets
        self.max_images = max_images_per_property
        
        # Set up transforms
        if transform is None:
            import torchvision.transforms as transforms
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                    std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transform
    
    def __len__(self):
        return len(self.property_triplets)
    
    def __getitem__(self, idx):
        """
        Get a property triplet.
        
        Returns:
            Tuple of (anchor_images, positive_images, negative_images) where each is a tensor
            of shape [num_images, channels, height, width]
        """
        triplet = self.property_triplets[idx]
        
        # Process anchor images
        anchor_images = self._process_property_images(triplet['anchor_images'])
        positive_images = self._process_property_images(triplet['positive_images'])
        negative_images = self._process_property_images(triplet['negative_images'])
        
        return anchor_images, positive_images, negative_images
    
    def _process_property_images(self, images):
        """Process images for a property, limiting to max_images."""
        # Limit number of images if needed
        if len(images) > self.max_images:
            # Randomly select subset
            indices = np.random.choice(len(images), self.max_images, replace=False)
            selected_images = [images[i] for i in indices]
        else:
            selected_images = images
        
        # Convert to torch tensors and stack
        processed_images = []
        for img in selected_images:
            if isinstance(img, str):  # Image path
                # Load image
                try:
                    with Image.open(img) as pil_img:
                        pil_img = pil_img.convert('RGB')
                        # Apply transform
                        if self.transform:
                            img_tensor = self.transform(pil_img)
                        else:
                            # Convert to numpy and then tensor
                            img_array = np.array(pil_img) / 255.0
                            img_tensor = torch.from_numpy(img_array.transpose(2, 0, 1)).float()
                        processed_images.append(img_tensor)
                except Exception as e:
                    logger.warning(f"Error loading image {img}: {str(e)}")
                    # Create a small random tensor as fallback
                    processed_images.append(torch.rand(3, 224, 224))
            else:  # Numpy array
                # Apply transform if needed
                if self.transform:
                    img_tensor = self.transform(Image.fromarray((img * 255).astype(np.uint8)))
                else:
                    # Convert to tensor
                    img_tensor = torch.from_numpy(img.transpose(2, 0, 1)).float()
                processed_images.append(img_tensor)
        
        # If no valid images, create a dummy
        if not processed_images:
            logger.warning(f"No valid images processed for property in triplet {idx}")
            processed_images.append(torch.zeros(3, 224, 224))
        
        # Stack into a single tensor
        return torch.stack(processed_images)

class OnDemandTripletDataset(torch.utils.data.Dataset):
    """
    Dataset for triplets stored on disk with structure:
    root_dir/
        {triplet_id}/
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
            
    This dataset loads images on-demand from the filesystem rather than keeping them in memory.
    """
    def __init__(self, triplet_infos, root_dir, max_images_per_folder=10, transform=None):
        """
        Initialize the dataset.
        
        Args:
            triplet_infos: List of triplet information dictionaries or property UIDs
            root_dir: Root directory containing triplet folders
            max_images_per_folder: Maximum number of images to use per folder
            transform: Optional transform to apply to images
        """
        self.triplet_infos = triplet_infos
        self.root_dir = os.path.abspath(root_dir)
        self.max_images = max_images_per_folder
        
        # Set up transforms
        if transform is None:
            import torchvision.transforms as transforms
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                    std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transform
        
        logger.info(f"Initialized OnDemandTripletDataset with {len(triplet_infos)} triplets")
    
    def __len__(self):
        return len(self.triplet_infos)
    
    def __getitem__(self, idx):
        """
        Get a triplet.
        
        Returns:
            Tuple of (anchor_images, positive_images, negative_images) where each is a tensor
            of shape [num_images, channels, height, width]
        """
        info = self.triplet_infos[idx]
        
        # Determine triplet directories based on info format
        if isinstance(info, str):
            # Just a property/triplet UID string
            triplet_dir = os.path.join(self.root_dir, info)
            anchor_dir = os.path.join(triplet_dir, "anchor")
            positive_dir = os.path.join(triplet_dir, "positive")
            negative_dir = os.path.join(triplet_dir, "negative")
        elif isinstance(info, dict):
            # Dictionary with explicit directories or IDs
            if "anchor_dir" in info and "positive_dir" in info and "negative_dir" in info:
                # Directory paths are directly specified
                anchor_dir = info["anchor_dir"]
                positive_dir = info["positive_dir"]
                negative_dir = info["negative_dir"]
            elif "id" in info:
                # ID is specified, construct paths
                triplet_dir = os.path.join(self.root_dir, info["id"])
                anchor_dir = os.path.join(triplet_dir, "anchor")
                positive_dir = os.path.join(triplet_dir, "positive")
                negative_dir = os.path.join(triplet_dir, "negative")
            else:
                # Unknown format
                logger.warning(f"Unknown triplet info format: {info}")
                # Return dummy data
                dummy = torch.zeros(1, 3, 224, 224)
                return dummy, dummy, dummy
        else:
            # Unknown type
            logger.warning(f"Unknown triplet info type: {type(info)}")
            # Return dummy data
            dummy = torch.zeros(1, 3, 224, 224)
            return dummy, dummy, dummy
        
        # Load images from each directory
        logger.debug(f"Loading triplet {idx} from anchor: {anchor_dir}, positive: {positive_dir}, negative: {negative_dir}")
        anchor_images = self._load_images_from_dir(anchor_dir)
        positive_images = self._load_images_from_dir(positive_dir)
        negative_images = self._load_images_from_dir(negative_dir)
        
        # Ensure we have at least one image in each category
        if len(anchor_images) == 0 or len(positive_images) == 0 or len(negative_images) == 0:
            logger.warning(f"Missing images in triplet {triplet_dir}, using dummy data")
            dummy = torch.zeros(1, 3, 224, 224)
            if len(anchor_images) == 0:
                anchor_images = dummy
            if len(positive_images) == 0:
                positive_images = dummy
            if len(negative_images) == 0:
                negative_images = dummy
        
        return anchor_images, positive_images, negative_images
    
    def _load_images_from_dir(self, directory):
        """Load images from a directory and return stacked tensors in a more memory-efficient way."""
        if not os.path.exists(directory):
            logger.warning(f"Directory does not exist: {directory}")
            return torch.zeros(1, 3, 224, 224)
        
        # Get image paths
        image_paths = []
        for filename in os.listdir(directory):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_paths.append(os.path.join(directory, filename))
        
        # Skip hidden files/directories (like .ipynb_checkpoints)
        image_paths = [p for p in image_paths if not os.path.basename(p).startswith('.')]
        
        # Handle empty directory
        if not image_paths:
            logger.warning(f"No image files found in {directory}")
            return torch.zeros(1, 3, 224, 224)
        
        # Limit number of images if needed
        if len(image_paths) > self.max_images:
            # Randomly select subset
            indices = np.random.choice(len(image_paths), self.max_images, replace=False)
            selected_paths = [image_paths[i] for i in indices]
        else:
            selected_paths = image_paths
        
        # Load and transform images with improved memory efficiency
        tensors = []
        for path in selected_paths:
            try:
                # Open the image with PIL in a memory-efficient way
                with Image.open(path) as img:
                    # Immediately resize to 224x224 to reduce memory usage
                    img = img.convert('RGB').resize((224, 224), Image.BILINEAR)
                    
                    # Apply transform or manual normalization
                    if self.transform:
                        img_tensor = self.transform(img)
                    else:
                        # Convert to numpy - this will be smaller now that we've resized
                        img_array = np.array(img) / 255.0
                        img_tensor = torch.from_numpy(img_array.transpose(2, 0, 1)).float()
                        
                        # Apply normalization manually
                        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                        img_tensor = (img_tensor - mean) / std
                    
                    tensors.append(img_tensor)
                    
                    # Clean up to reduce memory pressure
                    img = None
            except Exception as e:
                logger.warning(f"Error loading image {path}: {str(e)}")
        
        # Clear memory explicitly
        import gc
        gc.collect()
        
        # If no images were loaded, return a dummy
        if not tensors:
            logger.warning(f"No valid images loaded from {directory}")
            return torch.zeros(1, 3, 224, 224)
        
        # Stack images into a single tensor [num_images, channels, height, width]
        try:
            return torch.stack(tensors)
        except RuntimeError as e:
            # If we encounter an error during stacking, try to save memory
            logger.warning(f"Error stacking tensors: {str(e)}. Reducing to 1 image.")
            if tensors:
                return torch.stack([tensors[0]])
            else:
                return torch.zeros(1, 3, 224, 224)

# Custom collate function for variable-sized batches
def triplet_collate_fn(batch):
    """
    Custom collate function that handles variable numbers of images per property.
    Instead of stacking tensors of different first dimensions, it simply returns a list of tensors.
    
    Args:
        batch: List of (anchor, positive, negative) tuples, where each element is a tensor of shape 
              [num_images, channels, height, width] with potentially different num_images
    
    Returns:
        Tuple of (anchors, positives, negatives) where each element is a list of tensors
    """
    # Transpose the batch from batch_size x 3 to 3 x batch_size
    transposed = list(zip(*batch))
    
    # Don't stack, just return lists
    anchors = list(transposed[0])
    positives = list(transposed[1])
    negatives = list(transposed[2])
    
    return anchors, positives, negatives

def train_model(args, train_dataset=None, val_dataset=None):
    """Train the Siamese model with optional pre-created datasets."""
    logger.info("Preparing data for training...")
    
    # Record start time
    start_time = time.time()
    
    # Set random seed for reproducibility
    set_seed(args.seed)
    
    # Enable GPU optimizations
    # For RTX A5000 (24GB VRAM)
    torch.backends.cudnn.benchmark = True  # Enable cuDNN auto-tuner
    torch.backends.cuda.matmul.allow_tf32 = True  # Enable TF32 for faster matrix multiplications
    torch.backends.cudnn.allow_tf32 = True  # Enable TF32 for convolutions
    
    # Force GPU synchronization to detect CUDA errors early
    torch.cuda.synchronize()
    
    # Check GPU availability and stats
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # Convert to GB
        logger.info(f"Using GPU: {gpu_name} with {gpu_mem:.1f}GB memory")
        
        # Optimize for RTX A5000
        # Set optimal persistent workers and prefetch factor
        num_workers = min(8, os.cpu_count() or 4)
        persistent_workers = True
        prefetch_factor = 2
    else:
        logger.warning("No GPU available, using CPU for training (this will be slow)")
        num_workers = 0
        persistent_workers = False
        prefetch_factor = 2
    
    # Check if mixed precision is available and enable it
    mixed_precision_available = torch.cuda.is_available() and hasattr(torch.cuda, 'amp')
    # Override if user specifically disabled it
    use_mixed_precision = mixed_precision_available if args.mixed_precision else False
    
    if use_mixed_precision:
        logger.info("Using mixed precision training (faster with lower memory usage)")
    else:
        logger.info("Mixed precision training disabled")
    
    # Ensure output directory exists
    if args.output_dir is None:
        args.output_dir = args.model_dir
    ensure_dir_exists(args.output_dir)
    
    # If datasets are not provided, create them using the existing pipeline
    if train_dataset is None:
        # Get training data
        train_data_result = get_training_data(args.train_data_dir)
        
        # Unpack the result - get_training_data returns (data_dict, source_type)
        if isinstance(train_data_result, tuple) and len(train_data_result) == 2:
            train_data_source, source_type = train_data_result
        else:
            train_data_source = train_data_result
            source_type = "unknown"
        
        # Check if we're working with a triplet structure
        using_triplet_structure = False
        triplet_data = None
        
        if 'triplets' in train_data_source:
            using_triplet_structure = True
            logger.info("Detected triplet structure in the input data")
            triplet_data = train_data_source['triplets']
        
        # Get validation data if needed
        val_data_source = None
        if args.val_data_dir or (args.triplet_dir and args.val_split > 0):
            val_data_source = get_validation_data(args, source_type)
        
        # Create the training dataset
        if args.triplet_dir:
            # If triplet directory is specified, use OnDemandTripletDataset
            # This is handled in main()
            train_dataset = train_dataset
            val_dataset = val_dataset
        else:
            # Regular dataset creation from triplets
            if triplet_data:
                train_dataset = PropertyTripletDataset(
                    triplet_data, 
                    max_images_per_property=10,  # Use up to 10 images per property
                    transform=None  # Use default transforms
                )
            else:
                raise ValueError("No triplet data found for training")
            
            # Create validation dataset if data is available
            if val_data_source and 'triplets' in val_data_source:
                val_dataset = PropertyTripletDataset(
                    val_data_source['triplets'],
                    max_images_per_property=10,
                    transform=None
                )
    
    # Now we should have datasets ready
    logger.info("Creating data loaders...")
    
    # Calculate optimal batch size based on GPU memory if batch_size is not specified
    # RTX A5000 has 24GB, but ResNet50 needs more memory per sample
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        
        # Set PyTorch CUDA memory allocation strategy to reduce fragmentation
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
        
        # For ResNet50, we need to use smaller batches
        if args.backbone == 'resnet50' and args.batch_size > 24:
            original_batch_size = args.batch_size
            # Use more conservative batch size for ResNet50
            args.batch_size = 16  # Much more conservative for ResNet50
            logger.info(f"Reduced batch size from {original_batch_size} to {args.batch_size} for ResNet50 backbone")
        elif use_mixed_precision and args.batch_size == 32 and args.backbone == 'efficientnet':
            # Only increase batch size for EfficientNet
            args.batch_size = 64
            logger.info(f"Optimized batch size for EfficientNet with mixed precision: {args.batch_size}")
    
    # Fix for shared memory (shm) issues in containerized environments
    # Reduce number of workers and prefetch factor for stability
    if torch.cuda.is_available():
        # Check if we're likely in a container with limited shm
        try:
            # Try to detect container environment and shm size
            with open('/proc/self/mountinfo', 'r') as f:
                mounts = f.read()
                if 'shm' in mounts:
                    # Likely in a container with limited shm
                    logger.warning("Running in containerized environment with limited shared memory")
                    # Use significantly reduced workers to avoid shared memory issues
                    num_workers = 2
                    persistent_workers = True
                    prefetch_factor = 2
                    logger.info(f"Reduced DataLoader workers to {num_workers} to avoid shared memory issues")
        except:
            # If we can't check, play it safe anyway
            num_workers = 2
            logger.info("Using conservative DataLoader settings for stability")
    
    # Create data loaders with optimized settings for GPU and container environments
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
        pin_memory=torch.cuda.is_available(),
        collate_fn=triplet_collate_fn if hasattr(train_dataset, '_load_images_from_dir') else None,
        multiprocessing_context='spawn'  # Use spawn instead of fork for better memory handling
    )
    
    val_loader = None
    if val_dataset:
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=num_workers,
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor,
            pin_memory=torch.cuda.is_available(),
            collate_fn=triplet_collate_fn if hasattr(val_dataset, '_load_images_from_dir') else None,
            multiprocessing_context='spawn'  # Use spawn instead of fork for better memory handling
        )
    
    # Create and initialize the model
    model = SiameseNetwork(
        embedding_dim=args.embedding_dim,
        margin=args.margin,
        learning_rate=args.learning_rate,
        device='cuda' if torch.cuda.is_available() else 'cpu',  # Explicitly set device
        scheduler_type=args.scheduler,
        backbone=args.backbone,
        weight_decay=args.weight_decay
    )
    
    # Load checkpoint if available
    if args.load_checkpoint:
        logger.info(f"Loading checkpoint from {args.load_checkpoint}")
        model.load_model(args.load_checkpoint)
    
    # Ask for early stopping or use default
    use_early_stopping = True
    if not args.no_prompt:
        response = input("Do you want to use early stopping during training? (yes/no, default: yes): ")
        use_early_stopping = response.lower() not in ['n', 'no']
    
    # Start training
    logger.info(f"Training model for {args.epochs} epochs with batch size {args.batch_size}")
    logger.info(f"Using {args.backbone} backbone with property-level={args.property_level}, mixed_precision={use_mixed_precision}")
    
    # Define model save directory
    model_save_dir = os.path.join(args.output_dir, f"siamese_{args.backbone}_{time.strftime('%Y%m%d_%H%M%S')}")
    
    try:
        # Train the model
        training_result = model.train(
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            epochs=args.epochs,
            save_dir=model_save_dir,
            checkpoint_freq=args.checkpoint_freq,
            patience=10,  # Stop after 10 epochs without improvement
            use_early_stopping=use_early_stopping,
            mixed_precision=use_mixed_precision  # Use mixed precision based on availability
        )
        
        # Record end time and calculate duration
        end_time = time.time()
        training_duration = end_time - start_time
        logger.info(f"Training completed in {training_duration/60:.2f} minutes")
        
        # Save training metadata
        metadata_path = os.path.join(model_save_dir, "training_metadata.json")
        save_training_metadata(training_result, metadata_path, args)
        
        # Evaluate on test pairs if requested
        if args.eval_model and args.val_data_dir:
            logger.info("Evaluating model on test property pairs...")
            eval_results = evaluate_on_test_pairs(model, args.val_data_dir, num_pairs=args.eval_pairs)
            
            # Save evaluation results
            eval_path = os.path.join(model_save_dir, "evaluation_results.json")
            save_json(eval_results, eval_path)
        
        # Clean up memory
        clear_memory()
        
        return True
    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        logger.error(traceback.format_exc())
        return False

def increase_shared_memory():
    """Attempt to increase shared memory limit for containerized environments like RunPods."""
    try:
        # Check current shared memory size
        result = subprocess.run(['df', '-h', '/dev/shm'], capture_output=True, text=True)
        if result.returncode == 0:
            shm_info = result.stdout.strip().split('\n')
            if len(shm_info) > 1:
                # Output format is like "Filesystem Size Used Avail Use% Mounted on"
                shm_size = shm_info[1].split()[1]
                logging.info(f"Current shared memory size: {shm_size}")
                
                # If shared memory is too small (less than 2GB), warn the user
                if 'M' in shm_size or ('G' in shm_size and float(shm_size.replace('G', '')) < 2):
                    logging.warning(
                        "Shared memory is limited. This may cause issues with DataLoader workers.\n"
                        "For RunPods, increase the shared memory in the pod configuration.\n"
                        "Try adding this to your RunPod startup script:\n"
                        "    sudo mount -o remount,size=4G /dev/shm\n"
                    )
        
        # Try to increase shared memory if we have permissions (won't work in most cloud environments)
        try:
            import os
            if os.geteuid() == 0:  # Only try if we're root
                subprocess.run(['mount', '-o', 'remount,size=4G', '/dev/shm'], check=False)
                logging.info("Attempted to increase shared memory to 4GB")
        except Exception as e:
            # This will likely fail in most environments, which is fine
            pass
            
    except Exception as e:
        logging.warning(f"Failed to check shared memory size: {str(e)}")

def main():
    """Main function to handle training flow."""
    # Parse command line arguments
    args = parse_arguments()
    setup_logging(args)
    
    # Check and try to increase shared memory
    increase_shared_memory()
    
    try:
        logging.info("Starting training process")
        logging.info(f"Arguments: {args}")
        
        # Train the model using the appropriate data source
        if args.triplet_dir:
            # Normalize triplet_dir to absolute path
            triplet_dir = args.triplet_dir
            if not os.path.isabs(triplet_dir):
                triplet_dir = os.path.join(os.getcwd(), triplet_dir)
            args.triplet_dir = triplet_dir

            # If triplet directory is specified, use OnDemandTripletDataset directly
            logging.info(f"Using pre-structured triplet directory: {args.triplet_dir}")
            
            # Get list of property UIDs (folder names) in the triplet directory
            property_uids = [d for d in os.listdir(args.triplet_dir) 
                            if os.path.isdir(os.path.join(args.triplet_dir, d))]
            logging.info(f"Found {len(property_uids)} property triplets in {args.triplet_dir}")
            
            if not property_uids:
                raise ValueError(f"No property triplets found in {args.triplet_dir}")
            
            # If we need validation data, split the property UIDs
            train_uids = property_uids
            val_uids = None
            
            if args.val_split > 0 and not args.val_data_dir:
                # Determine validation split size
                val_size = int(len(property_uids) * args.val_split)
                train_size = len(property_uids) - val_size
                
                # Randomly shuffle property UIDs
                random.shuffle(property_uids)
                
                # Split into training and validation
                train_uids = property_uids[val_size:]
                val_uids = property_uids[:val_size]
                
                logging.info(f"Split {len(property_uids)} triplets into {len(train_uids)} for training and {len(val_uids)} for validation")
            
            # Create dataset directly using OnDemandTripletDataset
            train_dataset = OnDemandTripletDataset(
                triplet_infos=train_uids,
                root_dir=args.triplet_dir,
                max_images_per_folder=args.batch_size  # Use batch size as default
            )
            val_dataset = None
            if val_uids:
                val_dataset = OnDemandTripletDataset(
                    triplet_infos=val_uids,
                    root_dir=args.triplet_dir,
                    max_images_per_folder=args.batch_size
                )
            
            # Set args.property_level to True since we're using property triplets
            args.property_level = True
            
            # Train with the created datasets
            success = train_model(args, train_dataset=train_dataset, val_dataset=val_dataset)
        else:
            # Use original method for other data sources
            success = train_model(args)
        
        if success:
            logger.info("Training completed successfully!")
            
            # Create a README in the output directory with run details
            readme_path = os.path.join(args.output_dir, "README.md")
            with open(readme_path, 'w') as f:
                f.write(f"# Training Run: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                f.write("## Configuration\n\n")
                f.write(f"- Model: Siamese Network with {args.backbone} backbone\n")
                f.write(f"- Embedding dimension: {args.embedding_dim}\n")
                f.write(f"- Property-level learning: {'Yes' if args.property_level else 'No'}\n")
                f.write(f"- Mixed precision: {'Yes' if args.mixed_precision else 'No'}\n")
                f.write(f"- Learning rate: {args.learning_rate}\n")
                f.write(f"- Scheduler: {args.scheduler}\n")
                f.write(f"- Batch size: {args.batch_size}\n")
                f.write(f"- Epochs: {args.epochs}\n\n")
                f.write("## Results\n\n")
                f.write("See training_metadata.json for detailed results.\n")
            
            return 0
        else:
            logger.error("Training failed")
            return 1
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        logger.error(traceback.format_exc())
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 