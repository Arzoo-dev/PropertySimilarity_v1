#!/usr/bin/env python3
"""
Dataset Utilities Module
=========================

Consolidated dataset-related utility functions used across multiple analysis scripts.
This module eliminates duplication of dataset creation and processing logic.

Contains:
- create_validation_dataset_same_as_train: Standardized validation dataset creation
- extract_features_from_validation_dataset: Feature extraction from validation sets
- Dataset helper functions
"""

import os
import torch
import numpy as np
from tqdm import tqdm
from typing import Dict, List, Any, Tuple, Optional

# Import dataset classes (these need to be available)
try:
    from dataset.train_dataset import TripletDataset
except ImportError:
    print("Warning: Could not import TripletDataset. Make sure the dataset module is available.")
    TripletDataset = None

def create_validation_dataset_same_as_train(config: Dict[str, Any], split_file_path: str) -> Tuple[Optional[Any], Optional[Any]]:
    """
    Create validation dataset following the EXACT same process as train.py:
    1. Dataset Creation from Subject Folders
    2. Random Split at Triplet Level  
    3. Key Configuration Parameters (train_ratio: 0.8, random_seed: 42)
    4. Triplet Generation Process
    
    Originally duplicated in run_tsne_comparison.py and margin_analysis.py.
    
    Args:
        config: Configuration dictionary containing dataset parameters
        split_file_path: Path to the split file containing subject folders
        
    Returns:
        Tuple of (train_dataset, val_dataset) or (None, None) if failed
    """
    if TripletDataset is None:
        print("ERROR: TripletDataset not available. Cannot create datasets.")
        return None, None
        
    print("="*60)
    print("FOLLOWING EXACT TRAIN.PY VALIDATION SET CREATION PROCESS")
    print("="*60)
    
    # Step 1: Load subject folders from split file (same as train.py)
    subject_folders = None
    if split_file_path and os.path.isfile(split_file_path):
        with open(split_file_path, 'r') as f:
            subject_folders = [line.strip() for line in f if line.strip()]
        print(f"✓ Loaded {len(subject_folders)} subject folders from split file: {split_file_path}")
    else:
        print(f"ERROR: Invalid split file path: {split_file_path}")
        return None, None
    
    # Step 2: Create TripletDataset (same as train.py)
    print(f"✓ Creating TripletDataset from subject folders...")
    
    # Get parameters from config (same as train.py)
    max_memory_gb = config['data'].get('max_memory_cached', 4)
    memory_efficient = config.get('optimization', {}).get('memory_efficient', True)
    
    # Check for hard negatives (same logic as train.py)
    hard_neg_dir = None
    if 'hard_negatives' in config and config['hard_negatives'].get('use_hard_negatives', False):
        hard_neg_dir = config['hard_negatives']['hard_negative_dir']
    # Default fallback
    if hard_neg_dir is None:
        hard_neg_dir = "hard_negative_output/mining_results"
    
    # Only use hard negatives if directory exists
    use_hard_negatives = hard_neg_dir is not None and os.path.exists(hard_neg_dir)
    if use_hard_negatives:
        print(f"✓ Using hard negatives from: {hard_neg_dir}")
    else:
        print("✓ Using random negatives (no hard negative mining)")
        hard_neg_dir = None
    
    print(f"✓ Using memory limit of {max_memory_gb}GB for dataset cache")
    
    # Create TripletDataset (EXACT same parameters as train.py)
    dataset = TripletDataset(
        config['data']['root_dir'], 
        subject_folders,
        max_memory_gb=max_memory_gb,
        chunk_size=config['data']['chunk_size'],  # Read from config
        cache_flush_threshold=config.get('data', {}).get('cache_flush_threshold', 1000),  # Read from config with safe default
        hard_negative_dir=hard_neg_dir,
        memory_efficient=memory_efficient
    )
    
    print(f"✓ Created TripletDataset with {len(dataset)} triplets total")
    
    # Step 3: Random Split at Triplet Level (EXACT same as train.py)
    train_size = int(config['data']['train_ratio'] * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(config.get('data', {}).get('random_seed', 42))
    )
    
    print(f"✓ Split dataset using train_ratio={config['data']['train_ratio']}, random_seed={config['data']['random_seed']}")
    print(f"✓ Training set: {len(train_dataset)} triplets")
    print(f"✓ Validation set: {len(val_dataset)} triplets")
    print("="*60)
    
    return train_dataset, val_dataset

def extract_features_from_validation_dataset(
    model: torch.nn.Module, 
    val_dataset: torch.utils.data.Dataset, 
    device: torch.device, 
    max_triplets: int = 500
) -> Dict[str, List[np.ndarray]]:
    """Extract features from validation dataset triplets
    
    Originally used in run_tsne_comparison.py with slight variations.
    
    Args:
        model: The model to use for feature extraction
        val_dataset: Validation dataset to extract features from
        device: Device to run inference on
        max_triplets: Maximum number of triplets to process
        
    Returns:
        Dictionary with keys 'anchor', 'positive', 'negative' containing feature lists
    """
    features_dict = {'anchor': [], 'positive': [], 'negative': []}
    
    print(f"Extracting features from {min(len(val_dataset), max_triplets)} validation triplets...")
    
    # Sample triplets from validation set
    indices = torch.randperm(len(val_dataset))[:max_triplets]
    
    for i, idx in enumerate(tqdm(indices, desc="Processing validation triplets")):
        try:
            anchor, positive, negative = val_dataset[idx]
            
            # Extract features for each image in the triplet
            with torch.no_grad():
                # Process anchor
                anchor_tensor = anchor.unsqueeze(0).to(device)
                if hasattr(model, 'forward_features'):
                    anchor_features = model.forward_features(anchor_tensor)
                    anchor_feature = anchor_features[:, 0].cpu().numpy()[0]  # CLS token
                else:
                    anchor_feature = model(anchor_tensor).cpu().numpy()[0]
                features_dict['anchor'].append(anchor_feature)
                
                # Process positive
                positive_tensor = positive.unsqueeze(0).to(device)
                if hasattr(model, 'forward_features'):
                    positive_features = model.forward_features(positive_tensor)
                    positive_feature = positive_features[:, 0].cpu().numpy()[0]  # CLS token
                else:
                    positive_feature = model(positive_tensor).cpu().numpy()[0]
                features_dict['positive'].append(positive_feature)
                
                # Process negative
                negative_tensor = negative.unsqueeze(0).to(device)
                if hasattr(model, 'forward_features'):
                    negative_features = model.forward_features(negative_tensor)
                    negative_feature = negative_features[:, 0].cpu().numpy()[0]  # CLS token
                else:
                    negative_feature = model(negative_tensor).cpu().numpy()[0]
                features_dict['negative'].append(negative_feature)
                
        except Exception as e:
            print(f"Error processing triplet {idx}: {e}")
            continue
    
    print(f"✓ Extracted {len(features_dict['anchor'])} anchor features")
    print(f"✓ Extracted {len(features_dict['positive'])} positive features") 
    print(f"✓ Extracted {len(features_dict['negative'])} negative features")
    
    return features_dict

def validate_dataset_config(config: Dict[str, Any]) -> bool:
    """Validate that the configuration contains required dataset parameters
    
    Args:
        config: Configuration dictionary to validate
        
    Returns:
        True if config is valid, False otherwise
    """
    required_keys = [
        'data.root_dir',
        'data.train_ratio', 
        'data.random_seed',
        'data.chunk_size',
        'data.cache_flush_threshold',
        'data.max_memory_cached',
        'optimization.memory_efficient'
    ]
    
    for key_path in required_keys:
        keys = key_path.split('.')
        current = config
        
        try:
            for key in keys:
                current = current[key]
        except KeyError:
            print(f"ERROR: Missing required config key: {key_path}")
            return False
    
    return True

def get_dataset_statistics(dataset: torch.utils.data.Dataset) -> Dict[str, Any]:
    """Get basic statistics about a dataset
    
    Args:
        dataset: Dataset to analyze
        
    Returns:
        Dictionary containing dataset statistics
    """
    stats = {
        'total_triplets': len(dataset),
        'dataset_type': type(dataset).__name__
    }
    
    # Try to get additional info if available
    if hasattr(dataset, 'dataset') and hasattr(dataset.dataset, 'subject_folders'):
        stats['num_subjects'] = len(dataset.dataset.subject_folders)
    
    if hasattr(dataset, 'dataset') and hasattr(dataset.dataset, 'use_hard_negatives'):
        stats['uses_hard_negatives'] = dataset.dataset.use_hard_negatives
    
    return stats

def load_subject_folders_from_split_file(split_file_path: str) -> Optional[List[str]]:
    """Load subject folders from a split file
    
    Args:
        split_file_path: Path to the split file
        
    Returns:
        List of subject folder names or None if failed
    """
    if not split_file_path or not os.path.isfile(split_file_path):
        print(f"ERROR: Invalid split file path: {split_file_path}")
        return None
    
    try:
        with open(split_file_path, 'r') as f:
            subject_folders = [line.strip() for line in f if line.strip()]
        print(f"✓ Loaded {len(subject_folders)} subject folders from: {split_file_path}")
        return subject_folders
    except Exception as e:
        print(f"ERROR: Failed to load split file {split_file_path}: {e}")
        return None 