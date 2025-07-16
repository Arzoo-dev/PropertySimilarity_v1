#!/usr/bin/env python3
"""
Consolidated DINOv2 Training Pipeline
=====================================

This script combines hard negative mining and training into a single, streamlined pipeline:
1. Load configuration and select subject split
2. Extract features using pre-trained DINOv2
3. Mine hard negatives based on similarity criteria
4. Create triplet dataset with hard negatives
5. Train custom DINOv2 model with frozen backbone
6. Visualize embeddings and save results

Features:
- Single entry point for entire workflow
- Unified configuration and argument handling
- Memory efficient - no subprocess overhead
- Direct integration between mining and training phases
"""

import os
import logging
import yaml
import torch
import psutil
import argparse
import gc
import time
import threading
import platform
import json
import h5py
import numpy as np
from pathlib import Path
from tqdm import tqdm
from collections import OrderedDict
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers.optimization import get_cosine_schedule_with_warmup
from utils.training_utils import DataLoaderMonitor, TripletLoss
import torch.nn.functional as F
from torchvision.io import read_image
import timm

# Import existing components
from models.model_builder import DINOv2Retrieval
from dataset.train_dataset import TripletDataset, ChunkAwareSampler
from memory_efficient_trainer import MemoryEfficientTrainer
from utils.visualization import TrainingVisualizer
from utils.common import (
    setup_centralized_logging, 
    memory_monitor, 
    set_memory_threshold,
    start_memory_monitoring,
    stop_memory_monitoring
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



class FeatureExtractor:
    """
    Extracts features from images using DINOv2 model.
    Used for hard negative mining to compute embeddings for all images.
    """
    def __init__(self, model_name="vit_base_patch14_dinov2", device=None, batch_size=32):
        self.model_name = model_name
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = batch_size
        logger.info(f"Initializing FeatureExtractor with {model_name} on {self.device}")
        
        # Initialize the model
        self.model = timm.create_model(model_name, pretrained=True, num_classes=0)
        self.model.to(self.device)
        self.model.eval()
        
        # Get embedding dimension
        self.embedding_dim = self.model.embed_dim
        logger.info(f"Model embedding dimension: {self.embedding_dim}")
    
    def _load_and_preprocess_image(self, img_path):
        """Load and preprocess a single image"""
        try:
            img = read_image(img_path).float() / 255.0
            
            # Check if image is grayscale (1 channel) and convert to RGB (3 channels) if needed
            if img.shape[0] == 1:
                img = img.repeat(3, 1, 1)
            # Check if image has 4 channels (RGBA) and convert to RGB (3 channels)
            elif img.shape[0] == 4:
                img = img[:3, :, :]
                
            # Resize to match model input
            img = F.interpolate(img.unsqueeze(0), size=(518, 518), mode='bilinear', align_corners=False)[0]
            
            # Normalize
            mean = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)
            std = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)
            img = (img - mean) / std
            
            return img
        except Exception as e:
            logger.error(f"Error loading image {img_path}: {str(e)}")
            return None
    
    def extract_features_for_subject(self, subject_path, save_path=None):
        """
        Extract features for all images in a subject folder (anchor, positive, negative)
        """
        subject_path = Path(subject_path)
        subject_id = subject_path.name
        logger.info(f"Extracting features for subject {subject_id}")
        
        # Collect all image paths
        image_paths = []
        categories = []
        for category in ['anchor', 'positive', 'negative']:
            category_dir = subject_path / category
            if not category_dir.exists():
                logger.warning(f"Category {category} not found for subject {subject_id}")
                continue
                
            for img_file in category_dir.glob('*.jpg'):
                image_paths.append(str(img_file))
                categories.append(category)
            for img_file in category_dir.glob('*.jpeg'):
                image_paths.append(str(img_file))
                categories.append(category)
            for img_file in category_dir.glob('*.png'):
                image_paths.append(str(img_file))
                categories.append(category)
        
        if not image_paths:
            logger.warning(f"No images found for subject {subject_id}")
            return None
            
        logger.info(f"Found {len(image_paths)} images for subject {subject_id}")
        
        # Process in batches
        all_embeddings = {}
        for i in tqdm(range(0, len(image_paths), self.batch_size), desc=f"Processing {subject_id}"):
            batch_paths = image_paths[i:i+self.batch_size]
            batch_imgs = []
            valid_paths = []
            
            # Load and preprocess images
            for path in batch_paths:
                img = self._load_and_preprocess_image(path)
                if img is not None:
                    batch_imgs.append(img)
                    valid_paths.append(path)
            
            if not batch_imgs:
                continue
                
            # Stack images into a batch
            batch_tensor = torch.stack(batch_imgs).to(self.device)
            
            # Extract features
            with torch.no_grad():
                if hasattr(self.model, 'forward_features') and callable(self.model.forward_features):
                    features = self.model.forward_features(batch_tensor)
                else:
                    features = self.model(batch_tensor)
                cls_token = features[:, 0]  # Get CLS token
                embeddings = F.normalize(cls_token, p=2, dim=1)
            
            # Store embeddings
            for path, embedding in zip(valid_paths, embeddings.cpu()):
                all_embeddings[path] = embedding.numpy()
            
            # Free up GPU memory
            del batch_tensor, features, cls_token, embeddings
            torch.cuda.empty_cache()
        
        # Save embeddings if requested
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with h5py.File(save_path, 'w') as f:
                # Group by category
                for category in ['anchor', 'positive', 'negative']:
                    grp = f.create_group(category)
                    category_embeddings = {
                        os.path.basename(path): emb 
                        for path, emb in all_embeddings.items() 
                        if f"/{category}/" in path.replace("\\", "/")
                    }
                    for img_name, embedding in category_embeddings.items():
                        grp.create_dataset(img_name, data=embedding)
            
            logger.info(f"Saved embeddings for subject {subject_id} to {save_path}")
        
        # Organize by category for convenience
        organized_embeddings = {
            'anchor': {
                os.path.basename(path): emb 
                for path, emb in all_embeddings.items() 
                if "/anchor/" in path.replace("\\", "/")
            },
            'positive': {
                os.path.basename(path): emb 
                for path, emb in all_embeddings.items() 
                if "/positive/" in path.replace("\\", "/")
            },
            'negative': {
                os.path.basename(path): emb 
                for path, emb in all_embeddings.items() 
                if "/negative/" in path.replace("\\", "/")
            }
        }
        
        return organized_embeddings
    
    def load_embeddings_from_h5(self, file_path):
        """Load embeddings from an H5 file"""
        subject_embeddings = {'anchor': {}, 'positive': {}, 'negative': {}}
        
        with h5py.File(file_path, 'r') as f:
            for category in ['anchor', 'positive', 'negative']:
                if category in f:
                    for img_name in list(f[category].keys()):  # type: ignore
                        embedding = f[category][img_name][()]  # type: ignore
                        subject_embeddings[category][img_name] = embedding
        
        return subject_embeddings

def mine_hard_negatives(subject_embeddings, top_k=5, similarity_threshold=0.0, save_path=None):
    """Mine hard negatives for a subject from embeddings"""
    anchor_embeddings = subject_embeddings.get('anchor', {})
    negative_embeddings = subject_embeddings.get('negative', {})
    
    if not anchor_embeddings or not negative_embeddings:
        logger.warning("Missing anchor or negative embeddings")
        return {}
    
    logger.info(f"Mining hard negatives from {len(negative_embeddings)} negatives for {len(anchor_embeddings)} anchors")
    
    # Convert embeddings to numpy arrays for faster computation
    anchor_data = {name: np.array(emb) for name, emb in anchor_embeddings.items()}
    negative_data = {name: np.array(emb) for name, emb in negative_embeddings.items()}
    
    results = {}
    
    # For each anchor, find the most similar negatives
    for anchor_name, anchor_emb in tqdm(anchor_data.items(), desc="Finding hard negatives"):
        # Compute similarity with all negatives
        similarities = []
        
        for neg_name, neg_emb in negative_data.items():
            # Compute cosine similarity: dot product of normalized vectors
            similarity = float(np.dot(anchor_emb, neg_emb))
            similarities.append((neg_name, similarity))
        
        # Sort by similarity in descending order (highest similarity first)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Keep top-k hard negatives above threshold
        hard_negatives = [
            (name, sim) for name, sim in similarities[:top_k * 2]  # Get more than needed for filtering
            if sim >= similarity_threshold
        ][:top_k]  # Then take just top-k after filtering
        
        results[anchor_name] = {'hard_negatives': hard_negatives}
    
    # Save results if requested
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            json_results = {
                anchor: {
                    'hard_negatives': [(name, float(sim)) for name, sim in data['hard_negatives']]
                }
                for anchor, data in results.items()
            }
            json.dump(json_results, f, indent=2)
        
        logger.info(f"Saved hard negative mining results to {save_path}")
    
    return results

def select_subject_split_file(config, args):
    """Unified subject split file selection logic"""
    split_file_path = None
    
    # 1. Priority: command-line argument
    if args.subject_split_file_location:
        split_file_path = args.subject_split_file_location
    # 2. Fallback: config file
    elif 'training' in config and config['training'].get('subject_split_file_location'):
        split_file_path = config['training']['subject_split_file_location']
    
    subject_folders = None
    if split_file_path:
        split_file_path = os.path.abspath(split_file_path)
        if os.path.isdir(split_file_path):
            txt_files = sorted([f for f in os.listdir(split_file_path) if f.endswith('.txt')])
            if not txt_files:
                print(f"No .txt split files found in directory: {split_file_path}")
            else:
                print("Available split files:")
                for idx, fname in enumerate(txt_files, 1):
                    print(f"  {idx}. {fname}")
                while True:
                    try:
                        choice = input(f"Select a split file by number (1-{len(txt_files)}): ").strip()
                        if not choice.isdigit() or not (1 <= int(choice) <= len(txt_files)):
                            print("Invalid selection. Please enter a valid number.")
                            continue
                        selected_file = txt_files[int(choice)-1]
                        split_file_path = os.path.join(split_file_path, selected_file)
                        break
                    except Exception:
                        print("Invalid input. Please try again.")
        # If it's a file, use it directly
        elif os.path.isfile(split_file_path):
            pass  # already set
        else:
            print(f"Provided split file path is invalid: {split_file_path}")
            split_file_path = None
    
    if split_file_path and os.path.isfile(split_file_path):
        with open(split_file_path, 'r') as f:
            subject_folders = [line.strip() for line in f if line.strip()]
        logger.info(f"Loaded {len(subject_folders)} subject folders from split file: {split_file_path}")
        batch_name = os.path.splitext(os.path.basename(split_file_path))[0]  # e.g., 'train_batch_1'
    else:
        user_input = input('No valid subject split file provided. Do you want to continue with all subject folders? (y/n): ').strip().lower()
        if user_input != 'y':
            print('Exiting. Please provide a split file or confirm to use all subjects.')
            exit(0)
        data_dir = config['data']['root_dir']
        subject_folders = [d for d in os.listdir(data_dir) 
                          if os.path.isdir(os.path.join(data_dir, d))]
        batch_name = None
    
    return subject_folders, batch_name, split_file_path

def get_mining_parameters_from_config(config):
    """Get mining parameters from config"""
    # Check both possible locations in config
    if 'hard_negative_mining' in config:
        hn_config = config['hard_negative_mining']
        top_k = hn_config.get('max_negatives_per_anchor', 5)
        similarity_threshold = hn_config.get('similarity_threshold', 0.0)
    elif 'hard_negatives' in config:
        hn_config = config['hard_negatives']
        top_k = hn_config.get('top_k', 5)
        similarity_threshold = hn_config.get('similarity_threshold', 0.0)
    else:
        # Default values
        top_k = 5
        similarity_threshold = 0.0
        
    logger.info(f"Using mining parameters: top_k={top_k}, similarity_threshold={similarity_threshold}")
    return top_k, similarity_threshold

def run_hard_negative_mining(args, config, subject_folders, output_dir):
    """Run the hard negative mining process"""
    data_dir = Path(config['data']['root_dir'])
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create output directories
    embeddings_dir = Path(output_dir) / 'embeddings'
    mining_dir = Path(output_dir) / 'mining_results'
    os.makedirs(embeddings_dir, exist_ok=True)
    os.makedirs(mining_dir, exist_ok=True)
    
    logger.info(f"Data directory: {data_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Device: {device}")
    
    # Get mining parameters
    top_k, similarity_threshold = get_mining_parameters_from_config(config)
    
    # Initialize feature extractor
    extractor = FeatureExtractor(
        model_name=config['model']['name'],
        device=device,
        batch_size=args.batch_size
    )
    
    # Create initial metadata for tracking
    metadata = {
        'processed_subjects': [],
        'total_subjects': len(subject_folders),
        'top_k': top_k,
        'similarity_threshold': similarity_threshold,
        'model_name': config['model']['name']
    }
    
    # Process each subject
    subject_folder_paths = [data_dir / subject for subject in subject_folders if (data_dir / subject).exists()]
    logger.info(f"Processing {len(subject_folder_paths)} subjects for hard negative mining")
    
    for subject_folder in subject_folder_paths:
        subject_id = subject_folder.name
        logger.info(f"Processing subject {subject_id}")
        
        # Define output paths
        embedding_path = embeddings_dir / f"{subject_id}_embeddings.h5"
        mining_path = mining_dir / f"{subject_id}_hard_negatives.json"
        
        try:
            # Extract features or load existing
            if embedding_path.exists() and not args.force:
                logger.info(f"Embeddings already exist for {subject_id}, loading")
                subject_embeddings = extractor.load_embeddings_from_h5(embedding_path)
            else:
                logger.info(f"Extracting features for subject {subject_id}")
                subject_embeddings = extractor.extract_features_for_subject(subject_folder, embedding_path)
                if subject_embeddings is None:
                    logger.warning(f"No embeddings generated for {subject_id}, skipping")
                    continue
            
            # Mine hard negatives
            if mining_path.exists() and not args.force:
                logger.info(f"Mining results already exist for {subject_id}, skipping mining")
            else:
                logger.info(f"Mining hard negatives for subject {subject_id}")
                mining_results = mine_hard_negatives(
                    subject_embeddings, 
                    top_k=top_k,
                    similarity_threshold=similarity_threshold,
                    save_path=mining_path
                )
                if not mining_results:
                    logger.warning(f"No hard negatives found for {subject_id}")
            
            # Add to processed subjects
            metadata['processed_subjects'].append(subject_id)
            
        except Exception as e:
            logger.error(f"Error processing subject {subject_id}: {str(e)}")
            continue
    
    # Save metadata
    with open(Path(output_dir) / 'hard_negative_metadata.json', 'w') as f:
        json.dump({
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
            'settings': {
                'model_name': config['model']['name'],
                'top_k': top_k,
                'similarity_threshold': similarity_threshold,
                'data_dir': str(data_dir)
            },
            'stats': {
                'total_subjects': len(subject_folder_paths),
                'processed_subjects': len(metadata['processed_subjects']),
                'subject_ids': metadata['processed_subjects']
            }
        }, f, indent=2)
    
    logger.info(f"Hard negative mining completed for {len(metadata['processed_subjects'])} out of {len(subject_folder_paths)} subjects")
    logger.info(f"Results saved to {output_dir}")
    
    return str(mining_dir)

def run_training(args, config, subject_folders, batch_name, hard_negative_dir, checkpoint_dir):
    """Run the training process"""
    device = torch.device(args.device if args.device else ('cuda' if torch.cuda.is_available() else 'cpu'))
    
    # Log device information
    logger.info(f"Using device: {device}")
    
    if torch.cuda.is_available() and device.type == 'cuda':
        # Log GPU information
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9  # Convert to GB
        logger.info(f"GPU: {gpu_name} with {gpu_memory:.2f} GB memory")
        
        # Set optimal CUDA settings
        torch.backends.cudnn.benchmark = config['optimization'].get('cuda_benchmark', True)
        if config['optimization'].get('max_split_size_mb'):
            torch.cuda.set_per_process_memory_fraction(0.95)  # Reserve some GPU memory for system
    
    # Initialize visualizer with config parameters
    viz_enabled = config.get('visualization', {}).get('enabled', True)
    visualizer = TrainingVisualizer(
        log_dir=os.path.join('runs', f"{batch_name}_hard_neg_training_{time.strftime('%Y%m%d_%H%M%S')}_hard_neg" if batch_name else f"run_{time.strftime('%Y%m%d_%H%M%S')}_hard_neg")
    ) if viz_enabled else None
    
    # Create datasets with memory management
    logger.info("Creating datasets...")
    
    # Get max memory limit from config
    max_memory_gb = config['data']['max_memory_cached']
    logger.info(f"Using memory limit of {max_memory_gb}GB for dataset cache")
    
    # Extract memory_efficient flag from config
    memory_efficient = config['optimization']['memory_efficient']
    
    # Force garbage collection before dataset creation
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    torch.set_num_threads(max(1, config['data']['num_workers']))  # Ensure positive integer for torch.set_num_threads
    dataset = TripletDataset(
        config['data']['root_dir'], 
        subject_folders,
        max_memory_gb=max_memory_gb,
        chunk_size=config['data']['chunk_size'],  # Read from config
        cache_flush_threshold=config['data']['cache_flush_threshold'],  # Read from config
        hard_negative_dir=hard_negative_dir,
        memory_efficient=memory_efficient
    )
    
    train_size = int(config['data']['train_ratio'] * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(config['data']['random_seed'])
    )
    
    # Use multiprocessing context from config (defaults set above based on platform)
    mp_context = config['data']['multiprocessing_context']
    
    logger.info(f"Using multiprocessing context: {mp_context}")
    
    # Create data loaders with OPTIMIZED settings
    # Use ChunkAwareSampler to reduce chunk switching
    train_sampler = ChunkAwareSampler(
        train_dataset, 
        chunk_size=config['data']['chunk_size'],  # Read from config
        shuffle=True,
        drop_last=config['data']['drop_last']
    )
    
    # OPTIMIZED: Better DataLoader settings for GPU training
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        sampler=train_sampler,  # Use custom sampler instead of shuffle
        num_workers=min(8, config['data']['num_workers']),  # Cap at 8 workers for stability
        pin_memory=config['data']['pin_memory'],
        prefetch_factor=config['data']['prefetch_factor'] if config['data']['num_workers'] > 0 and config['data']['prefetch_factor'] is not None else None,
        persistent_workers=config['data']['persistent_workers'] if config['data']['num_workers'] > 0 else False,
        multiprocessing_context=mp_context
    )
    
    # OPTIMIZED: Validation DataLoader with fewer workers
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=max(1, min(4, config['data']['num_workers'] // 2)),  # Cap at 4 workers for validation
        pin_memory=config['data']['pin_memory'],
        prefetch_factor=max(1, (config['data']['prefetch_factor'] or 2) // 2) if config['data']['num_workers'] > 0 and config['data']['prefetch_factor'] is not None else None,
        persistent_workers=config['data']['persistent_workers'] if config['data']['num_workers'] > 0 else False,
        drop_last=config['data']['drop_last'],
        multiprocessing_context=mp_context
    )
    
    logger.info(f"Created data loaders. Training batches: {len(train_loader)}, Validation batches: {len(val_loader)}")
    
    # Prompt user for checkpoint resume
    resume_from_checkpoint = False
    checkpoint_path = None
    while True:
        user_resume = input("Do you want to resume training from a previous checkpoint? (y/n): ").strip().lower()
        if user_resume == 'y':
            checkpoint_path = input("Please enter the path to the checkpoint file: ").strip()
            if os.path.isfile(checkpoint_path):
                resume_from_checkpoint = True
                break
            else:
                print(f"Checkpoint file '{checkpoint_path}' not found. Please try again or enter 'n' to start from scratch.")
        elif user_resume == 'n':
            break
        else:
            print("Invalid input. Please enter 'y' or 'n'.")

    # Initialize model
    logger.info("Initializing model...")
    model = DINOv2Retrieval(
        model_name=config['model']['name'],
        pretrained=config['model']['pretrained'],
        embedding_dim=int(config['model']['embedding_dim']),
        dropout=float(config['model']['dropout']),
        freeze_backbone=config['model']['freeze_backbone']
    ).to(device)
    
    # Add GPU device diagnostics
    logger.info(f"Model device: {next(model.parameters()).device}")
    logger.info(f"CUDA device count: {torch.cuda.device_count()}")
    if torch.cuda.is_available():
        logger.info(f"Current CUDA device: {torch.cuda.current_device()}")
        logger.info(f"CUDA device name: {torch.cuda.get_device_name()}")
        logger.info(f"CUDA memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        logger.info(f"CUDA memory cached: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
    
    # Initialize loss function
    criterion = TripletLoss(margin=float(config['training']['margin'])).to(device)
    
    # Test model forward pass
    logger.info("Testing model forward pass...")
    model.eval()
    model.train()  # Set back to training mode
    
    # Initialize optimizer
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(
        trainable_params,
        lr=float(config['training']['learning_rate']),
        weight_decay=float(config['training']['weight_decay'])
    )
    
    # Calculate total steps for warmup
    total_steps = len(train_loader) * int(config['training']['epochs'])
    warmup_steps = len(train_loader) * int(config['training'].get('warmup_epochs', 0))
    
    # Initialize scheduler
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    # If resuming from checkpoint, load states
    if resume_from_checkpoint and checkpoint_path:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        if 'optimizer_state_dict' in checkpoint:
            try:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            except ValueError:
                print("WARNING: Optimizer state could not be loaded due to parameter mismatch (likely due to freeze_backbone change). Initializing optimizer from scratch.")
        if 'scheduler_state_dict' in checkpoint and scheduler is not None:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        logger.info(f"Resumed training from checkpoint: {checkpoint_path}")
    
    # Add memory threshold to config for the trainer
    config['optimization']['memory_alert_threshold'] = args.ram_threshold
    
    # Create trainer with visualization support
    trainer = MemoryEfficientTrainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
        device=str(device),
        visualizer=visualizer,
        checkpoint_dir=checkpoint_dir,
        batch_name=batch_name  # Pass batch_name to trainer for checkpoint naming
    )
    
    # Train model
    run_name = f"{batch_name}_hard_neg_training_{time.strftime('%Y%m%d_%H%M%S')}_hard_neg" if batch_name else f"run_{time.strftime('%Y%m%d_%H%M%S')}_hard_neg"
    logger.info(f"Starting training run: {run_name}...")
    logger.info("Using hard negatives for more efficient training")
    
    try:
        results = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=config['training']['epochs']
        )
        
        # Create final visualizations and report
        logger.info("Creating final training report...")
        
        if visualizer is not None:
            visualizer.plot_loss_curves(
                [h['train_loss'] for h in results['training_history']],
                [h['val_loss'] for h in results['training_history']],
                len(results['training_history'])
            )
            
            # Get sample embeddings for visualization - OPTIMIZED for batched processing
            sample_data = next(iter(val_loader))
            with torch.no_grad():
                model.eval()
                # OPTIMIZED: Use batched forward pass
                batch_size = sample_data[0].size(0)
                sample_gpu_data = [d.to(device) for d in sample_data]
                combined_input = torch.cat(sample_gpu_data, dim=0)
                combined_output = model(combined_input)
                
                # Split outputs
                anchor_embeddings = combined_output[:batch_size]
                positive_embeddings = combined_output[batch_size:2*batch_size]
                negative_embeddings = combined_output[2*batch_size:]
                
                # Combine them for visualization
                embeddings = torch.cat([
                    anchor_embeddings,
                    positive_embeddings,
                    negative_embeddings
                ], dim=0)
                
                # Create meaningful categorical labels: 0=anchor, 1=positive, 2=negative
                category_labels = np.array([0]*batch_size + [1]*batch_size + [2]*batch_size)
                visualizer.visualize_embeddings(
                    embeddings.cpu().numpy(),
                    category_labels
                )
                
                # Plot example triplets (move this here)
                visualizer.plot_example_triplets(
                    sample_data[0][:5],  # anchors
                    sample_data[1][:5],  # positives
                    sample_data[2][:5]   # negatives
                )
                
                # Clean up
                del sample_data, sample_gpu_data, combined_input, combined_output
                del anchor_embeddings, positive_embeddings, negative_embeddings, embeddings
                torch.cuda.empty_cache()
            
            gc.collect()
            
            # Save final report
            visualizer.save_final_report({
                'best_val_loss': results['best_val_loss'],
                'final_train_loss': results['training_history'][-1]['train_loss'],
                'final_val_loss': results['training_history'][-1]['val_loss'],
                'total_epochs': len(results['training_history']),
                'model_config': config['model'],
                'use_hard_negatives': True
            })
            
            # Create HTML report
            visualizer.create_html_report()
            visualizer.close()
            
            logger.info(f"Training completed! Best validation loss: {results['best_val_loss']:.4f}")
            logger.info(f"Training visualizations and report saved in {visualizer.log_dir}")
            logger.info(f"Model checkpoints saved in {checkpoint_dir}")
        else:
            logger.info(f"Training completed! Best validation loss: {results['best_val_loss']:.4f}")
            logger.info("Visualization disabled - no plots generated")
            logger.info(f"Model checkpoints saved in {checkpoint_dir}")
        
        return results

    except Exception as e:
        logger.error(f"Training error: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return None

def get_args():
    parser = argparse.ArgumentParser(description="Consolidated DINOv2 Training Pipeline with Hard Negative Mining")
    
    # Configuration
    parser.add_argument("--config", type=str, default="config/training_config.yaml", 
                        help="Path to the configuration file (default: config/training_config.yaml)")
    
    # Pipeline control
    parser.add_argument("--skip-mining", action="store_true",
                        help="Skip hard negative mining (use existing results)")
    parser.add_argument("--skip-training", action="store_true",
                        help="Skip training (only run hard negative mining)")
    parser.add_argument("--force", action="store_true",
                        help="Force re-extraction and re-mining even if files exist")
    
    # Hardware settings
    parser.add_argument("--device", type=str, default=None,
                        help="Device to use (cuda or cpu)")
    parser.add_argument("--ram-threshold", type=int, default=85,
                        help="RAM usage threshold percentage (0-100) to trigger memory cleanup")
    
    # Data settings
    parser.add_argument("--subject-split-file-location", type=str, default=None,
                        help="Path to a .txt file listing subject folders to use (one per line)")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size for feature extraction and training")
    parser.add_argument("--workers", type=int, default=None,
                        help="Override number of workers from config")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Number of epochs for training (overrides config)")
    
    # Output settings
    parser.add_argument("--output-dir", type=str, default="hard_negative_output",
                        help="Path to output directory for hard negative mining results")
    parser.add_argument("--run-name", type=str, default=None,
                        help="Custom name for this training run (for logs and checkpoints)")
    
    # Advanced settings (rarely used, but available for override)
    parser.add_argument("--learning-rate", type=float, default=None,
                        help="Override learning rate from config")
    parser.add_argument("--margin", type=float, default=None,
                        help="Override triplet loss margin from config")
    parser.add_argument("--similarity-threshold", type=float, default=None,
                        help="Override hard negative similarity threshold from config")
    parser.add_argument("--max-negatives", type=int, default=None,
                        help="Override max negatives per anchor from config")
    
    return parser.parse_args()

def main():
    # Parse command line arguments
    args = get_args()
    set_memory_threshold(args.ram_threshold)
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Apply comprehensive defaults for parameters moved from config to code
    # This allows using simplified config files with only essential user parameters
    
    # Data defaults (implementation details)
    data_defaults = {
        'random_seed': 42,
        'pin_memory': True,
        'prefetch_factor': None,  # Auto-determined based on workers
        'persistent_workers': True,
        'drop_last': True,
        'multiprocessing_context': 'spawn',  # Platform will be handled below
        'cache_flush_threshold': 95
    }
    for key, value in data_defaults.items():
        config['data'].setdefault(key, value)
    
    # Platform-specific multiprocessing context
    if platform.system() != 'Windows':
        config['data']['multiprocessing_context'] = 'fork'  # Better performance on Linux/Mac
    
    # Model defaults (rarely changed)
    model_defaults = {
        'pretrained': True,
        'dropout': 0.1
    }
    for key, value in model_defaults.items():
        config['model'].setdefault(key, value)
    
    # Training defaults (calculated or rarely changed)
    config['training'].setdefault('warmup_epochs', 1)
    
    # Optimization defaults (implementation optimizations)
    config.setdefault('optimization', {
        'mixed_precision': True,
        'memory_efficient': True,
        'memory_alert_threshold': 85,
        'cuda_benchmark': True,
        'max_split_size_mb': 512
    })
    
    # Logging defaults (sensible defaults)
    config.setdefault('logging', {
        'level': 'INFO',
        'save_checkpoints': True,
        'checkpoint_interval': 5,
        'log_interval': 100
    })
    
    # Visualization defaults (sensible defaults)
    config.setdefault('visualization', {
        'enabled': True,
        'plot_interval': 1000,
        'save_embeddings': True,
        'embedding_plot_interval': 5
    })
    
    # Hard negative mining defaults
    config.setdefault('hard_negative_mining', {
        'enabled': True,
        'directory': 'hard_negative_output/mining_results',
        'similarity_threshold': 0.0,
        'max_negatives_per_anchor': 5
    })
    
    # Override config with command line arguments if provided
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size
        # Recalculate gradient accumulation steps
        config['training']['gradient_accumulation_steps'] = max(1, config['training']['effective_batch_size'] // args.batch_size)
    
    if args.workers:
        config['data']['num_workers'] = args.workers
        
    if args.epochs:
        config['training']['epochs'] = args.epochs
    
    # Override advanced config parameters if provided
    if args.learning_rate:
        config['training']['learning_rate'] = args.learning_rate
    
    if args.margin:
        config['training']['margin'] = args.margin
        
    if args.similarity_threshold:
        config.setdefault('hard_negative_mining', {})['similarity_threshold'] = args.similarity_threshold
        
    if args.max_negatives:
        config.setdefault('hard_negative_mining', {})['max_negatives_per_anchor'] = args.max_negatives
    
    # Calculate gradient accumulation steps if not provided (must be after batch size overrides)
    if 'gradient_accumulation_steps' not in config['training']:
        config['training']['gradient_accumulation_steps'] = max(1, 
            config['training']['effective_batch_size'] // config['training']['batch_size'])
    
    # Set device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Unified subject selection - ask once, use everywhere
    subject_folders, batch_name, split_file_path = select_subject_split_file(config, args)
    
    # Set run name with timestamp
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name or f"pipeline_{timestamp}"
    if batch_name:
        run_name = f"{batch_name}_hard_neg_training_{timestamp}_hard_neg"
    else:
        run_name += "_hard_neg"
    
    # Create checkpoint directory
    checkpoint_dir = os.path.join('checkpoints', run_name)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Set up centralized logging to file in real time
    log_file_path = os.path.join(checkpoint_dir, 'training_run.log')
    file_handler = setup_centralized_logging(log_file_path)
    
    # Apply logging level from config
    if 'logging' in config and 'level' in config['logging']:
        log_level = getattr(logging, config['logging']['level'].upper(), logging.INFO)
        logging.getLogger().setLevel(log_level)
    
    logger.info(f"Centralized logging setup complete. Logging to file: {log_file_path}")
    
    # Save full config to checkpoint directory
    with open(os.path.join(checkpoint_dir, 'config.yaml'), 'w') as f:
        # Add pipeline settings to config
        config['pipeline'] = {
            'hard_negative_dir': args.output_dir,
            'subject_split_file': split_file_path,
            'skip_mining': args.skip_mining,
            'skip_training': args.skip_training
        }
        yaml.dump(config, f)
    
    # Start memory monitoring thread
    memory_thread = start_memory_monitoring()
    
    # Display pipeline settings
    logger.info("="*60)
    logger.info("CONSOLIDATED DINOV2 TRAINING PIPELINE")
    logger.info("="*60)
    logger.info(f"Memory management settings:")
    logger.info(f"  - RAM threshold: {args.ram_threshold}%")
    logger.info(f"  - Dataset cache limit: {config['data']['max_memory_cached']} GB")
    logger.info(f"  - Batch size: {config['training']['batch_size']}")
    logger.info(f"  - Workers: {config['data']['num_workers']}")
    logger.info(f"  - Gradient accumulation steps: {config['training']['gradient_accumulation_steps']}")
    logger.info(f"  - Dataset chunk size: {config['data']['chunk_size']} triplets")
    logger.info(f"  - Cache flush threshold: {config['data']['cache_flush_threshold']}% RAM usage")
    logger.info(f"Pipeline settings:")
    logger.info(f"  - Output directory: {args.output_dir}")
    logger.info(f"  - Subject split file: {split_file_path}")
    logger.info(f"  - Selected subjects: {len(subject_folders)}")
    logger.info(f"  - Skip mining: {args.skip_mining}")
    logger.info(f"  - Skip training: {args.skip_training}")
    logger.info(f"  - Device: {device}")
    
    logger.info(f"Training parameters:")
    logger.info(f"  - Learning rate: {config['training']['learning_rate']}")
    logger.info(f"  - Weight decay: {config['training']['weight_decay']}")
    logger.info(f"  - Triplet margin: {config['training']['margin']}")
    logger.info(f"  - Warmup epochs: {config['training'].get('warmup_epochs', 0)}")
    
    logger.info(f"Model configuration:")
    logger.info(f"  - Model name: {config['model']['name']}")
    logger.info(f"  - Embedding dim: {config['model']['embedding_dim']}")
    logger.info(f"  - Dropout: {config['model']['dropout']}")
    logger.info(f"  - Freeze backbone: {config['model']['freeze_backbone']}")
    
    logger.info(f"Hard negative mining:")
    hn_config = config.get('hard_negative_mining', {})
    logger.info(f"  - Enabled: {hn_config.get('enabled', True)}")
    logger.info(f"  - Similarity threshold: {hn_config.get('similarity_threshold', 0.0)}")
    logger.info(f"  - Max negatives per anchor: {hn_config.get('max_negatives_per_anchor', 5)}")
    
    logger.info(f"Visualization & Logging:")
    viz_config = config.get('visualization', {})
    log_config = config.get('logging', {})
    logger.info(f"  - Visualization enabled: {viz_config.get('enabled', True)}")
    logger.info(f"  - Save embeddings: {viz_config.get('save_embeddings', True)}")
    logger.info(f"  - Checkpoint interval: {log_config.get('checkpoint_interval', 5)} epochs")
    logger.info(f"  - Log level: {log_config.get('level', 'INFO')}")
    
    hard_negative_dir = None
    
    # Check if hard negative mining is enabled in config
    hn_enabled = config.get('hard_negative_mining', {}).get('enabled', True)
    
    # Step 1: Hard Negative Mining
    if not args.skip_mining and hn_enabled:
        logger.info("="*60)
        logger.info("STEP 1: HARD NEGATIVE MINING")
        logger.info("="*60)
        hard_negative_dir = run_hard_negative_mining(args, config, subject_folders, args.output_dir)
        logger.info(f"Hard negative mining completed. Results saved to: {hard_negative_dir}")
    elif not hn_enabled:
        logger.info("Hard negative mining disabled in config. Skipping mining step.")
        hard_negative_dir = None
    else:
        # Use config directory if available, otherwise fallback to args
        default_dir = config.get('hard_negative_mining', {}).get('directory', 'hard_negative_output/mining_results')
        hard_negative_dir = os.path.join(os.path.dirname(default_dir), 'mining_results') if os.path.dirname(default_dir) else args.output_dir + '/mining_results'
        if not os.path.exists(hard_negative_dir):
            logger.error(f"Mining results directory not found: {hard_negative_dir}")
            logger.error("Please run without --skip-mining first or ensure mining results exist.")
            return
        logger.info(f"Skipping mining, using existing results from: {hard_negative_dir}")
    
    # Step 2: Training
    if not args.skip_training:
        logger.info("="*60)
        logger.info("STEP 2: TRAINING WITH HARD NEGATIVES")
        logger.info("="*60)
        results = run_training(args, config, subject_folders, batch_name, hard_negative_dir, checkpoint_dir)
        if results:
            logger.info("="*60)
            logger.info("PIPELINE COMPLETED SUCCESSFULLY!")
            logger.info("="*60)
            logger.info(f"Best validation loss: {results['best_val_loss']:.4f}")
            logger.info(f"Training checkpoints: {checkpoint_dir}")
            logger.info(f"Hard negative results: {hard_negative_dir}")
        else:
            logger.error("Training failed!")
    else:
        logger.info("Skipping training step as requested.")
        logger.info("="*60)
        logger.info("HARD NEGATIVE MINING COMPLETED!")
        logger.info("="*60)
        logger.info(f"Hard negative results: {hard_negative_dir}")
        logger.info("To run training, execute this script again without --skip-training")
    
    # Stop memory monitor thread
    stop_memory_monitoring()
    memory_thread.join(timeout=1.0)
    
    # Final cleanup
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    logger.info("Pipeline execution completed.")

if __name__ == "__main__":
    main() 