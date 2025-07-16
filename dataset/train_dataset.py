import os
import torch
import gc
from torch.utils.data import Dataset, DataLoader, Sampler
import torch.nn.functional as F
from torchvision.io import read_image
from torchvision import transforms
import random
import json
from typing import List, Tuple, Dict, Optional
import logging
from functools import partial
import numpy as np
import psutil
import time
from collections import OrderedDict

logger = logging.getLogger(__name__)

class LRUCache:
    """Efficient LRU cache implementation for image caching"""
    def __init__(self, max_size: int):
        self.max_size = max_size
        self.cache = OrderedDict()
    
    def get(self, key):
        """Get item from cache and mark as recently used"""
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        return None
    
    def put(self, key, value):
        """Add item to cache with LRU eviction"""
        if key in self.cache:
            self.cache.move_to_end(key)
        else:
            if len(self.cache) >= self.max_size:
                self.cache.popitem(last=False)  # O(1) operation
        self.cache[key] = value
    
    def clear(self):
        """Clear all items from cache"""
        self.cache.clear()
    
    def __len__(self):
        return len(self.cache)

class ChunkAwareSampler(Sampler):
    """
    Sampler that ensures samples from the same chunk are processed together
    to minimize cache misses and improve memory efficiency.
    """
    def __init__(self, data_source, chunk_size, shuffle=True, drop_last=False):
        self.data_source = data_source
        self.chunk_size = chunk_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        
        # Calculate number of chunks
        self.num_chunks = (len(data_source) + chunk_size - 1) // chunk_size
        
    def __iter__(self):
        # Generate chunk indices
        chunk_indices = list(range(self.num_chunks))
        
        if self.shuffle:
            random.shuffle(chunk_indices)
        
        # Generate indices within each chunk
        indices = []
        for chunk_id in chunk_indices:
            start_idx = chunk_id * self.chunk_size
            end_idx = min(start_idx + self.chunk_size, len(self.data_source))
            
            chunk_indices_list = list(range(start_idx, end_idx))
            if self.shuffle:
                random.shuffle(chunk_indices_list)
            
            indices.extend(chunk_indices_list)
        
        return iter(indices)
    
    def __len__(self):
        return len(self.data_source)

class TripletDataset(Dataset):
    """
    TripletDataset optimized for large-memory systems (e.g., H100 SXM, 125GB RAM):
    - chunk_size and cache_flush_threshold should be provided from config file via train.py
    - chunk_size controls number of triplets per memory chunk
    - cache_flush_threshold controls when to clear cache based on RAM usage
    """
    def __init__(self, root_dir: str, subject_folders: List[str], transform=None, max_memory_gb=4, 
                 chunk_size=None, cache_flush_threshold=None, hard_negative_dir=None, memory_efficient=True):
        logger.info(f"Initializing TripletDataset with {len(subject_folders)} subjects")
        self.root_dir = root_dir
        self.subject_folders = subject_folders
        self.preload_size = (224, 224)  # Smaller initial load size to save memory
        self.final_size = (518, 518)
        self.transform = transform or self._get_default_transform()
        self.max_memory_gb = max_memory_gb
        
        # Hard negative settings
        self.hard_negative_dir = hard_negative_dir
        self.use_hard_negatives = hard_negative_dir is not None and os.path.exists(hard_negative_dir)
        if self.use_hard_negatives:
            logger.info(f"Using hard negatives from {hard_negative_dir}")
            self.hard_negative_mappings = self._load_hard_negative_mappings()
        else:
            logger.info("Using random negatives (no hard negative mining)")
            self.hard_negative_mappings = {}
        
        # Chunked loading settings - values should be passed from config
        if chunk_size is None or cache_flush_threshold is None:
            raise ValueError("chunk_size and cache_flush_threshold must be provided from config file")
        
        self.chunk_size = chunk_size
        self.cache_flush_threshold = cache_flush_threshold
        self.current_chunk_indices = set()  # Track indices in the current active chunk
        self.last_chunk_id = -1
        
        # OPTIMIZED: Use LRU cache instead of OrderedDict for better performance
        max_cache_size = min(self.chunk_size * 3, 10000)  # Cap cache size
        self.cache = LRUCache(max_cache_size)
        self.cache_hits = 0
        self.cache_misses = 0
        self.last_memory_check = time.time()
        self.memory_check_interval = 10  # Check memory every 10 seconds (increased from 5)
        
        logger.info("Generating triplets...")
        self.triplets = self._generate_triplets()
        logger.info(f"Generated {len(self.triplets)} triplets in total")

        # Save triplets as JSON in hard_negative_output/triplets
        triplet_save_dir = os.path.join('hard_negative_output', 'triplets')
        os.makedirs(triplet_save_dir, exist_ok=True)
        triplet_save_path = os.path.join(triplet_save_dir, 'all_triplets.json')
        triplet_dicts = [
            {"anchor": t[0], "positive": t[1], "negative": t[2]} for t in self.triplets
        ]
        with open(triplet_save_path, 'w') as f:
            json.dump(triplet_dicts, f, indent=2)
        logger.info(f"Saved all triplets to {triplet_save_path}")
        
        # Calculate number of chunks
        self.num_chunks = (len(self.triplets) + self.chunk_size - 1) // self.chunk_size
        logger.info(f"Data will be processed in {self.num_chunks} chunks of {self.chunk_size} triplets each")
        
        # Force garbage collection after initialization
        gc.collect()

    def _get_default_transform(self):
        logger.info("Using optimized image transforms")
        return transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def _load_hard_negative_mappings(self):
        """Load hard negative mappings from JSON files in the hard negative directory"""
        mappings = {}
        
        if not self.hard_negative_dir or not os.path.exists(self.hard_negative_dir):
            logger.warning("Hard negative directory not found or not specified")
            return mappings
            
        count = 0
        for subject in self.subject_folders:
            subject_id = os.path.basename(subject)
            hn_file = os.path.join(self.hard_negative_dir, f"{subject_id}_hard_negatives.json")
            
            if os.path.exists(hn_file):
                try:
                    with open(hn_file, 'r') as f:
                        subject_mappings = json.load(f)
                    mappings[subject_id] = subject_mappings
                    count += 1
                except Exception as e:
                    logger.error(f"Error loading hard negative file {hn_file}: {str(e)}")
        
        logger.info(f"Loaded hard negative mappings for {count} subjects")
        return mappings

    def _generate_triplets(self) -> List[Tuple[str, str, str]]:
        triplets = []
        total_triplets = 0
        hard_negative_count = 0
        # random_negative_count = 0  # No longer used
        
        for subject in self.subject_folders:
            logger.info(f"Processing subject: {subject}")
            subject_path = os.path.join(self.root_dir, subject)
            subject_id = os.path.basename(subject_path)
            
            # Get all images in each folder
            anchor_imgs = [f for f in os.listdir(os.path.join(subject_path, 'anchor')) 
                         if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            positive_imgs = [f for f in os.listdir(os.path.join(subject_path, 'positive')) 
                           if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            negative_imgs = [f for f in os.listdir(os.path.join(subject_path, 'negative')) 
                           if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            logger.info(f"Found {len(anchor_imgs)} anchor, {len(positive_imgs)} positive, {len(negative_imgs)} negative images for {subject}")

            # Use hard negatives if available for this subject
            subject_hard_negatives = {}
            if self.use_hard_negatives and subject_id in self.hard_negative_mappings:
                subject_hard_negatives = self.hard_negative_mappings[subject_id]

            # Create triplets with hard negatives only
            subject_triplets = 0
            for anchor in anchor_imgs:
                for positive in positive_imgs:
                    if subject_hard_negatives and anchor in subject_hard_negatives:
                        try:
                            hard_neg_list = subject_hard_negatives[anchor]['hard_negatives']
                            if hard_neg_list and len(hard_neg_list) > 0:
                                hard_neg_name, similarity = hard_neg_list[0]
                                triplets.append((
                                    os.path.join(subject_path, 'anchor', anchor),
                                    os.path.join(subject_path, 'positive', positive),
                                    os.path.join(subject_path, 'negative', hard_neg_name)
                                ))
                                hard_negative_count += 1
                                subject_triplets += 1
                        except Exception as e:
                            logger.warning(f"Error using hard negative for {anchor}: {str(e)}")
                    # No random negative fallback!
            total_triplets += subject_triplets
            logger.info(f"Generated {subject_triplets} triplets for subject {subject}")

        if self.use_hard_negatives:
            logger.info(f"Triplet generation complete. Total: {total_triplets}, Hard negatives: {hard_negative_count}")
        else:
            logger.info(f"Triplet generation complete. Total triplets (all random): {total_triplets}")
        
        return triplets

    def __len__(self):
        return len(self.triplets)

    def _check_memory_usage(self, force_check=False):
        """Check memory usage and clear cache if needed - OPTIMIZED for less overhead"""
        current_time = time.time()
        
        # Only check periodically unless forced
        if not force_check and current_time - self.last_memory_check < self.memory_check_interval:
            return
            
        self.last_memory_check = current_time
        mem = psutil.virtual_memory()
        mem_usage_percent = mem.percent
        mem_usage_gb = mem.used / 1e9
        
        # Clear cache if memory usage is above threshold
        if mem_usage_percent > self.cache_flush_threshold:
            cache_size_before = len(self.cache)
            logger.info(f"System RAM usage high ({mem_usage_percent:.1f}% ({mem_usage_gb:.1f} GB) > {self.cache_flush_threshold}%), clearing cache")
            self.cache.clear()
            self.current_chunk_indices.clear()
            gc.collect()
            mem_after = psutil.virtual_memory()
            mem_usage_after_percent = mem_after.percent
            mem_usage_after_gb = mem_after.used / 1e9
            logger.info(f"Cache cleared: {cache_size_before} items removed. System RAM: {mem_usage_percent:.1f}% ({mem_usage_gb:.1f} GB) → {mem_usage_after_percent:.1f}% ({mem_usage_after_gb:.1f} GB)")

    def _get_chunk_id(self, idx):
        """Get the chunk ID for a given index"""
        return idx // self.chunk_size

    def _load_chunk(self, chunk_id):
        """Load a specific chunk of data into memory - OPTIMIZED"""
        if chunk_id == self.last_chunk_id:
            return  # Chunk already loaded
            
        # Clear previous chunk data
        if len(self.cache) > 0:
            logger.info(f"Switching from chunk {self.last_chunk_id} to {chunk_id}, clearing cache")
            self.cache.clear()
            self.current_chunk_indices.clear()
            gc.collect()
        
        # Calculate chunk range
        start_idx = chunk_id * self.chunk_size
        end_idx = min(start_idx + self.chunk_size, len(self.triplets))
        
        logger.info(f"Loading chunk {chunk_id} ({start_idx}-{end_idx-1})")
        self.last_chunk_id = chunk_id
        
        # We don't preload the images here, just mark the chunk as active
        # Images will be loaded on demand in _load_and_process_image

    def _load_and_process_image(self, img_path):
        """Load and process a single image with OPTIMIZED memory management"""
        # Check if image is in cache
        cached_img = self.cache.get(img_path)
        if cached_img is not None:
            self.cache_hits += 1
            return cached_img
            
        # Not in cache, load it
        self.cache_misses += 1
        try:
            img = read_image(img_path).float() / 255.0
            
            # OPTIMIZED: Handle channel conversion more efficiently
            if img.shape[0] == 1:
                img = img.expand(3, -1, -1)  # More efficient than repeat
            elif img.shape[0] == 4:
                img = img[:3, :, :]  # Keep only the first 3 channels (RGB)
                
            # OPTIMIZED: Resize more efficiently
            img = F.interpolate(img.unsqueeze(0), size=self.final_size, 
                              mode='bilinear', align_corners=False).squeeze(0)
            
            # Add to cache using LRU
            self.cache.put(img_path, img)
            
            # Check memory usage less frequently
            self._check_memory_usage()
                
            return img
        except Exception as e:
            logger.error(f"Error loading image {img_path}: {str(e)}")
            raise

    def _load_and_process_image_batch(self, img_paths):
        """OPTIMIZED: Load multiple images efficiently in batch"""
        images = []
        uncached_paths = []
        uncached_indices = []
        
        # Check cache first
        for i, path in enumerate(img_paths):
            cached_img = self.cache.get(path)
            if cached_img is not None:
                images.append(cached_img)
                self.cache_hits += 1
            else:
                images.append(None)  # Placeholder
                uncached_paths.append(path)
                uncached_indices.append(i)
                self.cache_misses += 1
        
        # Load uncached images
        if uncached_paths:
            for i, path in enumerate(uncached_paths):
                try:
                    img = read_image(path).float() / 255.0
                    
                    # Handle channel conversion efficiently
                    if img.shape[0] == 1:
                        img = img.expand(3, -1, -1)
                    elif img.shape[0] == 4:
                        img = img[:3, :, :]
                    
                    # Resize efficiently
                    img = F.interpolate(img.unsqueeze(0), size=self.final_size, 
                                      mode='bilinear', align_corners=False).squeeze(0)
                    
                    # Add to cache and result list
                    self.cache.put(path, img)
                    images[uncached_indices[i]] = img
                    
                except Exception as e:
                    logger.error(f"Error loading image {path}: {str(e)}")
                    raise
        
        return images

    def __getitem__(self, idx):
        # Check if we need to load a different chunk
        chunk_id = self._get_chunk_id(idx)
        if chunk_id != self.last_chunk_id:
            self._load_chunk(chunk_id)
        
        # Add this index to current chunk indices
        self.current_chunk_indices.add(idx)
        
        anchor_path, positive_path, negative_path = self.triplets[idx]

        try:
            # OPTIMIZED: Load images with batch processing when possible
            anchor = self._load_and_process_image(anchor_path)
            positive = self._load_and_process_image(positive_path)
            negative = self._load_and_process_image(negative_path)
            
            if self.transform:
                anchor = self.transform(anchor)
                positive = self.transform(positive)
                negative = self.transform(negative)

            # Log cache stats less frequently
            if idx % 2000 == 0:  # Increased from 1000
                total = self.cache_hits + self.cache_misses
                hit_rate = self.cache_hits / total if total > 0 else 0
                mem = psutil.virtual_memory()
                mem_usage_percent = mem.percent
                mem_usage_gb = mem.used / 1e9
                logger.info(f"Cache hit rate: {hit_rate:.2%}, System RAM: {mem_usage_percent:.1f}% ({mem_usage_gb:.1f} GB), Cache size: {len(self.cache)}")
                # Force memory check
                self._check_memory_usage(force_check=True)
                
            return anchor, positive, negative
        except Exception as e:
            logger.error(f"Error loading triplet at index {idx}: {str(e)}")
            logger.error(f"Paths - Anchor: {anchor_path}, Positive: {positive_path}, Negative: {negative_path}")
            raise
