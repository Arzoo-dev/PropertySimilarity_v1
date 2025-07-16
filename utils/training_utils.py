#!/usr/bin/env python3
"""
Training Utilities Module
=========================

Consolidated training-related utilities combining metrics and scheduling functionality.
This module eliminates the need for separate tiny files while keeping training utilities organized.

Contains:
- TripletLoss: Memory-efficient triplet loss with cosine similarity
- DataLoaderMonitor: Performance monitoring and bottleneck identification  
- get_cosine_schedule_with_warmup: Learning rate scheduling with warmup

Originally consolidated from:
- utils/metrics.py (TripletLoss class)
- utils/schedulers.py (DataLoaderMonitor and scheduler functions)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time
import threading
from collections import deque
import logging
from typing import Dict
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR

# =============================================================================
# LOSS FUNCTIONS
# =============================================================================

class TripletLoss(nn.Module):
    """Memory-efficient implementation of triplet loss with cosine similarity
    
    Originally from utils/metrics.py. Uses cosine similarity instead of Euclidean
    distance for better performance with normalized embeddings.
    """
    def __init__(self, margin=0.2):
        super().__init__()
        self.margin = margin
    
    def forward(self, anchor, positive, negative):
        # Normalize embeddings for cosine similarity
        anchor = F.normalize(anchor, p=2, dim=1)
        positive = F.normalize(positive, p=2, dim=1)
        negative = F.normalize(negative, p=2, dim=1)
        
        # Compute similarities
        pos_sim = torch.sum(anchor * positive, dim=1)
        neg_sim = torch.sum(anchor * negative, dim=1)
        
        # Compute loss with margin
        losses = F.relu(self.margin - (pos_sim - neg_sim))
        loss = losses.mean()
        
        # Free memory
        del anchor, positive, negative
        del pos_sim, neg_sim
        
        return loss

# =============================================================================
# PERFORMANCE MONITORING
# =============================================================================

class DataLoaderMonitor:
    """Monitor DataLoader performance and identify bottlenecks
    
    Originally from utils/schedulers.py. Provides detailed timing analysis
    of data loading, GPU transfer, forward/backward passes, and optimization steps.
    """
    
    def __init__(self, window_size=100):
        self.window_size = window_size
        self.lock = threading.Lock()
        self.reset_data()
        self.logger = logging.getLogger(__name__)
        
    def reset_data(self):
        """Reset all timing data - call this at start of each epoch"""
        with self.lock:
            # Clear existing deques if they exist, otherwise create new ones
            if hasattr(self, 'load_times'):
                self.load_times.clear()
                self.transfer_times.clear()
                self.forward_times.clear()
                self.backward_times.clear()
                self.optimizer_times.clear()
                self.total_times.clear()
            else:
                # First time initialization
                self.load_times = deque(maxlen=self.window_size)
                self.transfer_times = deque(maxlen=self.window_size)
                self.forward_times = deque(maxlen=self.window_size)
                self.backward_times = deque(maxlen=self.window_size)
                self.optimizer_times = deque(maxlen=self.window_size)
                self.total_times = deque(maxlen=self.window_size)
    
    def record_batch_timing(self, load_time, transfer_time, forward_time, backward_time, optimizer_time, total_time):
        """Record timing for a batch - store only primitive values"""
        with self.lock:
            # Store only primitive float values, not tensor references
            self.load_times.append(float(load_time))
            self.transfer_times.append(float(transfer_time))
            self.forward_times.append(float(forward_time))
            self.backward_times.append(float(backward_time))
            self.optimizer_times.append(float(optimizer_time))
            self.total_times.append(float(total_time))
    
    def get_statistics(self):
        """Get performance statistics"""
        with self.lock:
            if not self.total_times:
                return None
                
            stats = {
                'load_time': {
                    'mean': sum(self.load_times) / len(self.load_times),
                    'max': max(self.load_times),
                    'min': min(self.load_times)
                },
                'transfer_time': {
                    'mean': sum(self.transfer_times) / len(self.transfer_times),
                    'max': max(self.transfer_times),
                    'min': min(self.transfer_times)
                },
                'forward_time': {
                    'mean': sum(self.forward_times) / len(self.forward_times),
                    'max': max(self.forward_times),
                    'min': min(self.forward_times)
                },
                'backward_time': {
                    'mean': sum(self.backward_times) / len(self.backward_times),
                    'max': max(self.backward_times),
                    'min': min(self.backward_times)
                },
                'optimizer_time': {
                    'mean': sum(self.optimizer_times) / len(self.optimizer_times),
                    'max': max(self.optimizer_times),
                    'min': min(self.optimizer_times)
                },
                'total_time': {
                    'mean': sum(self.total_times) / len(self.total_times),
                    'max': max(self.total_times),
                    'min': min(self.total_times)
                }
            }
            
            # Calculate bottlenecks
            total_mean = stats['total_time']['mean']
            load_pct = (stats['load_time']['mean'] / total_mean) * 100
            transfer_pct = (stats['transfer_time']['mean'] / total_mean) * 100
            forward_pct = (stats['forward_time']['mean'] / total_mean) * 100
            backward_pct = (stats['backward_time']['mean'] / total_mean) * 100
            optimizer_pct = (stats['optimizer_time']['mean'] / total_mean) * 100
            
            stats['bottlenecks'] = {
                'load_pct': load_pct,
                'transfer_pct': transfer_pct,
                'forward_pct': forward_pct,
                'backward_pct': backward_pct,
                'optimizer_pct': optimizer_pct
            }
            
            return stats
    
    def log_performance_report(self):
        """Log a comprehensive performance report"""
        stats = self.get_statistics()
        if not stats:
            return
            
        self.logger.info("=== DataLoader Performance Report ===")
        self.logger.info(f"Total batches analyzed: {len(self.total_times)}")
        self.logger.info(f"Average batch time: {stats['total_time']['mean']:.3f}s")
        self.logger.info(f"Data loading: {stats['load_time']['mean']:.3f}s ({stats['bottlenecks']['load_pct']:.1f}%)")
        self.logger.info(f"GPU transfer: {stats['transfer_time']['mean']:.3f}s ({stats['bottlenecks']['transfer_pct']:.1f}%)")
        self.logger.info(f"Forward pass: {stats['forward_time']['mean']:.3f}s ({stats['bottlenecks']['forward_pct']:.1f}%)")
        self.logger.info(f"Backward pass: {stats['backward_time']['mean']:.3f}s ({stats['bottlenecks']['backward_pct']:.1f}%)")
        self.logger.info(f"Optimizer step: {stats['optimizer_time']['mean']:.3f}s ({stats['bottlenecks']['optimizer_pct']:.1f}%)")
        
        # Identify main bottleneck
        bottlenecks = stats['bottlenecks']
        main_bottleneck = max(bottlenecks.items(), key=lambda x: x[1])
        self.logger.info(f"Main bottleneck: {main_bottleneck[0]} ({main_bottleneck[1]:.1f}%)")
        
        # Provide recommendations
        if bottlenecks['load_pct'] > 50:
            self.logger.warning("Data loading is the main bottleneck. Consider:")
            self.logger.warning("- Increasing num_workers")
            self.logger.warning("- Increasing prefetch_factor")
            self.logger.warning("- Using pin_memory=True")
            self.logger.warning("- Reducing chunk switching frequency")
        elif bottlenecks['transfer_pct'] > 30:
            self.logger.warning("GPU transfer is slow. Consider:")
            self.logger.warning("- Using pin_memory=True")
            self.logger.warning("- Reducing batch size")
            self.logger.warning("- Using mixed precision training")
        elif bottlenecks['forward_pct'] > 60:
            self.logger.warning("Forward pass is the bottleneck. Consider:")
            self.logger.warning("- Using mixed precision training")
            self.logger.warning("- Reducing model complexity")
            self.logger.warning("- Using gradient checkpointing")
        
        self.logger.info("=====================================")

# =============================================================================
# LEARNING RATE SCHEDULING
# =============================================================================

def get_cosine_schedule_with_warmup(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float = 0.5,
    last_epoch: int = -1
):
    """
    Creates a cosine learning rate schedule with linear warmup.
    
    Originally from utils/schedulers.py. Provides smooth learning rate scheduling
    with an initial warmup phase followed by cosine annealing.
    
    Args:
        optimizer: The optimizer for which to schedule the learning rate
        num_warmup_steps: The number of steps for the warmup phase
        num_training_steps: The total number of training steps
        num_cycles: The number of cycles in the cosine decay
        last_epoch: The index of the last epoch
        
    Returns:
        A scheduler with the appropriate schedule
    """
    def lr_lambda(current_step):
        # Linear warmup phase
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        
        # Cosine decay phase
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return LambdaLR(optimizer, lr_lambda, last_epoch) 