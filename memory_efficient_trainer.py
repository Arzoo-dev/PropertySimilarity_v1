#!/usr/bin/env python3
"""
Memory Efficient Trainer
========================

Optimized trainer for DINOv2-based triplet learning with comprehensive memory management,
asynchronous GPU transfers, batched forward passes, and enhanced monitoring.

Key Features:
- Memory-efficient training with automatic cleanup
- Asynchronous GPU transfers using CUDA streams
- Batched forward passes for better GPU utilization
- Comprehensive performance monitoring
- Early stopping and checkpoint management
- Integration with visualization and logging systems
"""

import os
import gc
import logging
import torch
import torch.cuda.amp as amp
from torch.utils.data import DataLoader
from typing import Dict, Any, cast
import psutil
import time
from utils.training_utils import DataLoaderMonitor

logger = logging.getLogger(__name__)

class MemoryEfficientTrainer:
    def __init__(self, model, criterion, optimizer, scheduler, config, device='cuda', visualizer=None, checkpoint_dir=None, batch_name=None):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config
        self.device = device
        self.visualizer = visualizer
        self.checkpoint_dir = checkpoint_dir or os.path.join('checkpoints', 'default')
        self.batch_name = batch_name
        
        # Initialize mixed precision training
        self.scaler = amp.GradScaler(enabled=config['optimization']['mixed_precision'])
        
        # Memory management settings - OPTIMIZED for better performance
        mem_eff = config['optimization'].get('memory_efficient', True)
        if mem_eff:
            # Less frequent cleanup for better performance
            self.gc_interval = 200        # Every 200 steps instead of 50
            self.clear_cache_interval = 50 # Every 50 steps instead of 10
            self.empty_cache_threshold = 0.85  # Slightly higher threshold
        else:
            # Relaxed (infrequent) memory management
            self.gc_interval = 500
            self.clear_cache_interval = 100
            self.empty_cache_threshold = 0.95
        
        self.memory_alert_threshold = config['optimization'].get('memory_alert_threshold', 85)
        
        # Training settings
        self.micro_batch_size = config['training']['batch_size']
        self.effective_batch_size = config['training']['effective_batch_size']
        self.grad_accum_steps = config['training']['gradient_accumulation_steps']
        
        # Verify settings
        if self.effective_batch_size != self.micro_batch_size * self.grad_accum_steps:
            logger.warning(
                f"Effective batch size ({self.effective_batch_size}) does not match "
                f"micro_batch_size ({self.micro_batch_size}) * grad_accum_steps ({self.grad_accum_steps})"
            )
        
        # Track memory stats
        self.peak_memory_usage = 0
        self.emergency_gc_count = 0
        self.last_memory_log = time.time()
        
        # Ensure checkpoint directory exists
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        self.data_monitor = DataLoaderMonitor(window_size=100)
        
        # Create CUDA stream for async transfers
        if torch.cuda.is_available():
            self.transfer_stream = torch.cuda.Stream()
        else:
            self.transfer_stream = None

    def _manage_memory(self, step: int):
        """Enhanced memory management routine - OPTIMIZED for less overhead"""
        # Always check system memory
        system_memory = psutil.virtual_memory().percent
        
        # Track peak memory usage
        if system_memory > self.peak_memory_usage:
            self.peak_memory_usage = system_memory
        
        # Log memory stats periodically (less frequent)
        current_time = time.time()
        if current_time - self.last_memory_log > 120:  # Log every 2 minutes instead of 1
            gpu_mem = f"GPU: {torch.cuda.memory_allocated() / 1e9:.2f}GB / {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f}GB" if torch.cuda.is_available() else "GPU: N/A"
            logger.info(f"Memory stats - RAM: {system_memory:.1f}% | {gpu_mem} | Peak RAM: {self.peak_memory_usage:.1f}% | Emergency GCs: {self.emergency_gc_count}")
            self.last_memory_log = current_time
        
        # Critical memory situation
        if system_memory > self.memory_alert_threshold:
            logger.warning(f"⚠️ Memory usage critical: {system_memory}% > threshold {self.memory_alert_threshold}%")
            self.emergency_gc_count += 1
            
            # Aggressive cleanup
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # If still critical after cleanup, pause briefly to let system recover
            if psutil.virtual_memory().percent > self.memory_alert_threshold:
                logger.warning("Still above threshold after cleanup, pausing briefly...")
                time.sleep(1)  # Reduced pause time
            
            return
            
        # Regular memory management - LESS FREQUENT for better performance
        if step % self.gc_interval == 0:
            gc.collect()
            
        if step % self.clear_cache_interval == 0 and torch.cuda.is_available():
            # Get current GPU memory usage
            allocated = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated() if torch.cuda.max_memory_allocated() > 0 else 0
            if allocated > self.empty_cache_threshold:
                torch.cuda.empty_cache()

    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Single epoch training with OPTIMIZED memory management and async transfers"""
        self.model.train()
        total_loss = 0
        num_batches = len(train_loader)
        
        # Log training start
        logger.info(f"Starting training epoch with {num_batches} batches")
        
        for batch_idx, (anchors, positives, negatives) in enumerate(train_loader):
            try:
                if batch_idx == 0:
                    logger.info(f"[Batch 0] anchors: {anchors.shape}, positives: {positives.shape}, negatives: {negatives.shape}")
                batch_start = time.time()
                
                # OPTIMIZED: Asynchronous GPU transfer with CUDA stream
                load_time = time.time() - batch_start
                transfer_start = time.time()
                
                if self.transfer_stream is not None:
                    with torch.cuda.stream(cast(torch.cuda.Stream, self.transfer_stream)):
                        # Non-blocking transfers
                        anchors = anchors.to(self.device, non_blocking=True)
                        positives = positives.to(self.device, non_blocking=True)
                        negatives = negatives.to(self.device, non_blocking=True)
                    
                    # Wait for transfers to complete before forward pass
                    torch.cuda.current_stream().wait_stream(self.transfer_stream)
                else:
                    # Fallback for CPU-only
                    anchors = anchors.to(self.device)
                    positives = positives.to(self.device)
                    negatives = negatives.to(self.device)
                
                transfer_time = time.time() - transfer_start
                forward_start = time.time()
                
                # OPTIMIZED: Single batched forward pass instead of three separate passes
                with amp.autocast(enabled=self.scaler.is_enabled()):
                    # Combine all inputs for single forward pass
                    batch_size = anchors.size(0)
                    combined_input = torch.cat([anchors, positives, negatives], dim=0)
                    combined_output = self.model(combined_input)
                    
                    # Split outputs back to individual tensors
                    anchor_embeddings = combined_output[:batch_size]
                    positive_embeddings = combined_output[batch_size:2*batch_size]
                    negative_embeddings = combined_output[2*batch_size:]
                    
                    loss = self.criterion(
                        anchor_embeddings,
                        positive_embeddings,
                        negative_embeddings
                    )
                
                forward_time = time.time() - forward_start
                backward_start = time.time()
                scaled_loss = loss.item() / self.grad_accum_steps
                self.scaler.scale(loss).backward()
                backward_time = time.time() - backward_start
                optimizer_time = 0
                if (batch_idx + 1) % self.grad_accum_steps == 0:
                    optimizer_start = time.time()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
                    
                    if self.scheduler is not None:
                        self.scheduler.step()
                    optimizer_time = time.time() - optimizer_start
                
                total_time = time.time() - batch_start
                self.data_monitor.record_batch_timing(
                    load_time, transfer_time, forward_time, backward_time, optimizer_time, total_time
                )
                
                # Update metrics
                total_loss += scaled_loss
                
                # Log progress instead of using tqdm
                logger.info(f"Training batch {batch_idx+1}/{num_batches} - Loss: {scaled_loss:.4f}")
                
                # Enhanced memory management
                self._manage_memory(batch_idx)
                
                # Explicitly delete tensors after use
                del anchors, positives, negatives
                del anchor_embeddings, positive_embeddings, negative_embeddings
                del combined_input, combined_output
                del load_time, transfer_time, forward_time, backward_time, optimizer_time, total_time
                
                # Free CPU RAM after transfer
                gc.collect()
                
            except RuntimeError as e:
                if "CUDA out of memory" in str(e) or "allocate" in str(e):
                    logger.error(f"CUDA OOM or memory allocation error: {str(e)}")
                    logger.error(f"At batch {batch_idx}/{num_batches}")
                    
                    # Emergency cleanup
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    gc.collect()
                    
                    # Reduce batch size dynamically if possible
                    if self.micro_batch_size > 1:
                        new_batch_size = max(1, self.micro_batch_size // 2)
                        logger.warning(f"Reducing batch size from {self.micro_batch_size} to {new_batch_size} for recovery")
                        self.micro_batch_size = new_batch_size
                        # Adjust grad accumulation to maintain effective batch size
                        self.grad_accum_steps = max(1, self.effective_batch_size // new_batch_size)
                        logger.warning(f"Adjusted gradient accumulation steps to {self.grad_accum_steps}")
                        
                        # Skip this batch and continue
                        continue
                    else:
                        # Can't reduce batch size further
                        logger.error("Cannot reduce batch size further. Training failed.")
                        raise
                else:
                    # Other runtime error
                    raise
        
        logger.info(f"Training epoch completed. Average loss: {total_loss / num_batches:.4f}")
        return {'loss': total_loss / num_batches}

    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validation step with OPTIMIZED memory management and async transfers"""
        self.model.eval()
        total_loss = 0
        num_batches = len(val_loader)
        
        # Log validation start
        logger.info(f"Starting validation with {num_batches} batches")
        
        with torch.no_grad():
            for batch_idx, (anchors, positives, negatives) in enumerate(val_loader):
                try:
                    if batch_idx == 0:
                        logger.info(f"[Batch 0] anchors: {anchors.shape}, positives: {positives.shape}, negatives: {negatives.shape}")
                    
                    # OPTIMIZED: Asynchronous GPU transfer
                    if self.transfer_stream is not None:
                        with torch.cuda.stream(cast(torch.cuda.Stream, self.transfer_stream)):
                            anchors = anchors.to(self.device, non_blocking=True)
                            positives = positives.to(self.device, non_blocking=True)
                            negatives = negatives.to(self.device, non_blocking=True)
                        
                        torch.cuda.current_stream().wait_stream(self.transfer_stream)
                    else:
                        anchors = anchors.to(self.device)
                        positives = positives.to(self.device)
                        negatives = negatives.to(self.device)
                    
                    # OPTIMIZED: Single batched forward pass
                    batch_size = anchors.size(0)
                    combined_input = torch.cat([anchors, positives, negatives], dim=0)
                    combined_output = self.model(combined_input)
                    
                    anchor_embeddings = combined_output[:batch_size]
                    positive_embeddings = combined_output[batch_size:2*batch_size]
                    negative_embeddings = combined_output[2*batch_size:]
                    
                    loss = self.criterion(
                        anchor_embeddings,
                        positive_embeddings,
                        negative_embeddings
                    )
                    
                    # Update metrics
                    total_loss += loss.item()
                    
                    # Log progress instead of using tqdm
                    logger.info(f"Validation batch {batch_idx+1}/{num_batches} - Loss: {loss.item():.4f}")
                    
                    # Enhanced memory management
                    self._manage_memory(batch_idx)
                    
                    # Explicitly delete tensors after use
                    del anchors, positives, negatives
                    del anchor_embeddings, positive_embeddings, negative_embeddings
                    del combined_input, combined_output
                    
                    # Free CPU RAM after transfer
                    gc.collect()
                    
                except RuntimeError as e:
                    if "CUDA out of memory" in str(e) or "allocate" in str(e):
                        logger.error(f"CUDA OOM or memory allocation error during validation: {str(e)}")
                        
                        # Emergency cleanup
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        gc.collect()
                        
                        # Skip this batch and continue if possible
                        if batch_idx < num_batches - 1:
                            continue
                        else:
                            # If last batch, use average of previous batches
                            if batch_idx > 0:
                                return {'loss': total_loss / batch_idx}
                            else:
                                raise
                    else:
                        # Other runtime error
                        raise
        
        logger.info(f"Validation completed. Average loss: {total_loss / num_batches:.4f}")
        return {'loss': total_loss / num_batches}

    def train(self, train_loader: DataLoader, val_loader: DataLoader, num_epochs: int) -> Dict[str, Any]:
        """Full training loop with memory management and early stopping"""
        best_val_loss = float('inf')
        training_history = []
        self.current_epoch = 0
        patience = 6
        epochs_no_improve = 0
        stop_epoch = num_epochs
        
        # Create checkpoint directory
        os.makedirs('checkpoints', exist_ok=True)
        
        # Initial memory cleanup
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            
            # Clear DataLoaderMonitor data at start of each epoch
            self.data_monitor.reset_data()
            
            logger.info("\n" + "="*60)
            logger.info(f" EPOCH {epoch+1}/{num_epochs} ".center(60, "="))
            logger.info("="*60 + "\n")
            logger.info(f"\nEpoch {epoch+1}/{num_epochs}")

            mem_before = psutil.virtual_memory().used / 1e9  # in GB
            vram_used = torch.cuda.memory_allocated(0) / 1e9 if torch.cuda.is_available() else 0
            logger.info(f"[Epoch {epoch}] RAM at Epoch Start: {mem_before:.2f} GB")
            logger.info(f"[Epoch {epoch}] GPU VRAM at Epoch Start: {vram_used:.1f} GB")
            
            # Reset peak memory tracking per epoch
            self.peak_memory_usage = psutil.virtual_memory().percent
            self.emergency_gc_count = 0
            
            # Training phase
            train_metrics = self.train_epoch(train_loader)
            self.data_monitor.log_performance_report()
            logger.info(f"Training Loss: {train_metrics['loss']:.4f}")
            
            # Validation phase
            val_metrics = self.validate(val_loader)
            logger.info(f"Validation Loss: {val_metrics['loss']:.4f}")
            
            # Early stopping logic
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                epochs_no_improve = 0
                logger.info(f"New best validation loss: {best_val_loss:.4f}")
                # Save checkpoint for best model
                checkpoint = {
                    'epoch': epoch + 1,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
                    'best_val_loss': best_val_loss,
                    'config': self.config
                }
                # Use batch_name in checkpoint filename if available
                if self.batch_name:
                    checkpoint_filename = f'best_model_epoch_{epoch+1}_{self.batch_name}.pth'
                else:
                    checkpoint_filename = f'best_model_epoch_{epoch+1}.pth'
                checkpoint_path = os.path.join(self.checkpoint_dir, checkpoint_filename)
                torch.save(checkpoint, checkpoint_path)
                logger.info(f"Saved checkpoint to {checkpoint_path}")
            else:
                epochs_no_improve += 1
                logger.info(f"No improvement in validation loss for {epochs_no_improve} epoch(s). Patience: {patience}")
                if epochs_no_improve >= patience:
                    logger.info(f"Early stopping triggered after {epoch+1} epochs. Best validation loss: {best_val_loss:.4f}")
                    stop_epoch = epoch + 1
                    break
            
            # Log metrics to visualizer at epoch level (reduces memory usage by ~1000x)
            if self.visualizer is not None:
                step = epoch + 1
                self.visualizer.log_metrics({
                    'train/loss': train_metrics['loss'],
                    'val/loss': val_metrics['loss'],
                    'train/learning_rate': self.optimizer.param_groups[0]['lr']
                }, step)
            
            # Save checkpoint every 5 epochs
            if (epoch + 1) % 5 == 0:
                checkpoint = {
                    'epoch': epoch + 1,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
                    'best_val_loss': best_val_loss,
                    'config': self.config
                }
                if self.batch_name:
                    checkpoint_filename = f'checkpoint_epoch_{epoch+1}_{self.batch_name}.pth'
                else:
                    checkpoint_filename = f'checkpoint_epoch_{epoch+1}.pth'
                checkpoint_path = os.path.join(self.checkpoint_dir, checkpoint_filename)
                torch.save(checkpoint, checkpoint_path)
                logger.info(f"Saved periodic checkpoint to {checkpoint_path}")
            
            # Record history
            history_entry = {
                'epoch': epoch + 1,
                'train_loss': train_metrics['loss'],
                'val_loss': val_metrics['loss'],
                'peak_memory': self.peak_memory_usage,
                'emergency_gc_count': self.emergency_gc_count
            }
            training_history.append(history_entry)
            
            # Keep only last 50 epochs in memory to prevent unlimited growth
            if len(training_history) > 50:
                training_history = training_history[-50:]
            
            # Memory stats at end of epoch
            mem_after = psutil.virtual_memory().used / 1e9  # in GB
            logger.info(f"[Epoch {epoch}] RAM at Epoch Start: {mem_before:.2f} GB, RAM after Epoch End: {mem_after:.2f} GB, diff: {mem_after - mem_before:.2f} GB")
            
            vram_used_after = torch.cuda.memory_allocated(0) / 1e9 if torch.cuda.is_available() else 0
            logger.info(f"Epoch VRAM stats - GPU VRAM at Epoch START: {vram_used:.1f} GB, GPU VRAM at Epoch END: {vram_used_after:.1f} GB, diff: {vram_used_after - vram_used:.1f} GB")
            
            # Force garbage collection between epochs
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            mem_after = psutil.virtual_memory().used / 1e9  # in GB
            logger.info(f"[Epoch {epoch}] RAM after CLEANING at Epoch End: {mem_after:.2f} GB")
            vram_used = torch.cuda.memory_allocated(0) / 1e9 if torch.cuda.is_available() else 0
            logger.info(f"GPU VRAM after CLEANING at Epoch End: {vram_used:.1f} GB")

            del mem_before, mem_after, vram_used, vram_used_after
        
        return {
            'best_val_loss': best_val_loss,
            'training_history': training_history,
            'stopped_epoch': stop_epoch
        } 