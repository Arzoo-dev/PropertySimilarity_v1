#!/usr/bin/env python3
"""
Common Utilities Module
=======================

Consolidated utility functions and classes used across multiple modules.
This module eliminates code duplication and provides a single source of truth
for shared functionality.

Contains:
- TeeOutput: Output redirection to both terminal and file
- setup_centralized_logging: Centralized logging configuration  
- memory_monitor: Background memory usage monitoring
- Config loading utilities
- Common helper functions
"""

import os
import sys
import time
import logging
import yaml
import gc
import threading
import psutil
import torch
from typing import Optional, Dict, Any, List

# Global variables for memory monitoring
stop_memory_monitor = False
memory_alert_threshold = 85
file_handler = None

class TeeOutput:
    """Class to redirect output to both terminal and file
    
    This class is used for capturing console output while still displaying
    it to the user. Originally duplicated in run_tsne_comparison.py and 
    margin_analysis.py.
    """
    def __init__(self, file_path: str):
        self.terminal = sys.stdout
        self.log_file = open(file_path, 'w', encoding='utf-8')
    
    def write(self, message: str):
        self.terminal.write(message)
        self.log_file.write(message)
        self.log_file.flush()  # Ensure immediate write to file
    
    def flush(self):
        self.terminal.flush()
        self.log_file.flush()
    
    def close(self):
        if self.log_file:
            self.log_file.close()

def setup_centralized_logging(log_file_path: str):
    """Setup centralized logging that all modules will use
    
    Originally duplicated in train_pipeline.py and train.py.
    
    Args:
        log_file_path: Path to the log file
        
    Returns:
        File handler for the log file
    """
    global file_handler
    
    # Clear any existing handlers from the root logger
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    
    # Create file handler
    file_handler = logging.FileHandler(log_file_path, mode='a', encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    
    # Add handlers to root logger
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)
    root_logger.setLevel(logging.INFO)
    
    # Disable propagation for specific loggers that might have their own handlers
    logging.getLogger('PIL').setLevel(logging.WARNING)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('tqdm').setLevel(logging.WARNING)
    
    return file_handler

def memory_monitor():
    """Background thread to monitor memory usage
    
    Originally duplicated in train_pipeline.py and train.py.
    Monitors system memory and triggers cleanup when thresholds are exceeded.
    """
    global stop_memory_monitor
    logger = logging.getLogger(__name__)
    prev_log_time = time.time()
    
    while not stop_memory_monitor:
        # Check memory usage
        mem_usage = psutil.virtual_memory().percent
        cpu_usage = psutil.cpu_percent(interval=None)
        
        # Log every 30 seconds or when approaching threshold
        current_time = time.time()
        if mem_usage > (memory_alert_threshold - 5) or current_time - prev_log_time > 30:
            logger.info(f"Memory: {mem_usage:.1f}% | CPU: {cpu_usage:.1f}%")
            prev_log_time = current_time
            
        # Emergency actions if memory is critically high
        if mem_usage > memory_alert_threshold:
            logger.warning(f"⚠️ MEMORY ALERT: Usage at {mem_usage:.1f}% (threshold: {memory_alert_threshold}%)")
            logger.warning("Forcing garbage collection...")
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Sleep to prevent high CPU usage from monitoring
        time.sleep(5)

def load_config(config_path: str = "config/training_config.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file
    
    Standardized config loading function used across multiple modules.
    Originally duplicated in run_tsne_comparison.py and margin_analysis.py.
    
    Args:
        config_path: Path to the configuration YAML file
        
    Returns:
        Dictionary containing configuration data
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def get_all_batch_files(split_dir: str = "dataset/training_splits") -> List[str]:
    """Get all available batch files automatically
    
    Originally duplicated in multiple analysis scripts.
    
    Args:
        split_dir: Directory containing split files
        
    Returns:
        List of full paths to batch files
    """
    if not os.path.isdir(split_dir):
        print(f"Training splits directory not found: {split_dir}")
        return []
    
    # Get all .txt files
    txt_files = sorted([f for f in os.listdir(split_dir) if f.endswith('.txt')])
    if not txt_files:
        print(f"No .txt split files found in directory: {split_dir}")
        return []
    
    # Return full paths
    batch_files = [os.path.join(split_dir, f) for f in txt_files]
    return batch_files

def setup_analysis_logging(output_dir: str, log_prefix: str, batch_name: str) -> tuple:
    """Setup logging for analysis scripts with consistent naming
    
    Args:
        output_dir: Directory to save logs
        log_prefix: Prefix for log filename (e.g., 'tsne_comparison', 'margin_analysis')
        batch_name: Name of the current batch being processed
        
    Returns:
        Tuple of (TeeOutput instance, log file path)
    """
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_filename = f"{log_prefix}_logs_{batch_name}_{timestamp}.txt"
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save log file in output directory
    log_path = os.path.join(output_dir, log_filename)
    
    # Create TeeOutput instance to redirect stdout
    tee = TeeOutput(log_path)
    sys.stdout = tee
    
    print("="*80)
    print(f"{log_prefix.upper().replace('_', ' ')} LOG - {batch_name.upper()}")
    print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Log file: {log_filename}")
    print(f"Log directory: {output_dir}")
    print("="*80)
    
    return tee, log_path

def set_memory_threshold(threshold: int):
    """Set the global memory alert threshold
    
    Args:
        threshold: Memory usage percentage (0-100) to trigger alerts
    """
    global memory_alert_threshold
    memory_alert_threshold = threshold

def stop_memory_monitoring():
    """Stop the background memory monitoring thread"""
    global stop_memory_monitor
    stop_memory_monitor = True

def start_memory_monitoring() -> threading.Thread:
    """Start background memory monitoring thread
    
    Returns:
        The monitoring thread
    """
    global stop_memory_monitor
    stop_memory_monitor = False
    memory_thread = threading.Thread(target=memory_monitor, daemon=True)
    memory_thread.start()
    return memory_thread 