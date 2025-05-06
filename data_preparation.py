"""
Data preparation wrapper for Siamese network training on RunPods.
This script combines the data gathering and preprocessing steps.
"""

import os
import sys
import logging
from typing import Dict, List, Tuple, Optional, Any
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import gc
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global memory cache
_image_cache = {}
_MAX_CACHE_SIZE = 500  # Limit cache size to prevent memory overflow

def clear_memory():
    """Force aggressive memory cleanup"""
    # Clear the image cache
    if '_image_cache' in globals():
        globals()['_image_cache'].clear()
    
    # Multiple garbage collection passes to ensure thorough cleanup
    for _ in range(3):
        gc.collect()
    
    logger.info("Performed aggressive memory cleanup")
    
    # Report memory usage if psutil is available
    try:
        import psutil
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        sys_mem = psutil.virtual_memory()
        logger.info(f"Memory usage after cleanup: Process {mem_info.rss / (1024*1024):.1f}MB, System {sys_mem.percent:.1f}%")
    except ImportError:
        pass

def check_memory_usage(threshold_pct=80.0):
    """Check memory usage and clean up if needed"""
    try:
        import psutil
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        sys_mem = psutil.virtual_memory()
        
        if sys_mem.percent > threshold_pct:
            logger.warning(f"Memory usage high ({sys_mem.percent:.1f}%), performing cleanup")
            clear_memory()
            return True
    except ImportError:
        pass
    return False

class PropertyDataset(Dataset):
    """
    Dataset for property-level similarity with expert ratings.
    Each item is a pair of properties (subject and comparable) with an expert similarity rating.
    Each property consists of multiple images.
    """
    
    def __init__(
        self,
        property_pairs: List[Dict],
        root_dir: str = "properties",
        transform=None,
        max_images_per_property: int = 10,
        room_matching: bool = False,
        low_memory_mode: bool = True
    ):
        """
        Initialize property dataset with pairs of properties and their similarity ratings.
        
        Args:
            property_pairs (List[Dict]): List of dictionaries containing property pairs and ratings
                Each dict should have: {
                    'subject_uid': str,
                    'comparable_uid': str,
                    'similarity_score': float, # Expert rating (0.0 to 1.0)
                    'subject_images': List[str], # Optional, relative paths to images
                    'comparable_images': List[str] # Optional, relative paths to images
                }
            root_dir (str): Base directory containing property image folders
            transform: Optional transform to be applied to images
            max_images_per_property (int): Maximum number of images to use per property
            room_matching (bool): Whether to use room-type matching (not implemented yet)
            low_memory_mode (bool): Whether to use memory-efficient processing
        """
        self.property_pairs = property_pairs
        self.root_dir = root_dir
        self.max_images = max_images_per_property
        self.room_matching = room_matching
        self.low_memory_mode = low_memory_mode
        
        # Set up transforms
        if transform:
            self.transform = transform
        else:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        
        # Check memory before validation
        check_memory_usage()
        
        # Validate and prepare the dataset in batches if low memory mode is enabled
        if low_memory_mode and len(property_pairs) > 100:
            self._validate_and_prepare_batched()
        else:
            self._validate_and_prepare()
        
        logger.info(f"Initialized PropertyDataset with {len(self.property_pairs)} valid property pairs")
        
        # Clear memory after initialization
        check_memory_usage()
    
    def _validate_and_prepare_batched(self):
        """Validate property pairs and prepare image paths in batches to save memory"""
        valid_pairs = []
        batch_size = 50  # Process 50 pairs at a time
        
        # Get total number of batches
        total_pairs = len(self.property_pairs)
        total_batches = (total_pairs + batch_size - 1) // batch_size
        
        logger.info(f"Processing {total_pairs} property pairs in {total_batches} batches")
        
        for batch_idx in range(total_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, total_pairs)
            
            logger.info(f"Processing batch {batch_idx+1}/{total_batches} (pairs {start_idx}-{end_idx-1})")
            
            # Process this batch
            batch_pairs = self.property_pairs[start_idx:end_idx]
            batch_valid_pairs = []
            
            for pair in batch_pairs:
                subject_uid = pair.get('subject_uid')
                comparable_uid = pair.get('comparable_uid')
                similarity_score = pair.get('similarity_score')
                
                if not subject_uid or not comparable_uid or similarity_score is None:
                    logger.warning(f"Skipping invalid property pair: {subject_uid}-{comparable_uid}")
                    continue
                
                # If image paths not directly provided, look for them in the root directory
                if 'subject_images' not in pair or 'comparable_images' not in pair:
                    subject_dir = os.path.join(self.root_dir, subject_uid)
                    comparable_dir = os.path.join(self.root_dir, comparable_uid)
                    
                    if not os.path.isdir(subject_dir) or not os.path.isdir(comparable_dir):
                        logger.warning(f"Property directory not found for {subject_uid} or {comparable_uid}")
                        continue
                    
                    # Get image paths for both properties
                    subject_images = self._get_property_images(subject_dir)
                    comparable_images = self._get_property_images(comparable_dir)
                    
                    if not subject_images or not comparable_images:
                        logger.warning(f"No images found for {subject_uid} or {comparable_uid}")
                        continue
                    
                    pair['subject_images'] = subject_images
                    pair['comparable_images'] = comparable_images
                
                # Limit the number of images per property if needed
                if len(pair['subject_images']) > self.max_images:
                    pair['subject_images'] = pair['subject_images'][:self.max_images]
                
                if len(pair['comparable_images']) > self.max_images:
                    pair['comparable_images'] = pair['comparable_images'][:self.max_images]
                
                batch_valid_pairs.append(pair)
            
            # Add valid pairs from this batch
            valid_pairs.extend(batch_valid_pairs)
            
            # Check memory after each batch
            check_memory_usage()
            
            # Sleep briefly to allow other processes to run
            time.sleep(0.01)
        
        self.property_pairs = valid_pairs
        logger.info(f"Validated {len(valid_pairs)} property pairs out of {total_pairs} original pairs")
    
    def _validate_and_prepare(self):
        """Validate property pairs and prepare image paths (original method)"""
        valid_pairs = []
        
        for pair in self.property_pairs:
            subject_uid = pair.get('subject_uid')
            comparable_uid = pair.get('comparable_uid')
            similarity_score = pair.get('similarity_score')
            
            if not subject_uid or not comparable_uid or similarity_score is None:
                logger.warning(f"Skipping invalid property pair: {pair}")
                continue
            
            # If image paths not directly provided, look for them in the root directory
            if 'subject_images' not in pair or 'comparable_images' not in pair:
                subject_dir = os.path.join(self.root_dir, subject_uid)
                comparable_dir = os.path.join(self.root_dir, comparable_uid)
                
                if not os.path.isdir(subject_dir) or not os.path.isdir(comparable_dir):
                    logger.warning(f"Property directory not found for {subject_uid} or {comparable_uid}")
                    continue
                
                # Get image paths for both properties
                subject_images = self._get_property_images(subject_dir)
                comparable_images = self._get_property_images(comparable_dir)
                
                if not subject_images or not comparable_images:
                    logger.warning(f"No images found for {subject_uid} or {comparable_uid}")
                    continue
                
                pair['subject_images'] = subject_images
                pair['comparable_images'] = comparable_images
            
            # Limit the number of images per property if needed
            if len(pair['subject_images']) > self.max_images:
                pair['subject_images'] = pair['subject_images'][:self.max_images]
            
            if len(pair['comparable_images']) > self.max_images:
                pair['comparable_images'] = pair['comparable_images'][:self.max_images]
            
            valid_pairs.append(pair)
        
        self.property_pairs = valid_pairs
        logger.info(f"Validated {len(valid_pairs)} property pairs out of {len(self.property_pairs)} original pairs")
    
    def _get_property_images(self, property_dir: str) -> List[str]:
        """Get all image paths for a property directory"""
        if not os.path.isdir(property_dir):
            return []
        
        image_extensions = ['.jpg', '.jpeg', '.png']
        images = []
        
        for filename in sorted(os.listdir(property_dir)):
            if any(filename.lower().endswith(ext) for ext in image_extensions):
                images.append(os.path.join(property_dir, filename))
        
        logger.debug(f"Found {len(images)} images in property directory {property_dir}")
        return images
    
    def __len__(self):
        return len(self.property_pairs)
    
    def __getitem__(self, idx):
        """
        Get a property pair with processed images and similarity score.
            
        Returns:
            Tuple containing:
                - List of subject property images (as tensors)
                - List of comparable property images (as tensors)
                - Similarity score (tensor)
                - Subject property UID (string)
                - Comparable property UID (string)
        """
        pair = self.property_pairs[idx]
        
        # Check memory before loading images
        if idx % 20 == 0:  # Check every 20 items
            check_memory_usage()
        
        subject_images = self._load_and_process_images(pair['subject_images'])
        comparable_images = self._load_and_process_images(pair['comparable_images'])
        
        similarity_score = torch.tensor(float(pair['similarity_score']), dtype=torch.float32)
        
        return (
            subject_images,
            comparable_images,
            similarity_score,
            pair['subject_uid'],
            pair['comparable_uid']
        )
    
    def _load_and_process_images(self, image_paths: List[str]) -> torch.Tensor:
        """Load and process all images for a property, returning as batch tensor"""
        processed_images = []
        
        for img_path in image_paths:
            try:
                # Check if image is in cache
                if self.low_memory_mode:
                    if img_path in _image_cache:
                        processed_images.append(_image_cache[img_path])
                        continue
                
                # Load and process image
                img = Image.open(img_path).convert('RGB')
                if self.transform:
                    img = self.transform(img)
                processed_images.append(img)
                
                # Cache the processed image if in low memory mode and cache isn't too big
                if self.low_memory_mode and len(_image_cache) < _MAX_CACHE_SIZE:
                    _image_cache[img_path] = img
            except Exception as e:
                logger.warning(f"Error processing image {img_path}: {str(e)}")
        
        # If no images were successfully processed, create a dummy image
        if not processed_images:
            logger.warning(f"No valid images processed. Creating a dummy image.")
            dummy = torch.zeros((3, 224, 224))
            processed_images.append(dummy)
        
        # Stack all images into a single batch tensor
        return torch.stack(processed_images)

def load_property_pairs_from_json(json_path: str) -> List[Dict]:
    """
    Load property pairs and their similarity ratings from a JSON file.
    
    Expected JSON format:
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
        List of property pair dictionaries
    """
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        pairs = data.get('property_pairs', [])
        logger.info(f"Loaded {len(pairs)} property pairs from {json_path}")
        
        # Clean up memory after loading large JSON
        if len(pairs) > 1000:
            gc.collect()
            
        return pairs
    except Exception as e:
        logger.error(f"Error loading property pairs from {json_path}: {str(e)}")
        return []

def create_property_dataloaders(
    json_path: str,
    root_dir: str = "properties",
    batch_size: int = 8,
    train_ratio: float = 0.8,
    max_images_per_property: int = 10,
    random_seed: int = 42,
    low_memory_mode: bool = True,
    num_workers: int = 2  # Reduced from 4 to save memory
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Create training and validation dataloaders for property-level similarity.
    
    Args:
        json_path: Path to JSON file with property pairs
        root_dir: Directory containing property image folders
        batch_size: Batch size for dataloaders
        train_ratio: Ratio of data to use for training
        max_images_per_property: Maximum images to use per property
        random_seed: Random seed for reproducibility
        low_memory_mode: Whether to use memory-efficient processing
        num_workers: Number of worker processes for data loading
        
    Returns:
        Tuple of (train_dataloader, val_dataloader)
    """
    # Set random seed for reproducibility
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    
    # Clear memory before loading data
    clear_memory()
    
    # Load property pairs
    property_pairs = load_property_pairs_from_json(json_path)
    
    if not property_pairs:
        logger.error("No valid property pairs found")
        return None, None
    
    # Shuffle and split the pairs
    np.random.shuffle(property_pairs)
    split_idx = int(len(property_pairs) * train_ratio)
    
    train_pairs = property_pairs[:split_idx]
    val_pairs = property_pairs[split_idx:]
    
    # Clean up original property_pairs to save memory
    del property_pairs
    gc.collect()
    
    logger.info(f"Split dataset: {len(train_pairs)} training pairs, {len(val_pairs)} validation pairs")
    
    # Create training dataset
    logger.info("Creating training dataset...")
    train_dataset = PropertyDataset(
        property_pairs=train_pairs,
        root_dir=root_dir,
        max_images_per_property=max_images_per_property,
        low_memory_mode=low_memory_mode
    )
    
    # Clean up to save memory before creating validation dataset
    del train_pairs
    gc.collect()
    
    # Create validation dataset
    logger.info("Creating validation dataset...")
    val_dataset = PropertyDataset(
        property_pairs=val_pairs,
        root_dir=root_dir,
        max_images_per_property=max_images_per_property,
        low_memory_mode=low_memory_mode
    )
    
    # Clean up to save memory
    del val_pairs
    gc.collect()
    
    # Create dataloaders with reduced workers to save memory
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False  # Disable pin_memory to save RAM
    )
    
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False  # Disable pin_memory to save RAM
    )
    
    return train_dataloader, val_dataloader

if __name__ == "__main__":
    print("\n===== Property-Level Similarity: Data Preparation =====\n")
    print("This script combines data gathering and preprocessing for property-level similarity training.")
    print("The process is now split into two steps:\n")
    print("1. Data Gathering: Downloads property sets from the API")
    print("2. Data Processing: Prepares property pairs with expert ratings\n")
    
    # Enable memory optimization by default
    low_memory_mode = True
    
    # Ask if user wants to enable memory optimization
    memory_opt = input("Enable memory optimization? (y/n, default: y): ").strip().lower()
    if memory_opt == 'n':
        low_memory_mode = False
        print("Memory optimization disabled (may use more RAM)")
    else:
        print("Memory optimization enabled (recommended for large datasets)")
    
    # Initial memory cleanup
    clear_memory()
    
    # Ask which steps to run
    print("Which steps would you like to run?")
    print("[1] Property data gathering only")
    print("[2] Property data processing only")
    print("[3] Both gathering and processing")
    
    choice = ''
    while choice not in ['1', '2', '3']:
        choice = input("Enter your choice (1, 2, or 3): ").strip()
    
    if choice in ['1', '3']:
        print("\n=== Running Property Data Gathering ===\n")
        from data_gathering import get_user_input_for_api_download, fetch_training_and_testing_data
        
        # Get user input for API parameters
        params = get_user_input_for_api_download()
        
        # Set low memory mode in params
        params["low_memory_mode"] = low_memory_mode
        
        # Fetch training and testing property data
        train_properties, test_properties = fetch_training_and_testing_data(
            api_url=params["api_url"],
            auth_token=params["auth_token"],
            train_properties=params["train_properties"],
            test_properties=params["test_properties"],
            cache_dir=params["cache_dir"],
            max_workers=params["max_workers"],
            low_memory_mode=params["low_memory_mode"]
        )
        
        # Clean up memory after data gathering
        clear_memory()
        
        if not train_properties and not test_properties:
            print("\nNo property data was obtained. What would you like to do?")
            print("[1] Retry with different parameters")
            print("[2] Continue with dummy data")
            print("[3] Exit")
            
            retry_choice = ''
            while retry_choice not in ['1', '2', '3']:
                retry_choice = input("Enter your choice (1, 2, or 3): ").strip()
            
            if retry_choice == '1':
                # Restart property gathering by recursively calling the main function
                print("\nRestarting property data gathering...\n")
                from sys import argv
                os.execv(sys.executable, [sys.executable] + argv)
                sys.exit(0)  # This line will not be reached if exec succeeds
            elif retry_choice == '2':
                # Create dummy data
                print("\nCreating dummy property data for testing...\n")
                from data_gathering import create_dummy_property_data
                
                train_properties = create_dummy_property_data(params["train_properties"])
                test_properties = create_dummy_property_data(params["test_properties"])
            else:
                print("Exiting.")
                sys.exit(0)
        
        # Save property data for processing
        from runpods_utils import ensure_dir_exists
        import json
        
        output_dir = params["output_dir"]
        ensure_dir_exists(output_dir)
        
        # Calculate total number of images
        train_image_count = sum(len(images) for images in train_properties.values())
        test_image_count = sum(len(images) for images in test_properties.values())
        
        data_paths = {
            "train_properties": train_properties,
            "test_properties": test_properties,
            "num_train_properties": len(train_properties),
            "num_test_properties": len(test_properties),
            "num_train_images": train_image_count,
            "num_test_images": test_image_count
        }
        
        property_data_file = os.path.join(output_dir, "property_data.json")
        
        # Save the data in batches if it's large
        if train_image_count + test_image_count > 10000:
            print("Large dataset detected, saving with memory optimization...")
            
            # Save train and test properties separately to reduce memory usage
            train_data = {
                "train_properties": train_properties,
                "num_train_properties": len(train_properties),
                "num_train_images": train_image_count
            }
            
            train_file = os.path.join(output_dir, "train_properties.json")
            with open(train_file, 'w') as f:
                json.dump(train_data, f)
            
            # Clean up train data to save memory
            del train_data
            gc.collect()
            
            test_data = {
                "test_properties": test_properties,
                "num_test_properties": len(test_properties),
                "num_test_images": test_image_count
            }
            
            test_file = os.path.join(output_dir, "test_properties.json")
            with open(test_file, 'w') as f:
                json.dump(test_data, f)
                
            # Clean up test data to save memory
            del test_data
            gc.collect()
            
            # Now create the combined file with minimal data
            with open(property_data_file, 'w') as f:
                json.dump({
                    "num_train_properties": len(train_properties),
                    "num_test_properties": len(test_properties),
                    "num_train_images": train_image_count,
                    "num_test_images": test_image_count,
                    "train_file": train_file,
                    "test_file": test_file
                }, f, indent=2)
        else:
            # For smaller datasets, save everything together
            with open(property_data_file, 'w') as f:
                json.dump(data_paths, f, indent=2)
        
        # Clean up memory after saving
        del data_paths
        clear_memory()
        
        print(f"\nProperty data gathering complete!")
        print(f"Downloaded {len(train_properties)} training property sets with {train_image_count} images")
        print(f"Downloaded {len(test_properties)} testing property sets with {test_image_count} images")
        print(f"Property data saved to {property_data_file}")
        
        # Check if we have property ratings
        use_ratings = input("\nDo you have expert property ratings? (y/n, default: n): ").strip().lower() == 'y'
        
        if use_ratings:
            from data_gathering import prepare_property_pairs_dataset
            
            ratings_path = input("Enter path to property ratings JSON file: ").strip()
            
            if os.path.exists(ratings_path):
                # Combine train and test properties for validation
                print("Combining properties for ratings validation...")
                all_properties = {**train_properties, **test_properties}
                
                # Clean up individual property dictionaries to save memory
                del train_properties
                del test_properties
                clear_memory()
                
                # Prepare property pairs dataset
                pairs_dir = os.path.join(output_dir, "property_pairs")
                prepare_property_pairs_dataset(all_properties, ratings_path, pairs_dir)
                
                print(f"\nProperty pairs prepared in {pairs_dir}")
                
                # Clean up all_properties
                del all_properties
                clear_memory()
            else:
                print(f"Error: Ratings file {ratings_path} not found")
    
    if choice in ['2', '3']:
        print("\n=== Running Property Data Processing ===\n")
        
        # Clear memory before processing
        clear_memory()
        
        if choice == '2':
            # If only processing, ask for input directory
            default_input_dir = "/workspace/data/properties"
            input_dir = input(f"Enter directory containing property_data.json [default: {default_input_dir}]: ").strip() or default_input_dir
        else:
            # If we just ran gathering, use the same directory
            input_dir = params["output_dir"]
        
        # First ensure the directory exists
        from runpods_utils import ensure_dir_exists
        ensure_dir_exists(input_dir)
        
        # Load property data
        property_data_file = os.path.join(input_dir, "property_data.json")
        
        if not os.path.exists(property_data_file):
            property_data_file = os.path.join(input_dir, "data_paths.json")  # Try legacy name
            
        if not os.path.exists(property_data_file):
            print(f"Error: property_data.json not found in {input_dir}")
            sys.exit(1)
            
        import json
        with open(property_data_file, 'r') as f:
            property_data = json.load(f)
        
        # Check if we have split files for large datasets
        if "train_file" in property_data and "test_file" in property_data:
            print("Loading property data from split files for memory efficiency...")
            
            # Load train properties
            with open(property_data["train_file"], 'r') as f:
                train_data = json.load(f)
                train_properties = train_data.get("train_properties", {})
            
            # Load test properties
            with open(property_data["test_file"], 'r') as f:
                test_data = json.load(f)
                test_properties = test_data.get("test_properties", {})
                
            # Update property_data
            property_data["train_properties"] = train_properties
            property_data["test_properties"] = test_properties
            
            # Clean up
            del train_data
            del test_data
            clear_memory()
        
        # Convert from legacy format if needed
        if "train_images" in property_data and "train_properties" not in property_data:
            print("Converting from legacy data format...")
            property_data["train_properties"] = property_data.get("train_images", {})
            property_data["test_properties"] = property_data.get("test_images", {})
        
        # Check if we have property ratings to process
        ratings_path = os.path.join(input_dir, "property_pairs", "property_pairs.json")
        if os.path.exists(ratings_path):
            print(f"Found property ratings at {ratings_path}")
            
            # Get settings for property pair processing
            from data_preprocessing import generate_property_pairs
            import time
            
            max_pairs_str = input("Maximum number of property pairs to process [default: 1000]: ").strip()
            max_pairs = int(max_pairs_str) if max_pairs_str.isdigit() else 1000
            
            max_images_str = input("Maximum images per property [default: 10]: ").strip()
            max_images = int(max_images_str) if max_images_str.isdigit() else 10
            
            augment_prob_str = input("Augmentation probability [default: 0.3]: ").strip()
            augment_prob = float(augment_prob_str) if augment_prob_str else 0.3
            
            # Ask for low memory mode for preprocessing
            low_memory_opt = input("Enable low memory mode for preprocessing? (y/n, default: y): ").strip().lower()
            preproc_low_memory = False if low_memory_opt == 'n' else True
            
            processed_dir = os.path.join(input_dir, "processed_property_pairs")
            ensure_dir_exists(processed_dir)
            
            # Combine train and test properties
            print("Combining properties for processing...")
            all_properties = {**property_data.get("train_properties", {}), **property_data.get("test_properties", {})}
            
            # Clean up property_data to save memory
            del property_data
            clear_memory()
            
            # Load ratings
            print(f"Loading ratings from {ratings_path}...")
            with open(ratings_path, 'r') as f:
                ratings_data = json.load(f)
            
            ratings = ratings_data.get("property_pairs", [])
            
            # Clean up ratings_data to save memory
            del ratings_data
            clear_memory()
            
            print(f"\nProcessing up to {max_pairs} property pairs with augmentation probability {augment_prob}...")
            print(f"Low memory mode is {'enabled' if preproc_low_memory else 'disabled'}")
            
            # Call generate_property_pairs with all settings
            generate_property_pairs(
                property_data=all_properties,
                ratings_data=ratings,
                num_pairs=max_pairs,
                max_images_per_property=max_images,
                augment_probability=augment_prob,
                output_dir=processed_dir,
                # Add low_memory_mode if the function supports it
                **({"low_memory_mode": preproc_low_memory} if "low_memory_mode" in 
                   inspect.signature(generate_property_pairs).parameters else {})
            )
            
            # Clean up
            del all_properties
            del ratings
            clear_memory()
            
            print(f"\nProperty pairs processed and saved to {processed_dir}")
            print("You can now train your model using the PropertyDataset class with these processed pairs.")
        else:
            print("\nNo property ratings found. To train a property-level similarity model, you need:")
            print("1. Property data (gathered in the first step)")
            print("2. Expert property ratings in JSON format")
            print("\nOnce you have these, you can run the processing step again.")
            
        # Legacy support for triplet-based approach
        triplet_option = input("\nDo you want to generate triplets for legacy training? (y/n, default: n): ").strip().lower() == 'y'
        
        if triplet_option:
            from data_preprocessing import (
                ensure_multiple_property_types,
                generate_triplets_with_augmentation,
                TripletDataset
            )
            
            # Clean up before triplet generation
            clear_memory()
            
            # Check if property_data is still available, reload if not
            if 'property_data' not in locals() or not property_data:
                print("Reloading property data...")
                with open(property_data_file, 'r') as f:
                    property_data = json.load(f)
            
            # Get train and test image sets
            train_images = property_data.get("train_properties", property_data.get("train_images", {}))
            test_images = property_data.get("test_properties", property_data.get("test_images", {}))
            
            # Clean up property_data to save memory
            del property_data
            clear_memory()
            
            # Ensure we have multiple property types
            train_images = ensure_multiple_property_types(train_images)
            
            # Get number of triplets
            num_triplets_str = input("Number of triplets to generate [default: 10000]: ").strip()
            num_triplets = int(num_triplets_str) if num_triplets_str.isdigit() else 10000
            
            triplets_dir = os.path.join(input_dir, "triplets")
            ensure_dir_exists(triplets_dir)
            
            print(f"\nGenerating {num_triplets} triplets...")
            
            # Use low memory generation
            print(f"Using memory-optimized triplet generation (low_memory_mode=True)")
            generate_triplets_with_augmentation(
                train_images,
                num_triplets=num_triplets,
                batch_size=50,  # Use smaller batches to reduce memory usage
                output_dir=triplets_dir,
                prefix="triplet",
                max_workers=1,  # Disable multiprocessing to avoid pickle errors
                augment_probability=0.5,
                save_triplets=True
            )
            
            # Clean up
            del train_images
            del test_images
            clear_memory()
            
            print(f"\nTriplets generated and saved to {triplets_dir}")
            print("You can use these triplets with the legacy triplet-based training approach.")
    
    # Final memory cleanup
    clear_memory()
    print("\nData preparation complete!") 