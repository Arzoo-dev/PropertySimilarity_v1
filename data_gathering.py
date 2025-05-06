"""
Data gathering utilities for Siamese network training on RunPods.

This module handles:
1. Getting user input for API parameters
2. Fetching images from the API
3. Caching images locally
4. Organizing images by property ID
5. Splitting properties into training and testing sets
"""

import os
import logging
import requests
import io
import time
import json
import random
import concurrent.futures
import uuid
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Tuple, Set, Optional, Any
from PIL import Image
import gc

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define directories for training and testing properties
TRAIN_DIR = "/workspace/train_properties"
TEST_DIR = "/workspace/test_properties"

# Create directories if they don't exist
os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(TEST_DIR, exist_ok=True)

# Utility functions for JSON handling
def load_json_from_file(file_path):
    """Load JSON data from a local file."""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        logger.info(f"Successfully loaded JSON from {file_path}")
        return data
    except Exception as e:
        logger.error(f"Error loading JSON from {file_path}: {e}")
        return None

def load_json_from_endpoint(endpoint, auth_token=None):
    """Fetch JSON data from an API endpoint."""
    headers = {}
    if auth_token:
        headers["Authorization"] = f"Bearer {auth_token}"
    
    try:
        response = requests.get(endpoint, headers=headers, timeout=30)
        response.raise_for_status()  # Raise an error on bad status
        data = response.json()
        logger.info(f"Successfully fetched JSON from {endpoint}")
        return data
    except Exception as e:
        logger.error(f"Error fetching JSON from endpoint: {e}")
        return None

def organize_properties_by_uid(data, property_key="properties"):
    """
    Extract properties from JSON data and organize them by UID.
    
    Args:
        data: JSON data containing property information
        property_key: Key in JSON data that contains the property list
        
    Returns:
        Dictionary mapping property UIDs to property data
    """
    properties_by_uid = {}
    
    if not data:
        logger.error("No data provided to organize_properties_by_uid")
        return properties_by_uid
    
    # Handle different property data structures
    properties = data.get(property_key, [])
    
    # If property_key doesn't exist or is empty, try to find properties elsewhere
    if not properties:
        # Check for subject_property and comps
        subject_property = data.get("subject_property")
        if subject_property and "uid" in subject_property:
            properties = [subject_property]
        
        comps = data.get("comps", [])
        if comps:
            properties.extend(comps)
    
    for prop in properties:
        if "uid" not in prop or not prop["uid"]:
            logger.warning("Property missing UID, skipping")
            continue
        
        uid = prop["uid"]
        properties_by_uid[uid] = prop
    
    logger.info(f"Organized {len(properties_by_uid)} properties by UID")
    return properties_by_uid

def extract_property_image_urls(properties_by_uid):
    """
    Extract image URLs from properties organized by UID.
    
    Args:
        properties_by_uid: Dictionary mapping property UIDs to property data
        
    Returns:
        Dictionary mapping property UIDs to lists of image URLs
    """
    property_images = {}
    
    for uid, prop_data in properties_by_uid.items():
        # Try different possible locations for photos
        photos = prop_data.get("photos", [])
        if not photos:
            photos = prop_data.get("images", [])
        
        # Extract URLs from photo objects
        image_urls = []
        for photo in photos:
            url = None
            # Handle different photo data structures
            if isinstance(photo, dict):
                url = photo.get("url")
                if not url:
                    url = photo.get("image_url")
            elif isinstance(photo, str):
                # Direct URL string
                url = photo
            
            if url and url.lower().endswith(('.jpg', '.jpeg', '.png', '.gif')):
                image_urls.append(url)
        
        if image_urls:
            property_images[uid] = image_urls
        else:
            logger.warning(f"No valid image URLs found for property {uid}")
    
    total_images = sum(len(urls) for urls in property_images.values())
    logger.info(f"Extracted {total_images} image URLs from {len(property_images)} properties")
    
    return property_images

def extract_image_urls_from_api_response(data):
    """
    Extract image URLs from various API response formats.
    Handles multiple common response structures.
    
    Args:
        data: JSON data from API response
        
    Returns:
        Dictionary mapping property UIDs to lists of image URLs
    """
    property_images = {}
    
    # Case 1: Premier Brokerage System format (subject_property and comps)
    subject_property = data.get("subject_property", {})
    if subject_property and "uid" in subject_property:
        subject_uid = subject_property.get("uid")
        subject_photos = subject_property.get("photos", [])
        
        # Extract URLs
        image_urls = []
        for photo in subject_photos:
            url = None
            if isinstance(photo, dict):
                url = photo.get("url")
            elif isinstance(photo, str):
                url = photo
                
            if url and url.lower().endswith(('.jpg', '.jpeg', '.png', '.gif')):
                image_urls.append(url)
        
        if image_urls:
            property_images[subject_uid] = image_urls
    
    # Extract from comps
    comps = data.get("comps", [])
    for comp in comps:
        if "uid" in comp:
            comp_uid = comp.get("uid")
            comp_photos = comp.get("photos", [])
            
            # Extract URLs
            image_urls = []
            for photo in comp_photos:
                url = None
                if isinstance(photo, dict):
                    url = photo.get("url")
                elif isinstance(photo, str):
                    url = photo
                    
                if url and url.lower().endswith(('.jpg', '.jpeg', '.png', '.gif')):
                    image_urls.append(url)
            
            if image_urls:
                property_images[comp_uid] = image_urls
    
    # If we found properties in the Premier Brokerage format, return them
    if property_images:
        logger.info(f"Extracted {len(property_images)} properties from Premier Brokerage format")
        return property_images
    
    # Case 2: Properties array format
    properties = data.get("properties", [])
    if properties:
        for i, prop in enumerate(properties):
            prop_uid = prop.get("uid", f"property_{i+1}")
            photos = prop.get("photos", []) or prop.get("images", [])
            
            # Extract URLs
            image_urls = []
            for photo in photos:
                url = None
                if isinstance(photo, dict):
                    url = photo.get("url", photo.get("image_url"))
                elif isinstance(photo, str):
                    url = photo
                    
                if url and url.lower().endswith(('.jpg', '.jpeg', '.png', '.gif')):
                    image_urls.append(url)
            
            if image_urls:
                property_images[prop_uid] = image_urls
        
        if property_images:
            logger.info(f"Extracted {len(property_images)} properties from properties array format")
            return property_images
    
    # Case 3: Results array format (treat each result as a property)
    results = data.get("results", [])
    if results:
        for i, result in enumerate(results):
            result_uid = result.get("uid", f"property_{i+1}")
            photos = result.get("photos", []) or result.get("images", [])
            
            # Extract URLs
            image_urls = []
            for photo in photos:
                url = None
                if isinstance(photo, dict):
                    url = photo.get("url", photo.get("image_url"))
                elif isinstance(photo, str):
                    url = photo
                    
                if url and url.lower().endswith(('.jpg', '.jpeg', '.png', '.gif')):
                    image_urls.append(url)
            
            if image_urls:
                property_images[result_uid] = image_urls
        
        if property_images:
            logger.info(f"Extracted {len(property_images)} properties from results array format")
            return property_images
    
    # Case 4: Direct images array (group all into a single property)
    images = data.get("images", []) or data.get("photos", [])
    if images:
        image_urls = []
        for image in images:
            url = None
            if isinstance(image, dict):
                url = image.get("url", image.get("image_url"))
            elif isinstance(image, str):
                url = image
                
            if url and url.lower().endswith(('.jpg', '.jpeg', '.png', '.gif')):
                image_urls.append(url)
        
        if image_urls:
            property_images["property_default"] = image_urls
            logger.info(f"Extracted 1 property with {len(image_urls)} images from direct images array")
            return property_images
    
    # Case 5: Last resort - search recursively for URL strings
    def extract_urls_recursive(obj, urls=None):
        if urls is None:
            urls = []
        
        if isinstance(obj, dict):
            for key, value in obj.items():
                if key in ["url", "image_url", "src", "path"] and isinstance(value, str) and value.startswith("http"):
                    if value.lower().endswith(('.jpg', '.jpeg', '.png', '.gif')):
                        urls.append(value)
                else:
                    extract_urls_recursive(value, urls)
        elif isinstance(obj, list):
            for item in obj:
                extract_urls_recursive(item, urls)
        
        return urls
    
    image_urls = extract_urls_recursive(data)
    if image_urls:
        property_images["property_default"] = image_urls
        logger.info(f"Extracted 1 property with {len(image_urls)} images from recursive search")
        return property_images
    
    logger.warning("Could not extract any property images from the API response")
    return {}

# Function to explicitly clean up memory
def clear_memory():
    """Force aggressive memory cleanup"""
    # Clear any module-level caches
    if 'image_cache' in globals():
        globals()['image_cache'].clear()
    
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

def download_property_dataset(
    endpoint: str = None,
    json_file: str = None,
    save_dir: str = "properties",
    auth_token: str = None,
    timeout: int = 60,
    max_retries: int = 5,
    retry_delay: int = 3,
    max_retry_delay: int = 60,  # Maximum delay between retries
    max_api_calls: int = 100,  # Maximum number of API calls to make
    max_properties: int = None,  # Maximum number of properties to download
    low_memory_mode: bool = True,  # Enable memory-saving features
    memory_limit_percentage: float = 70.0,  # Percentage of system memory to use before pausing
    batch_size: int = 10  # Number of properties to process at once in low memory mode
) -> Dict[str, List[str]]:
    """
    Download property dataset from API or JSON file and organize by property UID.
    Optimized for memory efficiency with large datasets.
    
    Args:
        endpoint: API endpoint for property dataset
        json_file: JSON file containing property dataset
        save_dir: Directory to save property images (default: "properties")
        auth_token: Authentication token for API access
        timeout: Timeout for API requests in seconds
        max_retries: Maximum number of retry attempts (use -1 for unlimited retries)
        retry_delay: Initial delay between retries in seconds
        max_retry_delay: Maximum delay between retries in seconds
        max_api_calls: Maximum number of API calls to make
        max_properties: Maximum number of properties to download (None for unlimited)
        low_memory_mode: Enable memory-saving features for large datasets
        memory_limit_percentage: Pause processing when system memory exceeds this percentage
        batch_size: Number of properties to process at once in low memory mode
        
    Returns:
        Dictionary mapping property UIDs to lists of image paths
    """
    all_properties_dict = {}
    
    # Import psutil for memory monitoring if available
    try:
        import psutil
        have_psutil = True
        process = psutil.Process(os.getpid())
        logger.info(f"Memory monitoring enabled, limit set to {memory_limit_percentage}% of system memory")
    except ImportError:
        have_psutil = False
        logger.warning("psutil not available, memory monitoring disabled. Install with: pip install psutil")
    
    # Validate inputs
    if not endpoint and not json_file:
        raise ValueError("Either endpoint or json_file must be provided")
    
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Set up headers for API requests
    headers = {}
    if auth_token:
        headers["Authorization"] = f"Bearer {auth_token}"
    
    # Helper function to check memory usage
    def check_memory_usage():
        """Check memory usage and pause if needed"""
        if have_psutil:
            # Get memory info
            mem_info = psutil.virtual_memory()
            process_memory = process.memory_info().rss / (1024 * 1024)  # MB
            
            # Log memory usage
            logger.debug(f"Memory usage: Process {process_memory:.1f}MB, System {mem_info.percent:.1f}%")
            
            # Pause if memory usage is too high
            if mem_info.percent > memory_limit_percentage:
                logger.warning(f"Memory usage high ({mem_info.percent:.1f}%), pausing to let system recover")
                # Release memory
                clear_memory()  # Use more aggressive cleanup
                # Sleep to allow OS to reclaim memory
                time.sleep(5)
                return True
        return False
    
    # Continuously fetch data until we have enough properties or reach the max calls limit
    api_calls = 0
    
    # Print message if max_properties is set
    if max_properties is not None:
        logger.info(f"Will download exactly {max_properties} property sets as requested")
    
    # Track properties we've processed to avoid duplicates
    processed_property_ids = set()
    temp_data_file = os.path.join(save_dir, "_temp_property_data.json")
    
    # Load any previously processed property data if it exists
    if os.path.exists(temp_data_file):
        try:
            with open(temp_data_file, 'r') as f:
                temp_data = json.load(f)
                processed_property_ids = set(temp_data.get("processed_ids", []))
                
                # Optionally load already downloaded properties
                if low_memory_mode:
                    for prop_id in processed_property_ids:
                        prop_dir = os.path.join(save_dir, prop_id)
                        if os.path.isdir(prop_dir):
                            # Just store the directory path, not image lists to save memory
                            all_properties_dict[prop_id] = True  # Use as marker
                
                logger.info(f"Loaded {len(processed_property_ids)} previously processed property IDs")
        except Exception as e:
            logger.error(f"Error loading temp property data file: {str(e)}")
    
    # Save processed IDs to resume later if needed
    def save_processed_ids():
        try:
            with open(temp_data_file, 'w') as f:
                json.dump({
                    "processed_ids": list(processed_property_ids),
                    "total_properties": len(processed_property_ids),
                    "last_update": time.strftime("%Y-%m-%d %H:%M:%S")
                }, f, indent=2)
            logger.debug("Saved processed property IDs to temp file")
        except Exception as e:
            logger.error(f"Error saving temp property data: {str(e)}")
    
    try:
        while api_calls < max_api_calls:
            # Check if we already have enough properties
            if max_properties is not None and len(processed_property_ids) >= max_properties:
                logger.info(f"Reached target of {max_properties} properties. Stopping download.")
                break
            
            # Calculate how many more properties we need
            if max_properties is not None:
                properties_needed = max_properties - len(processed_property_ids)
                logger.info(f"Need {properties_needed} more property sets to reach target of {max_properties}")
            
            # Check memory before API call
            check_memory_usage()
            
            api_calls += 1
            logger.info(f"API call {api_calls}/{max_api_calls if max_api_calls > 0 else 'unlimited'}")
            
            # Load data from API endpoint or JSON file
            data = None
            if endpoint:
                logger.info(f"Fetching property dataset from endpoint: {endpoint}")
                
                # Implement retry logic for API requests
                attempt = 0
                success = False
                
                while (max_retries == -1 or attempt < max_retries) and not success:
                    attempt += 1
                    try:
                        logger.info(f"API request attempt {attempt}{' (unlimited retries)' if max_retries == -1 else f'/{max_retries}'}")
                        response = requests.get(endpoint, headers=headers, timeout=timeout)
                        response.raise_for_status()
                        data = response.json()
                        success = True
                    except requests.exceptions.RequestException as e:
                        # Calculate delay with exponential backoff, but cap at max_retry_delay
                        wait_time = min(retry_delay * (2 ** (attempt - 1)), max_retry_delay)
                        logger.warning(f"Request failed: {str(e)}. Retrying in {wait_time} seconds...")
                        time.sleep(wait_time)
                
                if not success:
                    logger.error(f"Failed to fetch data after {attempt} attempts on API call {api_calls}")
                    if json_file:
                        logger.info(f"Falling back to JSON file: {json_file}")
                    else:
                        # Clean up memory from failed attempt before continuing
                        clear_memory()
                        continue  # Try the next API call
            
            if not data and json_file:
                logger.info(f"Loading property dataset from JSON file: {json_file}")
                try:
                    with open(json_file, 'r') as f:
                        data = json.load(f)
                except Exception as e:
                    logger.error(f"Failed to load JSON file: {str(e)}")
                    clear_memory()  # Clean up before returning
                    return all_properties_dict
            
            if not data:
                logger.warning("No data obtained from API or JSON file")
                # Add random delay before next attempt
                delay = retry_delay + random.random() * 2
                logger.info(f"Waiting {delay:.1f} seconds before next attempt...")
                clear_memory()  # Clean up from failed attempt
                time.sleep(delay)
                continue
            
            # Extract property images from the data using the enhanced extraction
            try:
                property_images = extract_image_urls_from_api_response(data)
                # Clear the original response data to free memory
                del data
                gc.collect()
            except Exception as e:
                logger.error(f"Error extracting images from response: {str(e)}")
                property_images = {}
                # Clear the failed data
                del data
                gc.collect()
            
            # Check memory after extraction
            check_memory_usage()
            
            if not property_images:
                logger.warning("No properties with images found in the data")
                # Add random delay before next attempt
                delay = retry_delay + random.random() * 2
                logger.info(f"Waiting {delay:.1f} seconds before next attempt...")
                clear_memory()  # Clean up after failed attempt
                time.sleep(delay)
                continue
            
            # Calculate how many more properties we can download
            remaining_slots = float('inf')  # Default to unlimited
            if max_properties is not None:
                remaining_slots = max_properties - len(processed_property_ids)
                logger.info(f"Can download {remaining_slots} more properties to reach limit of {max_properties}")
                
                # If no slots left, break out early
                if remaining_slots <= 0:
                    logger.info("Already reached property limit, stopping download")
                    # Release the property_images dictionary to free memory
                    del property_images
                    gc.collect()
                    break
            
            # Get a list of property IDs (exclude already processed ones to avoid duplicates)
            available_property_ids = [
                pid for pid in property_images.keys() 
                if pid not in processed_property_ids
            ]
            
            if not available_property_ids:
                logger.warning("No new properties found that haven't been processed already")
                # Clean up before continuing
                del property_images
                gc.collect()
                continue
                
            logger.info(f"Found {len(available_property_ids)} new properties to process")
            
            # Shuffle for random selection
            random.shuffle(available_property_ids)
            
            # Limit to just what we need
            if max_properties is not None:
                available_property_ids = available_property_ids[:remaining_slots]
            
            # Process in batches if using low memory mode
            if low_memory_mode:
                # Process batches of properties
                for batch_start in range(0, len(available_property_ids), batch_size):
                    # Check if we've reached the maximum properties limit
                    if max_properties is not None and len(processed_property_ids) >= max_properties:
                        logger.info(f"Reached target of {max_properties} properties. Stopping batch processing.")
                        break
                    
                    # Get batch of property IDs
                    batch_ids = available_property_ids[batch_start:batch_start + batch_size]
                    logger.info(f"Processing batch of {len(batch_ids)} properties ({batch_start+1}-{batch_start+len(batch_ids)} of {len(available_property_ids)})")
                    
                    # Process each property in batch
                    for property_id in batch_ids:
                        # Skip if already processed
                        if property_id in processed_property_ids:
                            continue
                            
                        try:
                            image_urls = property_images[property_id]
                            property_dir = os.path.join(save_dir, property_id)
                            os.makedirs(property_dir, exist_ok=True)
                            
                            logger.info(f"Processing property {property_id} with {len(image_urls)} images ({len(processed_property_ids)+1}/{max_properties if max_properties is not None else 'unlimited'})")
                            
                            # Download images one by one
                            property_images_list = []
                            for i, url in enumerate(image_urls):
                                # Check memory usage more frequently (every image instead of every 5)
                                check_memory_usage()
                                    
                                image_filename = f"{i+1:03d}.jpg"
                                image_path = os.path.join(property_dir, image_filename)
                                
                                # Skip if image already exists and is valid
                                if os.path.exists(image_path) and os.path.getsize(image_path) > 0:
                                    try:
                                        # Quick check if file is valid (avoid full load)
                                        with open(image_path, 'rb') as f:
                                            header = f.read(32)  # Read just the header
                                            if header.startswith(b'\xff\xd8'):  # JPEG signature
                                                property_images_list.append(image_path)
                                                continue
                                    except Exception:
                                        # If any error, try re-downloading
                                        pass
                                
                                # Download with retry
                                success = False
                                attempt = 0
                                while (max_retries == -1 or attempt < max_retries) and not success:
                                    attempt += 1
                                    try:
                                        img_response = requests.get(url, stream=True, timeout=timeout)
                                        img_response.raise_for_status()
                                        
                                        with open(image_path, 'wb') as img_file:
                                            for chunk in img_response.iter_content(chunk_size=8192):
                                                if chunk:
                                                    img_file.write(chunk)
                                        
                                        # Quick validation (just check the file exists and is non-empty)
                                        if os.path.exists(image_path) and os.path.getsize(image_path) > 0:
                                            property_images_list.append(image_path)
                                            success = True
                                        else:
                                            raise ValueError("Empty image file")
                                    except Exception as e:
                                        wait_time = min(retry_delay * (2 ** (attempt - 1)), max_retry_delay)
                                        logger.warning(f"Failed to download image {i+1}/{len(image_urls)}: {str(e)}. Retrying in {wait_time} seconds...")
                                        time.sleep(wait_time)
                                
                                # If failed after all retries, log and continue
                                if not success:
                                    logger.error(f"Failed to download image after {attempt} attempts: {url}")
                            
                            # Add property to processed set if it has at least one image
                            if property_images_list:
                                # In low memory mode, just store the folder path
                                all_properties_dict[property_id] = property_dir
                                processed_property_ids.add(property_id)
                                
                                # Save progress more frequently
                                if len(processed_property_ids) % 5 == 0:
                                    save_processed_ids()
                                
                                logger.info(f"Added property {property_id} with {len(property_images_list)} images. Total properties: {len(processed_property_ids)}")
                                
                                # Clear the property_images_list to free memory
                                del property_images_list
                            else:
                                logger.warning(f"No images downloaded for property {property_id}")
                                try:
                                    # Remove empty property directory
                                    os.rmdir(property_dir)
                                except:
                                    pass
                        except Exception as e:
                            logger.error(f"Error processing property {property_id}: {str(e)}")
                    
                    # More aggressive memory cleanup after each batch
                    check_memory_usage()
                    clear_memory()
                    
                    # Save progress after each batch
                    save_processed_ids()
            else:
                # Original approach (load all in memory)
                # Process each property
                for property_id in available_property_ids:
                    # Check if we've reached the maximum properties limit
                    if max_properties is not None and len(processed_property_ids) >= max_properties:
                        logger.info(f"Reached target of {max_properties} properties. Stopping download.")
                        break
                        
                    # Skip if we already have this property
                    if property_id in processed_property_ids:
                        logger.info(f"Property {property_id} already processed, skipping")
                        continue
                    
                    image_urls = property_images[property_id]
                    property_dir = os.path.join(save_dir, property_id)
                    os.makedirs(property_dir, exist_ok=True)
                    
                    property_images_list = []
                    logger.info(f"Processing property {property_id} with {len(image_urls)} images ({len(processed_property_ids)+1}/{max_properties if max_properties is not None else 'unlimited'})")
                    
                    # Download images
                    for i, url in enumerate(image_urls):
                        image_filename = f"{i+1:03d}.jpg"
                        image_path = os.path.join(property_dir, image_filename)
                        
                        # Skip if image already exists
                        if os.path.exists(image_path) and os.path.getsize(image_path) > 0:
                            try:
                                # Verify the image is valid
                                with Image.open(image_path) as img:
                                    img.verify()  # Verify it's a valid image
                                property_images_list.append(image_path)
                                logger.debug(f"Image already exists and verified: {image_path}")
                                continue
                            except Exception:
                                logger.warning(f"Existing image is corrupt: {image_path}. Will re-download.")
                                # Continue to re-download since the image is corrupt
                        
                        # Download image with retry logic
                        success = False
                        attempt = 0
                        
                        while (max_retries == -1 or attempt < max_retries) and not success:
                            attempt += 1
                            try:
                                logger.debug(f"Downloading image {i+1}/{len(image_urls)} for property {property_id}, attempt {attempt}{' (unlimited retries)' if max_retries == -1 else f'/{max_retries}'}")
                                
                                img_response = requests.get(url, stream=True, timeout=timeout)
                                img_response.raise_for_status()
                                
                                with open(image_path, 'wb') as img_file:
                                    for chunk in img_response.iter_content(chunk_size=8192):
                                        if chunk:
                                            img_file.write(chunk)
                                
                                # Verify the downloaded file is valid
                                if os.path.exists(image_path) and os.path.getsize(image_path) > 0:
                                    try:
                                        with Image.open(image_path) as img:
                                            # Just verify it's a valid image
                                            img.verify()
                                        property_images_list.append(image_path)
                                        success = True
                                        logger.debug(f"Successfully downloaded and verified image: {image_path}")
                                    except Exception as e:
                                        logger.warning(f"Downloaded invalid image file: {str(e)}")
                                        os.remove(image_path)  # Remove invalid image
                                        raise ValueError("Invalid image file")
                                else:
                                    raise ValueError("Empty or missing image file")
                                
                            except Exception as e:
                                # Calculate delay with exponential backoff, but cap at max_retry_delay
                                wait_time = min(retry_delay * (2 ** (attempt - 1)), max_retry_delay)
                                logger.warning(f"Failed to download image: {str(e)}. Retrying in {wait_time} seconds...")
                                time.sleep(wait_time)
                        
                        if not success:
                            logger.error(f"Failed to download image after {attempt} attempts: {url}")
                    
                    # Add property to dictionary if it has at least one image
                    if property_images_list:
                        all_properties_dict[property_id] = property_images_list
                        processed_property_ids.add(property_id)
                        logger.info(f"Added property {property_id} with {len(property_images_list)} images. Total properties: {len(processed_property_ids)}")
                        
                        # Check if we've reached the maximum properties limit
                        if max_properties is not None and len(processed_property_ids) >= max_properties:
                            logger.info(f"Reached target of {max_properties} properties. Stopping download.")
                            break
                    else:
                        logger.warning(f"No images downloaded for property {property_id}")
                        try:
                            # Remove empty property directory
                            os.rmdir(property_dir)
                        except:
                            pass
            
            # After processing all properties from this API call, clean up
            logger.info("Finished processing properties from current API call, cleaning up memory")
            # Explicitly release the property_images dictionary
            del property_images
            clear_memory()
            
            # For JSON file, we only process once
            if json_file:
                break
            
            # Check if we have enough properties
            if endpoint and len(processed_property_ids) > 0:
                if max_properties is not None and len(processed_property_ids) >= max_properties:
                    logger.info(f"Successfully downloaded {len(processed_property_ids)} properties with images")
                    break
                else:
                    # Add delay between API calls to avoid rate limiting
                    delay = 2 + random.random() * 2  # 2-4 seconds
                    logger.info(f"Waiting {delay:.1f} seconds before next API call...")
                    time.sleep(delay)
                    
                    # Save progress before next call
                    save_processed_ids()
        
        # Final aggressive memory cleanup before building the return dictionary
        logger.info("Download completed. Performing final memory cleanup before building result dictionary.")
        clear_memory()
        
        # If we're in low_memory_mode, we need to build full image lists for each property at the end
        if low_memory_mode:
            all_properties_dict_full = {}
            
            # Only load and process all the property files we need
            if max_properties is not None:
                processed_property_ids = list(processed_property_ids)[:max_properties]
            
            logger.info(f"Building final property dictionary with {len(processed_property_ids)} properties...")
            
            # Process properties in smaller batches for better memory management
            batch_size = 20  # Smaller batch size for final processing
            for batch_start in range(0, len(processed_property_ids), batch_size):
                batch_end = min(batch_start + batch_size, len(processed_property_ids))
                logger.info(f"Processing final batch {batch_start//batch_size + 1}/{(len(processed_property_ids) + batch_size - 1)//batch_size}")
                
                # Process each property in this batch
                for i, property_id in enumerate(list(processed_property_ids)[batch_start:batch_end]):
                    property_dir = os.path.join(save_dir, property_id)
                    
                    # Skip if property directory doesn't exist
                    if not os.path.isdir(property_dir):
                        continue
                    
                    # Get all image files in this property directory
                    image_paths = []
                    for filename in os.listdir(property_dir):
                        file_path = os.path.join(property_dir, filename)
                        if os.path.isfile(file_path) and filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                            image_paths.append(file_path)
                    
                    if image_paths:
                        all_properties_dict_full[property_id] = image_paths
                
                # Clean up memory after each batch
                gc.collect()
                # More frequent progress reports
                if batch_start % 100 == 0 and batch_start > 0:
                    logger.info(f"Processed {batch_end}/{len(processed_property_ids)} properties for final dictionary")
            
            # Replace the marker dictionary with the full paths dictionary
            all_properties_dict = all_properties_dict_full
            
            # Clear the temporary marker data
            del all_properties_dict_full
            clear_memory()
        
        total_images = sum(len(images) for images in all_properties_dict.values())
        logger.info(f"Downloaded {total_images} images across {len(all_properties_dict)} properties")
        
        # If max_properties is set, ensure we don't return more than requested
        if max_properties is not None and len(all_properties_dict) > max_properties:
            # Get a list of property IDs
            property_ids = list(all_properties_dict.keys())
            # Take only the first max_properties
            selected_ids = property_ids[:max_properties]
            # Create a new dictionary with only the selected properties
            selected_properties = {pid: all_properties_dict[pid] for pid in selected_ids}
            
            logger.info(f"Limiting to exactly {max_properties} properties as requested")
            # Explicitly release the full dictionary
            del all_properties_dict
            clear_memory()
            all_properties_dict = selected_properties
        
        # Clean up temp files if all went well
        if os.path.exists(temp_data_file):
            try:
                os.remove(temp_data_file)
                logger.info(f"Removed temporary data file: {temp_data_file}")
            except Exception as e:
                logger.warning(f"Failed to remove temporary data file: {str(e)}")
        
        # Final cleanup before returning results
        logger.info("Property organization complete. Performing final memory cleanup.")
        gc.collect()
        
        return all_properties_dict
        
    except KeyboardInterrupt:
        logger.warning("Download interrupted by user. Saving progress...")
        save_processed_ids()
        
        # Clean up memory before returning
        clear_memory()
        
        # Return what we have so far
        logger.info(f"Returning {len(all_properties_dict)} properties processed so far")
        return all_properties_dict

def list_properties_in_directory(directory: str) -> Dict[str, List[str]]:
    """
    List all property directories and their image files.
    
    Args:
        directory: The directory to list properties from
        
    Returns:
        Dictionary mapping property UIDs to lists of image file paths
    """
    if not os.path.exists(directory):
        logger.warning(f"Directory does not exist: {directory}")
        return {}
    
    properties = {}
    for prop_id in os.listdir(directory):
        prop_dir = os.path.join(directory, prop_id)
        if os.path.isdir(prop_dir):
            # Get all image files in this property directory
            image_paths = []
            for filename in os.listdir(prop_dir):
                file_path = os.path.join(prop_dir, filename)
                if os.path.isfile(file_path) and filename.lower().endswith(('.jpg', '.jpeg', '.png', '.gif')):
                    image_paths.append(file_path)
            
            if image_paths:
                properties[prop_id] = image_paths
                logger.debug(f"Found property {prop_id} with {len(image_paths)} images")
    
    logger.info(f"Found {len(properties)} properties with images in {directory}")
    return properties

def load_property_ratings(ratings_path):
    """
    Load property similarity ratings from a JSON file.
    
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
        ratings_path: Path to the JSON file with ratings
        
    Returns:
        List of rating dictionaries
    """
    try:
        with open(ratings_path, 'r') as f:
            data = json.load(f)
        
        ratings = data.get("property_pairs", [])
        logger.info(f"Loaded {len(ratings)} property ratings from {ratings_path}")
        return ratings
    except Exception as e:
        logger.error(f"Error loading property ratings from {ratings_path}: {str(e)}")
        return []

def verify_properties_have_images(
    property_ratings: List[Dict],
    property_data: Dict[str, List[str]]
) -> List[Dict]:
    """
    Verify that all properties in the ratings have images available.
    
    Args:
        property_ratings: List of rating dictionaries
        property_data: Dictionary mapping property UIDs to lists of image paths
        
    Returns:
        List of valid rating dictionaries
    """
    valid_ratings = []
    
    for rating in property_ratings:
        subject_uid = rating.get("subject_uid")
        comparable_uid = rating.get("comparable_uid")
        
        if not subject_uid or not comparable_uid:
            logger.warning(f"Rating missing subject or comparable UID: {rating}")
            continue
        
        if subject_uid not in property_data:
            logger.warning(f"Subject property {subject_uid} has no images")
            continue
            
        if comparable_uid not in property_data:
            logger.warning(f"Comparable property {comparable_uid} has no images")
            continue
        
        valid_ratings.append(rating)
    
    logger.info(f"Verified {len(valid_ratings)} valid ratings out of {len(property_ratings)}")
    return valid_ratings

def organize_properties_for_training(
    property_data: Dict[str, List[str]],
    train_ratio: float = 0.8,
    random_seed: int = 42
) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    """
    Split properties into training and testing sets.
    
    Args:
        property_data: Dictionary mapping property UIDs to lists of image paths
        train_ratio: Ratio of properties to use for training
        random_seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_properties, test_properties) dictionaries
    """
    random.seed(random_seed)
    
    # Get all property UIDs
    property_uids = list(property_data.keys())
    random.shuffle(property_uids)
    
    # Split into training and testing
    split_idx = int(len(property_uids) * train_ratio)
    train_uids = property_uids[:split_idx]
    test_uids = property_uids[split_idx:]
    
    # Create dictionaries for training and testing
    train_properties = {uid: property_data[uid] for uid in train_uids}
    test_properties = {uid: property_data[uid] for uid in test_uids}
    
    logger.info(f"Organized properties: {len(train_properties)} for training, {len(test_properties)} for testing")
    return train_properties, test_properties

def prepare_property_pairs_dataset(
    property_data: Dict[str, List[str]],
    ratings_path: str,
    output_dir: str = "property_pairs"
) -> str:
    """
    Prepare a dataset of property pairs with expert ratings.
    
    Args:
        property_data: Dictionary mapping property UIDs to lists of image paths
        ratings_path: Path to the JSON file with expert ratings
        output_dir: Directory to save the processed property pairs
        
    Returns:
        Path to the output directory
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load property ratings
    ratings = load_property_ratings(ratings_path)
    if not ratings:
        logger.error(f"No property ratings found in {ratings_path}")
        return output_dir
    
    # Verify properties have images
    valid_ratings = verify_properties_have_images(ratings, property_data)
    if not valid_ratings:
        logger.error("No valid property pairs found")
        return output_dir
    
    # Save valid ratings to output directory
    pairs_path = os.path.join(output_dir, "property_pairs.json")
    with open(pairs_path, 'w') as f:
        json.dump({
            "property_pairs": valid_ratings,
            "creation_date": time.strftime("%Y-%m-%d %H:%M:%S")
        }, f, indent=2)
    
    # Create a symlink or copy images to make dataset more portable (optional)
    for rating in valid_ratings:
        subject_uid = rating["subject_uid"]
        comparable_uid = rating["comparable_uid"]
        
        # Add image paths to rating data
        rating["subject_images"] = property_data[subject_uid]
        rating["comparable_images"] = property_data[comparable_uid]
    
    # Save updated ratings with image paths
    detailed_pairs_path = os.path.join(output_dir, "property_pairs_with_paths.json")
    with open(detailed_pairs_path, 'w') as f:
        json.dump({
            "property_pairs": valid_ratings,
            "creation_date": time.strftime("%Y-%m-%d %H:%M:%S")
        }, f, indent=2)
    
    logger.info(f"Prepared property pairs dataset with {len(valid_ratings)} pairs in {output_dir}")
    return output_dir

# Maintain the original function for backward compatibility
def download_images(urls, save_dir="images", start_index=0, timeout=10):
    """
    Legacy function for downloading individual images.
    Maintained for backward compatibility.
    """
    os.makedirs(save_dir, exist_ok=True)
    downloaded_paths = []
    
    for i, url in enumerate(urls):
        try:
            index = start_index + i
            image_path = os.path.join(save_dir, f"image_{index:04d}.jpg")
            
            if os.path.exists(image_path) and os.path.getsize(image_path) > 0:
                logger.info(f"Image already exists: {image_path}")
                downloaded_paths.append(image_path)
                continue
            
            response = requests.get(url, timeout=timeout)
            response.raise_for_status()
            
            with open(image_path, 'wb') as f:
                f.write(response.content)
            
            downloaded_paths.append(image_path)
            logger.info(f"Downloaded image {i+1}/{len(urls)}")
            
        except Exception as e:
            logger.error(f"Error downloading image {url}: {str(e)}")
    
    return downloaded_paths

# Keep the rest of the functions for backward compatibility

def get_user_input_for_api_download():
    """
    Get user input for API parameters focused on property-level data gathering.
    
    Returns:
        Dictionary of API parameters
    """
    print("\n=== Property Download Configuration ===\n")
    
    # Default values
    default_api_url = "https://dash.premierbrokeragesystem.com/api/get_random_listing_photos"
    default_auth_token = ""
    default_train_count = 100
    default_test_count = 20
    default_cache_dir = "/workspace/property_cache"
    default_output_dir = "/workspace/properties"
    default_max_workers = 4
    default_low_memory_mode = "y"
    default_memory_limit = 70
    
    # Get user input with defaults
    api_url = input(f"Enter API URL [default: {default_api_url}]: ").strip() or default_api_url
    
    auth_token = input(f"Enter authentication token [default: none]: ").strip() or default_auth_token
    
    train_properties_str = input(f"Number of property sets for training [default: {default_train_count}]: ").strip()
    train_properties = int(train_properties_str) if train_properties_str.isdigit() else default_train_count
    
    test_properties_str = input(f"Number of property sets for testing [default: {default_test_count}]: ").strip()
    test_properties = int(test_properties_str) if test_properties_str.isdigit() else default_test_count
    
    # Ask if user wants to use the cache directory
    use_cache = input("Use separate cache directory? (y/n, default: n): ").strip().lower() == 'y'
    
    if use_cache:
        cache_dir = input(f"Cache directory for downloaded property images [default: {default_cache_dir}]: ").strip() or default_cache_dir
    else:
        # Use output directory as cache to avoid duplication
        cache_dir = None
    
    output_dir = input(f"Output directory for processed data [default: {default_output_dir}]: ").strip() or default_output_dir
    
    # If cache_dir is None, use output_dir as cache
    if cache_dir is None:
        cache_dir = output_dir
        print(f"Using output directory {output_dir} as cache directory")
    
    max_workers_str = input(f"Maximum number of concurrent downloads [default: {default_max_workers}]: ").strip()
    max_workers = int(max_workers_str) if max_workers_str.isdigit() else default_max_workers
    
    # Get memory optimization options
    print("\n=== Memory Optimization Settings ===")
    print("Enable for large datasets or limited RAM to avoid memory overflow")
    
    low_memory_mode_choice = input(f"Enable memory optimization? (y/n, default: {default_low_memory_mode}): ").strip().lower() or default_low_memory_mode
    low_memory_mode = low_memory_mode_choice == 'y'
    
    if low_memory_mode:
        memory_limit_str = input(f"Memory usage limit (%) [default: {default_memory_limit}]: ").strip()
        memory_limit_percentage = float(memory_limit_str) if memory_limit_str and memory_limit_str.replace('.', '').isdigit() else default_memory_limit
        
        # Get batch size
        batch_size_str = input(f"Processing batch size [default: 10]: ").strip()
        batch_size = int(batch_size_str) if batch_size_str.isdigit() else 10
    else:
        memory_limit_percentage = default_memory_limit
        batch_size = 10
    
    # Create configuration
    params = {
        "api_url": api_url,
        "auth_token": auth_token,
        "train_properties": train_properties,
        "test_properties": test_properties,
        "cache_dir": cache_dir,
        "output_dir": output_dir,
        "max_workers": max_workers,
        "low_memory_mode": low_memory_mode,
        "memory_limit_percentage": memory_limit_percentage,
        "batch_size": batch_size
    }
    
    print("\nProperty Download Configuration:")
    for key, value in params.items():
        print(f"  {key}: {value}")
    
    return params

def fetch_training_and_testing_data(
    api_url: str,
    auth_token: Optional[str] = None,
    train_properties: int = 100,
    test_properties: int = 20,
    cache_dir: str = "/workspace/property_cache",
    max_workers: int = 4,
    low_memory_mode: bool = True  # Use memory-optimized processing
) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    """
    Fetch training and testing property sets from API with memory optimization.
    
    Args:
        api_url: URL of the API endpoint
        auth_token: Authentication token for API
        train_properties: Number of property sets for training
        test_properties: Number of property sets for testing
        cache_dir: Directory to cache downloaded property images
        max_workers: Maximum number of concurrent downloads
        low_memory_mode: Use memory-optimized processing for large datasets
        
    Returns:
        Tuple of (train_properties_dict, test_properties_dict) dictionaries mapping property UIDs to image paths
    """
    logger.info(f"Fetching data from API: {api_url}")
    logger.info(f"Will download exactly {train_properties} training property sets and {test_properties} testing property sets")
    logger.info(f"Memory optimization is {'enabled' if low_memory_mode else 'disabled'}")
    
    # Define directories
    train_dir = TRAIN_DIR
    test_dir = TEST_DIR
    
    # Create directories
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    # Clear memory before starting download
    clear_memory()
    
    # Ask if user wants to use the API
    use_api = input("Do you want to use the API? (y/n, default: n): ").strip().lower() == 'y'
    
    if use_api:
        # Total properties to download
        total_properties_needed = train_properties + test_properties
        logger.info(f"Starting download process for exactly {total_properties_needed} property sets")
        
        # First, try to download training properties
        train_properties_dict = {}
        test_properties_dict = {}
        
        # Step 1: Download all properties together using memory-optimized approach
        all_properties_dict = download_property_dataset(
            endpoint=api_url,
            auth_token=auth_token,
            save_dir=cache_dir,
            timeout=60,
            max_retries=-1,  # Unlimited retries
            max_api_calls=100,  # Keep trying with multiple API calls
            max_properties=total_properties_needed,  # Set maximum properties to download
            low_memory_mode=low_memory_mode,  # Use memory optimization
            memory_limit_percentage=70.0,  # Conservative memory limit
            batch_size=10  # Process in small batches
        )
        
        # Clear memory after download, before processing for train/test split
        logger.info("Download complete. Cleaning memory before processing train/test split.")
        clear_memory()
        
        if all_properties_dict:
            # Verify we got exactly the right number of properties
            actual_property_count = len(all_properties_dict)
            if actual_property_count != total_properties_needed:
                logger.warning(f"Downloaded {actual_property_count} properties, but needed exactly {total_properties_needed}")
                
                # If we got too many, trim the list
                if actual_property_count > total_properties_needed:
                    property_uids = list(all_properties_dict.keys())
                    selected_uids = property_uids[:total_properties_needed]
                    all_properties_dict = {uid: all_properties_dict[uid] for uid in selected_uids}
                    logger.info(f"Trimmed to exactly {total_properties_needed} properties")
                    
                # If we got too few, warn but continue
                elif actual_property_count < total_properties_needed:
                    logger.warning(f"Could only download {actual_property_count} properties, which is less than the requested {total_properties_needed}")
            
            logger.info(f"Splitting {len(all_properties_dict)} properties into {train_properties} training and {test_properties} testing sets")
            
            # Process properties in memory-efficient way if needed
            if low_memory_mode and len(all_properties_dict) > 50:  # Only use this for larger sets
                logger.info("Using memory-efficient property splitting...")
                
                # Get all property UIDs
                property_uids = list(all_properties_dict.keys())
                # Shuffle them for random selection
                random.shuffle(property_uids)
                
                # Determine train/test split
                train_uids = []
                test_uids = []
                
                # Fill training sets first
                remaining_train = min(train_properties, len(property_uids))
                if remaining_train > 0:
                    train_uids = property_uids[:remaining_train]
                    property_uids = property_uids[remaining_train:]
                
                # Then fill testing sets
                remaining_test = min(test_properties, len(property_uids))
                if remaining_test > 0:
                    test_uids = property_uids[:remaining_test]
                elif train_uids and test_properties > 0:
                    # If we don't have enough for test but have training, use some from training
                    logger.warning(f"Not enough properties for separate test set, using some from training")
                    test_count = min(test_properties, max(1, len(train_uids) // 3))
                    test_uids = train_uids[-test_count:]
                    train_uids = train_uids[:-test_count]
                
                logger.info(f"Split into {len(train_uids)} training and {len(test_uids)} testing property sets")
                
                # Process and copy properties in batches to reduce memory usage
                train_properties_dict = {}
                test_properties_dict = {}
                
                # Process training properties in batches
                batch_size = 10
                for batch_start in range(0, len(train_uids), batch_size):
                    batch_uids = train_uids[batch_start:batch_start + batch_size]
                    logger.info(f"Processing training batch {batch_start//batch_size + 1}/{(len(train_uids) + batch_size - 1)//batch_size}")
                    
                    for uid in batch_uids:
                        if uid not in all_properties_dict:
                            continue
                            
                        src_dir = os.path.join(cache_dir, uid)
                        dst_dir = os.path.join(train_dir, uid)
                        
                        # Create destination directory
                        os.makedirs(dst_dir, exist_ok=True)
                        
                        # Copy or link files
                        try:
                            # Get source files
                            if isinstance(all_properties_dict[uid], list):
                                # We have a list of image paths
                                source_images = all_properties_dict[uid]
                            else:
                                # We might have a directory path or other placeholder
                                # Scan the directory for images
                                source_images = []
                                if os.path.isdir(src_dir):
                                    for filename in os.listdir(src_dir):
                                        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                                            source_images.append(os.path.join(src_dir, filename))
                            
                            # Create destination images list
                            dest_images = []
                            for src_path in source_images:
                                # Get just the filename
                                filename = os.path.basename(src_path)
                                dst_path = os.path.join(dst_dir, filename)
                                
                                # Copy the file if not already there
                                if not os.path.exists(dst_path):
                                    import shutil
                                    shutil.copyfile(src_path, dst_path)
                                
                                dest_images.append(dst_path)
                            
                            # Store the destination paths
                            if dest_images:
                                train_properties_dict[uid] = dest_images
                        except Exception as e:
                            logger.error(f"Error copying training property {uid}: {str(e)}")
                    
                    # Clear memory after each batch
                    gc.collect()
                
                # Process testing properties in batches
                for batch_start in range(0, len(test_uids), batch_size):
                    batch_uids = test_uids[batch_start:batch_start + batch_size]
                    logger.info(f"Processing testing batch {batch_start//batch_size + 1}/{(len(test_uids) + batch_size - 1)//batch_size}")
                    
                    for uid in batch_uids:
                        if uid not in all_properties_dict:
                            continue
                            
                        src_dir = os.path.join(cache_dir, uid)
                        dst_dir = os.path.join(test_dir, uid)
                        
                        # Create destination directory
                        os.makedirs(dst_dir, exist_ok=True)
                        
                        # Copy or link files
                        try:
                            # Get source files
                            if isinstance(all_properties_dict[uid], list):
                                # We have a list of image paths
                                source_images = all_properties_dict[uid]
                            else:
                                # We might have a directory path or other placeholder
                                # Scan the directory for images
                                source_images = []
                                if os.path.isdir(src_dir):
                                    for filename in os.listdir(src_dir):
                                        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                                            source_images.append(os.path.join(src_dir, filename))
                            
                            # Create destination images list
                            dest_images = []
                            for src_path in source_images:
                                # Get just the filename
                                filename = os.path.basename(src_path)
                                dst_path = os.path.join(dst_dir, filename)
                                
                                # Copy the file if not already there
                                if not os.path.exists(dst_path):
                                    import shutil
                                    shutil.copyfile(src_path, dst_path)
                                
                                dest_images.append(dst_path)
                            
                            # Store the destination paths
                            if dest_images:
                                test_properties_dict[uid] = dest_images
                        except Exception as e:
                            logger.error(f"Error copying testing property {uid}: {str(e)}")
                    
                    # Clear memory after each batch
                    gc.collect()
            else:
                # Original approach
                # Shuffle property UIDs for random train/test split
                property_uids = list(all_properties_dict.keys())
                random.shuffle(property_uids)
                
                # Step 2: Split into training and testing sets with exact counts
                train_uids = []
                test_uids = []
                
                # Fill training sets first
                remaining_train = min(train_properties, len(property_uids))
                if remaining_train > 0:
                    train_uids = property_uids[:remaining_train]
                    property_uids = property_uids[remaining_train:]
                
                # Then fill testing sets
                remaining_test = min(test_properties, len(property_uids))
                if remaining_test > 0:
                    test_uids = property_uids[:remaining_test]
                elif train_uids and test_properties > 0:
                    # If we don't have enough for test but have training, use some from training
                    logger.warning(f"Not enough properties for separate test set, using some from training")
                    test_count = min(test_properties, max(1, len(train_uids) // 3))
                    test_uids = train_uids[-test_count:]
                    train_uids = train_uids[:-test_count]
                
                # Verify we have exact counts
                logger.info(f"Split into exactly {len(train_uids)} training and {len(test_uids)} testing property sets")
                
                # Step 3: Create dictionaries with only the properties we want
                train_properties_dict = {uid: all_properties_dict[uid] for uid in train_uids}
                test_properties_dict = {uid: all_properties_dict[uid] for uid in test_uids}
                
                # Step 4: Clean out existing directories to start fresh
                logger.info("Preparing train and test directories")
                import shutil
                
                # Clear train directory
                for item in os.listdir(train_dir):
                    item_path = os.path.join(train_dir, item)
                    if os.path.isdir(item_path):
                        try:
                            shutil.rmtree(item_path)
                        except:
                            logger.warning(f"Failed to remove directory: {item_path}")
                
                # Clear test directory
                for item in os.listdir(test_dir):
                    item_path = os.path.join(test_dir, item)
                    if os.path.isdir(item_path):
                        try:
                            shutil.rmtree(item_path)
                        except:
                            logger.warning(f"Failed to remove directory: {item_path}")
                
                # Step 5: Copy properties to the appropriate directories
                logger.info(f"Copying {len(train_uids)} property sets to training directory")
                train_properties_copy = {}
                
                for uid in train_uids:
                    src_dir = os.path.join(cache_dir, uid)
                    dst_dir = os.path.join(train_dir, uid)
                    
                    try:
                        # Create destination directory structure
                        os.makedirs(os.path.dirname(dst_dir), exist_ok=True)
                        
                        # Copy directory
                        shutil.copytree(src_dir, dst_dir)
                        
                        # Update paths
                        old_paths = train_properties_dict[uid]
                        new_paths = [os.path.join(dst_dir, os.path.basename(p)) for p in old_paths]
                        train_properties_copy[uid] = new_paths
                        
                        logger.debug(f"Copied property {uid} to training directory")
                    except Exception as e:
                        logger.error(f"Failed to copy property {uid}: {str(e)}")
                        train_properties_copy[uid] = train_properties_dict[uid]
                
                logger.info(f"Copying {len(test_uids)} property sets to testing directory")
                test_properties_copy = {}
                
                for uid in test_uids:
                    src_dir = os.path.join(cache_dir, uid)
                    dst_dir = os.path.join(test_dir, uid)
                    
                    try:
                        # Create destination directory structure
                        os.makedirs(os.path.dirname(dst_dir), exist_ok=True)
                        
                        # Copy directory
                        shutil.copytree(src_dir, dst_dir)
                        
                        # Update paths
                        old_paths = test_properties_dict[uid]
                        new_paths = [os.path.join(dst_dir, os.path.basename(p)) for p in old_paths]
                        test_properties_copy[uid] = new_paths
                        
                        logger.debug(f"Copied property {uid} to testing directory")
                    except Exception as e:
                        logger.error(f"Failed to copy property {uid}: {str(e)}")
                        test_properties_copy[uid] = test_properties_dict[uid]
                
                # Use the copies
                train_properties_dict = train_properties_copy
                test_properties_dict = test_properties_copy
            
            logger.info(f"Successfully organized {len(train_properties_dict)} training and {len(test_properties_dict)} testing property sets")
            
            # Final verification
            if len(train_properties_dict) != train_properties:
                logger.warning(f"Warning: Expected {train_properties} training sets but got {len(train_properties_dict)}")
            
            if len(test_properties_dict) != test_properties:
                logger.warning(f"Warning: Expected {test_properties} testing sets but got {len(test_properties_dict)}")
            
            # Final cleanup before returning results
            logger.info("Property organization complete. Performing final memory cleanup.")
            clear_memory()
            
            return train_properties_dict, test_properties_dict
            
        else:
            # If we couldn't get any properties, use dummy data
            logger.warning("Failed to download any properties, using dummy data instead")
            train_dummy = create_dummy_property_data(train_properties)
            test_dummy = create_dummy_property_data(test_properties)
            return train_dummy, test_dummy
    
    # If user chose not to use API, use local data
    logger.info("Using local property directories instead of API")
    
    use_existing = input("Do you have existing property directories? (y/n, default: n): ").strip().lower() == 'y'
    
    if use_existing:
        train_dir_input = input(f"Enter directory with training property sets [default: {train_dir}]: ").strip() or train_dir
        test_dir_input = input(f"Enter directory with testing property sets [default: {test_dir}]: ").strip() or test_dir
        
        train_properties_dict = list_properties_in_directory(train_dir_input) if train_dir_input else {}
        test_properties_dict = list_properties_in_directory(test_dir_input) if test_dir_input else {}
        
        # If specified directories don't have data, create dummy data
        if not train_properties_dict:
            logger.warning(f"No property data found in {train_dir_input}, creating dummy data")
            train_properties_dict = create_dummy_property_data(train_properties)
            
        if not test_properties_dict:
            logger.warning(f"No property data found in {test_dir_input}, creating dummy data")
            test_properties_dict = create_dummy_property_data(test_properties)
    else:
        # Use sample data for testing
        logger.info("Using sample data for testing")
        
        # Option to use sample data from directories
        train_dir_input = input(f"Enter directory for training property sets (leave empty for random data) [default: {train_dir}]: ").strip() or train_dir
        test_dir_input = input(f"Enter directory for testing property sets (leave empty for random data) [default: {test_dir}]: ").strip() or test_dir
        
        if train_dir_input and os.path.isdir(train_dir_input):
            # Get all property subdirectories
            train_properties_dict = list_properties_in_directory(train_dir_input)
            
            if not train_properties_dict:
                logger.warning("No valid property sets found in the training directory.")
                train_properties_dict = create_dummy_property_data(train_properties)
        else:
            # Create dummy data for testing
            logger.warning("No valid training directory. Creating dummy property data for testing.")
            train_properties_dict = create_dummy_property_data(train_properties)
        
        if test_dir_input and os.path.isdir(test_dir_input):
            # Get all property subdirectories
            test_properties_dict = list_properties_in_directory(test_dir_input)
            
            if not test_properties_dict:
                logger.warning("No valid property sets found in the testing directory.")
                test_properties_dict = create_dummy_property_data(test_properties)
        else:
            # Create dummy data for testing
            logger.warning("No valid testing directory. Creating dummy property data for testing.")
            test_properties_dict = create_dummy_property_data(test_properties)
    
    # Final cleanup before returning results
    logger.info("Data gathering complete. Performing final memory cleanup.")
    clear_memory()
    
    return train_properties_dict, test_properties_dict

def create_dummy_property_data(num_properties: int = 10) -> Dict[str, List[str]]:
    """
    Create dummy property data for testing.
    
    Args:
        num_properties: Number of properties to create
        
    Returns:
        Dictionary mapping property UIDs to lists of dummy image paths
    """
    logger.warning("Creating dummy property data for testing purposes")
    
    # Create a dummy directory in the workspace
    dummy_dir = "/workspace/dummy_properties"
    os.makedirs(dummy_dir, exist_ok=True)
    
    dummy_properties = {}
    for i in range(num_properties):
        uid = f"property_{i+1:04d}"
        property_dir = os.path.join(dummy_dir, uid)
        os.makedirs(property_dir, exist_ok=True)
        
        # Create 3-10 dummy image paths per property
        num_images = random.randint(3, 10)
        dummy_images = []
        
        # Create empty image files to make it more realistic
        for j in range(num_images):
            image_path = os.path.join(property_dir, f"image_{j+1:03d}.jpg")
            
            # Create an empty file
            try:
                with open(image_path, 'w') as f:
                    f.write("dummy image data")
                dummy_images.append(image_path)
            except Exception as e:
                logger.warning(f"Failed to create dummy image: {str(e)}")
        
        dummy_properties[uid] = dummy_images
    
    logger.info(f"Created {num_properties} dummy properties with a total of {sum(len(imgs) for imgs in dummy_properties.values())} dummy images")
    return dummy_properties

def fetch_local_images(directory: str, max_images: int = None) -> Dict[str, List[str]]:
    """
    Fetch images from a local directory.
    
    Args:
        directory: Path to the directory containing images
        max_images: Maximum number of images to fetch (None for all)
        
    Returns:
        Dictionary mapping property types to lists of image paths
    """
    property_images = {}
    
    # Check if directory exists
    if not os.path.exists(directory):
        logger.error(f"Directory does not exist: {directory}")
        return {}
    
    # Collect subdirectories (property types)
    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)
        if os.path.isdir(item_path):
            images = []
            for img_file in os.listdir(item_path):
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(item_path, img_file)
                    images.append(img_path)
                    
                    # Stop if we reached max_images
                    if max_images and len(images) >= max_images:
                        break
            
            if images:
                property_images[item] = images
                logger.info(f"Found {len(images)} images for property type: {item}")
    
    return property_images

def fetch_from_premier_brokerage_api(
    api_url: str,
    auth_token: str,
    property_type: str,
    count: int,
    cache_dir: str,
    max_workers: int
) -> List[str]:
    """
    Fetch images from the Premier Brokerage API for a specific property type.
    
    Args:
        api_url: API endpoint URL
        auth_token: Authentication token
        property_type: Property type to fetch
        count: Number of images to fetch
        cache_dir: Directory to cache images
        max_workers: Maximum number of concurrent download threads
        
    Returns:
        List of paths to downloaded images
    """
    logger.info(f"Fetching {count} images for property type {property_type} from API")
    
    # Ensure cache directory exists
    os.makedirs(cache_dir, exist_ok=True)
    
    # Construct API request
    headers = {"Authorization": f"Bearer {auth_token}"}
    params = {
        "property_type": property_type,
        "count": count
    }
    
    # Fetch data from API
    try:
        response = requests.get(api_url, headers=headers, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        logger.error(f"Error fetching data from API: {str(e)}")
        return []
    
    # Extract image URLs
    property_images = {}
    properties_by_uid = organize_properties_by_uid(data)
    property_images = extract_property_image_urls(properties_by_uid)
    
    if not property_images:
        logger.warning(f"No image URLs found for property type {property_type}")
        return []
    
    # Download images
    image_paths = []
    for uid, urls in property_images.items():
        # Create directory for this property
        property_dir = os.path.join(cache_dir, uid)
        os.makedirs(property_dir, exist_ok=True)
        
        # Download images for this property
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for i, url in enumerate(urls):
                image_path = os.path.join(property_dir, f"{i}.jpg")
                futures.append(executor.submit(download_image, url, image_path))
            
            # Collect results
            for future in futures:
                try:
                    result = future.result()
                    if result:
                        image_paths.append(result)
                except Exception as e:
                    logger.error(f"Error in download worker: {str(e)}")
    
    logger.info(f"Downloaded {len(image_paths)} images for property type {property_type}")
    return image_paths

def download_image(url: str, save_path: str) -> Optional[str]:
    """Download an image from a URL and save it to disk."""
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        # Save image to disk
        with open(save_path, 'wb') as f:
            f.write(response.content)
        
        return save_path
    except Exception as e:
        logger.warning(f"Failed to download image from {url}: {str(e)}")
        return None

if __name__ == "__main__":
    print("\n=== Running Property Data Gathering ===\n")
    
    try:
        # Get user input for API parameters
        params = get_user_input_for_api_download()
        
        # Create cache directory if it doesn't exist
        os.makedirs(params.get("cache_dir", "/workspace/property_cache"), exist_ok=True)
        
        # Clear memory before starting the main process
        if 'clear_memory' in globals():
            clear_memory()
            logger.info("Cleared memory before starting data gathering")
        
        # Fetch training and testing data with memory optimization
        train_properties, test_properties = fetch_training_and_testing_data(
            api_url=params.get("api_url"),
            auth_token=params.get("auth_token"),
            train_properties=params["train_properties"],
            test_properties=params["test_properties"],
            cache_dir=params.get("cache_dir", "/workspace/property_cache"),
            max_workers=params["max_workers"],
            low_memory_mode=params.get("low_memory_mode", True)  # Use memory optimization by default
        )
        
        # Clear memory after fetching data and before saving
        if 'clear_memory' in globals():
            clear_memory()
            logger.info("Cleared memory after data fetching, before saving results")
        
        # Print summary of downloaded data
        print("\n=== Data Gathering Results ===")
        print(f"Training property sets: {len(train_properties)}")
        print(f"Testing property sets: {len(test_properties)}")
        
        # Save paths for preprocessing
        try:
            from runpods_utils import ensure_dir_exists
            ensure_dir_exists(params["output_dir"])
        except ImportError:
            # Fallback if runpods_utils is not available
            os.makedirs(params["output_dir"], exist_ok=True)
        
        # Save to JSON for later use
        data_paths = {
            "train_properties": train_properties,
            "test_properties": test_properties,
            "num_train_properties": len(train_properties),
            "num_test_properties": len(test_properties),
            "total_train_images": sum(len(imgs) for imgs in train_properties.values()),
            "total_test_images": sum(len(imgs) for imgs in test_properties.values())
        }
        
        data_paths_file = os.path.join(params["output_dir"], "property_data.json")
        with open(data_paths_file, 'w') as f:
            json.dump(data_paths, f, indent=2)
        
        print(f"\nProperty data saved to {data_paths_file}")
        
        # Check if we have property ratings
        use_ratings = input("\nDo you have expert property ratings? (y/n, default: n): ").strip().lower() == 'y'
        
        if use_ratings:
            ratings_path = input("Enter path to property ratings JSON file: ").strip()
            
            if os.path.exists(ratings_path):
                # Combine train and test properties for validation
                all_properties = {**train_properties, **test_properties}
                
                # Prepare property pairs dataset
                pairs_dir = os.path.join(params["output_dir"], "property_pairs")
                prepare_property_pairs_dataset(all_properties, ratings_path, pairs_dir)
                
                print(f"\nProperty pairs prepared in {pairs_dir}")
            else:
                print(f"Error: Ratings file {ratings_path} not found")
        
        # Final memory cleanup before exiting
        if 'clear_memory' in globals():
            clear_memory()
            logger.info("Final memory cleanup before exiting")
        
        print("\nProperty data gathering complete!")
        print("You can now proceed to processing property sets with expert ratings for similarity training.")
    
    except KeyboardInterrupt:
        print("\nProcess interrupted by user. Exiting...")
        # Clean up memory on exit
        if 'clear_memory' in globals():
            clear_memory()
    except Exception as e:
        print(f"\nAn error occurred during data gathering: {str(e)}")
        import traceback
        traceback.print_exc()
        # Clean up memory on error
        if 'clear_memory' in globals():
            clear_memory() 