"""
Services for property comparison model inference
"""

import os
import sys
import logging
import torch
import numpy as np
import re
from typing import List, Dict, Tuple, Any, Optional
import aiohttp
import asyncio
from pathlib import Path

# Add parent directory to path to import model modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import model and utility functions
from siamese_network import SiameseNetwork
from api.utils import (
    download_property_images,
    preprocess_images,
    compute_accuracy_metrics,
    download_blob_to_temp,
    logger
)

# Default model paths - support both /app (Docker) and /workspace (RunPods) paths
if os.path.exists('/app/final_model'):
    DEFAULT_MODEL_DIR = '/app/final_model/'
elif os.path.exists('/workspace/final_model'):
    DEFAULT_MODEL_DIR = '/workspace/final_model/'
else:
    DEFAULT_MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'final_model/')

MODEL_CHECKPOINT = "siamese_embedding_model.pt"


class ModelService:
    """Service for property comparison model inference"""
    
    def __init__(self, model_dir: str = None, model_gcs_path: str = None):
        """
        Initialize the model service
        
        Args:
            model_dir: Directory containing model checkpoint
            model_gcs_path: Google Cloud Storage path to model (format: gs://bucket-name/path/to/model.pth.tar)
        """
        # Use provided model_dir if given, otherwise use default
        self.model_dir = model_dir or DEFAULT_MODEL_DIR
        
        # Log the model directory we're using
        logger.info(f"Using model directory: {self.model_dir}")
        
        self.model_gcs_path = model_gcs_path
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.temp_files = []  # Track temporary files to clean up
        
        # Limit images per property for inference
        self.max_images_per_property = 10
        
        # Load the model
        # We need to run the async load method in the event loop
        if asyncio.get_event_loop().is_running():
            # If we're already in an event loop, create a task
            asyncio.create_task(self._async_load_model())
        else:
            # Otherwise, run the load method to completion
            asyncio.run(self._async_load_model())
        
    async def _async_load_model(self):
        """Load the SiameseNetwork model from local file system or GCS"""
        try:
            model_path = None
            
            # Check if model needs to be loaded from GCS
            if self.model_gcs_path:
                if self.model_gcs_path.startswith("gs://"):
                    # Parse bucket and blob path
                    gcs_regex = r"gs://([^/]+)/(.+)"
                    match = re.match(gcs_regex, self.model_gcs_path)
                    
                    if match:
                        bucket_name = match.group(1)
                        blob_path = match.group(2)
                        
                        logger.info(f"Attempting to download model from GCS: {self.model_gcs_path}")
                        # Download model to a temporary file
                        temp_path = await download_blob_to_temp(bucket_name, blob_path)
                        
                        if temp_path:
                            model_path = temp_path
                            self.temp_files.append(temp_path)  # Track for cleanup
                    else:
                        logger.error(f"Invalid GCS path format: {self.model_gcs_path}")
                else:
                    logger.error(f"Model GCS path must start with gs://: {self.model_gcs_path}")
            
            # Fall back to local file if GCS download failed or wasn't requested
            if not model_path:
                model_path = os.path.join(self.model_dir, MODEL_CHECKPOINT)
                
                # Check if the file exists
                if not os.path.exists(model_path):
                    # Try some alternative locations
                    alt_locations = [
                        os.path.join('/app/final_model', MODEL_CHECKPOINT),
                        os.path.join('/workspace/final_model', MODEL_CHECKPOINT),
                        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'final_model', MODEL_CHECKPOINT)
                    ]
                    
                    for alt_path in alt_locations:
                        if os.path.exists(alt_path):
                            model_path = alt_path
                            logger.info(f"Found model at alternative location: {model_path}")
                            break
            
            # Check if model exists
            if not os.path.exists(model_path):
                logger.error(f"Model checkpoint not found at: {model_path}")
                raise FileNotFoundError(f"Model checkpoint not found at: {model_path}")
            
            # Initialize model
            self.model = SiameseNetwork(embedding_dim=256, backbone="resnet50")
            # Load model state and configuration (handles both directory and single-file paths)
            self.model.load_model(model_path)
            # Move embedding and aggregator modules to device
            self.model.embedding_model.to(self.device)
            self.model.property_aggregator.to(self.device)
            # Set modules to evaluation mode
            self.model.embedding_model.eval()
            self.model.property_aggregator.eval()
            
            logger.info(f"Model loaded successfully from {model_path} to {self.device}")
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    async def compare_properties(
        self,
        subject_photos: List[str],
        comp_properties: List[Dict[str, Any]],
        threshold: float = 5.0,
        max_comps: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Compare subject property against multiple comp properties
        
        Args:
            subject_photos: List of subject property photo URLs
            comp_properties: List of comparable properties with UIDs and photo URLs
            threshold: Similarity threshold (0-10)
            max_comps: Maximum number of comp properties to process
            
        Returns:
            Dictionary with comparison results
        """
        # Ensure model is loaded
        if self.model is None:
            logger.error("Model not loaded yet")
            return self._create_error_response("Model not loaded yet")
            
        # Limit number of comp properties if specified
        if max_comps and max_comps < len(comp_properties):
            comp_properties = comp_properties[:max_comps]
        
        # Create aiohttp session for image downloads
        async with aiohttp.ClientSession() as session:
            # Download subject property images
            subject_images = await download_property_images(
                session, subject_photos, self.max_images_per_property
            )
            
            if not subject_images:
                logger.error("Failed to download any subject property images")
                return self._create_error_response("Failed to download subject property images")
            
            # Process subject images
            subject_tensor = preprocess_images(subject_images)
            
            # Process each comp property
            comp_results = []
            true_labels = []
            predicted_labels = []
            similarity_scores = []
            similar_scores = []
            dissimilar_scores = []
            
            for idx, comp in enumerate(comp_properties):
                comp_uid = comp.get('uid', f'comp_{idx}')
                comp_photos = [photo.get('url') for photo in comp.get('photos', [])]
                comp_address = comp.get('address', '')
                
                # Download comp property images
                comp_images = await download_property_images(
                    session, comp_photos, self.max_images_per_property
                )
                
                if not comp_images:
                    logger.warning(f"Failed to download any images for comp property {comp_uid}")
                    continue
                
                # Process comp images
                comp_tensor = preprocess_images(comp_images)
                
                # Set all as "similar" for demonstration
                # In a real implementation, this should be based on property type or other criteria
                is_similar = True
                
                # Run model inference
                with torch.no_grad():
                    similarity_score = self._calculate_similarity(subject_tensor, comp_tensor)
                
                # Apply threshold to determine predicted label
                predicted_similar = similarity_score > threshold
                
                # Store results for metrics calculation
                true_labels.append(is_similar)
                predicted_labels.append(predicted_similar)
                similarity_scores.append(similarity_score)
                
                # Track scores by similarity
                if is_similar:
                    similar_scores.append(similarity_score)
                else:
                    dissimilar_scores.append(similarity_score)
                
                # Add to results
                comp_results.append({
                    "pair_id": idx,
                    "subject_property_id": "subject",
                    "comp_property_id": comp_uid,
                    "subject_images": len(subject_images),
                    "comp_images": len(comp_images),
                    "true_label": "similar" if is_similar else "dissimilar",
                    "predicted_label": "similar" if predicted_similar else "dissimilar",
                    "similarity_score": float(similarity_score),
                    "correct_prediction": (is_similar == predicted_similar),
                    "address": comp_address
                })
            
            # Calculate metrics
            metrics = compute_accuracy_metrics(true_labels, predicted_labels)
            
            # Add average scores
            metrics["avg_similar_score"] = float(np.mean(similar_scores)) if similar_scores else 0
            metrics["avg_dissimilar_score"] = float(np.mean(dissimilar_scores)) if dissimilar_scores else 0
            
            # Create final result
            result = {
                "subject_property_id": "subject",
                "metrics": metrics,
                "comp_pairs": comp_results,
                "threshold": threshold
            }
            
            return result
    
    def _calculate_similarity(self, subject_tensor: torch.Tensor, comp_tensor: torch.Tensor) -> float:
        """
        Calculate similarity score between subject and comp properties
        
        Args:
            subject_tensor: Preprocessed subject property images
            comp_tensor: Preprocessed comp property images
            
        Returns:
            Similarity score (0-10)
        """
        if subject_tensor.size(0) == 0 or comp_tensor.size(0) == 0:
            logger.warning("Empty tensor detected in similarity calculation")
            return 0.0
        
        try:
            # Move tensors to device
            subject_tensor = subject_tensor.to(self.device)
            comp_tensor = comp_tensor.to(self.device)
            
            # Get embeddings using the embedding_model
            subject_embedding = self.model.embedding_model(subject_tensor)
            comp_embedding = self.model.embedding_model(comp_tensor)
            
            # Calculate average embedding for each property
            subject_avg = torch.mean(subject_embedding, dim=0, keepdim=True)
            comp_avg = torch.mean(comp_embedding, dim=0, keepdim=True)
            
            # Calculate distance (Euclidean)
            distance = torch.nn.functional.pairwise_distance(subject_avg, comp_avg).item()
            
            # Convert distance to similarity score (0-10)
            # Lower distance means higher similarity
            similarity = 10.0 * (1.0 - min(distance, 1.0))
            
            return similarity
            
        except Exception as e:
            logger.error(f"Error calculating similarity: {str(e)}")
            return 0.0
    
    def _create_error_response(self, error_message: str) -> Dict[str, Any]:
        """Create an error response"""
        return {
            "subject_property_id": "subject",
            "error": error_message,
            "metrics": {
                "num_test_pairs": 0,
                "accuracy": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "f1_score": 0.0,
                "tp": 0,
                "fp": 0,
                "tn": 0,
                "fn": 0,
                "avg_similar_score": 0.0,
                "avg_dissimilar_score": 0.0
            },
            "comp_pairs": [],
            "threshold": 5.0
        }
        
    def __del__(self):
        """Clean up temporary files when service is destroyed"""
        for temp_file in self.temp_files:
            try:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)
                    logger.info(f"Deleted temporary file: {temp_file}")
            except Exception as e:
                logger.error(f"Error deleting temporary file {temp_file}: {str(e)}") 