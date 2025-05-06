#!/usr/bin/env python
"""
Test script for Siamese network model.
Tests the model on property-level (multiple images per property) 
and computes similarity scores with visualizations.
Enhanced for property-level similarity testing.
"""

import os
import sys
import json
import logging
import argparse
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from typing import Dict, List, Tuple, Optional, Any, Generator
import torch
import shutil
from tqdm import tqdm
import cv2
import random
from itertools import combinations
from collections import defaultdict
import gc

# Import local modules
from runpods_utils import ensure_dir_exists, load_json, save_json
from siamese_network import SiameseNetwork

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("testing.log")
    ]
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Test Siamese network model on property images")
    
    parser.add_argument("--model-dir", type=str, required=True,
                        help="Directory containing the trained model")
    parser.add_argument("--test-data-dir", type=str, required=True,
                        help="Directory containing test images or triplets")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Directory to save test results")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Batch size for testing")
    parser.add_argument("--embedding-dim", type=int, default=256,
                        help="Embedding dimension (must match trained model)")
    parser.add_argument("--num-pairs", type=int, default=100,
                        help="Number of image pairs to test")
    parser.add_argument("--model-type", type=str, default="final",
                        choices=["best", "final", "checkpoint", "siamese_embedding_model"],
                        help="Which model to use (best, final, checkpoint, or direct model filename)")
    parser.add_argument("--threshold", type=float, default=5.0,
                        help="Similarity threshold (0-10) for categorizing images")
    parser.add_argument("--save-visualizations", action="store_true",
                        help="Save visualizations of image pairs (can be memory intensive)")
    parser.add_argument("--max-workers", type=int, default=2,
                        help="Maximum number of worker processes for data loading")
    parser.add_argument("--property-level", action="store_true",
                        help="Test on property level (multiple images per property)")
    parser.add_argument("--images-per-property", type=int, default=10,
                        help="Maximum number of images to use per property for property-level testing")
    parser.add_argument("--backbone", type=str, default="efficientnet",
                        choices=["efficientnet", "resnet50"],
                        help="Backbone network architecture (must match trained model)")
    
    return parser.parse_args()

def clear_memory():
    """Force aggressive memory cleanup"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info("Performed memory cleanup")

def load_test_properties(test_data_dir: str) -> Dict[str, List[str]]:
    """
    Load test properties with their images from a directory.
    Assumes each subdirectory is a property.
    
    Args:
        test_data_dir: Directory containing test property folders
        
    Returns:
        Dictionary mapping property UIDs to lists of image paths
    """
    # Check if test_data_dir contains property_data.json
    property_data_file = os.path.join(test_data_dir, "property_data.json")
    if os.path.exists(property_data_file):
        logger.info(f"Loading test properties from property_data.json at {property_data_file}")
        with open(property_data_file, 'r') as f:
            data = json.load(f)
        
        if "test_properties" in data:
            logger.info(f"Found {len(data['test_properties'])} test properties in property_data.json")
            return data["test_properties"]
    
    # If no property_data.json, scan directories
    logger.info(f"Scanning directories in {test_data_dir} for test properties")
    
    # Look for test_properties directory
    test_props_dir = os.path.join(test_data_dir, "test_properties")
    if os.path.exists(test_props_dir) and os.path.isdir(test_props_dir):
        logger.info(f"Found test_properties directory at {test_props_dir}")
        test_data_dir = test_props_dir
    
    # Assume each subdirectory is a property
    properties = {}
    for property_uid in os.listdir(test_data_dir):
        property_dir = os.path.join(test_data_dir, property_uid)
        if os.path.isdir(property_dir):
            # Get all image files in this property directory
            image_paths = []
            for filename in os.listdir(property_dir):
                if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                    image_paths.append(os.path.join(property_dir, filename))
            
            if image_paths:
                properties[property_uid] = image_paths
                logger.debug(f"Found property {property_uid} with {len(image_paths)} images")
    
    logger.info(f"Found {len(properties)} properties with {sum(len(imgs) for imgs in properties.values())} total images")
    return properties

def load_and_preprocess_image(image_path: str, target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
    """
    Load and preprocess an image for model input.
    
    Args:
        image_path: Path to the image file
        target_size: Target size for resizing
        
    Returns:
        Preprocessed image as numpy array
    """
    try:
        # Load image (using PIL instead of cv2 to reduce memory usage)
        with Image.open(image_path) as img:
            img = img.convert('RGB')
            img = img.resize(target_size)
            img_array = np.array(img).astype(np.float32) / 255.0
        
        # Convert to model input format (NCHW)
        img_array = np.transpose(img_array, (2, 0, 1))
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
    except Exception as e:
        logger.error(f"Error loading image {image_path}: {str(e)}")
        # Return a small random array as a fallback
        return np.random.rand(1, 3, 224, 224).astype(np.float32)

def generate_property_test_pairs(properties: Dict[str, List[str]], num_pairs: int = 100) -> List[Tuple[str, str, bool]]:
    """
    Generate pairs of properties for testing, with balanced similar/dissimilar pairs.
    For property-level testing, we return the property UIDs instead of specific image paths.
    
    Args:
        properties: Dictionary mapping property UIDs to lists of image paths
        num_pairs: Number of pairs to generate
        
    Returns:
        List of tuples (property1_uid, property2_uid, is_similar)
    """
    # Ensure we have enough properties
    if len(properties) < 2:
        raise ValueError(f"Not enough test properties: {len(properties)} (need at least 2)")
    
    # Adjust num_pairs if needed
    max_possible_pairs = len(properties) * (len(properties) - 1) // 2
    if num_pairs > max_possible_pairs:
        logger.warning(f"Requested {num_pairs} pairs, but only {max_possible_pairs} are possible. Adjusting.")
        num_pairs = max_possible_pairs
    
    # Group properties by type (assume type is first part of UID before underscore)
    properties_by_type = defaultdict(list)
    for uid in properties.keys():
        prop_type = uid.split('_')[0] if '_' in uid else 'unknown'
        properties_by_type[prop_type].append(uid)
    
    # If we have distinguishable property types
    has_multiple_types = len(properties_by_type) > 1
    
    # Generate similar and dissimilar pairs
    similar_pairs = []
    dissimilar_pairs = []
    
    # Count how many similar pairs we want
    similar_pairs_target = min(num_pairs // 2, sum(len(props) * (len(props) - 1) // 2 for props in properties_by_type.values()))
    
    if has_multiple_types:
        # Generate similar pairs (same property type)
        logger.info("Generating similar pairs within property types...")
        
        # Try to distribute similar pairs evenly across property types
        similar_per_type = max(1, similar_pairs_target // len(properties_by_type))
        
        for prop_type, props in properties_by_type.items():
            if len(props) < 2:
                continue  # Skip property types with only one property
                
            # Generate pairs for this property type
            type_pairs = list(combinations(props, 2))
            
            # Randomly select up to similar_per_type pairs
            num_to_select = min(similar_per_type, len(type_pairs))
            selected_pairs = random.sample(type_pairs, num_to_select)
            
            for prop1, prop2 in selected_pairs:
                similar_pairs.append((prop1, prop2, True))
                
            if len(similar_pairs) >= similar_pairs_target:
                break
        
        # Generate dissimilar pairs (different property types)
        logger.info("Generating dissimilar pairs across different property types...")
        dissimilar_pairs_target = num_pairs - len(similar_pairs)
        
        prop_types = list(properties_by_type.keys())
        pairs_added = 0
        
        # Try to get balanced pairs
        while pairs_added < dissimilar_pairs_target and len(prop_types) >= 2:
            # Select two different property types
            type1, type2 = random.sample(prop_types, 2)
            
            # Select a random property from each type
            if properties_by_type[type1] and properties_by_type[type2]:
                prop1 = random.choice(properties_by_type[type1])
                prop2 = random.choice(properties_by_type[type2])
                
                dissimilar_pairs.append((prop1, prop2, False))
                pairs_added += 1
    else:
        logger.warning("Could not identify distinct property types. Using random pairs.")
        # Fallback to random pairs without type information
        all_properties = list(properties.keys())
        
        # Generate all possible pairs
        all_pairs = list(combinations(all_properties, 2))
        random.shuffle(all_pairs)
        
        # Split randomly into similar and dissimilar
        similar_count = min(num_pairs // 2, len(all_pairs))
        similar_pairs = [(p1, p2, True) for p1, p2 in all_pairs[:similar_count]]
        
        dissimilar_count = min(num_pairs - similar_count, len(all_pairs) - similar_count)
        if dissimilar_count > 0:
            dissimilar_pairs = [(p1, p2, False) for p1, p2 in all_pairs[similar_count:similar_count + dissimilar_count]]
    
    # Combine and shuffle all pairs
    all_pairs = similar_pairs + dissimilar_pairs
    random.shuffle(all_pairs)
    
    logger.info(f"Generated {len(all_pairs)} test pairs: {len(similar_pairs)} similar, {len(dissimilar_pairs)} dissimilar")
    return all_pairs

def test_property_similarity(
    model: SiameseNetwork, 
    property_pairs: List[Tuple[str, str, bool]],
    properties: Dict[str, List[str]],
    output_dir: str, 
    batch_size: int,
    images_per_property: int,
    threshold: float = 5.0,
    save_visualizations: bool = False
) -> Dict:
    """
    Test the model on pairs of properties (each with multiple images).
    
    Args:
        model: Trained Siamese network
        property_pairs: List of tuples (property1_uid, property2_uid, is_similar)
        properties: Dictionary mapping property UIDs to lists of image paths
        output_dir: Directory to save results
        batch_size: Batch size for processing
        images_per_property: Maximum number of images to use per property
        threshold: Similarity threshold (0-10) for categorizing as similar/dissimilar
        save_visualizations: Whether to save visualizations of property pairs
        
    Returns:
        Dictionary with test results
    """
    # Create output directories
    results_dir = os.path.join(output_dir, "test_results")
    similar_dir = os.path.join(results_dir, "similar_pairs")
    dissimilar_dir = os.path.join(results_dir, "dissimilar_pairs")
    
    ensure_dir_exists(results_dir)
    if save_visualizations:
        ensure_dir_exists(similar_dir)
        ensure_dir_exists(dissimilar_dir)
    
    # Track results
    all_scores = []
    true_labels = []
    predicted_labels = []
    all_results = []
    
    # Set model to evaluation mode
    model.eval()
    
    # Process in batches
    logger.info(f"Testing {len(property_pairs)} property pairs...")
    
    with torch.no_grad():
        for batch_idx in range(0, len(property_pairs), batch_size):
            batch_end = min(batch_idx + batch_size, len(property_pairs))
            batch = property_pairs[batch_idx:batch_end]
            
            logger.info(f"Processing batch {batch_idx//batch_size + 1}/{(len(property_pairs) + batch_size - 1)//batch_size}")
            
            batch_results = []
            
            for i, (prop1_uid, prop2_uid, is_similar) in enumerate(batch):
                try:
                    # Get property image paths
                    prop1_images = properties.get(prop1_uid, [])
                    prop2_images = properties.get(prop2_uid, [])
                    
                    if not prop1_images or not prop2_images:
                        logger.warning(f"Missing images for property {prop1_uid} or {prop2_uid}")
                        continue
                    
                    # Limit number of images if needed
                    if len(prop1_images) > images_per_property:
                        prop1_images = random.sample(prop1_images, images_per_property)
                    
                    if len(prop2_images) > images_per_property:
                        prop2_images = random.sample(prop2_images, images_per_property)
                    
                    # Load and preprocess images for both properties
                    prop1_processed = [load_and_preprocess_image(img_path) for img_path in prop1_images]
                    prop2_processed = [load_and_preprocess_image(img_path) for img_path in prop2_images]
                    
                    # Compute similarity between properties
                    similarity_score = model.compute_property_similarity(prop1_processed, prop2_processed)
                    
                    # Determine predicted label
                    predicted_similar = similarity_score >= threshold
                    
                    # Add to results
                    pair_result = {
                        "pair_id": batch_idx + i,
                        "property1_uid": prop1_uid,
                        "property2_uid": prop2_uid,
                        "property1_images": len(prop1_images),
                        "property2_images": len(prop2_images),
                        "true_label": "similar" if is_similar else "dissimilar",
                        "predicted_label": "similar" if predicted_similar else "dissimilar",
                        "similarity_score": float(similarity_score),
                        "correct_prediction": (is_similar == predicted_similar)
                    }
                    batch_results.append(pair_result)
                    
                    # Add to overall tracking
                    all_scores.append(similarity_score)
                    true_labels.append(1 if is_similar else 0)
                    predicted_labels.append(1 if predicted_similar else 0)
                    
                    # Save visualization if requested
                    if save_visualizations:
                        # Determine output directory based on correctness
                        vis_dir = similar_dir if predicted_similar else dissimilar_dir
                        save_property_visualization(
                            prop1_images[:min(3, len(prop1_images))],  # Use at most 3 images per property for visualization
                            prop2_images[:min(3, len(prop2_images))],
                            similarity_score,
                            is_similar,
                            predicted_similar,
                            vis_dir,
                            f"pair_{batch_idx + i:04d}"
                        )
                except Exception as e:
                    logger.error(f"Error processing property pair ({prop1_uid}, {prop2_uid}): {str(e)}")
                    continue
            
            # Add batch results
            all_results.extend(batch_results)
            
            # Clear memory after each batch
            clear_memory()
            
            # Save intermediate results
            if (batch_idx // batch_size) % 5 == 0 or batch_end == len(property_pairs):
                if true_labels:
                    # Compute metrics
                    metrics = compute_metrics(true_labels, predicted_labels, all_scores)
                    
                    # Save intermediate results
                    intermediate_path = os.path.join(output_dir, f"intermediate_results_batch_{batch_idx//batch_size + 1}.json")
                    save_json({
                        "metrics": metrics,
                        "pairs_processed": len(all_results),
                        "threshold": threshold
                    }, intermediate_path)
    
    # Compute final metrics
    metrics = compute_metrics(true_labels, predicted_labels, all_scores)
    
    # Save detailed results
    results_path = os.path.join(output_dir, "property_test_results.json")
    full_results = {
        "metrics": metrics,
        "property_pairs": all_results,
        "threshold": threshold
    }
    
    save_json(full_results, results_path)
    logger.info(f"Saved detailed results to {results_path}")
    
    # Generate and save similarity histogram
    create_similarity_histogram(all_scores, true_labels, threshold, output_dir)
    
    return metrics

def save_property_visualization(prop1_images: List[str], prop2_images: List[str], 
                              similarity_score: float, is_similar: bool, 
                              predicted_similar: bool, output_dir: str, filename_prefix: str) -> None:
    """
    Create and save a visualization of a property pair with results.
    Shows multiple images for each property.
    
    Args:
        prop1_images: List of image paths for property 1
        prop2_images: List of image paths for property 2
        similarity_score: Computed similarity score
        is_similar: True if properties are actually similar
        predicted_similar: True if model predicted similarity
        output_dir: Directory to save visualization
        filename_prefix: Prefix for output filename
    """
    try:
        # Create output directory
        ensure_dir_exists(output_dir)
        
        # Determine how many images to show per property (max 3)
        num_images1 = min(3, len(prop1_images))
        num_images2 = min(3, len(prop2_images))
        
        # Create a figure with 2 rows, one for each property
        fig, axes = plt.subplots(2, max(num_images1, num_images2), figsize=(15, 8))
        
        # Load and plot property 1 images
        for i in range(num_images1):
            with Image.open(prop1_images[i]) as img:
                img = img.convert('RGB')
                if num_images1 == 1:
                    ax = axes[0]
                else:
                    ax = axes[0, i]
                ax.imshow(img)
                ax.axis('off')
                ax.set_title(f"Property 1 - Image {i+1}")
        
        # Hide unused axes in first row
        for i in range(num_images1, max(num_images1, num_images2)):
            if num_images1 == 1:
                if i == 0:
                    continue
                axes[0].axis('off')
            else:
                axes[0, i].axis('off')
        
        # Load and plot property 2 images
        for i in range(num_images2):
            with Image.open(prop2_images[i]) as img:
                img = img.convert('RGB')
                if num_images2 == 1:
                    ax = axes[1]
                else:
                    ax = axes[1, i]
                ax.imshow(img)
                ax.axis('off')
                ax.set_title(f"Property 2 - Image {i+1}")
        
        # Hide unused axes in second row
        for i in range(num_images2, max(num_images1, num_images2)):
            if num_images2 == 1:
                if i == 0:
                    continue
                axes[1].axis('off')
            else:
                axes[1, i].axis('off')
        
        # Add results text as a figure title
        result_text = (
            f"Similarity Score: {similarity_score:.2f}/10 | "
            f"Actual: {'Similar' if is_similar else 'Dissimilar'} | "
            f"Predicted: {'Similar' if predicted_similar else 'Dissimilar'} | "
            f"{'✓ Correct' if is_similar == predicted_similar else '✗ Incorrect'}"
        )
        
        # Set color based on correctness
        text_color = 'green' if is_similar == predicted_similar else 'red'
        
        # Add as figure title
        plt.suptitle(result_text, fontsize=14, color=text_color)
        
        # Adjust layout
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Make room for suptitle
        
        # Save figure
        output_path = os.path.join(output_dir, f"{filename_prefix}_score_{similarity_score:.2f}.png")
        plt.savefig(output_path, dpi=100)
        plt.close(fig)
        
    except Exception as e:
        logger.error(f"Error creating property visualization: {str(e)}")

def compute_metrics(true_labels: List[int], predicted_labels: List[int], scores: List[float]) -> Dict:
    """
    Compute evaluation metrics from test results.
    
    Args:
        true_labels: Ground truth labels (1 for similar, 0 for dissimilar)
        predicted_labels: Predicted labels (1 for similar, 0 for dissimilar)
        scores: Similarity scores
        
    Returns:
        Dictionary with computed metrics
    """
    if not true_labels or len(true_labels) != len(predicted_labels):
        logger.warning("Cannot compute metrics: invalid or mismatched labels")
        return {}
    
    # Calculate confusion matrix values
    tp = sum(1 for t, p in zip(true_labels, predicted_labels) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(true_labels, predicted_labels) if t == 0 and p == 1)
    tn = sum(1 for t, p in zip(true_labels, predicted_labels) if t == 0 and p == 0)
    fn = sum(1 for t, p in zip(true_labels, predicted_labels) if t == 1 and p == 0)
    
    # Compute metrics
    accuracy = (tp + tn) / len(true_labels) if true_labels else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    # Separate scores by true label
    similar_scores = [score for score, label in zip(scores, true_labels) if label == 1]
    dissimilar_scores = [score for score, label in zip(scores, true_labels) if label == 0]
    
    # Calculate average scores
    avg_similar_score = sum(similar_scores) / len(similar_scores) if similar_scores else 0
    avg_dissimilar_score = sum(dissimilar_scores) / len(dissimilar_scores) if dissimilar_scores else 0
    
    # Return metrics
    return {
        "num_test_pairs": len(true_labels),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "avg_similar_score": avg_similar_score,
        "avg_dissimilar_score": avg_dissimilar_score
    }

def create_similarity_histogram(scores: List[float], true_labels: List[int], threshold: float, output_dir: str) -> None:
    """
    Create and save a histogram of similarity scores.
    
    Args:
        scores: List of similarity scores
        true_labels: Ground truth labels (1 for similar, 0 for dissimilar)
        threshold: Similarity threshold
        output_dir: Directory to save the histogram
    """
    try:
        # Create lists of scores for similar and dissimilar pairs
        similar_scores = [score for score, label in zip(scores, true_labels) if label == 1]
        dissimilar_scores = [score for score, label in zip(scores, true_labels) if label == 0]
        
        # Create figure
        plt.figure(figsize=(10, 6))
        
        # Plot histograms
        bins = np.linspace(0, 10, 40)  # 0-10 range with 40 bins
        
        if similar_scores:
            plt.hist(similar_scores, bins=bins, alpha=0.5, label='Similar', color='blue')
        
        if dissimilar_scores:
            plt.hist(dissimilar_scores, bins=bins, alpha=0.5, label='Dissimilar', color='red')
        
        # Plot threshold line
        plt.axvline(x=threshold, color='black', linestyle='--', 
                    label=f'Threshold ({threshold:.1f})')
        
        # Add labels and title
        plt.xlabel('Similarity Score')
        plt.ylabel('Frequency')
        plt.title('Distribution of Similarity Scores')
        plt.legend()
        plt.grid(alpha=0.3)
        
        # Add statistics as text
        if similar_scores:
            plt.figtext(0.15, 0.80, f"Similar pairs: {len(similar_scores)}\n"
                                   f"Mean: {np.mean(similar_scores):.2f}\n"
                                   f"Std: {np.std(similar_scores):.2f}", 
                       bbox=dict(facecolor='blue', alpha=0.1))
        
        if dissimilar_scores:
            plt.figtext(0.15, 0.65, f"Dissimilar pairs: {len(dissimilar_scores)}\n"
                                   f"Mean: {np.mean(dissimilar_scores):.2f}\n"
                                   f"Std: {np.std(dissimilar_scores):.2f}", 
                       bbox=dict(facecolor='red', alpha=0.1))
        
        # Save figure
        output_path = os.path.join(output_dir, "similarity_histogram.png")
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()
        
        logger.info(f"Saved similarity histogram to {output_path}")
    except Exception as e:
        logger.error(f"Error creating similarity histogram: {str(e)}")

def main():
    """Main function to run the testing."""
    args = parse_args()
    
    # Create output directory
    ensure_dir_exists(args.output_dir)
    
    # Load the trained model
    model_version = args.model_type
    
    # Check if the model_type is a specific file
    if model_version == "siamese_embedding_model":
        model_path = os.path.join(args.model_dir, "siamese_embedding_model.pt")
    else:
        model_path = os.path.join(args.model_dir, f"{model_version}_model")
    
    logger.info(f"Loading model from {model_path}")
    model = SiameseNetwork(embedding_dim=args.embedding_dim, backbone=args.backbone)
    
    try:
        model.load_model(model_path)
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        # Try alternate model path if standard one fails
        if model_version == "final" or model_version == "best":
            alt_path = os.path.join(args.model_dir, "siamese_embedding_model.pt")
            try:
                logger.info(f"Trying alternate path: {alt_path}")
                model.load_model(alt_path)
                logger.info("Model loaded successfully from alternate path")
            except Exception as e2:
                logger.error(f"Error loading model from alternate path: {str(e2)}")
                sys.exit(1)
        else:
            sys.exit(1)
    
    # Set model to evaluation mode
    model.eval()
    
    # Load test data
    if args.property_level:
        # Property-level testing
        logger.info("Running property-level testing (multiple images per property)")
        
        # Load test properties
        test_properties = load_test_properties(args.test_data_dir)
        
        if not test_properties:
            logger.error(f"No test properties found in {args.test_data_dir}")
            sys.exit(1)
        
        logger.info(f"Loaded {len(test_properties)} test properties with {sum(len(imgs) for imgs in test_properties.values())} total images")
        
        # Generate property pairs for testing
        property_pairs = generate_property_test_pairs(test_properties, args.num_pairs)
        
        # Run property-level testing
        metrics = test_property_similarity(
            model=model,
            property_pairs=property_pairs,
            properties=test_properties,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            images_per_property=args.images_per_property,
            threshold=args.threshold,
            save_visualizations=args.save_visualizations
        )
    else:
        # Original image-pair testing (loads test_images same as before)
        logger.info("Running standard image-level testing")
        
        from train import generate_test_pairs, load_test_images, test_model
        
        # Load test images
        test_images = load_test_images(args.test_data_dir)
        
        # Generate test pairs
        test_pairs = generate_test_pairs(test_images, args.num_pairs)
        
        # Test the model with memory-efficient processing
        metrics = test_model(
            model, 
            test_pairs, 
            args.output_dir, 
            args.batch_size, 
            args.threshold,
            args.save_visualizations
        )
    
    # Print results
    logger.info("====== Test Results ======")
    logger.info(f"Number of test pairs: {metrics.get('num_test_pairs', 0)}")
    logger.info(f"Accuracy: {metrics.get('accuracy', 0):.4f}")
    logger.info(f"Precision: {metrics.get('precision', 0):.4f}")
    logger.info(f"Recall: {metrics.get('recall', 0):.4f}")
    logger.info(f"F1 Score: {metrics.get('f1_score', 0):.4f}")
    logger.info(f"Average score for similar pairs: {metrics.get('avg_similar_score', 0):.2f}")
    logger.info(f"Average score for dissimilar pairs: {metrics.get('avg_dissimilar_score', 0):.2f}")
    logger.info("==========================")
    
    # Create summary README
    readme_path = os.path.join(args.output_dir, "README.md")
    with open(readme_path, 'w') as f:
        f.write(f"# Testing Results\n\n")
        f.write(f"## Configuration\n\n")
        f.write(f"- Model directory: {args.model_dir}\n")
        f.write(f"- Model type: {args.model_type}\n")
        f.write(f"- Test data: {args.test_data_dir}\n")
        f.write(f"- Property-level testing: {'Yes' if args.property_level else 'No'}\n")
        f.write(f"- Similarity threshold: {args.threshold}\n\n")
        
        f.write(f"## Results\n\n")
        f.write(f"- Number of test pairs: {metrics.get('num_test_pairs', 0)}\n")
        f.write(f"- Accuracy: {metrics.get('accuracy', 0):.4f}\n")
        f.write(f"- Precision: {metrics.get('precision', 0):.4f}\n")
        f.write(f"- Recall: {metrics.get('recall', 0):.4f}\n")
        f.write(f"- F1 Score: {metrics.get('f1_score', 0):.4f}\n")
        f.write(f"- Average score for similar pairs: {metrics.get('avg_similar_score', 0):.2f}\n")
        f.write(f"- Average score for dissimilar pairs: {metrics.get('avg_dissimilar_score', 0):.2f}\n\n")
        
        f.write(f"See `property_test_results.json` for detailed results.\n")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 