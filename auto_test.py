#!/usr/bin/env python
"""
Auto Testing script for Siamese network model.
Compares a subject property against a specified number of comp properties.
"""

import os
import sys
import json
import logging
import argparse
import numpy as np
import matplotlib.pyplot as plt
import random
from PIL import Image
from typing import Dict, List, Tuple, Optional, Any
import torch
import shutil
from tqdm import tqdm
import gc
import pandas as pd

# Import local modules
from runpods_utils import ensure_dir_exists, load_json, save_json
from siamese_network import SiameseNetwork

# Import functions from test.py 
from test import (load_test_properties, load_and_preprocess_image, 
                 compute_metrics, create_similarity_histogram, 
                 save_property_visualization, clear_memory)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("auto_testing.log")
    ]
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Auto-test Siamese network model on property images")
    
    parser.add_argument("--model-dir", type=str, required=True,
                        help="Directory containing the trained model")
    parser.add_argument("--test-data-dir", type=str, required=True,
                        help="Directory containing test images or properties")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Directory to save test results")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Batch size for testing")
    parser.add_argument("--embedding-dim", type=int, default=256,
                        help="Embedding dimension (must match trained model)")
    parser.add_argument("--model-type", type=str, default="final",
                        choices=["best", "final", "checkpoint", "siamese_embedding_model"],
                        help="Which model to use (best, final, checkpoint, or direct model filename)")
    parser.add_argument("--threshold", type=float, default=5.0,
                        help="Similarity threshold (0-10) for categorizing images")
    parser.add_argument("--max-preview-images", type=int, default=3,
                        help="Maximum number of images to show in each visualization preview")
    parser.add_argument("--backbone", type=str, default="efficientnet",
                        choices=["efficientnet", "resnet50"],
                        help="Backbone network architecture (must match trained model)")
    parser.add_argument("--num-comps", type=int, default=None,
                        help="Number of comp properties to compare subject property against (default: ask user)")
    parser.add_argument("--subject-property-id", type=str, default=None,
                        help="Specific subject property ID to test (if not provided, will prompt for input)")
    
    return parser.parse_args()

def generate_subject_vs_comps_pairs(properties, subject_property_id, num_comps):
    """
    Generate pairs where a specific subject property is compared against random comp properties.
    
    Args:
        properties: Dictionary of all properties
        subject_property_id: ID of the subject property to compare against others
        num_comps: Number of comp properties to compare against
    
    Returns:
        List of tuples (subject_property_id, comp_property_id, is_similar)
    """
    if subject_property_id not in properties:
        raise ValueError(f"Subject property {subject_property_id} not found")
        
    # Get all other property IDs
    comp_ids = [pid for pid in properties.keys() if pid != subject_property_id]
    
    # Limit number of comps if specified
    if num_comps and num_comps < len(comp_ids):
        comp_ids = random.sample(comp_ids, num_comps)
    
    # Determine if properties are similar (based on property type)
    subject_type = subject_property_id.split('_')[0] if '_' in subject_property_id else 'unknown'
    
    # Generate pairs
    pairs = []
    for comp_id in comp_ids:
        comp_type = comp_id.split('_')[0] if '_' in comp_id else 'unknown'
        is_similar = (comp_type == subject_type)
        pairs.append((subject_property_id, comp_id, is_similar))
    
    return pairs

def test_subject_against_comps(
    model: SiameseNetwork, 
    subject_property_id: str,
    property_pairs: List[Tuple[str, str, bool]],
    properties: Dict[str, List[str]],
    output_dir: str, 
    batch_size: int,
    threshold: float = 5.0,
    max_preview_images: int = 3
) -> Dict:
    """
    Test a subject property against multiple comp properties.
    
    Args:
        model: Trained Siamese network
        subject_property_id: ID of the subject property
        property_pairs: List of tuples (subject_property_id, comp_property_id, is_similar)
        properties: Dictionary mapping property UIDs to lists of image paths
        output_dir: Directory to save results
        batch_size: Batch size for processing
        threshold: Similarity threshold (0-10) for categorizing as similar/dissimilar
        max_preview_images: Maximum number of images to show in each visualization preview
        
    Returns:
        Dictionary with test results
    """
    # Create output directories for this subject property
    subject_output_dir = os.path.join(output_dir, f"subject_{subject_property_id}")
    ensure_dir_exists(subject_output_dir)
    
    results_dir = os.path.join(subject_output_dir, "results")
    similar_dir = os.path.join(results_dir, "similar_pairs")
    dissimilar_dir = os.path.join(results_dir, "dissimilar_pairs")
    
    ensure_dir_exists(results_dir)
    ensure_dir_exists(similar_dir)
    ensure_dir_exists(dissimilar_dir)
    
    # Track results
    all_scores = []
    true_labels = []
    predicted_labels = []
    all_results = []
    
    # Set model to evaluation mode (if the method exists)
    try:
        model.eval()
    except AttributeError:
        logger.debug("SiameseNetwork does not have eval() method, continuing without it...")
    
    # Process in batches
    logger.info(f"Testing subject property {subject_property_id} against {len(property_pairs)} comp properties...")
    
    with torch.no_grad():
        for batch_idx in range(0, len(property_pairs), batch_size):
            batch_end = min(batch_idx + batch_size, len(property_pairs))
            batch = property_pairs[batch_idx:batch_end]
            
            logger.info(f"Processing batch {batch_idx//batch_size + 1}/{(len(property_pairs) + batch_size - 1)//batch_size}")
            
            batch_results = []
            
            for i, (subject_id, comp_id, is_similar) in enumerate(batch):
                try:
                    # Get property image paths
                    subject_images = properties.get(subject_id, [])
                    comp_images = properties.get(comp_id, [])
                    
                    if not subject_images or not comp_images:
                        logger.warning(f"Missing images for property {subject_id} or {comp_id}")
                        continue
                    
                    # Use all images for both properties (no sampling/limiting)
                    
                    # Load and preprocess images for both properties
                    subject_processed = [load_and_preprocess_image(img_path) for img_path in subject_images]
                    comp_processed = [load_and_preprocess_image(img_path) for img_path in comp_images]
                    
                    # Compute similarity between properties
                    similarity_score = model.compute_property_similarity(subject_processed, comp_processed)
                    
                    # Determine predicted label
                    predicted_similar = similarity_score >= threshold
                    
                    # Add to results
                    pair_result = {
                        "pair_id": batch_idx + i,
                        "subject_property_id": subject_id,
                        "comp_property_id": comp_id,
                        "subject_images": len(subject_images),
                        "comp_images": len(comp_images),
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
                    
                    # Always save visualization (no longer optional)
                    # Determine output directory based on correctness
                    vis_dir = similar_dir if predicted_similar else dissimilar_dir
                    
                    # Use max_preview_images to limit visualization complexity
                    preview_subject_images = subject_images[:min(max_preview_images, len(subject_images))]
                    preview_comp_images = comp_images[:min(max_preview_images, len(comp_images))]
                    
                    save_property_visualization(
                        preview_subject_images,
                        preview_comp_images,
                        similarity_score,
                        is_similar,
                        predicted_similar,
                        vis_dir,
                        f"pair_{subject_id}_vs_{comp_id}"
                    )
                except Exception as e:
                    logger.error(f"Error processing property pair ({subject_id}, {comp_id}): {str(e)}")
                    continue
            
            # Add batch results
            all_results.extend(batch_results)
            
            # Clear memory after each batch
            clear_memory()
    
    # Compute final metrics
    metrics = compute_metrics(true_labels, predicted_labels, all_scores)
    
    # Save detailed results with explanatory comments
    results_path = os.path.join(subject_output_dir, f"{subject_property_id}_test_results.json")
    
    # Add explanatory comments to metrics
    metrics_with_comments = {
        # Overall metrics
        "num_test_pairs": metrics.get("num_test_pairs", 0),  # Total number of test pairs
        "accuracy": metrics.get("accuracy", 0),  # Proportion of correct predictions (TP+TN)/(TP+TN+FP+FN)
        "precision": metrics.get("precision", 0),  # TP/(TP+FP) - Of all predicted similar, how many were actually similar
        "recall": metrics.get("recall", 0),  # TP/(TP+FN) - Of all actually similar, how many were correctly predicted
        "f1_score": metrics.get("f1_score", 0),  # Harmonic mean of precision and recall: 2*(precision*recall)/(precision+recall)
        
        # Confusion matrix elements
        "tp": metrics.get("tp", 0),  # True positive - Correctly identified as similar
        "fp": metrics.get("fp", 0),  # False positive - Incorrectly identified as similar
        "tn": metrics.get("tn", 0),  # True negative - Correctly identified as dissimilar
        "fn": metrics.get("fn", 0),  # False negative - Incorrectly identified as dissimilar
        
        # Average scores
        "avg_similar_score": metrics.get("avg_similar_score", 0),  # Average similarity score for actually similar pairs
        "avg_dissimilar_score": metrics.get("avg_dissimilar_score", 0)  # Average similarity score for actually dissimilar pairs
    }
    
    full_results = {
        "subject_property_id": subject_property_id,  # ID of the subject property being tested
        "metrics": metrics_with_comments,  # Statistics about test performance
        "comp_pairs": all_results,  # Detailed results for each comp property comparison
        "threshold": threshold  # Similarity threshold used for classification (0-10)
    }
    
    save_json(full_results, results_path)
    logger.info(f"Saved detailed results for {subject_property_id} to {results_path}")
    
    # Generate and save similarity histogram
    if all_scores:
        create_similarity_histogram(all_scores, true_labels, threshold, subject_output_dir)
    
    return metrics_with_comments, all_results

def main():
    """Main function to run the auto testing."""
    args = parse_args()
    
    # Create output directory
    ensure_dir_exists(args.output_dir)
    
    # Ask for subject property ID if not provided
    subject_property_id = args.subject_property_id
    
    # Load test properties
    test_properties = load_test_properties(args.test_data_dir)
    
    if not test_properties:
        logger.error(f"No test properties found in {args.test_data_dir}")
        sys.exit(1)
    
    logger.info(f"Loaded {len(test_properties)} test properties with {sum(len(imgs) for imgs in test_properties.values())} total images")
    
    if subject_property_id is None:
        # Display available properties
        print("\nAvailable properties:")
        for i, prop_id in enumerate(test_properties.keys()):
            print(f"{i+1}. {prop_id} ({len(test_properties[prop_id])} images)")
        
        try:
            selection = int(input("\nEnter the number of the subject property to test: "))
            if 1 <= selection <= len(test_properties):
                subject_property_id = list(test_properties.keys())[selection-1]
            else:
                logger.error("Invalid selection. Using first property as default.")
                subject_property_id = list(test_properties.keys())[0]
        except ValueError:
            logger.error("Invalid input. Using first property as default.")
            subject_property_id = list(test_properties.keys())[0]
    
    logger.info(f"Selected subject property: {subject_property_id}")
    
    # Ask for number of comp properties if not provided
    num_comps = args.num_comps
    if num_comps is None:
        try:
            num_comps = int(input("Enter the number of comp properties to compare with the subject property: "))
        except ValueError:
            logger.error("Invalid number entered. Using 10 as default.")
            num_comps = 10
    
    # Check if we have enough properties for the requested number of comps
    if len(test_properties) <= num_comps:
        logger.warning(f"Only {len(test_properties)} properties available. Reducing comps to {len(test_properties) - 1}")
        num_comps = max(1, len(test_properties) - 1)
    
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
    
    # Set model to evaluation mode (if the method exists)
    try:
        model.eval()
    except AttributeError:
        logger.info("SiameseNetwork does not have eval() method, continuing without it...")
    
    # Generate pairs for subject property
    property_pairs = generate_subject_vs_comps_pairs(
        test_properties, 
        subject_property_id, 
        num_comps
    )
    
    # Test subject property against selected comps
    metrics, results = test_subject_against_comps(
        model=model,
        subject_property_id=subject_property_id,
        property_pairs=property_pairs,
        properties=test_properties,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        threshold=args.threshold,
        max_preview_images=args.max_preview_images
    )
    
    # Create summary DataFrame and save to CSV
    summary_df = pd.DataFrame([{
        "subject_property_id": subject_property_id,
        "num_comps": len(property_pairs),
        "accuracy": metrics.get("accuracy", 0),
        "precision": metrics.get("precision", 0),
        "recall": metrics.get("recall", 0),
        "f1_score": metrics.get("f1_score", 0),
        "avg_similar_score": metrics.get("avg_similar_score", 0),
        "avg_dissimilar_score": metrics.get("avg_dissimilar_score", 0),
        "similar_count": metrics.get("tp", 0) + metrics.get("fn", 0),
        "dissimilar_count": metrics.get("tn", 0) + metrics.get("fp", 0)
    }])
    
    summary_csv = os.path.join(args.output_dir, "auto_test_summary.csv")
    summary_df.to_csv(summary_csv, index=False)
    
    # Also save as JSON with comments
    summary_json = os.path.join(args.output_dir, "auto_test_summary.json")
    summary_dict = {
        "subject_property_id": subject_property_id,  # ID of the subject property tested
        "num_comps": len(property_pairs),  # Number of comp properties tested against the subject
        "accuracy": metrics.get("accuracy", 0),  # Proportion of correct predictions
        "precision": metrics.get("precision", 0),  # Of all predicted similar, how many were actually similar
        "recall": metrics.get("recall", 0),  # Of all actually similar, how many were correctly predicted
        "f1_score": metrics.get("f1_score", 0),  # Harmonic mean of precision and recall
        "tp": metrics.get("tp", 0),  # True positive - Correctly identified as similar
        "fp": metrics.get("fp", 0),  # False positive - Incorrectly identified as similar
        "tn": metrics.get("tn", 0),  # True negative - Correctly identified as dissimilar
        "fn": metrics.get("fn", 0),  # False negative - Incorrectly identified as dissimilar
        "avg_similar_score": metrics.get("avg_similar_score", 0),  # Average similarity score for actually similar pairs
        "avg_dissimilar_score": metrics.get("avg_dissimilar_score", 0),  # Average similarity score for actually dissimilar pairs
        "similar_count": metrics.get("tp", 0) + metrics.get("fn", 0),  # Total number of actually similar pairs
        "dissimilar_count": metrics.get("tn", 0) + metrics.get("fp", 0)  # Total number of actually dissimilar pairs
    }
    save_json(summary_dict, summary_json)
    
    # Create summary README
    readme_path = os.path.join(args.output_dir, "README.md")
    with open(readme_path, 'w') as f:
        f.write(f"# Auto Testing Results\n\n")
        f.write(f"## Configuration\n\n")
        f.write(f"- Model directory: {args.model_dir}\n")
        f.write(f"- Model type: {args.model_type}\n")
        f.write(f"- Test data: {args.test_data_dir}\n")
        f.write(f"- Subject property: {subject_property_id}\n")
        f.write(f"- Number of comp properties: {len(property_pairs)}\n")
        f.write(f"- Similarity threshold: {args.threshold}\n\n")
        
        f.write(f"## Summary\n\n")
        f.write(f"- Accuracy: {metrics.get('accuracy', 0):.4f}\n")
        f.write(f"- Precision: {metrics.get('precision', 0):.4f}\n")
        f.write(f"- Recall: {metrics.get('recall', 0):.4f}\n")
        f.write(f"- F1 Score: {metrics.get('f1_score', 0):.4f}\n")
        f.write(f"- Average similarity score for similar pairs: {metrics.get('avg_similar_score', 0):.2f}\n")
        f.write(f"- Average similarity score for dissimilar pairs: {metrics.get('avg_dissimilar_score', 0):.2f}\n\n")
        
        f.write(f"## Explanation of Metrics\n\n")
        f.write(f"- **TP (True Positive)**: {metrics.get('tp', 0)} - Correctly identified as similar\n")
        f.write(f"- **FP (False Positive)**: {metrics.get('fp', 0)} - Incorrectly identified as similar\n")
        f.write(f"- **TN (True Negative)**: {metrics.get('tn', 0)} - Correctly identified as dissimilar\n")
        f.write(f"- **FN (False Negative)**: {metrics.get('fn', 0)} - Incorrectly identified as dissimilar\n")
        f.write(f"- **Accuracy**: (TP+TN)/(TP+TN+FP+FN) - Proportion of correct predictions\n")
        f.write(f"- **Precision**: TP/(TP+FP) - Of all predicted similar, how many were actually similar\n")
        f.write(f"- **Recall**: TP/(TP+FN) - Of all actually similar, how many were correctly predicted\n")
        f.write(f"- **F1 Score**: 2*(precision*recall)/(precision+recall) - Harmonic mean of precision and recall\n\n")
        
        f.write(f"See `{subject_property_id}_test_results.json` for detailed results for each comp property comparison.\n")
    
    print(f"\nAuto-testing complete! Results saved to {args.output_dir}")
    return 0

if __name__ == "__main__":
    sys.exit(main()) 