"""
Client script for making inference requests to the Siamese model deployed on Vertex AI.
"""

import argparse
import base64
import json
import os
from pathlib import Path
from typing import List, Dict, Any
import time

import numpy as np
from google.cloud import aiplatform
from PIL import Image

def encode_image(image_path: str) -> str:
    """
    Read an image and encode it as base64 string.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Base64 encoded string of the image
    """
    with open(image_path, "rb") as f:
        image_bytes = f.read()
    
    encoded = base64.b64encode(image_bytes).decode("utf-8")
    return encoded

def predict_single_image(
    project: str,
    location: str,
    endpoint_id: str,
    image_path: str
) -> List[float]:
    """
    Get embedding for a single image from the deployed model.
    
    Args:
        project: Google Cloud project ID
        location: Region where endpoint is deployed
        endpoint_id: Vertex AI endpoint ID
        image_path: Path to the image file
        
    Returns:
        Embedding vector for the image
    """
    # Create client
    client = aiplatform.gapic.PredictionServiceClient(
        client_options={"api_endpoint": f"{location}-aiplatform.googleapis.com"}
    )
    
    # Read and encode image
    encoded_content = encode_image(image_path)
    
    # Format the request
    endpoint = client.endpoint_path(project, location, endpoint_id)
    instance = {"b64": encoded_content}
    
    # Make the request
    print(f"Sending prediction request for image: {image_path}")
    response = client.predict(
        endpoint=endpoint,
        instances=[instance],
        parameters={}
    )
    
    # Return the embedding
    embedding = response.predictions[0]
    return embedding

def compute_similarity(emb1: List[float], emb2: List[float]) -> float:
    """
    Compute cosine similarity between two embeddings.
    
    Args:
        emb1: First embedding vector
        emb2: Second embedding vector
        
    Returns:
        Similarity score between 0 and 1
    """
    # Convert to numpy arrays
    emb1_np = np.array(emb1)
    emb2_np = np.array(emb2)
    
    # Compute cosine similarity
    dot_product = np.dot(emb1_np, emb2_np)
    norm1 = np.linalg.norm(emb1_np)
    norm2 = np.linalg.norm(emb2_np)
    
    similarity = dot_product / (norm1 * norm2)
    
    # Normalize to 0-1 range
    normalized_similarity = (similarity + 1) / 2
    
    return normalized_similarity

def compare_properties(
    project: str,
    location: str,
    endpoint_id: str,
    property1_images: List[str],
    property2_images: List[str],
) -> Dict[str, Any]:
    """
    Compare two properties by getting embeddings for all images and computing
    average similarity.
    
    Args:
        project: Google Cloud project ID
        location: Region where endpoint is deployed
        endpoint_id: Vertex AI endpoint ID
        property1_images: List of paths to images for property 1
        property2_images: List of paths to images for property 2
        
    Returns:
        Dictionary with similarity score and processing time
    """
    start_time = time.time()
    
    # Get embeddings for property 1
    property1_embeddings = []
    for img_path in property1_images:
        embedding = predict_single_image(project, location, endpoint_id, img_path)
        property1_embeddings.append(embedding)
    
    # Get embeddings for property 2
    property2_embeddings = []
    for img_path in property2_images:
        embedding = predict_single_image(project, location, endpoint_id, img_path)
        property2_embeddings.append(embedding)
    
    # Compute pairwise similarities
    similarities = []
    for emb1 in property1_embeddings:
        for emb2 in property2_embeddings:
            sim = compute_similarity(emb1, emb2)
            similarities.append(sim)
    
    # Compute average similarity
    avg_similarity = np.mean(similarities)
    
    # Convert to 0-10 scale for property comparison
    property_score = avg_similarity * 10
    
    end_time = time.time()
    processing_time = end_time - start_time
    
    return {
        "similarity_score": property_score,
        "processing_time_seconds": processing_time,
        "num_images_property1": len(property1_images),
        "num_images_property2": len(property2_images),
    }

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Siamese model inference client for Vertex AI")
    parser.add_argument("--project", required=True, help="Google Cloud project ID")
    parser.add_argument("--location", default="us-central1", help="Region where endpoint is deployed")
    parser.add_argument("--endpoint-id", required=True, help="Vertex AI endpoint ID")
    parser.add_argument("--property1", required=True, help="Directory containing images for property 1")
    parser.add_argument("--property2", required=True, help="Directory containing images for property 2")
    
    return parser.parse_args()

def main():
    """Main function for running the inference client"""
    args = parse_args()
    
    # Get all image paths
    property1_images = [str(path) for path in Path(args.property1).glob("*.jpg")]
    property2_images = [str(path) for path in Path(args.property2).glob("*.jpg")]
    
    if not property1_images:
        raise ValueError(f"No images found in {args.property1}")
    if not property2_images:
        raise ValueError(f"No images found in {args.property2}")
    
    print(f"Found {len(property1_images)} images for property 1")
    print(f"Found {len(property2_images)} images for property 2")
    
    # Compare properties
    result = compare_properties(
        args.project,
        args.location,
        args.endpoint_id,
        property1_images,
        property2_images
    )
    
    # Print results
    print("\nComparison Results:")
    print(f"Similarity Score (0-10): {result['similarity_score']:.2f}")
    print(f"Processing Time: {result['processing_time_seconds']:.2f} seconds")
    
    # Interpret results
    if result['similarity_score'] > 7.5:
        print("Interpretation: Properties are very similar")
    elif result['similarity_score'] > 5.0:
        print("Interpretation: Properties have moderate similarity")
    else:
        print("Interpretation: Properties are dissimilar")

if __name__ == "__main__":
    main() 