# Vertex AI Deployment for Siamese Network

This folder contains all the necessary files and scripts to deploy the Siamese Neural Network property comparison model on Google Cloud's Vertex AI platform.

## Overview

The Siamese Neural Network model takes property images as input and generates embedding vectors that can be used to compare the similarity between properties. To deploy this model for production use, we package it as a Model Archive (.mar) file using TorchServe and deploy it on Vertex AI.

The deployment uses the **ResNet50 backbone** to match the implementation used in the API service.

## Contents

- **model_handler.py**: Custom handler for processing input/output for the TorchServe model
- **model_export.py**: Script to convert PyTorch model to TorchScript and create MAR file
- **inference_client.py**: Client script to demonstrate making inference requests
- **requirements.txt**: Dependencies required for deployment
- **vertex_deployment.md**: Detailed documentation on deployment steps

## Quick Start

1. Install requirements:
   ```
   pip install -r requirements.txt
   ```

2. Export the model:
   ```
   python model_export.py
   ```
   This script ensures the model uses the ResNet50 backbone to match the API implementation.

3. Deploy on Vertex AI (see vertex_deployment.md for details)

4. Run test inference:
   ```
   python inference_client.py --project YOUR_PROJECT --endpoint-id YOUR_ENDPOINT_ID --property1 path/to/property1_images --property2 path/to/property2_images
   ```

## Model Archive (MAR) File

The deployment uses TorchServe's Model Archive format to package the model with its handler. The archive includes:

- TorchScript model (converted from PyTorch with ResNet50 backbone)
- Custom handler for preprocessing and postprocessing
- Model metadata

## Preprocessing Details

The model handler uses the same preprocessing as the API implementation:
- Resize images to 224x224 pixels
- Normalize with ImageNet mean/std values: 
  - mean=[0.485, 0.456, 0.406]
  - std=[0.229, 0.224, 0.225]
- L2 normalization of embeddings for cosine similarity

## Inference Process

1. Client sends base64-encoded image(s) to Vertex AI endpoint
2. Handler preprocesses image (resize, normalize)
3. Model generates embedding vector
4. Handler formats and returns embedding
5. Client can compare embeddings to compute similarity scores

## Deployment Options

The deployment can be configured for different resource needs:

- **CPU-only**: Lower cost, suitable for development
- **GPU-accelerated**: Higher throughput, recommended for production
- **Autoscaling**: Scale based on demand patterns

Refer to vertex_deployment.md for complete configuration options. 