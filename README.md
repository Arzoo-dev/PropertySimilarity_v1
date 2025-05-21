# Siamese Network for Property Image Similarity (RunPods Version)

This directory contains a PyTorch implementation of a Siamese Network for property image similarity, optimized for training on RunPods with deployment options for Vertex AI.

## Overview

The model uses a Siamese Network architecture with an EfficientNet-B0 backbone to compute embeddings for property images. These embeddings are used to determine similarity between properties, which can be used for recommendations, clustering, or search.

## Directory Structure

```
RunPodsModel/
├── api/                  # FastAPI implementation for model deployment
│   ├── Dockerfile        # Multi-stage Docker build for efficient deployment
│   ├── main.py           # FastAPI entry point with API endpoints
│   ├── services.py       # Model service implementation
│   ├── schemas.py        # Pydantic models for request/response validation
│   ├── utils.py          # Helper functions for image processing
│   └── start.sh          # Container startup script
├── data_preparation.py   # Data loading and preprocessing utilities with API integration
├── data_preprocessing.py # Image triplet generation for training
├── runpods_utils.py      # File handling utilities for RunPods
├── siamese_network.py    # PyTorch Siamese Network implementation
├── train.py              # Training script
├── test.py               # Model evaluation script
├── final_model/          # Directory for trained model weights
├── requirements.txt      # Python dependencies
└── README.md             # This file
```

## Requirements

To install the required dependencies:

```bash
pip install -r requirements.txt
```

### Key Dependencies

- PyTorch 2.1.0 with CUDA support
- torchvision 0.16.0
- numpy, pandas, scikit-learn for data processing
- tqdm for progress monitoring
- albumentations for image augmentation
- matplotlib for visualization
- OpenCV (opencv-python) for image processing
- FastAPI and uvicorn for API serving

### System Dependencies

For OpenCV and image processing, these system packages are required:

```bash
apt-get install -y libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev
```

## Data Sources

The system supports two methods of obtaining training data:

1. **Local Files**: Load images from local directories
2. **API Endpoint**: Fetch images from a remote API endpoint

### Local Directory Structure

If using local files, organize them as follows:

```
data/
├── property_type_1/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── property_type_2/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── ...
```

Where each subfolder represents a different property type (e.g., house, apartment, etc.), and contains images of that property type.

### API Integration

The API client supports:
- Authentication via token
- Concurrent image downloads
- Local caching of downloaded images
- Organized directory structure by property type
- Configurable retry logic for robustness

## Training the Model

### Training with Local Files

```bash
python train.py \
  --train-data-dir /workspace/data/train \
  --val-data-dir /workspace/data/val \
  --model-dir /workspace/models/siamese \
  --num-triplets 10000 \
  --batch-size 32 \
  --epochs 50 \
  --save-triplets \
  --eval-model
```

### Training with API Endpoint

```bash
python train.py \
  --api-endpoint "https://your-api-endpoint.com/api/v1" \
  --api-token "your_auth_token" \
  --property-types "house" "apartment" "condo" \
  --images-per-type 100 \
  --api-val-property-types "duplex" "townhouse" \
  --model-dir /workspace/models/siamese \
  --num-triplets 10000 \
  --batch-size 32 \
  --epochs 50 \
  --save-triplets \
  --eval-model
```

### Command-line Arguments

#### Data Source Options
- `--train-data-dir`: Directory containing training data
- `--api-endpoint`: API endpoint URL for fetching training data
- `--api-token`: Authentication token for API
- `--property-types`: Property types to fetch from API
- `--images-per-type`: Number of images to fetch per property type (default: 100)
- `--api-cache-dir`: Directory to cache API images (default: "/workspace/data/api_cache")
- `--api-workers`: Maximum number of concurrent download threads (default: 8)

#### Validation Data Options
- `--val-data-dir`: Directory containing validation data
- `--api-val-property-types`: Property types to fetch from API for validation

#### Model and Training Options
- `--model-dir`: Directory to save model checkpoints (required)
- `--output-dir`: Directory for output files (defaults to model-dir)
- `--num-triplets`: Number of triplets to generate for training (default: 10000)
- `--val-triplets`: Number of triplets for validation (default: 1000)
- `--batch-size`: Batch size for training (default: 32)
- `--epochs`: Number of training epochs (default: 50)
- `--learning-rate`: Learning rate (default: 0.001)
- `--embedding-dim`: Dimension of the embedding vector (default: 128)
- `--margin`: Margin for triplet loss (default: 0.2)
- `--augment-prob`: Probability of applying augmentation (default: 0.7)
- `--seed`: Random seed for reproducibility (default: 42)
- `--save-triplets`: Save generated triplets to disk (flag)
- `--eval-model`: Evaluate model after training (flag)
- `--eval-pairs`: Number of pairs for evaluation (default: 500)
- `--checkpoint-freq`: Frequency to save checkpoints (default: 5 epochs)
- `--load-checkpoint`: Path to checkpoint to continue training from

## Using the Model

You can use the trained model to compute similarity between property images:

```python
from siamese_network import SiameseNetwork
from data_preparation import load_and_preprocess_image

# Load model
model = SiameseNetwork(embedding_dim=128)
model.load_model('/workspace/models/siamese/best_model')

# Load and preprocess images
img1 = load_and_preprocess_image('property1.jpg')
img2 = load_and_preprocess_image('property2.jpg')

# Compute similarity (returns a score from 0-10)
similarity = model.compute_similarity(img1, img2)
print(f"Similarity score: {similarity:.2f}/10")
```

## RunPods Setup

This code is designed to run on RunPods with PyTorch. When creating your RunPod:

1. Choose a container with PyTorch and CUDA support.
2. Mount your data directory to `/workspace/data` (or use the API endpoint).
3. Set the output directory to `/workspace/models`.

### Example RunPods Template

- **Container Image**: `pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime`
- **Volume Mounts**:
  - `/path/to/local/data:/workspace/data` (optional if using API)
  - `/path/to/local/models:/workspace/models`
- **Command**:
  ```bash
  cd /workspace && python RunPodsModel/train.py [OPTIONS]
  ```

## Model Evaluation

After training, the model is evaluated on validation data (if provided). The evaluation results include:

- Accuracy: How often the model correctly identifies similar/dissimilar pairs
- Precision: Proportion of predicted similar pairs that are actually similar
- Recall: Proportion of actual similar pairs that are correctly identified
- F1 Score: Harmonic mean of precision and recall

A similarity score above 5.0 (on a scale of 0-10) is considered a match.

## API Deployment

The model is packaged as a FastAPI service ready for deployment on RunPods, Vertex AI, or other container platforms.

### API Features

- **Property Comparison**: Compare a subject property against multiple comparable properties
- **Asynchronous Processing**: Concurrent image downloads for faster processing
- **Image Caching**: Improved performance through caching
- **Detailed Metrics**: Comprehensive comparison results and similarity scoring
- **Health Checks**: Monitoring endpoints for deployment health
- **Flexible Model Loading**: Load models from local files, mounted volumes, or cloud storage
- **Vertex AI Integration**: Compatible endpoints that handle Vertex AI's request/response format

### Docker Deployment

The provided multi-stage Dockerfile creates an optimized container with GPU support:

```bash
# Build the Docker image
docker build -t property-api-lightweight -f RunPodsModel/api/Dockerfile .

# Run with local model weights mounted
docker run -p 8080:8080 --gpus all -v /path/to/your/final_model:/app/weights property-api-lightweight
```

#### Docker Image Features

- Multi-stage build process to minimize final image size
- CUDA 11.8 with cuDNN8 for GPU acceleration
- OpenCV system dependencies pre-installed
- Extended timeout and retry settings for large package downloads
- Comprehensive verification to ensure all packages are properly installed

### Environment Variables

The API container supports the following environment variables:

- `MODEL_DIR`: Path to the directory containing model weights (default: `/app/final_model`)
- `MODEL_GCS_PATH`: Google Cloud Storage path to model in format `gs://bucket-name/path/to/model`
- `PORT`: Port to run the API server on (default: 8080, auto-detected on Vertex AI)
- `HOST`: Host to bind the server to (default: 0.0.0.0)
- `ENABLE_CLOUD_LOGGING`: Set to "true" to enable Google Cloud Logging

### API Endpoints

- `GET /health-check` or `GET /`: Health check endpoint
- `POST /api/compare-properties`: Main comparison endpoint
- `POST /predict`: Vertex AI compatible endpoint that handles the platform's request/response format

Example regular request:
```json
{
  "subject_property": {
    "photos": [
      {"url": "https://example.com/subject_photo1.jpg"},
      {"url": "https://example.com/subject_photo2.jpg"}
    ],
    "address": "123 Main St, Anytown, USA"
  },
  "comps": [
    {
      "uid": "comp1",
      "photos": [
        {"url": "https://example.com/comp1_photo1.jpg"}
      ],
      "address": "456 Oak St, Anytown, USA"
    }
  ],
  "threshold": 5.0
}
```

Example Vertex AI request:
```json
{
  "instances": [
    {
      "subject_property": { /* same content as regular request */ },
      "comps": [ /* same content as regular request */ ],
      "threshold": 5.0
    }
  ],
  "parameters": {
    "threshold": 5.0,
    "max_comps": 10
  }
}
```

Note: Parameter values in the instance take precedence over the global parameters field.

### Vertex AI Deployment

To deploy the model on Google Cloud Vertex AI:

1. **Upload model files to Google Cloud Storage**:
   ```bash
   gsutil cp -r final_model/* gs://your-bucket/models/
   ```

2. **Build and push the Docker image**:
   ```bash
   # Tag for Artifact Registry
   docker tag property-api-lightweight us-central1-docker.pkg.dev/your-project/property-api-repo/property-api:latest
   
   # Push to Artifact Registry
   docker push us-central1-docker.pkg.dev/your-project/property-api-repo/property-api:latest
   ```

3. **Deploy to Vertex AI** via Google Cloud Console:
   - Create a model in the Vertex AI Models section
   - Create an endpoint
   - Deploy the model to the endpoint with GPU acceleration
   - Set `MODEL_GCS_PATH` environment variable to your model's GCS path
   - Specify container settings:
     - Container port: 8080
     - Prediction route: /predict
     - Health route: /health-check

### Vertex AI Deployment Troubleshooting

If you encounter issues with your Vertex AI deployment:

1. **Package Dependency Issues**: Ensure tqdm, albumentations, and other required packages are installed
2. **OpenCV System Libraries**: Add necessary system packages (libgl1-mesa-glx, etc.) for OpenCV functionality
3. **GCS Access Permissions**: Grant your service account proper permissions to access your model bucket
4. **Large File Downloads**: Use --resume-retries and increased timeouts for downloading large models
5. **Memory Limitations**: Ensure your deployment has sufficient memory for both model and image processing 