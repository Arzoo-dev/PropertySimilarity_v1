# Property Comparison API

A FastAPI service for comparing real estate properties using a Siamese neural network.

## Features

- Compare a subject property against multiple comparable properties
- Asynchronous image downloading for faster processing
- Caching of downloaded images for improved performance
- Detailed similarity metrics and comparison results
- Configurable similarity threshold
- Google Cloud Storage integration for model loading
- Vertex AI deployment support
- Flexible model loading from multiple sources
- Optimized multi-stage Docker build

## Installation

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Ensure you have the model files in one of these locations:
   - Local `final_model` directory
   - Mounted volume at `/app/weights`
   - Google Cloud Storage (when using `MODEL_GCS_PATH`)

## Usage

### Running the API Locally

Start the API server:

```bash
python -m api.main
```

Or using uvicorn directly:

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

### Environment Variables

- `MODEL_DIR`: Path to the directory containing the model checkpoint (default: `/app/final_model`)
- `MODEL_GCS_PATH`: Google Cloud Storage path to model in format gs://bucket-name/path/to/model.pth.tar
- `PORT`: Port to run the API server on (default: 69)
- `HOST`: Host to bind the server to (default: 0.0.0.0)
- `ENABLE_CLOUD_LOGGING`: Set to "true" to enable Google Cloud Logging (default: false)

### API Endpoints

#### Health Check

```
GET /health-check
GET /
```

Returns the health status of the API and whether the model is loaded.

#### Property Comparison

```
POST /api/compare-properties
```

Compares a subject property against multiple comparable properties.

Example request:

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
        {"url": "https://example.com/comp1_photo1.jpg"},
        {"url": "https://example.com/comp1_photo2.jpg"}
      ],
      "address": "456 Oak St, Anytown, USA"
    },
    {
      "uid": "comp2",
      "photos": [
        {"url": "https://example.com/comp2_photo1.jpg"},
        {"url": "https://example.com/comp2_photo2.jpg"}
      ],
      "address": "789 Pine St, Anytown, USA"
    }
  ],
  "threshold": 5.0,
  "max_comps": 10
}
```

Example response:

```json
{
  "subject_property_id": "subject",
  "metrics": {
    "num_test_pairs": 2,
    "accuracy": 1.0,
    "precision": 1.0,
    "recall": 1.0,
    "f1_score": 1.0,
    "tp": 2,
    "fp": 0,
    "tn": 0,
    "fn": 0,
    "avg_similar_score": 9.85,
    "avg_dissimilar_score": 0
  },
  "comp_pairs": [
    {
      "pair_id": 0,
      "subject_property_id": "subject",
      "comp_property_id": "comp1",
      "subject_images": 2,
      "comp_images": 2,
      "true_label": "similar",
      "predicted_label": "similar",
      "similarity_score": 9.9,
      "correct_prediction": true,
      "address": "456 Oak St, Anytown, USA"
    },
    {
      "pair_id": 1,
      "subject_property_id": "subject",
      "comp_property_id": "comp2",
      "subject_images": 2,
      "comp_images": 2,
      "true_label": "similar",
      "predicted_label": "similar",
      "similarity_score": 9.8,
      "correct_prediction": true,
      "address": "789 Pine St, Anytown, USA"
    }
  ],
  "threshold": 5.0
}
```

## Documentation

Interactive documentation is available at:

- Swagger UI: `/docs`
- ReDoc: `/redoc`
- API documentation: `/api/docs`

## Testing

Run the tests:

```bash
pytest api/tests/
```

Or run the test file directly:

```bash
python -m api.tests.test_api
```

## Docker

The project includes a multi-stage Dockerfile that significantly reduces the final image size and improves security.

### Build the Docker image:

```bash
docker build -t property-api-lightweight -f RunPodsModel/api/Dockerfile .
```

### Run with different model loading options:

#### 1. Run with mounted model weights:

```bash
docker run -p 69:69 --gpus all -v /path/to/local/model/weights:/app/weights property-api-lightweight
```

#### 2. Run with Google Cloud Storage model:

```bash
docker run -p 69:69 --gpus all -e MODEL_GCS_PATH=gs://your-bucket/models/siamese_embedding_model.pt property-api-lightweight
```

#### 3. Run with model embedded in the image:

If you need to include the model in the image (not recommended for larger models):

```bash
# First modify the Dockerfile to uncomment the COPY line for model files
docker build -t property-api-with-model -f RunPodsModel/api/Dockerfile --build-arg INCLUDE_MODEL=true .
docker run -p 69:69 --gpus all property-api-with-model
```

## Vertex AI Deployment

To deploy the API to Vertex AI:

1. Store your model in a Google Cloud Storage bucket:

```bash
gsutil cp -r final_model/* gs://your-bucket/models/
```

2. Build the Docker image and push it to Artifact Registry:

```bash
# Configure Docker for Artifact Registry
gcloud auth configure-docker us-central1-docker.pkg.dev

# Build and tag the image
docker build -t property-api-lightweight -f RunPodsModel/api/Dockerfile .
docker tag property-api-lightweight us-central1-docker.pkg.dev/your-project/property-api-repo/property-api:latest

# Push to Artifact Registry
docker push us-central1-docker.pkg.dev/your-project/property-api-repo/property-api:latest
```

3. Deploy the container to Vertex AI:

### Using Google Cloud Console (UI):

1. Go to Vertex AI section in Google Cloud Console
2. Navigate to "Models" and click "Create"
3. Select "Import" tab
4. For model artifact, enter Artifact Registry container URI
5. Create an endpoint
6. Deploy model to endpoint with these settings:
   - Machine type: n1-standard-4 or appropriate size
   - Accelerator: NVIDIA T4 GPU (1 unit)
   - Min/max replicas: 1/3 (adjust as needed)
   - Container port: 69
   - Environment variables:
     - MODEL_GCS_PATH: gs://your-bucket/models/siamese_embedding_model.pt
     - ENABLE_CLOUD_LOGGING: true

### Using gcloud CLI:

```bash
gcloud ai endpoints create --display-name="property-comparison-api" --region=us-central1
gcloud ai endpoints deploy-model <endpoint-id> \
    --model=property-comparison-api \
    --display-name=property-comparison-api \
    --machine-type=n1-standard-4 \
    --accelerator=count=1,type=NVIDIA_TESLA_T4 \
    --min-replica-count=1 \
    --max-replica-count=3 \
    --container-image-uri=us-central1-docker.pkg.dev/your-project/property-api-repo/property-api:latest \
    --container-ports=69 \
    --container-env-vars=MODEL_GCS_PATH=gs://your-bucket/models/siamese_embedding_model.pt,ENABLE_CLOUD_LOGGING=true
```

4. The API will be available at the Vertex AI endpoint URL.

### Vertex AI Considerations

- Health checks are implemented on `/` and `/health-check` endpoints
- JSON-formatted logs are enabled with ENABLE_CLOUD_LOGGING=true
- Model loading order:
  1. Check mounted volume at /app/weights
  2. Check local directory at /app/final_model
  3. Download from GCS if MODEL_GCS_PATH is provided
- The start.sh script handles automatic worker scaling based on CPU cores 