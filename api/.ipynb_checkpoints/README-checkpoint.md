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

## Installation

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Ensure you have the model files in the `final_model` directory or specify a custom path using the `MODEL_DIR` environment variable.

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

- `MODEL_DIR`: Path to the directory containing the model checkpoint (default: `../final_model`)
- `MODEL_GCS_PATH`: Google Cloud Storage path to model in format gs://bucket-name/path/to/model.pth.tar
- `PORT`: Port to run the API server on (default: 8080 for Vertex AI compatibility)
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

Build the Docker image:

```bash
docker build -t property-comparison-api -f api/Dockerfile .
```

Run the container:

```bash
docker run -p 8080:8080 -e MODEL_DIR=/app/final_model property-comparison-api
```

## Vertex AI Deployment

To deploy the API to Vertex AI:

1. Store your model in a Google Cloud Storage bucket:

```bash
gsutil cp final_model/model_best.pth.tar gs://your-bucket/models/
```

2. Build the Docker image and push it to Google Container Registry:

```bash
docker build -t gcr.io/your-project/property-comparison-api -f api/Dockerfile .
docker push gcr.io/your-project/property-comparison-api
```

3. Deploy the container to Vertex AI:

```bash
gcloud ai endpoints create --display-name="property-comparison-api" --region=us-central1
gcloud ai endpoints deploy-model <endpoint-id> \
    --model=property-comparison-api \
    --display-name=property-comparison-api \
    --machine-type=n1-standard-4 \
    --min-replica-count=1 \
    --max-replica-count=3 \
    --container-image-uri=gcr.io/your-project/property-comparison-api \
    --container-ports=8080 \
    --container-env-vars=MODEL_GCS_PATH=gs://your-bucket/models/model_best.pth.tar,ENABLE_CLOUD_LOGGING=true
```

4. The API will be available at the Vertex AI endpoint URL.

### Vertex AI Considerations

- The API listens on port 8080 by default (Vertex AI standard)
- Health checks are implemented on `/` and `/health-check` endpoints
- JSON-formatted logs are enabled for Google Cloud Logging
- Model can be loaded directly from Google Cloud Storage 