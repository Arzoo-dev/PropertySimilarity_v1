# Deploying Siamese Network on Vertex AI

This document outlines the process for deploying the Siamese Neural Network model on Google Cloud's Vertex AI platform.

## Prerequisites

1. Google Cloud account with Vertex AI enabled
2. Google Cloud SDK installed and configured
3. PyTorch and TorchServe installed
4. Model trained and exported

## Deployment Steps

### 1. Create TorchScript Model

First, convert the PyTorch model to TorchScript format using the model_export.py script:

```bash
cd RunPodsModel
python VertexDeployment/model_export.py
```

This script will:
- Load the trained model from final_model/siamese_embedding_model.pt
- Convert it to TorchScript format
- Save it in VertexDeployment/torchscript_model/

### 2. Create Model Archive (.mar) File

The same script will also create a .mar file using torch-model-archiver:

```bash
torch-model-archiver -f \
    --model-name model \
    --version 1.0 \
    --serialized-file VertexDeployment/torchscript_model/siamese_embedding_model.pt \
    --handler VertexDeployment/model_handler.py \
    --export-path VertexDeployment/model_artifacts
```

This bundles the model and handler into a single deployable unit.

### 3. Upload Model to Google Cloud Storage

```bash
# Create a bucket if you don't have one
gsutil mb gs://YOUR_BUCKET_NAME

# Copy the model archive to GCS
gsutil cp VertexDeployment/model_artifacts/model.mar gs://YOUR_BUCKET_NAME/models/siamese/
```

### 4. Deploy Model on Vertex AI

#### Using Console:
1. Go to Vertex AI > Model Registry in the Google Cloud Console
2. Click "Upload Model"
3. Select "PyTorch" as framework
4. Specify the GCS location of your .mar file
5. Configure compute and scaling options
6. Deploy to an endpoint

#### Using gcloud CLI:
```bash
# Create a model
gcloud ai models upload \
  --region=us-central1 \
  --display-name=siamese-property-model \
  --container-image-uri=us-docker.pkg.dev/vertex-ai/prediction/pytorch-gpu.1-13:latest \
  --artifact-uri=gs://YOUR_BUCKET_NAME/models/siamese/

# Deploy to an endpoint
gcloud ai endpoints deploy-model ENDPOINT_ID \
  --region=us-central1 \
  --model=MODEL_ID \
  --display-name=siamese-property-endpoint \
  --machine-type=n1-standard-4 \
  --accelerator=count=1,type=NVIDIA_TESLA_T4 \
  --traffic-split=0=100
```

## Testing the Deployment

### Sample Request Format

```python
import base64
from google.cloud import aiplatform

def predict_image_similarity(project, location, endpoint_id, image_path):
    # Create client
    client = aiplatform.gapic.PredictionServiceClient(
        client_options={"api_endpoint": f"{location}-aiplatform.googleapis.com"}
    )
    
    # Read image and convert to base64
    with open(image_path, "rb") as f:
        image_bytes = f.read()
    
    encoded_content = base64.b64encode(image_bytes).decode("utf-8")
    
    # Format the request
    endpoint = client.endpoint_path(project, location, endpoint_id)
    instance = {"b64": encoded_content}
    
    # Make the request
    response = client.predict(
        endpoint=endpoint,
        instances=[instance],
        parameters={}
    )
    
    # Process the response
    embedding = response.predictions[0]
    return embedding
```

## Understanding the MAR File Format

The .mar (Model Archive) file is a packaging format used by TorchServe that contains:

1. **Model Artifacts**: The TorchScript model file
2. **Handler Code**: Custom Python code that processes inputs and outputs
3. **Manifest**: Metadata about the model, including dependencies

This format ensures that your model and its preprocessing/postprocessing logic are deployed together as a single unit, making deployment more consistent and reliable.

## Monitoring and Maintenance

- Enable logging for your Vertex AI endpoint to monitor performance
- Set up alerts for low throughput or high latency
- Consider implementing A/B testing for model updates
- Regularly evaluate model performance metrics in production 