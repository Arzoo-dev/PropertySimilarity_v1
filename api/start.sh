#!/bin/bash

# Display NVIDIA GPU information
echo "=========="
echo "== CUDA =="
echo "=========="
echo "NVIDIA GPU Information:"
nvidia-smi

# Set default values for environment variables if not set
if [ -z "$MODEL_DIR" ]; then
    export MODEL_DIR="/workspace/final_model"
    echo "MODEL_DIR not set, using default: $MODEL_DIR"
fi

# Handle PORT for Vertex AI compatibility
# Vertex AI sets the PORT environment variable automatically
if [ -z "$PORT" ]; then
    # Default to 8080 for Vertex AI
    export PORT="8080"
    echo "PORT not set, using default: $PORT"
fi

# Check for Vertex AI specific environment variables
if [ ! -z "$AIP_PREDICT_ROUTE" ]; then
    echo "Running in Vertex AI environment"
    echo "AIP_PREDICT_ROUTE: $AIP_PREDICT_ROUTE"
    echo "AIP_HEALTH_ROUTE: $AIP_HEALTH_ROUTE"
fi

if [ -z "$HOST" ]; then
    export HOST="0.0.0.0"
    echo "HOST not set, using default: $HOST"
fi

# Display environment variables
echo "Environment variables:"
echo "MODEL_DIR: $MODEL_DIR"
echo "PORT: $PORT"
echo "HOST: $HOST"

# GCS Path handling and verification
if [ ! -z "$MODEL_GCS_PATH" ]; then
    echo "MODEL_GCS_PATH provided: $MODEL_GCS_PATH"
    
    # Fix any spaces in the MODEL_GCS_PATH
    # Remove any leading/trailing whitespace
    MODEL_GCS_PATH=$(echo "$MODEL_GCS_PATH" | xargs)
    echo "Cleaned MODEL_GCS_PATH: $MODEL_GCS_PATH"
    
    # If MODEL_GCS_PATH points to a folder but doesn't end with a slash, add it
    if [[ "$MODEL_GCS_PATH" != */ ]] && [[ "$MODEL_GCS_PATH" != *.pt ]] && [[ "$MODEL_GCS_PATH" != *.pth ]]; then
        MODEL_GCS_PATH="${MODEL_GCS_PATH}/"
        echo "Updated MODEL_GCS_PATH with trailing slash: $MODEL_GCS_PATH"
    fi

    # Verify GCS access
    echo "Verifying GCS access..."
    if gsutil ls $MODEL_GCS_PATH > /dev/null 2>&1; then
        echo "✅ Successfully accessed GCS bucket"
    else
        echo "❌ Failed to access GCS bucket. Check permissions and bucket name"
        echo "Attempted to access: $MODEL_GCS_PATH"
    fi
fi

# Ensure model directory exists and copy model files if needed
mkdir -p $MODEL_DIR
echo "Created model directory: $MODEL_DIR"

# Check specifically for the main model file DINOv2_custom.pth
echo "Looking for main model file: DINOv2_custom.pth"

# Check for model in different locations with priority
if [ ! -f "$MODEL_DIR/DINOv2_custom.pth" ]; then
    echo "Main model file not found in $MODEL_DIR, checking alternative locations..."
    
    # Check for mounted volume at /app/weights
    if [ -d "/app/weights" ] && [ -f "/app/weights/DINOv2_custom.pth" ]; then
        echo "Found main model file in mounted volume, copying from /app/weights to $MODEL_DIR"
        cp -rv /app/weights/* $MODEL_DIR/
    # Check for model in default Docker container location
    elif [ -d "/app/final_model" ] && [ -f "/app/final_model/DINOv2_custom.pth" ]; then
        echo "Copying main model file from /app/final_model to $MODEL_DIR"
        cp -rv /app/final_model/* $MODEL_DIR/
    # Download from Google Cloud Storage if path is provided
    elif [ ! -z "$MODEL_GCS_PATH" ]; then
        echo "Attempting to download model from GCS..."
        
        # Special handling for property-comparison-model bucket
        if [[ "$MODEL_GCS_PATH" == *"property-comparison-model"* ]]; then
            echo "Detected property-comparison-model bucket"
            
            # If path points to the bucket or folder but not specific files
            if [[ "$MODEL_GCS_PATH" == */ ]] || [[ ! "$MODEL_GCS_PATH" == *".pth" ]]; then
                echo "Downloading model files from: ${MODEL_GCS_PATH}final_model/"
                
                # Download main model file first
                echo "Downloading main model file DINOv2_custom.pth..."
                if gsutil cp "${MODEL_GCS_PATH}final_model/DINOv2_custom.pth" "$MODEL_DIR/"; then
                    echo "✅ Successfully downloaded main model file"
                else
                    echo "❌ Failed to download main model file"
                    echo "Command attempted: gsutil cp ${MODEL_GCS_PATH}final_model/DINOv2_custom.pth $MODEL_DIR/"
                    # List contents of the GCS path for debugging
                    echo "Contents of GCS path:"
                    gsutil ls "${MODEL_GCS_PATH}final_model/" || echo "Failed to list directory contents"
                fi
                
                # Download auxiliary files if main model was successful
                if [ -f "$MODEL_DIR/DINOv2_custom.pth" ]; then
                    echo "Downloading auxiliary files..."
                    gsutil cp "${MODEL_GCS_PATH}final_model/model_config.json" "$MODEL_DIR/" && echo "✅ Downloaded model_config.json" || echo "⚠️ Could not download model_config.json"
                    gsutil cp "${MODEL_GCS_PATH}final_model/property_aggregator.pt" "$MODEL_DIR/" && echo "✅ Downloaded property_aggregator.pt" || echo "⚠️ Could not download property_aggregator.pt"
                fi
            else
                # If the path points directly to the model file
                if [[ "$MODEL_GCS_PATH" == *"DINOv2_custom.pth" ]]; then
                    echo "Downloading main model file directly..."
                    if gsutil cp "$MODEL_GCS_PATH" "$MODEL_DIR/"; then
                        echo "✅ Successfully downloaded main model file"
                    else
                        echo "❌ Failed to download model file"
                        echo "Command attempted: gsutil cp $MODEL_GCS_PATH $MODEL_DIR/"
                    fi
                fi
            fi
        fi
    else
        echo "❌ No valid model source found"
    fi
fi

# Final verification of model files
echo "Verifying downloaded model files..."
if [ -f "$MODEL_DIR/DINOv2_custom.pth" ]; then
    echo "✅ Main model file DINOv2_custom.pth is available"
    echo "Model file size: $(ls -lh $MODEL_DIR/DINOv2_custom.pth)"
    echo "Model directory contents:"
    ls -lh $MODEL_DIR/
else
    echo "❌ CRITICAL ERROR: Main model file DINOv2_custom.pth not found!"
    echo "The API will not work without this file."
    echo "Files in model directory ($MODEL_DIR):"
    ls -la $MODEL_DIR
fi

# If RUNPODS_GPU_COUNT is not set, default to 1
if [ -z "$RUNPODS_GPU_COUNT" ]; then
    export RUNPODS_GPU_COUNT=1
fi

echo "GPU Count: $RUNPODS_GPU_COUNT"

# Check if uvicorn is installed and in PATH
if ! command -v uvicorn &> /dev/null; then
    echo "ERROR: uvicorn not found in PATH"
    echo "Attempting to find uvicorn..."
    find / -name uvicorn 2>/dev/null || echo "No uvicorn binary found in the system"
    
    echo "Installing uvicorn as a last resort..."
    pip install uvicorn fastapi
fi

# Start the server with uvicorn
if [ "$ENVIRONMENT" = "development" ]; then
    echo "Starting in development mode..."
    exec uvicorn api.main:app --host $HOST --port $PORT --reload
else
    echo "Starting in production mode..."
    # Determine reasonable number of workers based on CPU cores
    CORES=$(nproc)
    WORKERS=$((CORES > 8 ? 8 : CORES))
    if [ "$WORKERS" -lt 1 ]; then
        WORKERS=1
    fi
    echo "Using $WORKERS workers (system has $CORES cores)"
    
    # Use uvicorn with workers
    exec uvicorn api.main:app \
        --host $HOST \
        --port $PORT \
        --workers $WORKERS \
        --timeout-keep-alive 120
fi 