#!/bin/bash

# Display NVIDIA GPU information
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
if [ ! -z "$MODEL_GCS_PATH" ]; then
    echo "MODEL_GCS_PATH: $MODEL_GCS_PATH"
fi

# Ensure model directory exists and copy model files if needed
mkdir -p $MODEL_DIR

# Check for model in different locations with priority
if [ ! -f "$MODEL_DIR/siamese_embedding_model.pt" ]; then
    # Check for mounted volume at /app/weights
    if [ -d "/app/weights" ] && [ -f "/app/weights/siamese_embedding_model.pt" ]; then
        echo "Found model in mounted volume, copying from /app/weights to $MODEL_DIR"
        cp -r /app/weights/* $MODEL_DIR/
    # Check for model in default Docker container location
    elif [ -d "/app/final_model" ] && [ -f "/app/final_model/siamese_embedding_model.pt" ]; then
    echo "Copying model files from /app/final_model to $MODEL_DIR"
    cp -r /app/final_model/* $MODEL_DIR/
    # Download from Google Cloud Storage if path is provided
    elif [ ! -z "$MODEL_GCS_PATH" ]; then
        echo "Downloading model from GCS: $MODEL_GCS_PATH"
        # Check if it's a directory or file path
        if [[ "$MODEL_GCS_PATH" == */ ]]; then
            # It's a directory, copy all contents
            gsutil -m cp -r "${MODEL_GCS_PATH}*" $MODEL_DIR/
        else
            # It's a file, copy just that file
            gsutil cp $MODEL_GCS_PATH $MODEL_DIR/
            # If it's a compressed file, extract it
            if [[ "$MODEL_GCS_PATH" == *.tar.gz ]] || [[ "$MODEL_GCS_PATH" == *.tgz ]]; then
                echo "Extracting downloaded archive..."
                tar -xzf $MODEL_DIR/$(basename $MODEL_GCS_PATH) -C $MODEL_DIR
            elif [[ "$MODEL_GCS_PATH" == *.zip ]]; then
                echo "Extracting downloaded zip..."
                apt-get update && apt-get install -y unzip && \
                apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
                unzip $MODEL_DIR/$(basename $MODEL_GCS_PATH) -d $MODEL_DIR
                # Remove the zip file after extraction to save space
                rm -f $MODEL_DIR/$(basename $MODEL_GCS_PATH)
            fi
        fi
    else
        echo "WARNING: No model files found. The API may not work correctly!"
    fi
fi

# List files in model directory to verify
echo "Files in model directory ($MODEL_DIR):"
ls -la $MODEL_DIR

# If RUNPODS_GPU_COUNT is not set, default to 1
if [ -z "$RUNPODS_GPU_COUNT" ]; then
    export RUNPODS_GPU_COUNT=1
fi

echo "GPU Count: $RUNPODS_GPU_COUNT"

# Start the server with uvicorn
if [ "$ENVIRONMENT" = "development" ]; then
    echo "Starting in development mode..."
    exec uvicorn api.main:app --host $HOST --port $PORT --reload
else
    echo "Starting in production mode..."
    # Determine reasonable number of workers based on CPU cores
    # Use a maximum of 8 workers to avoid too many workers
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