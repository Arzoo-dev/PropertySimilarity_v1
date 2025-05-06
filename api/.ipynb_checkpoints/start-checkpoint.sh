#!/bin/bash

# Display NVIDIA GPU information
echo "NVIDIA GPU Information:"
nvidia-smi

# Set default values for environment variables if not set
if [ -z "$MODEL_DIR" ]; then
    export MODEL_DIR="workspace/final_model/"
    echo "MODEL_DIR not set, using default: $MODEL_DIR"
fi

if [ -z "$PORT" ]; then
    export PORT="69"
    echo "PORT not set, using default: $PORT"
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

# If RUNPODS_GPU_COUNT is not set, default to 1
if [ -z "$RUNPODS_GPU_COUNT" ]; then
    export RUNPODS_GPU_COUNT=1
fi

echo "GPU Count: $RUNPODS_GPU_COUNT"

# Start the server with uvicorn
if [ "$ENVIRONMENT" = "development" ]; then
    echo "Starting in development mode..."
    exec uvicorn main:app --host $HOST --port $PORT --reload
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
    exec uvicorn main:app \
        --host $HOST \
        --port $PORT \
        --workers $WORKERS \
        --timeout-keep-alive 120
fi 