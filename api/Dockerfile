###############
# BUILDER STAGE
###############
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu20.04 as builder

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=America/New_York
ENV PYTHONUNBUFFERED=1

# Install Python and build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.9 \
    python3.9-dev \
    python3.9-distutils \
    build-essential \
    curl \
    wget \
    git \
    ca-certificates \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Set up Python 3.9 and pip properly
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.9 && \
    python3.9 -m pip install --upgrade pip setuptools wheel

# Create and set working directory
WORKDIR /build

# Copy requirements file
COPY api/requirements.txt ./requirements.txt

# Install dependencies in a single layer to reduce image size
RUN python3.9 -m pip install --no-cache-dir fastapi==0.104.1 uvicorn==0.24.0 httpx>=0.24.0 "pydantic==1.10.12" && \
    python3.9 -m pip install --no-cache-dir torch>=2.2.0 torchvision>=0.17.0 && \
    python3.9 -m pip install --no-cache-dir timm>=0.9.0 && \
    python3.9 -m pip install --no-cache-dir pillow numpy==1.26.1 requests python-multipart && \
    python3.9 -m pip install --no-cache-dir python-dotenv psutil && \
    rm -rf ~/.cache/pip

# Verify PyTorch installation and show package locations
RUN python3.9 -c "import torch; print(f'PyTorch installation location: {torch.__file__}')" && \
    find / -name "torch" -type d 2>/dev/null

##################
# RUNTIME STAGE
##################
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu20.04

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=America/New_York

# Install runtime dependencies and Google Cloud SDK
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.9 \
    python3.9-distutils \
    python3.9-dev \
    curl \
    gnupg \
    lsb-release \
    wget \
    ca-certificates \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list \
    && curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key --keyring /usr/share/keyrings/cloud.google.gpg add - \
    && apt-get update && apt-get install -y --no-install-recommends \
    google-cloud-sdk \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Set up Python 3.9 and pip properly
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.9 && \
    python3.9 -m pip install --upgrade pip setuptools wheel

# Create app directory and necessary subdirectories
WORKDIR /app

# Install PyTorch and other dependencies directly in runtime stage
RUN python3.9 -m pip install --no-cache-dir fastapi==0.104.1 uvicorn==0.24.0 httpx>=0.24.0 pydantic==1.10.12 && \
    python3.9 -m pip install --no-cache-dir torch>=2.2.0 torchvision>=0.17.0 && \
    python3.9 -m pip install --no-cache-dir timm>=0.9.0 && \
    python3.9 -m pip install --no-cache-dir pillow numpy==1.26.1 requests python-multipart && \
    python3.9 -m pip install --no-cache-dir httpx python-dotenv psutil google-cloud-storage google-cloud-logging && \
    rm -rf ~/.cache/pip

# Copy requirements for reference but don't install twice
COPY api/requirements.txt ./requirements.txt

# Create all necessary directories with proper permissions
RUN mkdir -p ./api ./models ./final_model ./weights /root/.config/gcloud && \
    chmod -R 755 /root/.config/gcloud

# Copy only the files needed for inference
COPY models/model_builder.py ./models/model_builder.py
COPY api/__init__.py ./api/__init__.py
COPY api/main.py ./api/main.py
COPY api/services.py ./api/services.py
COPY api/utils.py ./api/utils.py
COPY api/schemas.py ./api/schemas.py

# Create the model directory (model will be downloaded at runtime)
RUN mkdir -p ./final_model

# Install dos2unix for reliable line ending conversion
RUN apt-get update && apt-get install -y dos2unix && rm -rf /var/lib/apt/lists/*

# Copy and fix the startup script
COPY api/start.sh /start.sh
RUN dos2unix /start.sh && chmod +x /start.sh

# Set environment variables
ENV MODEL_DIR=/app/final_model
ENV HOST=0.0.0.0
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0
ENV TMPDIR=/var/tmp
ENV GOOGLE_APPLICATION_CREDENTIALS=/tmp/gcp-key.json
ENV MODEL_GCS_PATH=gs://property-comparison-model/

# Create and set permissions for directories
RUN mkdir -p $TMPDIR && chmod 1777 $TMPDIR && \
    mkdir -p $MODEL_DIR && chmod 777 $MODEL_DIR && \
    mkdir -p /root/.gsutil && chmod 777 /root/.gsutil

# Expose port - Vertex AI will use the PORT env variable
EXPOSE 8080

# Add healthcheck compatible with both Vertex AI and custom environments
HEALTHCHECK --interval=30s --timeout=30s --start-period=10s --retries=3 \
  CMD curl -f http://localhost:${PORT:-8080}/health-check || exit 1

# Verify installations and configurations
RUN echo "Verifying Python version..." && \
    python3.9 --version && \
    echo "Verifying PyTorch installation..." && \
    python3.9 -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')" && \
    echo "Verifying numpy installation..." && \
    python3.9 -c "import numpy; print(f'NumPy version: {numpy.__version__}')" && \
    echo "Verifying tqdm installation..." && \
    python3.9 -c "import tqdm; print(f'tqdm version: {tqdm.__version__}')" && \
    echo "Verifying other key packages..." && \
    python3.9 -c "import timm, fastapi, httpx; print('All key packages loaded successfully')" && \
    echo "Verifying timm installation..." && \
    python3.9 -c "import timm; print(f'timm version: {timm.__version__}')" && \
    echo "Verifying gsutil installation..." && \
    gsutil --version && \
    echo "Verifying GCS Python package..." && \
    python3.9 -c "from google.cloud import storage; print('GCS client library available')" && \
    echo "Verifying directory permissions..." && \
    ls -la $MODEL_DIR && \
    ls -la /root/.config/gcloud

# Run the application
CMD ["/start.sh"] 