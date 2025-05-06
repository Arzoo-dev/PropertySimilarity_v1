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
    python3.9 \
    python3.9-dev \
    python3-pip \
    build-essential \
    curl \
    wget \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Set Python aliases
RUN ln -sf /usr/bin/python3.9 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip

# Create and set working directory
WORKDIR /build

# Copy requirements file
COPY RunPodsModel/api/requirements.txt ./requirements.txt

# Install dependencies in a single layer to reduce image size
RUN pip install --no-cache-dir google-cloud-storage google-cloud-logging && \
    pip install --no-cache-dir --default-timeout=900 torch==2.1.0+cu118 torchvision==0.16.0+cu118 --extra-index-url https://download.pytorch.org/whl/cu118 && \
    pip install --no-cache-dir -r requirements.txt && \
    rm -rf ~/.cache/pip

##################
# RUNTIME STAGE
##################
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu20.04

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=America/New_York

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.9 \
    python3-pip \
    curl \
    gnupg \
    lsb-release \
    wget \
    && echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list \
    && curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key --keyring /usr/share/keyrings/cloud.google.gpg add - \
    && apt-get update && apt-get install -y --no-install-recommends \
    google-cloud-sdk \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Set Python aliases
RUN ln -sf /usr/bin/python3.9 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

# Create app directory
WORKDIR /app

# Copy Python packages from builder stage
COPY --from=builder /usr/local/lib/python3.9/dist-packages /usr/local/lib/python3.9/dist-packages
COPY --from=builder /usr/lib/python3/dist-packages /usr/lib/python3/dist-packages

# Create model directories
RUN mkdir -p ./api ./final_model ./weights

# Copy only the files needed for inference
COPY RunPodsModel/siamese_network.py ./
COPY RunPodsModel/api/__init__.py ./api/__init__.py
COPY RunPodsModel/api/main.py ./api/main.py
COPY RunPodsModel/api/services.py ./api/services.py
COPY RunPodsModel/api/utils.py ./api/utils.py
COPY RunPodsModel/api/schemas.py ./api/schemas.py

# Copy and make executable the startup script
COPY RunPodsModel/api/start.sh /start.sh
RUN chmod +x /start.sh

# Set environment variables
ENV MODEL_DIR=/app/final_model
ENV PORT=69
ENV HOST=0.0.0.0
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0
ENV TMPDIR=/var/tmp

# Create and set permissions for tmp directory
RUN mkdir -p $TMPDIR && chmod 1777 $TMPDIR

# Expose the port
EXPOSE 69

# Add healthcheck
HEALTHCHECK --interval=30s --timeout=30s --start-period=10s --retries=3 \
  CMD curl -f http://localhost:${PORT}/health-check || exit 1

# Run the application
CMD ["/start.sh"] 