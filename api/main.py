"""
FastAPI application for property comparison
"""

import os
import logging
import json
import sys
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, HTTPException, BackgroundTasks, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import time

from schemas import PropertyComparisonRequest, ComparisonResult
from services import ModelService
from utils import clear_cache, setup_json_logging, logger

# Setup JSON logging for Google Cloud Logging
setup_json_logging()

# Create FastAPI application
app = FastAPI(
    title="Property Comparison API",
    description="API for comparing real estate properties using a Siamese neural network",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update with specific origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize model service
model_service = None


@app.on_event("startup")
async def startup_event():
    """Initialize the model service on startup"""
    global model_service
    
    # Set model directory from environment variable or use default
    model_dir = os.getenv("MODEL_DIR", None)
    model_gcs_path = os.getenv("MODEL_GCS_PATH", None)
    
    # Initialize model service
    logger.info("Initializing model service...")
    try:
        model_service = ModelService(
            model_dir=model_dir, 
            model_gcs_path=model_gcs_path
        )
        logger.info("Model service initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing model service: {str(e)}")
        # Continue without model service - will return error responses
        model_service = None


@app.on_event("shutdown")
def shutdown_event():
    """Clean up resources on shutdown"""
    logger.info("Clearing cache on shutdown")
    clear_cache()


@app.get("/health-check")
async def health_check():
    """Health check endpoint for Vertex AI"""
    return {"status": "healthy", "model_loaded": model_service is not None}


@app.get("/")
async def root():
    """Root endpoint for Vertex AI health checks"""
    return {"status": "healthy", "model_loaded": model_service is not None}


@app.get("/health")
async def legacy_health_check():
    """Legacy health check endpoint"""
    return {"status": "healthy", "model_loaded": model_service is not None}


@app.post("/api/compare-properties", response_model=ComparisonResult)
async def compare_properties(
    request: PropertyComparisonRequest,
    background_tasks: BackgroundTasks
):
    """
    Compare a subject property against multiple comp properties.
    
    Uses the trained Siamese network to generate similarity scores.
    """
    # Check if model service is initialized
    if model_service is None:
        raise HTTPException(
            status_code=503,
            detail="Model service not initialized. Please try again later."
        )
    
    # Log request summary
    logger.info(f"Processing comparison request: 1 subject property with {len(request.comps)} comp properties")
    
    try:
        # Extract subject property photos - convert URLs to strings
        subject_photos = [str(photo.url) for photo in request.subject_property.photos]
        
        # Process comps - convert URLs to strings
        comp_properties = []
        for comp in request.comps:
            comp_properties.append({
                "uid": comp.uid,
                "photos": [{"url": str(photo.url)} for photo in comp.photos],
                "address": comp.address
            })
        
        # Run comparison
        result = await model_service.compare_properties(
            subject_photos=subject_photos,
            comp_properties=comp_properties,
            threshold=request.threshold,
            max_comps=request.max_comps
        )
        
        # Schedule cache cleanup after response
        background_tasks.add_task(clear_cache)
        
        return result
    
    except Exception as e:
        logger.error(f"Error processing comparison request: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing comparison request: {str(e)}"
        )


@app.get("/api/docs")
async def get_documentation():
    """Return API documentation"""
    return {
        "endpoints": [
            {
                "path": "/",
                "method": "GET",
                "description": "Root endpoint for Vertex AI health checks"
            },
            {
                "path": "/health-check",
                "method": "GET",
                "description": "Health check endpoint for Vertex AI"
            },
            {
                "path": "/api/compare-properties",
                "method": "POST",
                "description": "Compare a subject property against multiple comp properties",
                "request_format": "See /docs or /redoc for detailed schema"
            },
            {
                "path": "/docs",
                "method": "GET",
                "description": "Swagger UI documentation"
            },
            {
                "path": "/redoc",
                "method": "GET",
                "description": "ReDoc documentation"
            }
        ]
    }


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Middleware to log requests in a format suitable for Cloud Logging"""
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    
    log_dict = {
        "request_path": request.url.path,
        "request_method": request.method,
        "process_time_ms": process_time * 1000,
        "status_code": response.status_code,
    }
    
    logger.info("Request processed", extra=log_dict)
    
    return response


if __name__ == "__main__":
    # Run server if called directly
    # Use PORT environment variable provided by Vertex AI
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")
    
    # Configure uvicorn
    uvicorn.run(
        "main:app",  # Use relative import when running directly from this file
        host=host,
        port=port,
        log_level="info"
    ) 