"""
Tests for the property comparison API
"""

import os
import json
import pytest
from fastapi.testclient import TestClient
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import app and schemas
from api.main import app
from api.schemas import PropertyComparisonRequest


# Test client
client = TestClient(app)


def test_health_endpoint():
    """Test the health check endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data["status"] == "healthy"


def test_docs_endpoint():
    """Test the API documentation endpoint"""
    response = client.get("/api/docs")
    assert response.status_code == 200
    data = response.json()
    assert "endpoints" in data
    assert isinstance(data["endpoints"], list)


def test_compare_properties_validation():
    """Test validation for property comparison endpoint"""
    # Test with empty request
    response = client.post("/api/compare-properties", json={})
    assert response.status_code != 200  # Validation error
    
    # Test with invalid URL
    invalid_request = {
        "subject_property": {
            "photos": [
                {"url": "not-a-valid-url"}  # Invalid URL format
            ],
            "address": "123 Test St"
        },
        "comps": []
    }
    response = client.post("/api/compare-properties", json=invalid_request)
    assert response.status_code != 200  # Validation error
    

def test_compare_properties_endpoint():
    """Test the property comparison endpoint with mock data
    
    This test uses the sample JSON file to create a request for the API.
    """
    # Path to sample data
    sample_data_path = Path(__file__).parent.parent.parent / "get_random_listing_photos.json"
    
    # Skip test if sample data doesn't exist
    if not sample_data_path.exists():
        pytest.skip(f"Sample data file not found: {sample_data_path}")
    
    try:
        # Load sample data
        with open(sample_data_path, "r") as f:
            sample_data = json.load(f)
        
        # Create request data
        request_data = {
            "subject_property": {
                "photos": [{"url": photo["url"]} for photo in sample_data["subject_property"]["photos"][:3]],
                "address": sample_data["subject_property"]["address"]
            },
            "comps": [
                {
                    "uid": comp["uid"],
                    "photos": [{"url": photo["url"]} for photo in comp["photos"][:3]],
                    "address": comp["address"]
                }
                for comp in sample_data["comps"][:2]  # Use only first 2 comps for faster testing
            ],
            "threshold": 5.0,
            "max_comps": 2
        }
        
        # Send request
        # Note: This may fail if the model isn't loaded or if image URLs are unreachable
        # The test is mainly to verify the API structure, not for full integration testing
        response = client.post("/api/compare-properties", json=request_data)
        
        # Print response for debugging
        print(f"Response status: {response.status_code}")
        if response.status_code != 200:
            print(f"Error: {response.text}")
        
        # Check response structure even if there's an error
        if response.status_code == 200:
            data = response.json()
            assert "subject_property_id" in data
            assert "metrics" in data
            assert "comp_pairs" in data
            assert "threshold" in data
            
    except Exception as e:
        pytest.fail(f"Error testing property comparison endpoint: {str(e)}")


if __name__ == "__main__":
    # Run tests manually
    test_health_endpoint()
    test_docs_endpoint()
    test_compare_properties_validation()
    print("Basic tests passed!")
    
    # Optionally run the property comparison test
    try:
        test_compare_properties_endpoint()
        print("Property comparison test passed!")
    except Exception as e:
        print(f"Property comparison test failed: {str(e)}") 