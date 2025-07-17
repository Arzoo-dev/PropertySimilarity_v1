"""
Pydantic models for API request and response validation
"""

from typing import List, Dict, Optional, Any
from pydantic import BaseModel, HttpUrl, Field


class Photo(BaseModel):
    """Photo model with URL"""
    url: HttpUrl = Field(..., description="URL of the property photo")


class PropertyInput(BaseModel):
    """Property input model without UID"""
    photos: List[Photo] = Field(..., description="List of property photos")
    address: str = Field(..., description="Property address")


class CompProperty(BaseModel):
    """Comparable property model with UID"""
    uid: str = Field(..., description="Unique identifier for the property")
    photos: List[Photo] = Field(..., description="List of property photos")
    address: str = Field(..., description="Property address")


class PropertyComparisonRequest(BaseModel):
    """Request model for property comparison"""
    subject_property: PropertyInput = Field(..., description="Subject property to compare against comps")
    comps: List[CompProperty] = Field(..., description="List of comparable properties")
    threshold: Optional[float] = Field(7.5, description="Similarity threshold (0-10)")
    max_comps: Optional[int] = Field(None, description="Maximum number of comps to process")


class ComparisonPair(BaseModel):
    """Single property comparison result"""
    pair_id: int = Field(..., description="Unique identifier for the comparison pair")
    subject_property_id: str = Field("subject", description="Subject property identifier")
    comp_property_id: str = Field(..., description="Comparable property identifier")
    subject_images: int = Field(..., description="Number of subject property images")
    comp_images: int = Field(..., description="Number of comparable property images")
    true_label: str = Field(..., description="True label (similar/dissimilar)")
    predicted_label: str = Field(..., description="Predicted label (similar/dissimilar)")
    similarity_score: float = Field(..., description="Similarity score (0-10)")
    correct_prediction: bool = Field(..., description="Whether prediction matches true label")
    address: Optional[str] = Field(None, description="Comparable property address")


class Metrics(BaseModel):
    """Metrics for model performance"""
    num_test_pairs: int = Field(..., description="Number of comparison pairs")
    accuracy: float = Field(..., description="Accuracy of predictions")
    precision: float = Field(..., description="Precision of positive predictions")
    recall: float = Field(..., description="Recall of positive predictions")
    f1_score: float = Field(..., description="F1 score (harmonic mean of precision and recall)")
    tp: int = Field(..., description="True positives count")
    fp: int = Field(..., description="False positives count")
    tn: int = Field(..., description="True negatives count")
    fn: int = Field(..., description="False negatives count")
    avg_similar_score: float = Field(..., description="Average similarity score for similar pairs")
    avg_dissimilar_score: float = Field(..., description="Average similarity score for dissimilar pairs")


class ComparisonResult(BaseModel):
    """Response model for property comparison results"""
    subject_property_id: str = Field("subject", description="Subject property identifier")
    metrics: Metrics = Field(..., description="Comparison metrics")
    comp_pairs: List[ComparisonPair] = Field(..., description="List of comparison results")
    threshold: float = Field(..., description="Similarity threshold used (0-10)") 