from pydantic import BaseModel, Field
from typing import List, Optional

# --- Models for the Initial Analysis Endpoint ---

class ClinicalAssessment(BaseModel):
    """Defines the clinical data that can be sent in the request."""
    diabetes: bool = False
    peripheral_arterial_disease: bool = False
    autoimmune_disorder: bool = False
    malnutrition: bool = False
    ambulatory: bool = False
    wheelchair_dependent: bool = False
    bedbound: bool = False
    caregiver_support: bool = False
    drainage_heavy: bool = False
    sanguinous: bool = False
    purulent: bool = False
    odor_foul: bool = False
    temperature_warmer_hot: bool = False
    other_information: str = ""

# --- Models for the Expand Treatment Plan Endpoint ---

class ExpandTreatmentRequest(BaseModel):
    """
    Defines the request body for the treatment plan expansion.
    It requires only the full raw text from the initial analysis.
    """
    original_analysis: str = Field(..., description="The full, raw text output from the initial analysis.")

# --- Models for the Revise Products Endpoint ---

class ReviseProductsRequest(BaseModel):
    """Defines the request body for revising products."""
    original_analysis: str = Field(..., description="The full, raw text output from the initial analysis.")
    revision_reason: str = Field(..., description="The reason for revision, e.g., 'Too Costly'.")

