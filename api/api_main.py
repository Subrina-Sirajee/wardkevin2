from fastapi import FastAPI, File, UploadFile, Form, Depends, HTTPException
from fastapi.responses import JSONResponse
import shutil
import os
from typing import Dict

# Import our schemas and the main facade
from . import schemas
from main import NurseLensFacade

# --- App Initialization ---
app = FastAPI(
    title="NurseLens AI API",
    description="API for advanced wound analysis using AI.",
    version="1.0.0"
)

# --- Dependency Injection for the Facade ---
# This is a smart way to manage the facade instance.
# We can easily switch models here, e.g., by getting the model from a header or query param.
def get_ai_system() -> NurseLensFacade:
    # For now, we hardcode gpt-4o, but this could be made dynamic.
    return NurseLensFacade(model_name="gpt-4o")

# --- API Endpoints ---

@app.post("/analysis/initial", tags=["Analysis"])
async def create_initial_analysis(
    # We use Depends() to get the clinical data from the form
    assessment_data: schemas.ClinicalAssessment = Depends(),
    # The image is sent as a file upload
    image: UploadFile = File(..., description="The wound image file to be analyzed."),
    # The wound location is also a form field
    wound_location: str = Form("Right Arm", description="The location of the wound."),
    # Get an instance of our facade using dependency injection
    ai_system: NurseLensFacade = Depends(get_ai_system)
):
    """
    Performs the initial, comprehensive wound analysis from an image and clinical data.
    
    This endpoint accepts multipart/form-data. You must send the image as a file
    and the other parameters as form fields.
    """
    # Create a temporary directory to save the uploaded image
    temp_dir = "temp_images"
    os.makedirs(temp_dir, exist_ok=True)
    temp_image_path = os.path.join(temp_dir, image.filename)

    try:
        # Save the uploaded file to the temporary path
        with open(temp_image_path, "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)

        # Convert the Pydantic model to a dictionary for the facade
        assessment_params = assessment_data.model_dump()

        # Call our existing facade method with the saved image path and data
        result = ai_system.analyze_wound_with_image(
            image_path=temp_image_path,
            wound_location=wound_location,
            **assessment_params
        )

        if not result["success"]:
            raise HTTPException(status_code=500, detail=result.get("error", "An unknown error occurred during analysis."))
        
        return result

    finally:
        # Clean up: always remove the temporary image file
        if os.path.exists(temp_image_path):
            os.remove(temp_image_path)


@app.post("/analysis/expand-treatment-plan", tags=["Follow-up Actions"])
async def expand_treatment_plan(
    request: schemas.ExpandTreatmentRequest,
    ai_system: NurseLensFacade = Depends(get_ai_system)
):
    """
    Expands the treatment plan from a previous analysis.
    
    You must provide the raw text (`original_analysis`) from the response of the `/analysis/initial` endpoint.
    """
    # The facade needs the raw text analysis to be stored first
    ai_system.last_analysis = request.original_analysis
    
    result = ai_system.expand_last_treatment_plan()

    if not result["success"]:
        raise HTTPException(status_code=500, detail=result.get("error", "Failed to expand treatment plan."))
    
    return result


@app.post("/analysis/revise-products", tags=["Follow-up Actions"])
async def revise_products(
    request: schemas.ReviseProductsRequest,
    ai_system: NurseLensFacade = Depends(get_ai_system)
):
    """
    Revises the product recommendations from a previous analysis based on a constraint.
    
    You must provide the raw text (`original_analysis`) from the response of the
    `/analysis/initial` endpoint.
    """
    # Store the necessary context on the facade instance
    ai_system.last_analysis = request.original_analysis
    
    result = ai_system.revise_last_products(revision_reason=request.revision_reason)

    if not result["success"]:
        raise HTTPException(status_code=500, detail=result.get("error", "Failed to revise products."))
        
    return result

