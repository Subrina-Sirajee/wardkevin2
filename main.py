import json
from client_factory import get_ai_client
from data_formatter import ClinicalDataFormatter
from response_parser import AIResponseParser
from datetime import datetime
from pdf_generator import create_healing_history_pdf
import os

class NurseLensFacade:
    def __init__(self, model_name: str):
        """Initializes the coordinator."""
        self.client = get_ai_client(model_name)
        self.formatter = ClinicalDataFormatter()
        self.parser = AIResponseParser()
        self.last_analysis = None
        self.last_assessment_data = None

    def calculate_healing_progress(self, patient_id: str, history_records: list) -> dict:
        """
        Orchestrates the healing progress calculation based on a provided history.
        This method is now stateless and relies on the backend to provide the data.
        """
        print(f"\nDEBUG: Starting healing progress calculation for patient {patient_id}...")

        # Handle the "First Assessment" case
        if len(history_records) < 2:
            print("DEBUG: Less than 2 assessments found. Healing progress is 0%.")
            # Return a consistent JSON response structure
            return {
                "success": True,
                "timestamp": datetime.now().isoformat(),
                "model_used": self.client.model,
                "pdf_path": None,  # No PDF generated for single record
                "json_response": {"healing_progress_percentage": 0}
            }
        
        pdf_path = None  # Initialize to ensure it exists
        try:
            # Generate the multi-page PDF on the fly from the provided history
            pdf_path = create_healing_history_pdf(history_records, patient_id)

            # Call the AI client to analyze the PDF
            result = self.client.get_healing_progress(pdf_path)

            # --- PDF IS NOT DELETED ---
            # The os.remove(pdf_path) line has been removed as you requested.
            # You can find the generated PDF in the "generated_pdfs" folder.
            
            if not result["success"]:
                return result

            # Build the final, consistent API response
            return {
                "success": True,
                "timestamp": datetime.now().isoformat(),
                "model_used": self.client.model,
                "pdf_path": pdf_path,  # Optionally return the path for reference
                "json_response": result["healing_progress_json"]
            }
        except Exception as e:
            # If an error occurs, we still have the pdf_path if it was created
            error_response = {
                "success": False,
                "timestamp": datetime.now().isoformat(),  # Added for consistency
                "model_used": self.client.model,  # Added for consistency
                "error": f"Healing progress workflow failed: {str(e)}"
            }
            if pdf_path:
                error_response["pdf_path"] = pdf_path
            return error_response

    def analyze_wound_with_image(self, image_path: str, wound_location: str = "Right Arm", **assessment_params) -> dict:
        """Orchestrates the end-to-end wound analysis process using a single, powerful API call."""
        try:
            assessment_data = self.formatter.format_assessment_data(**assessment_params)
            current_date = datetime.now().strftime("%d/%m/%Y")
            main_prompt = self.formatter.create_main_analysis_prompt(assessment_data, wound_location, current_date)
            base64_image = self.client.encode_image(image_path)
            
            print("DEBUG: Making a single, comprehensive API call for all sections...")
            complete_ai_analysis = self.client.get_initial_analysis(main_prompt, base64_image)

            self.last_analysis = complete_ai_analysis
            self.last_assessment_data = assessment_data

            response_json = self.parser.parse_response_to_json(complete_ai_analysis)

            return {
                "success": True, "timestamp": datetime.now().isoformat(),
                "model_used": self.client.model, "assessment_data": assessment_data,
                "json_response": response_json,
            }
        except Exception as e:
            self.last_analysis = None
            self.last_assessment_data = None
            return {"success": False, "error": f"Workflow error: {str(e)}"}

    def expand_last_treatment_plan(self) -> dict:
        """
        Expands the treatment plan and returns a full, structured API response.
        """
        if not self.last_analysis:
            return {"success": False, "error": "You must run 'analyze_wound_with_image' first."}
        
        print("\nDEBUG: Calling client to expand treatment plan...")
        result = self.client.expand_treatment_plan(self.last_analysis)

        if not result["success"]:
            return result

        return {
            "success": True,
            "timestamp": datetime.now().isoformat(),
            "model_used": self.client.model,
            "json_response": result["expanded_plan_json"]
        }

    def revise_last_products(self, revision_reason: str) -> dict:
        """
        Revises products and returns a full, structured API response.
        """
        if not self.last_analysis:
            return {"success": False, "error": "You must run 'analyze_wound_with_image' first."}
        
        if revision_reason not in ["Patient Won't Tolerate", "Too Costly", "Products Unavailable", "Other"]:
            return {"success": False, "error": "Invalid revision reason."}
            
        print(f"\nDEBUG: Calling client to revise products (Reason: {revision_reason})...")
        result = self.client.revise_products(self.last_analysis, revision_reason)

        if not result["success"]:
            return result

        return {
            "success": True,
            "timestamp": datetime.now().isoformat(),
            "model_used": self.client.model,
            "json_response": result["revised_products_json"]
        }

if __name__ == "__main__":
    # --- SIMULATION SETUP ---
    patient_id = "hand_wound_case_001"
    # model_choice = "grok-4"  # Or "gemini-1.5-pro-latest" if you prefer
    # model_choice = "gpt-4o"  # Or "gemini-1.5-pro-latest" if you prefer
    model_choice = "gemini-flash-latest"  # Or "gemini-1.5-pro-latest" if you prefer
    ai_system = NurseLensFacade(model_name=model_choice)
    
    simulated_backend_database = []

    # --- STEP 1: FIRST ASSESSMENT (DAY 0 - Open Wound) ---
    print("\n" + "="*20 + " ASSESSMENT 1 (DAY 0) " + "="*20)
    initial_result = ai_system.analyze_wound_with_image(
        image_path="wound_1.png",
        wound_location="Volar Wrist/Palm",
        other_information="Initial assessment of an open abrasion on the hand."
    )
    if initial_result["success"]:
        record1 = {
            "image_path": "wound_1.png",
            "assessment_date": "2025-10-01 09:00:00",
            "analysis": initial_result["json_response"]
        }
        simulated_backend_database.append(record1)
        print("Assessment 1 successful. Data stored in backend.")
    else:
        print(f"Assessment 1 failed: {initial_result['error']}")

    # --- STEP 2: SECOND ASSESSMENT (DAY 10 - Stitches) ---
    print("\n" + "="*20 + " ASSESSMENT 2 (DAY 10) " + "="*20)
    reassessment_1 = ai_system.analyze_wound_with_image(
        image_path="wound_2.png",
        wound_location="Volar Wrist/Palm",
        other_information="Reassessment after primary closure with sutures. Wound edges are approximated."
    )
    if reassessment_1["success"]:
        record2 = {
            "image_path": "wound_2.png",
            "assessment_date": "2025-10-11 09:30:00", # 10 days later
            "analysis": reassessment_1["json_response"]
        }
        simulated_backend_database.append(record2)
        print("Assessment 2 successful. Data stored in backend.")
    else:
        print(f"Assessment 2 failed: {reassessment_1['error']}")

    # --- STEP 3: THIRD ASSESSMENT (DAY 30 - Healed Scar) ---
    print("\n" + "="*20 + " ASSESSMENT 3 (DAY 30) " + "="*20)
    reassessment_2 = ai_system.analyze_wound_with_image(
        image_path="wound_3.png",
        wound_location="Volar Wrist/Palm",
        other_information="Final follow-up. Sutures removed, wound is fully epithelialized, leaving a mature scar."
    )
    if reassessment_2["success"]:
        record3 = {
            "image_path": "wound_3.png",
            "assessment_date": "2025-10-31 10:00:00", # 20 days after the second assessment
            "analysis": reassessment_2["json_response"]
        }
        simulated_backend_database.append(record3)
        print("Assessment 3 successful. Data stored in backend.")
    else:
        print(f"Assessment 3 failed: {reassessment_2['error']}")

    # --- STEP 2: CALCULATING HEALING PROGRESS ---
    print("\n" + "="*20 + " CALCULATING HEALING PROGRESS " + "="*20)
    
    progress_result = ai_system.calculate_healing_progress(
        patient_id=patient_id, 
        history_records=simulated_backend_database
    )

    if progress_result["success"]:
        print("\n--- HEALING PROGRESS RESULT (JSON) ---")
        progress_json = progress_result['json_response']
        print(json.dumps(progress_json, indent=2))
        percentage = progress_json.get("healing_progress_percentage", "N/A")
        print(f"\n>>>>>> Overall Healing Progress: {percentage}% <<<<<<")
        if progress_result.get('pdf_path'):
            print(f"Review the generated PDF at: {progress_result.get('pdf_path')}")
    else:
        print(f"\n--- FAILED TO CALCULATE HEALING PROGRESS ---")
        print(f"Error: {progress_result.get('error', 'Unknown error')}")
