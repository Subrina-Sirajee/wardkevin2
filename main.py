# main.py - MODIFIED to run all functions

import json
from client_factory import get_ai_client
from data_formatter import ClinicalDataFormatter
from response_parser import AIResponseParser
from datetime import datetime

class NurseLensFacade:
    def __init__(self, model_name: str):
        """Initializes the coordinator."""
        self.client = get_ai_client(model_name)
        self.formatter = ClinicalDataFormatter()
        self.parser = AIResponseParser()
        self.last_analysis = None
        self.last_assessment_data = None

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

    # --- These are the follow-up action methods ---
    def expand_last_treatment_plan(self, wound_location: str) -> dict:
        """
        Expands the treatment plan and returns a full, structured API response.
        """
        if not self.last_analysis or not self.last_assessment_data:
            return {"success": False, "error": "You must run 'analyze_wound_with_image' first."}
        
        print("\nDEBUG: Calling client to expand treatment plan...")
        # Call the client to get the core JSON data
        result = self.client.expand_treatment_plan(
            self.last_analysis,
            self.last_assessment_data,
            wound_location
        )

        # --- THIS IS THE NEW PART ---
        # If the client call fails, return the error immediately.
        if not result["success"]:
            return result # Pass the error dictionary straight through

        # If successful, build the full, consistent response object.
        return {
            "success": True,
            "timestamp": datetime.now().isoformat(),
            "model_used": self.client.model,
            # The 'json_response' key now holds the specific result from this function.
            "json_response": result["expanded_plan_json"]
        }
        # ----------------------------

    def revise_last_products(self, revision_reason: str) -> dict:
        """
        Revises products and returns a full, structured API response.
        """
        if not self.last_analysis:
            return {"success": False, "error": "You must run 'analyze_wound_with_image' first."}
        
        if revision_reason not in ["Patient Won't Tolerate", "Too Costly", "Products Unavailable", "Other"]:
            return {"success": False, "error": "Invalid revision reason."}
            
        print(f"\nDEBUG: Calling client to revise products (Reason: {revision_reason})...")
        # Call the client to get the core JSON data
        result = self.client.revise_products(self.last_analysis, revision_reason)

        # --- THIS IS THE NEW PART ---
        # If the client call fails, return the error immediately.
        if not result["success"]:
            return result

        # If successful, build the full, consistent response object.
        return {
            "success": True,
            "timestamp": datetime.now().isoformat(),
            "model_used": self.client.model,
            # The 'json_response' key now holds the specific result from this function.
            "json_response": result["revised_products_json"]
        }

# --- Main Execution Block - MODIFIED to handle consistent JSON responses from all steps ---
if __name__ == "__main__":
    # --- STEP 1: INITIAL ANALYSIS ---
    print("="*30 + " STEP 1: INITIAL ANALYSIS " + "="*30)
    model_choice = "gpt-4o"
    ai_system = NurseLensFacade(model_name=model_choice)
    
    initial_result = ai_system.analyze_wound_with_image(
        image_path="wound2.jpg",
        wound_location="Lower Left Leg/Shin",
        diabetes=True,
        other_information="Chronic wound with signs of infection."
    )

    if not initial_result["success"]:
        print(f"\n--- ANALYSIS FAILED ---")
        print(f"Error: {initial_result['error']}")
    else:
        print(f"\nANALYSIS FROM {initial_result['model_used']} SUCCESSFUL!")
        # The core data is in the 'json_response' key
        final_json = initial_result['json_response']
        
        # Print the initial report
        print("\n" + "-"*25 + " INITIAL AI REPORT " + "-"*25)
        # for section, content in final_json.items():
        #     print(f"\n**{section}:**")
        #     print(content if content else "-> Not found.")
        # print("\n" + "-"*67)
        print(final_json)

        # --- STEP 2: EXPAND TREATMENT PLAN ---
        print("\n" + "="*25 + " STEP 2: EXPAND TREATMENT PLAN " + "="*25)
        expanded_result = ai_system.expand_last_treatment_plan(wound_location="Lower Left Leg/Shin")
        
        # --- THIS IS THE CORRECTED PART ---
        if expanded_result["success"]:
            print("\n--- EXPANDED TREATMENT PLAN (JSON) ---")
            # Access the data from the 'json_response' key and pretty-print it
            expanded_json = expanded_result['json_response']
            print(json.dumps(expanded_json, indent=2))
        else:
            print(f"\n--- FAILED TO EXPAND PLAN ---")
            print(f"Error: {expanded_result.get('error', 'Unknown error')}")
        # ------------------------------------

        # --- STEP 3: REVISE PRODUCTS ---
        print("\n" + "="*28 + " STEP 3: REVISE PRODUCTS " + "="*28)
        revision_reason = "Too Costly"
        print(f"Reason for revision: {revision_reason}")
        
        revised_result = ai_system.revise_last_products(revision_reason=revision_reason)
        
        # --- THIS IS ALSO THE CORRECTED PART ---
        if revised_result["success"]:
            print("\n--- REVISED PRODUCTS LIST (JSON) ---")
            # Access the data from the 'json_response' key and pretty-print it
            revised_json = revised_result['json_response']
            print(json.dumps(revised_json, indent=2))
        else:
            print(f"\n--- FAILED TO REVISE PRODUCTS ---")
            print(f"Error: {revised_result.get('error', 'Unknown error')}")
        # ---------------------------------------

