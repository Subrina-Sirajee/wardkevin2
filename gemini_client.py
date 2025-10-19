import os
import re
import json
import base64
import google.generativeai as genai
from dotenv import load_dotenv
from ai_client_interface import AIClientInterface
from typing import Dict, List

class GeminiClient(AIClientInterface):
    """The concrete implementation of the AIClientInterface for Google's Gemini models."""
    
    def __init__(self, api_key: str = None, model: str = "gemini-1.5-pro-latest"):
        load_dotenv()
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables.")
        
        genai.configure(api_key=self.api_key)
        
        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
        ]
        
        self.model_name = model
        self.model = genai.GenerativeModel(self.model_name, safety_settings=safety_settings)
        
        # --- THIS IS THE NEW, MORE FORCEFUL PROMPT ---
        self.clinical_protocol = """
        You are a world-class dermatologist AI. Your task is to analyze the provided wound image and clinical data.
        Your response MUST be a single block of text.
        You MUST use the following seven section headers and NOTHING ELSE:
        **Case Information:**
        **Clinical Observations:**
        **Wound Summary:**
        **Tissue Percentages Over Time:**
        **Treatment Plan:**
        **Recommended Products:**
        **Wound Tissue Evaluation:**

        DO NOT create your own headers like 'Health Risk Factors' or 'Overall Impression'.
        You MUST follow the exact format and style shown in the examples below for each section.

        --- EXAMPLE FORMATS ---

        **Clinical Observations:**
        Wound: Not determinable from provided data; the provided image does not display a visible epithelial break or focused wound field...

        **Wound Summary:**
        Wound: Not determinable from provided data; the provided image does not display a visible epithelial break or focused wound...

        **Tissue Percentages Over Time:**
        IMPORTANT: For this section, all values for Granulation, Slough, Eschar, and Epithelialization MUST be an integer percentage (e.g., "30%"). DO NOT use descriptive words like "Minimal". The four percentages for each day MUST add up to exactly 100%.
        **Day 0:**
        - Granulation: 60%
        - Slough: 30%
        - Eschar: 10%
        - Epithelialization: 0%
        **Day 7:**
        - Granulation: 55%
        - Slough: 25%
        - Eschar: 15%
        - Epithelialization: 5%
        **Day 14:**
        - Granulation: 50%
        - Slough: 20%
        - Eschar: 10%
        - Epithelialization: 20%
        **Day 21:**
        - Granulation: 45%
        - Slough: 15%
        - Eschar: 5%
        - Epithelialization: 35%

        **Treatment Plan:**
        **Wound Care Recommendations:**
        1. Perform focused in-person exam...
        **Ongoing Care:**
        - Change dressings every 48–72h...

        **Recommended Products:**
        - Sterile 0.9% saline
        - 35 mL syringe with 19-20G catheter

        **Wound Tissue Evaluation:**
        - **Granulation:** 90%
        - **Slough:** 10%
        - **Eschar:** 0%
        - **Epithelialization:** Early, minimal (~5%)
        - **Size:** 14 × 8 × 0.5 cm (≈60 cm²)
        - **Edges:** Advancing, epithelializing
        - **Exudate:** Moderate, serosanguinous, no odor
        - **Periwound:** Intact, healthy

        --- END EXAMPLES ---

        Your final output MUST contain all seven of the specified sections, formatted correctly.
        """

    def _make_api_call(self, prompt_parts: List, max_tokens: int, temperature: float, is_json: bool = False) -> str:
        """
        A single, reliable method for making all Gemini API calls,
        with robust checking for blocked responses.
        """
        try:
            generation_config = {
                "temperature": temperature,
                "max_output_tokens": max_tokens,
            }
            if is_json:
                generation_config["response_mime_type"] = "application/json"

            response = self.model.generate_content(prompt_parts, generation_config=generation_config)
            
            if not response.candidates or response.candidates[0].finish_reason.name != "STOP":
                reason = "UNKNOWN"
                if response.candidates:
                    reason = response.candidates[0].finish_reason.name
                raise Exception(f"AI response was empty or blocked by the provider for reason: {reason}.")

            return response.text
        except Exception as e:
            error_message = f"Google Gemini API call failed: {str(e)}"
            print(f"CRITICAL API ERROR: {error_message}")
            raise Exception(error_message)

    def get_initial_analysis(self, prompt: str, base64_image: str) -> str:
        """Performs the primary wound analysis with an image using Gemini."""
        print("DEBUG: Constructing multimodal message for Gemini.")
        image_part = {"mime_type": "image/jpeg", "data": base64.b64decode(base64_image)}
        text_part = self.clinical_protocol + "\n\n" + prompt
        prompt_parts = [text_part, image_part]
        
        print("DEBUG: Sending request to Gemini for image analysis...")
        return self._make_api_call(prompt_parts=prompt_parts, max_tokens=4096, temperature=0.2)

    def expand_treatment_plan(self, original_analysis: str) -> Dict:
        """Generates an expanded treatment plan as JSON using Gemini."""
        try:
            treatment_section_match = re.search(r'\*\*Treatment Plan:\*\*(.*?)\*\*Recommended Products:\*\*', original_analysis, re.DOTALL)
            if not treatment_section_match:
                return {"success": False, "error": "Could not find 'Treatment Plan' to expand."}
            treatment_section = treatment_section_match.group(1).strip()

            one_shot_example = """
            "recommendations": [{"action": "...", "rationale": "..."}],
            "ongoing_care": "...",
            "patient_education": "..."
            """
            
            system_prompt = "You are a JSON API that provides expanded wound care treatment plans. You always respond with a single, valid JSON object and nothing else."
            user_prompt = f"""
            Based on the original treatment plan below, provide a comprehensive expanded treatment plan.
            You MUST return a single, valid JSON object and nothing else.
            The JSON object must have three keys: "recommendations", "ongoing_care", and "patient_education".

            --- EXAMPLE of desired JSON structure ---
            {{ {one_shot_example} }}
            --- END EXAMPLE ---

            Now, generate the JSON for the following case:
            **Original Brief Treatment Plan:** {treatment_section}
            """
            full_prompt = system_prompt + "\n\n" + user_prompt
            
            response_text = self._make_api_call(prompt_parts=[full_prompt], max_tokens=2000, temperature=0.1)
            
            try:
                cleaned_text = re.sub(r'```json\s*|\s*```', '', response_text).strip()
                json_response = json.loads(cleaned_text)
                return {"success": True, "expanded_plan_json": json_response}
            except json.JSONDecodeError:
                return {"success": False, "error": "Failed to decode Gemini's JSON response.", "raw_response": response_text}
        
        except Exception as e:
            return {"success": False, "error": f"Error in Gemini expand_treatment_plan: {str(e)}"}

    def revise_products(self, original_analysis: str, revision_reason: str) -> Dict:
        """Revises product recommendations as JSON using Gemini."""
        try:
            if revision_reason == "Patient Won't Tolerate":
                instruction = "Focus on gentle, hypoallergenic products that are comfortable for sensitive patients. Avoid aggressive treatments and prioritize patient comfort."
            elif revision_reason == "Too Costly":
                instruction = "Recommend cost-effective, generic alternatives and basic wound care supplies. Focus on essential products only and suggest budget-friendly options."
            elif revision_reason == "Products Unavailable":
                instruction = "Suggest readily available alternatives that can be found in most pharmacies or medical supply stores. Include multiple product options."
            else:  # This handles the "Other" case or any unexpected values.
                instruction = "Provide alternative product recommendations with different mechanisms of action or formulations."
            products_match = re.search(r'\*\*Recommended Products:\*\*(.*?)\*\*Wound Tissue Evaluation:\*\*', original_analysis, re.DOTALL)
            if not products_match:
                return {"success": False, "error": "Could not find 'Recommended Products' to revise."}
            current_products = products_match.group(1).strip()

            system_prompt = "You are a JSON API that provides revised wound care product recommendations. You always respond with a single, valid JSON object and nothing else."
            user_prompt = f"""
            The current recommended products need to be revised based on the constraint: "{revision_reason}".
            Instruction: {instruction}
            Current Recommended Products: {current_products} 
            
            You MUST return a single, valid JSON object and nothing else.
            The JSON object must have one key: "revised_products", a list of objects(not more than 2 or 3), each with "product_name" and "rationale".

            --- EXAMPLE of desired JSON structure ---
            {{
              "revised_products": [
                {{"product_name": "Generic Sterile Saline (0.9%)", "rationale": "A cost-effective alternative..."}}
              ]
            }}
            --- END EXAMPLE ---
            """
            full_prompt = system_prompt + "\n\n" + user_prompt

            response_text = self._make_api_call(prompt_parts=[full_prompt], max_tokens=2048, temperature=0.1)
            
            try:
                cleaned_text = re.sub(r'```json\s*|\s*```', '', response_text).strip()
                json_response = json.loads(cleaned_text)
                return {"success": True, "revised_products_json": json_response}
            except json.JSONDecodeError:
                return {"success": False, "error": "Failed to decode Gemini's JSON response.", "raw_response": response_text}
        
        except Exception as e:
            return {"success": False, "error": f"Error in Gemini revise_products: {str(e)}"}
        

    def get_healing_progress(self, pdf_path: str) -> Dict:
        """
        Analyzes a PDF of wound history using Gemini and returns a healing percentage.
        This version correctly uses the _make_api_call helper.
        """
        try:
            print(f"DEBUG: Reading PDF {pdf_path} for Gemini multimodal prompt...")
            with open(pdf_path, "rb") as pdf_file:
                pdf_bytes = pdf_file.read()

            pdf_part = {"mime_type": "application/pdf", "data": pdf_bytes}

            nuanced_user_prompt = """
            You are a world-class wound care specialist. The attached PDF file contains the complete history of a single wound.
            Your task is to provide a nuanced 'Healing Progress Percentage' based on the LATEST image and data in the sequence.
            Use the following definitions: 0% is the initial state, 100% is a fully healed, pale scar. A sutured wound with redness is not 100%.
            You MUST respond with only a single, valid JSON object containing one key: 'healing_progress_percentage'.
            """
            
            prompt_parts = [nuanced_user_prompt, pdf_part]

            # --- THE CRITICAL FIX ---
            # Call the robust helper function instead of calling the model directly.
            # We also tell the helper to expect a JSON response.
            response_text = self._make_api_call(
                prompt_parts=prompt_parts, 
                max_tokens=1024, 
                temperature=0.0,
                is_json=True # Tell the helper to set the JSON mime type
            )
            # ------------------------
            
            json_response = json.loads(response_text)
            return {"success": True, "healing_progress_json": json_response}

        except Exception as e:
            return {"success": False, "error": f"Error in Gemini get_healing_progress: {str(e)}"}
