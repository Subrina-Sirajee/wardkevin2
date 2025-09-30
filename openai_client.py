import json
import os
import re # Make sure re is imported
from openai import OpenAI
from dotenv import load_dotenv
from ai_client_interface import AIClientInterface
from typing import Dict, List

class OpenAIClient(AIClientInterface):
    # ... (__init__ and clinical_protocol are fine)
    def __init__(self, api_key: str = None, model: str = "gpt-4o"):
        load_dotenv()
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables.")
        
        self.client = OpenAI(api_key=self.api_key)
        self.model = model
        # In the OpenAIClient class, inside the __init__ method:

        self.clinical_protocol = """
        You are a world-class dermatologist AI. Your task is to analyze the provided wound image and clinical data.
        You MUST provide a strictly structured response with the following sections EXACTLY as named:
        **Case Information:**
        **Clinical Observations:**
        **Treatment Plan:**
        **Recommended Products:**
        **Wound Tissue Evaluation:**
        **Wound Summary:**
        **Tissue Percentages Over Time:**

        Follow the format of the user's prompt for the Case Information section.
        For all other sections, use the following examples as a reference for format and style. Your own evaluation MUST be based on the image and data provided.

        --- EXAMPLE FORMATS ---

        **Clinical Observations:**
        Wound: Not determinable from provided data; the provided image does not display a visible epithelial break or focused wound field so wound location, size, depth, tissue composition (granulation/slough/eschar) and presence of foreign material cannot be assessed. Clinical tactile assessment and calibrated measurement are required for definitive description.

        **Treatment Plan:**
        **Wound Care Recommendations:**
        1. Perform focused in-person exam with calibrated measurements (L × W × D), probe-to-bone if indicated, photos, and culture only if infection suspected. Rationale: Ensures accurate characterization and avoids unnecessary antibiotics.
        2. Cleanse with sterile saline 0.9% via 35 mL syringe + 19–20G catheter; gently remove loose debris. Avoid routine cytotoxic antiseptics. Rationale: Reduces bioburden while preserving viable tissue.
        **Ongoing Care:**
        - Change dressings every 48–72h (sooner if saturated), maintaining moist environment without periwound maceration.
        - Urgent escalation if: purulent drainage, spreading erythema >2 cm, increased pain, fever/systemic signs, swelling, new necrosis, worsening odor.

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

        **Wound Summary:**
        Wound: Not determinable from provided data; the provided image does not display a visible epithelial break or focused wound. Clinical tactile assessment and calibrated measurement are required for definitive description.

        **Tissue Percentages Over Time:**
        IMPORTANT: For this section, all values for Granulation, Slough, Eschar, and Epithelialization MUST be an integer percentage (e.g., "30%", "0%", "15%"). DO NOT use descriptive words like "Minimal" or "None". The four percentages for each day MUST add up to exactly 100%.
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
        
        --- END EXAMPLES ---

        Base your entire analysis on the VISIBLE information in the image and the clinical data provided. Be specific. Do not invent data.
        """


    def _make_api_call(self, messages: List[Dict], max_tokens: int, temperature: float) -> str:
        """A single, reliable method for making all API calls."""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature
            )
            return response.choices[0].message.content
        except Exception as e:
            # --- THIS IS THE FIX ---
            # Instead of just printing, re-raise the exception so the program
            # knows a critical error occurred.
            error_message = f"OpenAI API call failed: {str(e)}"
            print(f"CRITICAL API ERROR: {error_message}")
            raise Exception(error_message)
            # -----------------------

    # The rest of the methods in this file (get_initial_analysis, expand_treatment_plan, etc.)
    # do not need to be changed. They correctly use the _make_api_call helper.
    def get_initial_analysis(self, prompt: str, base64_image: str) -> str:
        """
        Performs the primary wound analysis with an image.
        This method is now self-contained and guaranteed to be correct.
        """
        print("DEBUG: Constructing multimodal message for OpenAI.")
        messages = [
            {
                "role": "system",
                "content": self.clinical_protocol
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                            "detail": "high"
                        }
                    }
                ]
            }
        ]
        
        print("DEBUG: Sending request to OpenAI for image analysis...")
        return self._make_api_call(messages=messages, max_tokens=2000, temperature=0.2)

    # In the OpenAIClient class:

    # In the OpenAIClient class in openai_client.py:

    def expand_treatment_plan(self, original_analysis: str, assessment_data: Dict, wound_location: str) -> Dict:
        """Generates an expanded treatment plan and returns it as a structured JSON object."""
        try:
            # Safely extract the initial, brief treatment plan from the first analysis
            treatment_section_match = re.search(r'\*\*Treatment Plan:\*\*(.*?)\*\*Recommended Products:\*\*', original_analysis, re.DOTALL)
            if not treatment_section_match:
                return {"success": False, "error": "Could not find 'Treatment Plan' in the original analysis to expand."}
            treatment_section = treatment_section_match.group(1).strip()

            # Define the one-shot example for the desired JSON structure
            one_shot_example = """
            "recommendations": [
                {
                    "action": "Perform a focused in-person wound assessment including calibrated measurements...",
                    "rationale": "Accurate characterization and microbiology are necessary to direct appropriate therapy."
                },
                {
                    "action": "Irrigate the wound with sterile 0.9% saline...",
                    "rationale": "Mechanical irrigation reduces surface bioburden and aids assessment while preserving viable tissue."
                }
            ],
            "ongoing_care": "Change dressings every 48–72 hours or sooner if saturated... Escalate to urgent evaluation if any of the following occur: ...",
            "patient_education": "Educate the patient and caregiver on signs of infection and the importance of dressing changes."
            """

            # Build the full prompt for the AI
            expand_prompt = f"""
            Based on the original treatment plan and clinical context below, provide a comprehensive expanded treatment plan.
            You MUST return a single, valid JSON object and nothing else. Do not include any introductory text or markdown formatting.
            The JSON object must have three keys: "recommendations" (a list of objects, each with "action" and "rationale"), "ongoing_care" (a string), and "patient_education" (a string).

            --- EXAMPLE of desired JSON structure ---
            {{
            {one_shot_example}
            }}
            --- END EXAMPLE ---

            Now, generate the JSON for the following case:

            **Original Brief Treatment Plan:**
            {treatment_section}

            **Clinical Context:**
            - Wound Location: {wound_location}
            - Health Risk Factors: {', '.join(assessment_data['patient_overview']['health_risk_factors'])}
            """
            
            # Prepare the messages for the API call
            messages = [
                {"role": "system", "content": "You are a JSON API that provides expanded wound care treatment plans. You always respond with a single, valid JSON object and nothing else."},
                {"role": "user", "content": expand_prompt}
            ]

            # Make the API call
            response_text = self._make_api_call(messages=messages, max_tokens=2000, temperature=0.1)
            
            # Safely parse the JSON response from the AI
            try:
                cleaned_text = re.sub(r'```json\s*|\s*```', '', response_text).strip()
                json_response = json.loads(cleaned_text)
                # Return the dictionary with the correct key as expected by the facade
                return {"success": True, "expanded_plan_json": json_response}
            except json.JSONDecodeError:
                return {"success": False, "error": "Failed to decode AI's JSON response for the expanded plan.", "raw_response": response_text}
        
        except Exception as e:
            return {"success": False, "error": f"Error in OpenAI expand_treatment_plan: {str(e)}"}



    # In the OpenAIClient class:

    # In the OpenAIClient class in openai_client.py:

    def revise_products(self, original_analysis: str, revision_reason: str) -> Dict:
        """Revises product recommendations and returns them as a structured JSON object."""
        try:
            # Define instructions based on the revision reason
            instruction = {
                "Patient Won't Tolerate": "Focus on gentle, hypoallergenic products that are comfortable for sensitive patients.",
                "Too Costly": "Recommend cost-effective, generic alternatives and basic wound care supplies.",
                "Products Unavailable": "Suggest readily available alternatives that can be found in most pharmacies.",
                "Other": "Provide alternative product recommendations with different mechanisms of action."
            }.get(revision_reason, "Provide alternative product recommendations.")

            # Safely extract the current product list
            products_match = re.search(r'\*\*Recommended Products:\*\*(.*?)\*\*Wound Tissue Evaluation:\*\*', original_analysis, re.DOTALL)
            if not products_match:
                return {"success": False, "error": "Could not find 'Recommended Products' in the original analysis to revise."}
            current_products = products_match.group(1).strip()

            # Build the full prompt for the AI
            revision_prompt = f"""
            The current recommended products need to be revised based on the constraint: "{revision_reason}".
            
            Instruction: {instruction}
            
            Current Recommended Products:
            {current_products}
            
            You MUST return a single, valid JSON object and nothing else.
            The JSON object must have one key: "revised_products", which is a list of objects. Each object should have two keys: "product_name" and "rationale".

            --- EXAMPLE of desired JSON structure ---
            {{
              "revised_products": [
                {{
                  "product_name": "Generic Sterile Saline (0.9%)",
                  "rationale": "A cost-effective alternative for wound cleansing that is widely available."
                }},
                {{
                  "product_name": "Basic Non-adherent Gauze",
                  "rationale": "Provides a budget-friendly primary dressing to protect the wound bed."
                }}
              ]
            }}
            --- END EXAMPLE ---
            """
            
            # Prepare the messages for the API call
            messages = [
                {"role": "system", "content": "You are a JSON API that provides revised wound care product recommendations. You always respond with a single, valid JSON object and nothing else."},
                {"role": "user", "content": revision_prompt}
            ]

            # Make the API call
            response_text = self._make_api_call(messages=messages, max_tokens=1000, temperature=0.1)
            
            # Safely parse the JSON response from the AI
            try:
                cleaned_text = re.sub(r'```json\s*|\s*```', '', response_text).strip()
                json_response = json.loads(cleaned_text)
                # Return the dictionary with the correct key as expected by the facade
                return {"success": True, "revised_products_json": json_response}
            except json.JSONDecodeError:
                return {"success": False, "error": "Failed to decode AI's JSON response for revised products.", "raw_response": response_text}
        
        except Exception as e:
            return {"success": False, "error": f"Error in OpenAI revise_products: {str(e)}"}


    # def get_tissue_percentages_over_time(self, original_analysis: str, assessment_data: Dict, wound_location: str) -> Dict:
    #     """Generates tissue composition percentages for a healing trajectory using OpenAI."""
    #     try:
    #         tissue_section_match = re.search(r'\*\*Wound Tissue Evaluation:\*\*(.*)', original_analysis, re.DOTALL)
    #         if not tissue_section_match:
    #             return {"success": False, "error": "Could not find 'Wound Tissue Evaluation' in the original analysis."}
    #         tissue_section = tissue_section_match.group(1).strip()

    #         tissue_prompt = f"""
    #         Based on the wound characteristics and clinical data provided, estimate the tissue composition percentages for a healing trajectory over 4 time points: Day 0 (initial), Day 7, Day 14, and Day 21.

    #         Current Wound Assessment:
    #         {tissue_section}

    #         Clinical Context:
    #         - Wound Location: {wound_location}
    #         - Health Risk Factors: {', '.join(assessment_data['patient_overview']['health_risk_factors'])}
    #         - Clinical Notes: {assessment_data['clinical_assessment']['other_relevant_information']}

    #         Provide realistic percentages for each tissue type (Granulation, Slough, Eschar, Epithelialization) at each time point.
    #         Format EXACTLY as follows, with each day's percentages totaling 100%:
    #         **Day 0:**
    #         - Granulation: X%
    #         - Slough: Y%
    #         - Eschar: Z%
    #         - Epithelialization: W%
    #         **Day 7:**
    #         ...and so on for Day 14 and Day 21.
    #         """
            
    #         messages = [
    #             {"role": "system", "content": "You are an expert wound care specialist. Provide realistic tissue composition percentages over time in the exact format requested."},
    #             {"role": "user", "content": tissue_prompt}
    #         ]

    #         tissue_percentages = self._make_api_call(messages=messages, max_tokens=1000, temperature=0.1)
    #         return {"success": True, "tissue_percentages_over_time": tissue_percentages}
    #     except Exception as e:
    #         return {"success": False, "error": f"Error in OpenAI get_tissue_percentages_over_time: {str(e)}"}

    # def get_wound_summary(self, original_analysis: str, assessment_data: Dict, wound_location: str) -> Dict:
    #     """Generates a concise wound summary using OpenAI."""
    #     try:
    #         summary_prompt = f"""
    #         Based on the initial analysis and clinical data, provide a concise, professional wound summary.
    #         Describe what can and cannot be determined from the available data.

    #         Clinical Context:
    #         - Wound Location: {wound_location}
    #         - Clinical Findings: {assessment_data['clinical_assessment']['other_relevant_information']}
    #         - Health Factors: {', '.join(assessment_data['patient_overview']['health_risk_factors'])}
    #         - Initial AI Observations: {original_analysis}

    #         Generate a summary in the style of a clinical note.
    #         """
            
    #         messages = [
    #             {"role": "system", "content": "You are an expert wound care specialist. You write concise, clinical wound summaries that clearly state what can be assessed and what requires further evaluation."},
    #             {"role": "user", "content": summary_prompt}
    #         ]

    #         wound_summary = self._make_api_call(messages=messages, max_tokens=800, temperature=0.1)
    #         return {"success": True, "wound_summary": wound_summary}
    #     except Exception as e:
    #         return {"success": False, "error": f"Error in OpenAI get_wound_summary: {str(e)}"}
