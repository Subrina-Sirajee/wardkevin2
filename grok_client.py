import json
import os
import re
from openai import OpenAI
from dotenv import load_dotenv
from ai_client_interface import AIClientInterface
from typing import Dict, List
import base64
import time

class GrokClient(AIClientInterface):
    def __init__(self, api_key: str = None, model: str = "grok-4"):
        load_dotenv()
        self.api_key = api_key or os.getenv("XAI_API_KEY")
        if not self.api_key:
            raise ValueError("XAI_API_KEY not found in environment variables.")
        
        self.client = OpenAI(base_url="https://api.x.ai/v1", api_key=self.api_key)
        self.model = model
        
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
            error_message = f"Grok API call failed: {str(e)}"
            print(f"CRITICAL API ERROR: {error_message}")
            raise Exception(error_message)

    def get_initial_analysis(self, prompt: str, base64_image: str) -> str:
        """
        Performs the primary wound analysis with an image.
        """
        print("DEBUG: Constructing multimodal message for Grok.")
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
        
        print("DEBUG: Sending request to Grok for image analysis...")
        return self._make_api_call(messages=messages, max_tokens=2000, temperature=0.2)

    def expand_treatment_plan(self, original_analysis: str) -> Dict:
        """Generates an expanded treatment plan and returns it as a structured JSON object."""
        try:
            treatment_section_match = re.search(r'\*\*Treatment Plan:\*\*(.*?)\*\*Recommended Products:\*\*', original_analysis, re.DOTALL)
            if not treatment_section_match:
                return {"success": False, "error": "Could not find 'Treatment Plan' in the original analysis to expand."}
            treatment_section = treatment_section_match.group(1).strip()

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

            expand_prompt = f"""
            Based on the original treatment plan below, provide a comprehensive expanded treatment plan.
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
            """
            
            messages = [
                {"role": "system", "content": "You are a JSON API that provides expanded wound care treatment plans. You always respond with a single, valid JSON object and nothing else."},
                {"role": "user", "content": expand_prompt}
            ]

            response_text = self._make_api_call(messages=messages, max_tokens=2000, temperature=0.1)
            
            try:
                cleaned_text = re.sub(r'```json\s*|\s*```', '', response_text).strip()
                json_response = json.loads(cleaned_text)
                return {"success": True, "expanded_plan_json": json_response}
            except json.JSONDecodeError:
                return {"success": False, "error": "Failed to decode AI's JSON response for the expanded plan.", "raw_response": response_text}
        
        except Exception as e:
            return {"success": False, "error": f"Error in Grok expand_treatment_plan: {str(e)}"}

    def revise_products(self, original_analysis: str, revision_reason: str) -> Dict:
        """Revises product recommendations and returns them as a structured JSON object."""
        try:
            instruction = {
                "Patient Won't Tolerate": "Focus on gentle, hypoallergenic products that are comfortable for sensitive patients.",
                "Too Costly": "Recommend cost-effective, generic alternatives and basic wound care supplies.",
                "Products Unavailable": "Suggest readily available alternatives that can be found in most pharmacies.",
                "Other": "Provide alternative product recommendations with different mechanisms of action."
            }.get(revision_reason, "Provide alternative product recommendations.")

            products_match = re.search(r'\*\*Recommended Products:\*\*(.*?)\*\*Wound Tissue Evaluation:\*\*', original_analysis, re.DOTALL)
            if not products_match:
                return {"success": False, "error": "Could not find 'Recommended Products' in the original analysis to revise."}
            current_products = products_match.group(1).strip()

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
            
            messages = [
                {"role": "system", "content": "You are a JSON API that provides revised wound care product recommendations. You always respond with a single, valid JSON object and nothing else."},
                {"role": "user", "content": revision_prompt}
            ]

            response_text = self._make_api_call(messages=messages, max_tokens=1000, temperature=0.1)
            
            try:
                cleaned_text = re.sub(r'```json\s*|\s*```', '', response_text).strip()
                json_response = json.loads(cleaned_text)
                return {"success": True, "revised_products_json": json_response}
            except json.JSONDecodeError:
                return {"success": False, "error": "Failed to decode AI's JSON response for revised products.", "raw_response": response_text}
        
        except Exception as e:
            return {"success": False, "error": f"Error in Grok revise_products: {str(e)}"}

    def get_healing_progress(self, pdf_path: str) -> Dict:
        """
        Analyzes a PDF of wound history using Grok's multimodal capabilities (without file upload).
        This approach reads the PDF locally and sends it directly, avoiding permission issues.
        """
        try:
            print(f"DEBUG: Reading PDF {pdf_path} for Grok multimodal request...")
            with open(pdf_path, "rb") as pdf_file:
                pdf_bytes = pdf_file.read()

            # Encode PDF as base64 for the API
            pdf_base64 = base64.b64encode(pdf_bytes).decode('utf-8')

            nuanced_user_prompt = """
            You are a world-class wound care specialist. The attached PDF file contains the complete history of a single wound.
            Your task is to provide a nuanced 'Healing Progress Percentage' based on the LATEST image and data in the sequence compared to previous images and data.
            
            Use the following definitions:
            - 0% represents the initial state of the wound (the first image).
            - 100% represents a fully healed state, characterized by a faint, pale, non-erythematous (not red) scar with fully restored skin integrity.
            - A wound that is closed with sutures but still shows significant redness, swelling, or a prominent, fresh scar should be considered in an intermediate stage (e.g., 60-80%), NOT 100%.
            - A wound that is smaller but still open with granulation tissue might be around 50%.
            
            Analyze the LATEST image in the PDF and assess its state relative to the final goal of a fully mature, pale scar. Based on this, provide a single integer for the healing progress percentage.
            
            You MUST respond with only a single, valid JSON object containing one key: 'healing_progress_percentage'.
            """

            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": nuanced_user_prompt
                        },
                        {
                            "type": "document",
                            "source": {
                                "type": "base64",
                                "media_type": "application/pdf",
                                "data": pdf_base64
                            }
                        }
                    ]
                }
            ]

            print("DEBUG: Sending request to Grok for healing progress analysis...")
            response_text = self._make_api_call(messages=messages, max_tokens=1024, temperature=0.0)
            
            # Clean and parse JSON response
            cleaned_text = re.sub(r'```json\s*|\s*```', '', response_text).strip()
            json_response = json.loads(cleaned_text)
            
            return {"success": True, "healing_progress_json": json_response}

        except Exception as e:
            return {"success": False, "error": f"Error in Grok get_healing_progress: {str(e)}"}