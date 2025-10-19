import json
import os
import re # Make sure re is imported
from openai import OpenAI
from dotenv import load_dotenv
from ai_client_interface import AIClientInterface
from typing import Dict, List
import base64
import time

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


    def expand_treatment_plan(self, original_analysis: str) -> Dict:
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


    def get_healing_progress(self, pdf_path: str) -> Dict:
        """
        Analyzes a PDF of wound history using the OpenAI Assistants API and a more nuanced prompt.
        """
        try:
            # ... (The file upload and assistant creation logic is correct)
            print(f"DEBUG: Uploading PDF {pdf_path} to OpenAI's file store...")
            with open(pdf_path, "rb") as pdf_file:
                uploaded_file = self.client.files.create(file=pdf_file, purpose='assistants')
            print(f"DEBUG: PDF uploaded successfully. File ID: {uploaded_file.id}")

            assistant = self.client.beta.assistants.create(
                name="Wound Healing Analyst",
                instructions="You are a wound care specialist. Analyze the attached file which contains a wound's healing history. Respond with only a valid JSON object containing a single key: 'healing_progress_percentage'.",
                model=self.model,
                tools=[{"type": "file_search"}]
            )

            # --- THIS IS THE NEW, MORE NUANCED PROMPT ---
            nuanced_user_prompt = """
            The attached PDF contains the complete history of a single wound, with the first page being the baseline.
            Your task is to provide a nuanced 'Healing Progress Percentage' based on the LATEST image in the sequence.

            Use the following definitions for your calculation:
            - 0% represents the initial state of the wound (the first image).
            - 100% represents a fully healed state, characterized by a faint, pale, non-erythematous (not red) scar with fully restored skin integrity.
            - A wound that is closed with sutures but still shows significant redness, swelling, or a prominent, fresh scar should be considered in an intermediate stage (e.g., 60-80%), NOT 100%.
            - A wound that is smaller but still open with granulation tissue might be around 50%.

            Analyze the LATEST image in the PDF and assess its state relative to the final goal of a fully mature, pale scar. Based on this, provide a single integer for the healing progress percentage.
            """
            # ------------------------------------------------

            thread = self.client.beta.threads.create(
                messages=[
                    {
                        "role": "user",
                        "content": nuanced_user_prompt, # Use the new prompt
                        "attachments": [{"file_id": uploaded_file.id, "tools": [{"type": "file_search"}]}]
                    }
                ]
            )

            # ... (The rest of the run, wait, and cleanup logic is correct and does not need to change)
            run = self.client.beta.threads.runs.create(thread_id=thread.id, assistant_id=assistant.id)

            print("DEBUG: Waiting for the AI assistant to complete the analysis...")
            while run.status in ['queued', 'in_progress', 'cancelling']:
                time.sleep(1)
                run = self.client.beta.threads.runs.retrieve(thread_id=thread.id, run_id=run.id)

            if run.status == 'completed':
                messages = self.client.beta.threads.messages.list(thread_id=thread.id)
                response_text = messages.data[0].content[0].text.value
                
                cleaned_text = re.sub(r'```json\s*|\s*```', '', response_text).strip()
                json_response = json.loads(cleaned_text)
                
                print("DEBUG: Cleaning up OpenAI resources...")
                self.client.files.delete(uploaded_file.id)
                self.client.beta.assistants.delete(assistant.id)
                self.client.beta.threads.delete(thread.id)
                
                return {"success": True, "healing_progress_json": json_response}
            else:
                self.client.files.delete(uploaded_file.id)
                self.client.beta.assistants.delete(assistant.id)
                self.client.beta.threads.delete(thread.id)
                raise Exception(f"Assistant run failed with status: {run.status}")

        except Exception as e:
            return {"success": False, "error": f"Error in OpenAI get_healing_progress: {str(e)}"}
