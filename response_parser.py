import re

class AIResponseParser:
    """Parses the raw text output from the AI into a structured format."""

    def parse_response_to_json(self, ai_response: str) -> dict:
        """
        Parses the AI's full markdown response into a single JSON object.
        This version uses a more specific regex to avoid capturing sub-headers.
        """
        # For debugging
        print("\n--- RAW AI RESPONSE (SINGLE CALL) ---")
        print(ai_response)
        print("-------------------------------------\n")

        # Define all the section headers we expect in the single response
        section_keys = [
            "Case Information", "Clinical Observations",
            "Treatment Plan", "Recommended Products", "Wound Tissue Evaluation", "Wound Summary", "Tissue Percentages Over Time"
        ]
        
        response_json = {key: "" for key in section_keys}

        # --- THIS IS THE CORRECTED REGEX ---
        # 1. Create a string of our specific headers separated by '|' (which means OR in regex)
        #    This will look like: 'Case Information|Clinical Observations|Wound Summary|...'
        lookahead_headers = '|'.join(re.escape(key) for key in section_keys)
        
        # 2. Build the new, more specific pattern.
        #    It now looks ahead for `**` followed by one of our EXACT headers, then `:`
        pattern = re.compile(r'\*\*(.*?):\*\*(.*?)(?=\*\*(?:' + lookahead_headers + r'):|\Z)', re.DOTALL)
        # ------------------------------------
        
        matches = pattern.findall(ai_response)

        if not matches:
            print("WARNING: Parsing failed. No section headers found in the AI response.")
            response_json["error"] = "Parsing failed. AI response did not contain expected headers."
            return response_json

        for header, content in matches:
            header = header.strip()
            cleaned_content = content.strip()
            
            if header in response_json:
                response_json[header] = cleaned_content

        return response_json
