from datetime import datetime

class ClinicalDataFormatter:
    """Handles the formatting of clinical data and generation of prompts."""

    def format_assessment_data(self, **kwargs) -> dict:
        """
        Formats boolean flags and other inputs into a structured dictionary.
        This method is more dynamic than the original.
        """
        assessment_data = {
            "timestamp": datetime.now().isoformat(),
            "patient_overview": {
                "health_risk_factors": self._collect_items(kwargs, ["diabetes", "peripheral_arterial_disease"]),
                "mobility": self._collect_items(kwargs, ["ambulatory", "wheelchair_dependent", "bedbound"]),
                "living_situation": self._collect_items(kwargs, ["alone", "caregiver_support", "facility"]),
            },
            "clinical_assessment": {
                "drainage_amount": self._collect_items(kwargs, ["drainage_none", "drainage_scant"]),
                "drainage_type": self._collect_items(kwargs, ["serous", "sanguinous", "purulent"]),
                "odor_assessment": self._collect_items(kwargs, ["odor_absent", "odor_present", "odor_foul"]),
                "peri_wound_skin_temperature": self._collect_items(kwargs, ["temperature_same", "temperature_warmer_hot"]),
                "other_relevant_information": kwargs.get("other_information", "")
            }
        }
        # Note: The lists in _collect_items are abbreviated for clarity. Add all original items.
        return assessment_data

    def _collect_items(self, source: dict, keys: list) -> list:
        """Helper to collect true flags from a dictionary."""
        return [key.replace('_', ' ').title() for key in keys if source.get(key)]

    def create_main_analysis_prompt(self, assessment_data: dict, wound_location: str, current_date: str) -> str:
        """Creates the full text prompt for the initial AI analysis."""
        clinical = assessment_data['clinical_assessment']
        
        # Helper to safely get the first item or a default value
        def get_first_or_default(data_list, default="Not specified"):
            return data_list[0] if data_list else default

        # Construct the prompt with exact details for the 'Case Information' section
        prompt = f"""
        For the Case Information section, use these exact details:
        - Case Date: {current_date}
        - Wound Location: {wound_location}
        - Drainage Amount: {get_first_or_default(clinical['drainage_amount'])}
        - Drainage Type: {get_first_or_default(clinical['drainage_type'])}
        - Odor Assessment: {get_first_or_default(clinical['odor_assessment'])}
        - Additional Clinical Info: {clinical.get('other_relevant_information') or 'None'}

        WOUND ASSESSMENT REQUEST - {current_date}

        Analyze the wound image and integrate with the following clinical data to provide a structured response:
        
        {self._format_prompt_section("HEALTH RISK FACTORS", assessment_data['patient_overview']['health_risk_factors'])}
        {self._format_prompt_section("MOBILITY", assessment_data['patient_overview']['mobility'])}
        {self._format_prompt_section("LIVING SITUATION", assessment_data['patient_overview']['living_situation'])}

        CLINICAL ASSESSMENT:
        {self._format_prompt_section("Drainage Amount", clinical['drainage_amount'])}
        {self._format_prompt_section("Drainage Type", clinical['drainage_type'])}
        {self._format_prompt_section("Odor Assessment", clinical['odor_assessment'])}
        {self._format_prompt_section("Peri-wound Skin Temperature", clinical['peri_wound_skin_temperature'])}
        {self._format_prompt_section("Other Relevant Information", [clinical['other_relevant_information']])}

        Based on your visual analysis of the wound image and the clinical data provided, provide your assessment in the exact format shown in the example.
        """
        return prompt

    def _format_prompt_section(self, title: str, items: list) -> str:
        """Formats a section of the prompt if data is available."""
        if not items or not items[0]:
            return ""
        return f"{title.upper()}: {', '.join(items)}\n"
