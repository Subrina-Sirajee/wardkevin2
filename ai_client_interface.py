from abc import ABC, abstractmethod
from typing import Dict
import base64

class AIClientInterface(ABC):
    """
    Defines the abstract base class (the "contract") that all AI clients must follow.
    """

    @abstractmethod
    def get_initial_analysis(self, prompt: str, base64_image: str) -> str:
        """Performs the primary wound analysis with an image."""
        pass

    @abstractmethod
    def expand_treatment_plan(self, original_analysis: str, assessment_data: Dict, wound_location: str) -> Dict:
        """Generates an expanded, detailed treatment plan."""
        pass

    @abstractmethod
    def revise_products(self, original_analysis: str, revision_reason: str) -> Dict:
        """Regenerates product recommendations based on a specific constraint."""
        pass

    @abstractmethod
    def get_healing_progress(self, pdf_path: str) -> Dict:
        """Analyzes a multi-page PDF of wound history and returns a healing percentage."""
        pass

    # @abstractmethod
    # def get_tissue_percentages_over_time(self, original_analysis: str, assessment_data: Dict, wound_location: str) -> Dict:
    #     """Generates a projection of tissue composition percentages over time."""
    #     pass

    # @abstractmethod
    # def get_wound_summary(self, original_analysis: str, assessment_data: Dict, wound_location: str) -> Dict:
    #     """Generates a concise, clinical wound summary."""
    #     pass

    def encode_image(self, image_path: str) -> str:
        """Encodes an image file to a base64 string (concrete implementation)."""
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        except FileNotFoundError:
            raise FileNotFoundError(f"Image file not found at path: {image_path}")
        except Exception as e:
            raise IOError(f"Error encoding image at {image_path}: {str(e)}")
