from ai_client_interface import AIClientInterface
from openai_client import OpenAIClient
from gemini_client import GeminiClient
from grok_client import GrokClient

def get_ai_client(model_name: str) -> AIClientInterface:
    """
    Factory function to select and return the appropriate AI client instance
    based on the provided model name.

    Args:
        model_name (str): The name of the model to use (e.g., 'gpt-4o', 'gemini-pro-vision', 'grok-4').

    Returns:
        An instance of a class that implements AIClientInterface.
    
    Raises:
        ValueError: If the model_name is not supported.
    """
    model_name_lower = model_name.lower()

    if "gpt" in model_name_lower:
        print(f"Initializing OpenAI client for model: {model_name}")
        return OpenAIClient(model=model_name)
    
    elif "gemini" in model_name_lower:
        print(f"Initializing Gemini client for model: {model_name}")
        return GeminiClient(model=model_name)
    
    elif "grok" in model_name_lower:
        print(f"Initializing Grok client for model: {model_name}")
        return GrokClient(model=model_name)
    
    else:
        raise ValueError(f"Unsupported model: '{model_name}'. No client available.")
