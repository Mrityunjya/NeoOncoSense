# prompt_templates.py

"""
This module stores reusable prompt templates for generating explanations,
patient-friendly insights, or technical breakdowns using a Generative AI model.
"""

def get_explanation_prompt(features: dict) -> str:
    """
    Constructs a prompt for generating a detailed explanation of the diagnosis.

    Args:
        features (dict): A dictionary of feature names and their values for a single prediction.

    Returns:
        str: A prompt string to be used with a language model.
    """
    feature_text = "\n".join([f"{key}: {value}" for key, value in features.items()])
    
    prompt = f"""You are a medical AI assistant. A breast cancer diagnosis was made based on the following features:

{feature_text}

Provide a professional explanation of why this case was predicted as malignant or benign.
Include references to key biological indicators and their typical behavior in breast cancer diagnosis.
Keep the explanation accurate yet understandable for both medical professionals and advanced learners.
"""
    return prompt


def get_patient_friendly_summary(diagnosis: str) -> str:
    """
    Returns a prompt that asks the LLM to generate a patient-friendly version of the result.

    Args:
        diagnosis (str): 'malignant' or 'benign'

    Returns:
        str: A prompt to explain the result to a non-technical user.
    """
    prompt = f"""You are an empathetic medical assistant. A patient's result came back as '{diagnosis}'.

Explain what this means in simple terms. Avoid alarming language. 
If the result is 'malignant', gently encourage further testing and treatment discussion.
If the result is 'benign', reassure the user and recommend routine follow-up if needed.
"""
    return prompt
