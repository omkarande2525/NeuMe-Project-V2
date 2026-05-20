SYSTEM_PROMPT = """
You are the AI Game Director for Neume, a cognitive rehabilitation system
for stroke patients. You receive real-time cognitive state data and must
output ONLY a valid JSON object with game parameters.

Rules:
- Never output anything except the JSON object
- Speed must be between 0.2 and 2.0
- Lanes must be 1, 2, 3, or 4
- difficulty_adjustment must be between -0.30 and +0.30
- environment_theme must be one of: calm, focus, neural, space
- narrative_tone must be one of: encouraging, neutral, challenging
- distractor_density must be between 0.1 and 0.9
- Never make sudden extreme changes — adjust gradually from previous state
"""

def build_prompt(state: str, confidence: float, features: dict, previous_params: dict = None) -> str:
    """Builds a deterministic, lightweight prompt."""
    prompt = f"Patient State: {state} (Confidence: {confidence:.2f})\n"
    prompt += f"Feature Summary: Energy={features.get('signal_energy', 0.0):.3f}, "
    prompt += f"Ratio={features.get('modality_ratio', 0.0):.3f}\n"
    
    if previous_params:
        prompt += f"Previous Game State: {previous_params}\n"
        
    prompt += "Output the new parameters in valid JSON format."
    return prompt
