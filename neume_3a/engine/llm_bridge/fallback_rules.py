def smooth_update(prev: float, target: float) -> float:
    """Interpolate smoothly towards target to avoid jarring game transitions."""
    return prev * 0.7 + target * 0.3

FATIGUE_PARAMS = {
    "lanes": 2,
    "speed": 0.6,
    "difficulty_adjustment": -0.2,
    "environment_theme": "calm",
    "narrative_tone": "encouraging",
    "distractor_density": 0.3
}

FOCUSED_PARAMS = {
    "lanes": 3,
    "speed": 1.0,
    "difficulty_adjustment": 0.0,
    "environment_theme": "focus",
    "narrative_tone": "neutral",
    "distractor_density": 0.5
}

FLOW_PARAMS = {
    "lanes": 4,
    "speed": 1.6,
    "difficulty_adjustment": 0.2,
    "environment_theme": "neural",
    "narrative_tone": "challenging",
    "distractor_density": 0.75
}

def get_fallback_params(state: str, previous_params: dict = None) -> dict:
    if state == "FATIGUE":
        target = FATIGUE_PARAMS.copy()
    elif state == "FLOW":
        target = FLOW_PARAMS.copy()
    else:
        target = FOCUSED_PARAMS.copy()
        
    if not previous_params:
        return target
        
    # Interpolate smoothable values
    target["speed"] = smooth_update(previous_params.get("speed", target["speed"]), target["speed"])
    target["difficulty_adjustment"] = smooth_update(previous_params.get("difficulty_adjustment", target["difficulty_adjustment"]), target["difficulty_adjustment"])
    target["distractor_density"] = smooth_update(previous_params.get("distractor_density", target["distractor_density"]), target["distractor_density"])
    
    return validate_params(target)

def validate_params(params: dict) -> dict:
    """Clamp values to valid ranges."""
    params["speed"] = max(0.2, min(2.0, params.get("speed", 1.0)))
    params["difficulty_adjustment"] = max(-0.3, min(0.3, params.get("difficulty_adjustment", 0.0)))
    params["distractor_density"] = max(0.1, min(0.9, params.get("distractor_density", 0.5)))
    
    # Int clamp for lanes
    lanes = int(params.get("lanes", 3))
    params["lanes"] = max(1, min(4, lanes))
    
    # Ensure strings exist
    params["environment_theme"] = params.get("environment_theme", "focus")
    params["narrative_tone"] = params.get("narrative_tone", "neutral")
    
    return params
