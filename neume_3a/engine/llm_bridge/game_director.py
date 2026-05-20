import json
from shared.logger import get_logger
from .fallback_rules import get_fallback_params, validate_params

class GameDirector:
    """Central controller that maps cognitive states to game parameters."""
    
    def __init__(self, config: dict):
        self.config = config
        self.logger = get_logger("GameDirector")
        self.previous_params = None
        self.use_llm = False # Forced OFF for real-time stability
        
        if self.use_llm:
            try:
                import openai
                # self.client = openai.OpenAI()
                self.logger.info("OpenAI client initialized (LLM mode active).")
            except Exception as e:
                self.logger.warning(f"Failed to init OpenAI: {e}. Falling back to rule-based.")
                self.use_llm = False

    def get_params(self, state: str, confidence: float, z_vector: list) -> dict:
        """Main method to compute params safely and quickly."""
        if confidence < 0.6:
            self.logger.debug("Low confidence prediction. Using conservative fallback.")
            params = get_fallback_params(state, self.previous_params)
            self.previous_params = params
            return params
            
        if self.use_llm:
            try:
                # LLM call would go here
                params = self._call_llm(state, confidence)
                params = validate_params(params)
            except Exception as e:
                self.logger.warning(f"LLM failure: {e}. Using fallback.")
                params = get_fallback_params(state, self.previous_params)
        else:
            params = get_fallback_params(state, self.previous_params)
            
        self.previous_params = params
        return params

    def _call_llm(self, state, confidence) -> dict:
        """Mock LLM call - replace with real API if needed."""
        raise NotImplementedError("LLM API disabled for stability.")
