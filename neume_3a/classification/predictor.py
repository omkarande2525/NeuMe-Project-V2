import numpy as np
from shared.logger import get_logger
from shared.io_utils import load_pickle

class CognitiveStatePredictor:
    """Unified predictor wrapper for real-time inference using RF baseline."""
    
    def __init__(self, model_type: str = "rf"):
        self.logger = get_logger("Predictor")
        self.model = None
        self.label_encoder = None
        self._is_loaded = False
        
    def load(self, model_path: str, encoder_path: str = None) -> None:
        try:
            state = load_pickle(model_path)
            self.model = state["clf"]
            self.label_encoder = state["label_encoder"]
            self._is_loaded = True
            self.logger.info("Loaded RF Predictor successfully")
        except Exception as e:
            self.logger.error(f"Failed to load predictor from {model_path}: {e}")
            raise

    def predict(self, z: np.ndarray) -> tuple[str, float]:
        if not self._is_loaded:
            raise RuntimeError("Predictor not loaded")
            
        z_batch = z.reshape(1, -1)
        
        preds_encoded = self.model.predict(z_batch)
        probs = self.model.predict_proba(z_batch)
        
        confidence = float(np.max(probs))
        label_str = str(self.label_encoder.inverse_transform(preds_encoded)[0])
        
        return label_str, confidence
