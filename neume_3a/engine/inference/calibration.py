import time
import numpy as np
from shared.logger import get_logger
from shared.io_utils import save_pickle

class CalibrationProtocol:
    """
    Creates user-specific normalization from a baseline session.
    """
    def __init__(self, extractor, config: dict):
        self.extractor = extractor
        self.config = config
        self.logger = get_logger("CalibrationProtocol")

    def run(self, eeg_epochs: list[np.ndarray], fnirs_epochs: list[np.ndarray], user_id: str = "default") -> dict:
        self.logger.info(f"Running calibration for user {user_id}")
        t0 = time.perf_counter()
        
        # 1. Extract raw features (no normalization)
        raw_features = []
        for eeg, fnirs in zip(eeg_epochs, fnirs_epochs):
            feat = self.extractor.extract(eeg, fnirs)
            raw_features.append(feat)
            
        z_raw = np.array(raw_features)
        
        # 2. Compute Mean/Std
        mean = np.mean(z_raw, axis=0)
        std = np.std(z_raw, axis=0)
        std[std == 0] = 1e-8
        
        # 3. Save
        state = {
            "mean": mean,
            "std": std,
            "n_samples": len(z_raw)
        }
        
        save_path = f"data/processed/normalizer_{user_id}.pkl"
        save_pickle(state, save_path)
        
        profile = {
            "user_id": user_id,
            "n_samples": len(z_raw),
            "normalizer_path": save_path
        }
        
        elapsed = time.perf_counter() - t0
        self.logger.info(f"Calibration finished in {elapsed:.2f}s")
        return profile
