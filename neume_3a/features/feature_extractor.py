import time
import numpy as np
import pandas as pd
from shared.logger import get_logger
from shared.io_utils import save_pickle, load_pickle

class FeatureExtractor:
    """
    Extracts exactly 15 features from raw multimodal epochs to ensure 
    compatibility with downstream pipelines and models.
    
    Expected inputs:
    EEG epoch: (9, 256) array
    fNIRS epoch: (8, 10) array
    """

    def __init__(self):
        self._mean: np.ndarray = None
        self._std: np.ndarray = None
        self.n_features: int = 15
        self.logger = get_logger("FeatureExtractor")

    def extract(self, eeg_epoch: np.ndarray, fnirs_epoch: np.ndarray) -> np.ndarray:
        """
        eeg_epoch: np.ndarray shape (9, 256)
        fnirs_epoch: np.ndarray shape (8, 10)

        Returns: np.array(15,) dtype=float32, normalised if fit() was called.
        """
        # 1. Compute EEG features (8)
        # 0: F3 mean (ch 0)
        # 1: C3 mean (ch 3)
        # 2: P3 mean (ch 6)
        # 3: signal_energy mean (mean of squares across all channels)
        # 4: rolling_avg mean proxy (mean of Fz, ch 2)
        # 5: diff_1 std (std of differences along time for F3)
        # 6: combined_signal mean (mean of all channels)
        # 7: avg_signal mean (mean of all channels)
        
        eeg_f3_mean = np.mean(eeg_epoch[0, :])
        eeg_c3_mean = np.mean(eeg_epoch[3, :])
        eeg_p3_mean = np.mean(eeg_epoch[6, :])
        eeg_energy = np.mean(eeg_epoch ** 2)
        eeg_fz_mean = np.mean(eeg_epoch[2, :])
        eeg_diff_std = np.std(np.diff(eeg_epoch[0, :]))
        eeg_all_mean = np.mean(eeg_epoch)
        eeg_avg_signal = np.mean(eeg_epoch) # identical proxy for legacy
        
        # 2. Compute fNIRS features (4)
        # 8: ch1 mean (ch 0)
        # 9: ch2 mean (ch 1)
        # 10: ch3 mean (ch 2)
        # 11: rolling_avg slope (polyfit on ch 0)
        fnirs_ch1_mean = np.mean(fnirs_epoch[0, :])
        fnirs_ch2_mean = np.mean(fnirs_epoch[1, :])
        fnirs_ch3_mean = np.mean(fnirs_epoch[2, :])
        fnirs_slope = np.polyfit(np.arange(10), fnirs_epoch[0, :], 1)[0]
        
        # 3. Compute Combined features (3)
        # 12: modality_ratio
        # 13: interaction_term
        # 14: differential_baseline
        fnirs_all_mean = np.mean(fnirs_epoch)
        modality_ratio = eeg_all_mean / (fnirs_all_mean + 1e-10)
        interaction_term = eeg_energy * fnirs_all_mean
        differential_baseline = eeg_fz_mean - fnirs_ch1_mean
        
        # Assemble
        features = np.array([
            eeg_f3_mean, eeg_c3_mean, eeg_p3_mean, eeg_energy,
            eeg_fz_mean, eeg_diff_std, eeg_all_mean, eeg_avg_signal,
            fnirs_ch1_mean, fnirs_ch2_mean, fnirs_ch3_mean, fnirs_slope,
            modality_ratio, interaction_term, differential_baseline
        ], dtype=np.float32)
        
        # Replace NaNs/Infs
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Normalize
        if self._mean is not None and self._std is not None:
            features = (features - self._mean) / (self._std + 1e-8)
            
        return features

    def fit(self, z_matrix: np.ndarray) -> None:
        """Fit normaliser on the full training z_matrix."""
        t0 = time.perf_counter()
        self.logger.debug("[FeatureExtractor.fit] starting")
        
        self._mean = np.mean(z_matrix, axis=0)
        self._std = np.std(z_matrix, axis=0)
        self._std[self._std == 0] = 1e-8  # Prevent division by zero
        
        elapsed = time.perf_counter() - t0
        self.logger.debug(f"[FeatureExtractor.fit] done in {elapsed:.2f}s")

    def save(self, path: str) -> None:
        """Save mean, std, n_features to joblib pickle."""
        t0 = time.perf_counter()
        self.logger.debug("[FeatureExtractor.save] starting")
        state = {
            "mean": self._mean,
            "std": self._std,
            "n_features": self.n_features
        }
        save_pickle(state, path)
        elapsed = time.perf_counter() - t0
        self.logger.debug(f"[FeatureExtractor.save] done in {elapsed:.2f}s")

    def load(self, path: str) -> None:
        """Load mean, std, n_features from joblib pickle."""
        t0 = time.perf_counter()
        self.logger.debug("[FeatureExtractor.load] starting")
        state = load_pickle(path)
        self._mean = state["mean"]
        self._std = state["std"]
        self.n_features = state["n_features"]
        elapsed = time.perf_counter() - t0
        self.logger.debug(f"[FeatureExtractor.load] done in {elapsed:.2f}s")

    def extract_from_csv_window(self, eeg_window: pd.DataFrame, fnirs_window: pd.DataFrame) -> np.ndarray:
        """Fallback for CSV-based real-time demo."""
        # Maps the 15 features identically from the DataFrame columns
        features = np.zeros(15, dtype=np.float32)
        features[0] = eeg_window['sensor_1'].mean()
        features[1] = eeg_window['sensor_2'].mean()
        features[2] = eeg_window['sensor_3'].mean()
        features[3] = eeg_window['signal_energy'].mean()
        features[4] = eeg_window['rolling_avg'].mean()
        features[5] = eeg_window['diff_1'].std()
        features[6] = eeg_window['combined_signal'].mean()
        features[7] = eeg_window['avg_signal'].mean()
        
        features[8] = fnirs_window['fnirs_channel_1'].mean()
        features[9] = fnirs_window['fnirs_channel_2'].mean()
        features[10] = fnirs_window['fnirs_channel_3'].mean()
        
        y = fnirs_window['rolling_avg'].values
        features[11] = np.polyfit(np.arange(len(y)), y, 1)[0] if len(y) > 1 else 0.0
        
        features[12] = features[7] / (fnirs_window['avg_signal'].mean() + 1e-10)
        features[13] = features[3] * fnirs_window['combined_signal'].mean()
        features[14] = features[4] - fnirs_window['rolling_avg'].mean()
        
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        if self._mean is not None and self._std is not None:
            features = (features - self._mean) / (self._std + 1e-8)
        return features
