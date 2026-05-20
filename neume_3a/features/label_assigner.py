import time
import numpy as np
from shared.logger import get_logger

class LabelAssigner:
    """
    Derives 3-class cognitive labels from the signal energy percentiles.
    """

    def __init__(self, config: dict):
        self.config = config
        self.logger = get_logger("LabelAssigner")
        self.p33 = None
        self.p66 = None

    def fit(self, eeg_epochs: list[np.ndarray]) -> None:
        """
        Compute energy_33rd and energy_66th percentiles from the full dataset.
        eeg_epochs: list of np.ndarray of shape (9, 256)
        """
        t0 = time.perf_counter()
        self.logger.debug("[LabelAssigner.fit] starting")
        
        energies = [np.mean(epoch ** 2) for epoch in eeg_epochs]
        
        self.p33 = np.percentile(energies, 33)
        self.p66 = np.percentile(energies, 66)
        
        self.logger.info(f"Fitted energy thresholds: p33={self.p33:.4f}, p66={self.p66:.4f}")
        
        elapsed = time.perf_counter() - t0
        self.logger.debug(f"[LabelAssigner.fit] done in {elapsed:.2f}s")

    def assign(self, eeg_epoch: np.ndarray) -> str:
        """
        Given one epoch's EEG data, return label string.
        """
        if self.p33 is None or self.p66 is None:
            raise RuntimeError("LabelAssigner must be fit() before assign() is called.")
            
        energy = np.mean(eeg_epoch ** 2)
        
        if energy <= self.p33:
            return "FATIGUE"
        elif energy <= self.p66:
            return "FOCUSED"
        else:
            return "FLOW"

    def assign_batch(self, eeg_epochs: list[np.ndarray]) -> list[str]:
        """
        Calls assign() on each epoch in the list.
        """
        t0 = time.perf_counter()
        self.logger.debug("[LabelAssigner.assign_batch] starting")
        
        labels = [self.assign(ep) for ep in eeg_epochs]
        
        unique, counts = np.unique(labels, return_counts=True)
        self.logger.info(f"Class distribution after assignment: {dict(zip(unique, counts))}")
        
        elapsed = time.perf_counter() - t0
        self.logger.debug(f"[LabelAssigner.assign_batch] done in {elapsed:.2f}s")
        return labels
