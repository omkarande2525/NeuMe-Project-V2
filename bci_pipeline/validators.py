# ============================================================
# FILE: bci_pipeline/validators.py
# ============================================================
"""Data validation routines for ensuring payload integrity."""

import numpy as np
import mne
from typing import Tuple
from utils.logger import get_logger

log = get_logger(__name__)

def validate_eeg_epochs(epochs: mne.BaseEpochs, subject_id: int) -> None:
    """Validates processed EEG epochs.
    
    Args:
        epochs: The generated EEG Epochs object.
        subject_id: The ID of the subject being validated.
        
    Raises:
        AssertionError: If validation checks fail.
    """
    assert isinstance(epochs, mne.BaseEpochs), f"Subject {subject_id:03d} failed: Expected mne.BaseEpochs, got {type(epochs)}"
    assert epochs.times is not None and len(epochs.times) > 0, f"Subject {subject_id:03d} failed: Epochs must have time points"
    
    data = epochs.get_data()
    assert data.shape[0] > 0, f"Subject {subject_id:03d} failed: No EEG epochs found. Try adjusting epoch params."
    assert data.ndim == 3, f"Subject {subject_id:03d} failed: EEG epochs must be 3D (n_epochs, n_ch, n_times)"
    
    log.info(f"Validation Subject {subject_id:03d}: EEG shape {data.shape}, sfreq {epochs.info['sfreq']}Hz")

def validate_fnirs_array(arr: np.ndarray, subject_id: int) -> None:
    """Validates processed fNIRS numerical arrays.
    
    Args:
        arr: The generated fNIRS numerical array.
        subject_id: The ID of the subject being validated.
        
    Raises:
        AssertionError: If validation checks fail.
    """
    assert isinstance(arr, np.ndarray), f"Subject {subject_id:03d} failed: Expected np.ndarray, got {type(arr)}"
    assert arr.ndim == 3, f"Subject {subject_id:03d} failed: fNIRS array must be 3D (N_epochs, n_channels, timepoints)"
    assert arr.shape[0] > 0, f"Subject {subject_id:03d} failed: No fNIRS epochs found. Try adjusting epoch params."
    assert not np.isnan(arr).any(), f"Subject {subject_id:03d} failed: fNIRS array contains NaN values"
    assert not np.isinf(arr).any(), f"Subject {subject_id:03d} failed: fNIRS array contains Inf values"
    
    log.info(f"Validation Subject {subject_id:03d}: fNIRS array shape {arr.shape}")

def validate_alignment(
    eeg_epochs: mne.BaseEpochs,
    fnirs_arr: np.ndarray,
    config: dict,
    subject_id: int | None = None,
) -> Tuple[mne.BaseEpochs, np.ndarray]:
    """Validates alignment of EEG and fNIRS outputs and trims to minimum epoch count.
    
    Args:
        eeg_epochs: The EEG Epochs object.
        fnirs_arr: The fNIRS numpy array.
        config: Configuration dictionary.
        subject_id: The ID of the subject being validated.
        
    Returns:
        Tuple containing the aligned (trimmed) EEG Epochs and fNIRS array.
        
    Raises:
        ValueError: If either modality contains zero epochs after evaluation.
    """
    subject_label = f"{subject_id:03d}" if subject_id is not None else "N/A"
    eeg_count = len(eeg_epochs)
    fnirs_count = fnirs_arr.shape[0]
    min_epochs = min(eeg_count, fnirs_count)

    if min_epochs == 0:
        raise ValueError(
            f"Cannot align EEG and fNIRS for subject {subject_label} because one modality has zero epochs."
        )

    if eeg_count != fnirs_count:
        log.warning(
            f"Epoch count mismatch for subject {subject_label}: EEG={eeg_count}, "
            f"fNIRS={fnirs_count}. Trimming both to {min_epochs} for alignment."
        )
        eeg_epochs = eeg_epochs[:min_epochs]
        fnirs_arr = fnirs_arr[:min_epochs]
        
    log.info("Alignment check: epoch count match confirmed after robust checking")
    return eeg_epochs, fnirs_arr
