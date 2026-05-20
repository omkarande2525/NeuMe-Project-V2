# ============================================================
# FILE: bci_pipeline/tests/test_validators.py
# ============================================================
"""Tests for pipeline validators."""

import pytest
import mne
import numpy as np
from validators import validate_eeg_epochs, validate_fnirs_array, validate_alignment

def test_validate_eeg_epochs_passes():
    info = mne.create_info(ch_names=["C3"], sfreq=100, ch_types="eeg")  # type: ignore[call-arg]
    events = np.array([[0, 0, 1], [100, 0, 1]])
    data = np.random.randn(2, 1, 100)
    epochs = mne.EpochsArray(data, info, events=events)
    
    validate_eeg_epochs(epochs, 1)  # Should not raise

def test_validate_eeg_epochs_fails_on_non_epochs():
    data = np.random.randn(2, 1, 100)
    with pytest.raises(AssertionError):
        validate_eeg_epochs(data, 1)  # type: ignore[arg-type]

def test_validate_fnirs_array_passes():
    data = np.random.randn(19, 4, 256).astype(np.float32)
    validate_fnirs_array(data, 1)  # Should not raise

def test_validate_fnirs_array_fails_on_nan():
    data = np.random.randn(19, 4, 256).astype(np.float32)
    data[0, 0, 0] = np.nan
    with pytest.raises(AssertionError):
        validate_fnirs_array(data, 1)

def test_validate_alignment_fails_on_epoch_count_mismatch():
    info = mne.create_info(ch_names=["C3"], sfreq=100, ch_types="eeg")  # type: ignore[call-arg]
    events = np.zeros((10, 3), dtype=int)
    events[:, 0] = np.arange(0, 1000, 100)
    data_eeg = np.random.randn(10, 1, 100)
    epochs = mne.EpochsArray(data_eeg, info, events=events)
    
    data_fnirs = np.random.randn(15, 4, 256)
    
    aligned_epochs, aligned_fnirs = validate_alignment(
        epochs, data_fnirs, config={}, subject_id=1
    )

    assert len(aligned_epochs) == 10
    assert aligned_fnirs.shape[0] == 10
