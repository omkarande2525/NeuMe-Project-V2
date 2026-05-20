# ============================================================
# FILE: bci_pipeline/tests/test_eeg_pipeline.py
# ============================================================
"""Tests for the EEG processing pipeline."""

import pytest
import mne
import numpy as np
from pipelines.eeg_pipeline import run_eeg_pipeline

@pytest.fixture
def default_config():
    return {
        "pipeline": {
            "epoch_duration": 1.0,
            "epoch_overlap": 0.5,
            "eeg": {
                "channels": ["F3", "F4", "Fz", "C3", "C4", "Cz", "P3", "P4", "Pz"],
                "bandpass_low": 1.0,
                "bandpass_high": 40.0,
                "notch_freq": 50.0,
                "resample_hz": 256,
                "max_ica_components": 9
            }
        }
    }

def test_eeg_pipeline_output_is_epochs(default_config):
    ch_names = ["F3", "F4", "Fz", "C3", "C4", "Cz", "P3", "P4", "Pz"]
    sfreq = 256
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")  # type: ignore[call-arg]
    data = np.random.randn(len(ch_names), sfreq * 10)  # 10 seconds of data
    raw = mne.io.RawArray(data, info)
    
    epochs = run_eeg_pipeline(raw, default_config)
    assert isinstance(epochs, mne.Epochs)

def test_eeg_pipeline_epoch_count(default_config):
    ch_names = ["F3", "F4", "Fz", "C3", "C4", "Cz", "P3", "P4", "Pz"]
    sfreq = 256
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")  # type: ignore[call-arg]
    data = np.random.randn(len(ch_names), sfreq * 10)  # 10 seconds
    raw = mne.io.RawArray(data, info)
    
    epochs = run_eeg_pipeline(raw, default_config)
    assert epochs.get_data().shape[0] == 19

def test_ica_components_never_exceed_channels(default_config):
    ch_names = ["F3", "C3", "P3"]  # Only 3 channels
    sfreq = 256
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")  # type: ignore[call-arg]
    data = np.random.randn(len(ch_names), sfreq * 10)
    raw = mne.io.RawArray(data, info)
    
    from unittest.mock import patch
    with patch("mne.preprocessing.ICA") as mock_ica:
        mock_ica_instance = mock_ica.return_value
        mock_ica_instance.find_bads_eog.return_value = ([], [])
        
        epochs = run_eeg_pipeline(raw, default_config)
        
        call_kwargs = mock_ica.call_args[1]
        n_components = call_kwargs["n_components"]
        assert n_components <= 3
        # Strict checking following RULE 1:
        # n_components = min(max_ica_components, len(ch_names) - 1)
        assert n_components <= len(ch_names) - 1
