# ============================================================
# FILE: bci_pipeline/tests/test_fnirs_pipeline.py
# ============================================================
"""Tests for the fNIRS processing pipeline."""

import pytest
import mne
import numpy as np
from unittest.mock import patch
from pipelines.fnirs_pipeline import run_fnirs_pipeline

@pytest.fixture
def default_config():
    return {
        "pipeline": {
            "epoch_duration": 1.0,
            "epoch_overlap": 0.5,
            "fnirs": {
                "bandpass_low": 0.01,
                "bandpass_high": 0.1,
                "chromophore": "HbO"
            }
        }
    }

@patch("mne.preprocessing.nirs.optical_density")
@patch("mne.preprocessing.nirs.scalp_coupling_index")
@patch("mne.preprocessing.nirs.beer_lambert_law")
def test_fnirs_pipeline_output_shape(mock_beer_lambert, mock_sci, mock_od, default_config):
    ch_names = ["S1_D1 hbo", "S1_D2 hbo", "S2_D1 hbo", "S2_D2 hbo"]
    sfreq = 10
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="hbo")  # type: ignore[call-arg]
    data = np.random.randn(len(ch_names), sfreq * 30)  # 30 seconds
    raw = mne.io.RawArray(data, info)
    
    mock_od.return_value = raw
    mock_sci.return_value = [1.0, 1.0, 1.0, 1.0]
    mock_beer_lambert.return_value = raw
    
    result = run_fnirs_pipeline(raw, default_config)
    
    assert result.ndim == 3
    assert result.shape[0] > 0
    assert result.shape[1] == 4
    assert result.shape[2] == 10  # 1.0s * 10Hz

@patch("mne.preprocessing.nirs.optical_density")
@patch("mne.preprocessing.nirs.scalp_coupling_index")
@patch("mne.preprocessing.nirs.beer_lambert_law")
def test_fnirs_no_nan(mock_beer_lambert, mock_sci, mock_od, default_config):
    ch_names = ["S1_D1 hbo", "S1_D2 hbo", "S2_D1 hbo", "S2_D2 hbo"]
    info = mne.create_info(ch_names=ch_names, sfreq=10, ch_types="hbo")  # type: ignore[call-arg]
    data = np.random.randn(len(ch_names), 300)
    raw = mne.io.RawArray(data, info)
    
    mock_od.return_value = raw
    mock_sci.return_value = [1.0, 1.0, 1.0, 1.0]
    mock_beer_lambert.return_value = raw
    
    result = run_fnirs_pipeline(raw, default_config)
    assert not np.isnan(result).any()
