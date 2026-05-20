# ============================================================
# FILE: bci_pipeline/tests/test_loaders.py
# ============================================================
"""Tests for dataset loaders."""

import pytest
import mne
import numpy as np
import scipy.io
from unittest.mock import patch
from loaders.physionet_loader import load_physionet
from loaders.seed_loader import load_seed
from loaders.fnirs_loader import load_fnirs

@patch("mne.datasets.eegbci.load_data")
@patch("mne.io.read_raw_edf")
def test_physionet_loader_returns_raw(mock_read_raw_edf, mock_load_data):
    info = mne.create_info(ch_names=["O1", "O2", "C3", "C4"], sfreq=160, ch_types="eeg")  # type: ignore[call-arg]
    data = np.random.randn(4, 1600)
    synthetic_raw = mne.io.RawArray(data, info)
    
    mock_load_data.return_value = ["dummy_path.edf"]
    mock_read_raw_edf.return_value = synthetic_raw
    
    config = {"datasets": {"physionet": {"tasks": [3]}}}
    raw = load_physionet(1, config)
    
    assert isinstance(raw, mne.io.BaseRaw)

@patch("scipy.io.loadmat")
@patch("glob.glob")
def test_seed_loader_dynamic_key(mock_glob, mock_loadmat):
    mock_glob.return_value = ["dummy_subject_1.mat"]
    
    mock_dict = {
        "djc_eeg1": np.random.randn(62, 1000).astype(np.float32),
        "__header__": b"some header data"
    }
    mock_loadmat.return_value = mock_dict
    
    config = {"datasets": {"seed": {"base_path": "dummy_path"}}}
    raw = load_seed(1, config)
    
    assert isinstance(raw, mne.io.RawArray)

@patch("glob.glob")
@patch("pathlib.Path.exists")
def test_fnirs_loader_file_not_found(mock_exists, mock_glob):
    mock_glob.return_value = []
    mock_exists.return_value = False
    
    config = {"datasets": {"fnirs": {"base_path": "dummy_path"}}}
    
    with pytest.raises(FileNotFoundError) as exc_info:
        load_fnirs(1, config)
        
    assert "https://openneuro.org/datasets/ds003831" in str(exc_info.value)
