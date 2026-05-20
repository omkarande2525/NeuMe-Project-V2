# ============================================================
# FILE: bci_pipeline/loaders/seed_loader.py
# ============================================================
"""Loader for the SEED-IV EEG Dataset."""

import glob
import scipy.io
from pathlib import Path
import mne
from utils.logger import get_logger

log = get_logger(__name__)

def load_seed(subject_id: int, config: dict) -> mne.io.Raw:
    """Loads SEED-IV EEG MATLAB data for a specific subject.
    
    Args:
        subject_id: Integer ID of the subject.
        config: Configuration dictionary.
        
    Returns:
        mne.io.Raw: The converted EEG Raw object.
        
    Raises:
        FileNotFoundError: If the valid .mat file is missing.
        ValueError: If no valid EEG key is found in the MATLAB file.
    """
    base_path = Path(config["datasets"]["seed"]["base_path"]) / "eeg_raw_data"
    
    mat_files = glob.glob(str(base_path / f"*{subject_id}*.mat"))
    if not mat_files:
        mat_files = glob.glob(str(base_path / f"subject_{subject_id:02d}*.mat"))
        
    if not mat_files:
        raise FileNotFoundError(
            f"SEED-IV data not found for subject {subject_id}. "
            f"Request access and download from https://bcmi.sjtu.edu.cn/home/seed/seed-iv.html "
            f"and place MAT files under {base_path}."
        )
        
    filepath = mat_files[0]
    mat = scipy.io.loadmat(filepath)
    
    eeg_keys = [k for k in mat.keys() if k.endswith("_eeg1") and not k.startswith("__")]
    
    if not eeg_keys:
        available_keys = list(mat.keys())
        raise ValueError(
            f"Subject {subject_id:03d} failed: No valid '.endswith(\'_eeg1\')' "
            f"key found in SEED-IV file {filepath}. Available keys: {available_keys}"
        )
        
    eeg_key = eeg_keys[0]
    eeg_data = mat[eeg_key]
    
    n_channels, n_timepoints = eeg_data.shape
    sfreq = 200  # SEED-IV native sampling rate
    
    ch_names = [f"EEG{i:03d}" for i in range(n_channels)]
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")
    
    raw = mne.io.RawArray(eeg_data / 1e6, info, verbose=False)  # convert µV to V
    
    log.info(f"Subject {subject_id:03d}: SEED-IV used key '{eeg_key}', shape {eeg_data.shape}, sfreq {sfreq}Hz")
    return raw
