# ============================================================
# FILE: bci_pipeline/loaders/fnirs_loader.py
# ============================================================
"""Loader for the fNIRS Dataset (ds003831)."""

import glob
from pathlib import Path
import mne
from utils.logger import get_logger

log = get_logger(__name__)

import numpy as np

def generate_synthetic_fnirs(subject_id: int) -> mne.io.Raw:
    log.info(f"fNIRS Subject {subject_id:03d}: Generating synthetic fNIRS fallback data")
    sfreq = 10.0
    duration = 1800 # 30 mins to ensure enough data
    n_times = int(duration * sfreq)
    
    n_channels = 16
    ch_names = []
    for i in range(n_channels // 2):
        ch_names.append(f"S1_D{i+1} 760")
        ch_names.append(f"S1_D{i+1} 850")

    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="fnirs_cw_amplitude")
    t = np.arange(n_times) / sfreq
    heartbeat = 0.1 * np.sin(2 * np.pi * 1.2 * t)
    data = 1.0 + 0.01 * np.random.randn(n_channels, n_times) + heartbeat
    
    raw = mne.io.RawArray(data, info, verbose=False)
    for ch_name, ch in zip(raw.ch_names, raw.info['chs']):
        if '760' in ch_name:
            ch['loc'][9] = 760.0
        elif '850' in ch_name:
            ch['loc'][9] = 850.0
        # set source and detector locs so distance is 0.03 (3cm)
        ch['loc'][3:6] = [0.0, 0.0, 0.0]  # source
        ch['loc'][6:9] = [0.03, 0.0, 0.0] # detector
            
    return raw

def load_fnirs(subject_id: int, config: dict) -> mne.io.Raw:
    """Loads fNIRS SNIRF data for a specific subject.
    
    Args:
        subject_id: Integer ID of the subject.
        config: Configuration dictionary.
        
    Returns:
        mne.io.Raw: The preloaded fNIRS Raw object.
    """
    base_path = Path(config["datasets"]["fnirs"]["base_path"])
    expected_dir = base_path / f"sub-{subject_id:02d}" / "nirs"
    
    snirf_files = glob.glob(str(expected_dir / "*.snirf"))
    allow_synthetic = config["datasets"]["fnirs"].get("allow_synthetic", False)

    if not snirf_files:
        message = (
            f"Subject {subject_id:03d}: fNIRS dataset ds003831 not found at {expected_dir}. "
            "Visit https://openneuro.org/datasets/ds003831 to obtain the data."
        )
        if allow_synthetic:
            log.warning(f"{message} Using synthetic fallback as configured.")
            return generate_synthetic_fnirs(subject_id)
        raise FileNotFoundError(message)

    filepath = Path(snirf_files[0])
    raw = mne.io.read_raw_snirf(filepath, preload=True, verbose=False)
    
    log.info(f"Subject {subject_id:03d}: loaded {filepath}, sfreq {raw.info['sfreq']}Hz, {len(raw.ch_names)} channels")
    return raw
