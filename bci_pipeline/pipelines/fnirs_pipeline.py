# ============================================================
# FILE: bci_pipeline/pipelines/fnirs_pipeline.py
# ============================================================
"""Pipeline for preprocessing fNIRS data."""

import numpy as np
import mne
from typing import Any
from utils.logger import get_logger

log = get_logger(__name__)

def run_fnirs_pipeline(raw: Any, config: dict) -> np.ndarray:
    """Runs the full fNIRS preprocessing pipeline.
    
    Args:
        raw: The raw fNIRS continuous data.
        config: Configuration dictionary.
        
    Returns:
        np.ndarray: Epoched fNIRS data of shape (N_epochs, n_channels, timepoints).
        
    Raises:
        ValueError: If data cannot be epoched due to short length.
    """
    # STEP 1 — Optical density conversion
    raw_od = mne.preprocessing.nirs.optical_density(raw)
    log.info(f"fNIRS Pipeline STEP 1: Converted to optical density, {len(raw_od.ch_names)} channels")
    
    # STEP 2 — Scalp coupling index quality check
    sci = mne.preprocessing.nirs.scalp_coupling_index(raw_od)
    bad_channels = [
        raw_od.ch_names[i]
        for i, s in enumerate(sci)
        if s < 0.5
    ]
    raw_od.info["bads"].extend(bad_channels)
    log.info(f"fNIRS Pipeline STEP 2: Scalp coupling index check, marked bad channels: {bad_channels}")
    
    # STEP 3 — Modified Beer-Lambert Law (haemoglobin conversion)
    raw_haemo = mne.preprocessing.nirs.beer_lambert_law(raw_od, ppf=0.1)
    log.info("fNIRS Pipeline STEP 3: Applied modified Beer-Lambert Law conversion")
    
    # STEP 4 — Extract HbO only
    raw_hbo = raw_haemo.copy().pick(picks="hbo")
    log.info(f"fNIRS Pipeline STEP 4: Selected HbO channels, count: {len(raw_hbo.ch_names)}")
    
    # STEP 5 — Bandpass filter
    raw_hbo.filter(
        l_freq=config["pipeline"]["fnirs"]["bandpass_low"],
        h_freq=config["pipeline"]["fnirs"]["bandpass_high"],
        method="iir",
        iir_params={"order": 3, "ftype": "butter"},
        picks="hbo",
        verbose=False
    )
    log.info("fNIRS Pipeline STEP 5: Applied bandpass filter")
    
    # STEP 6 — Sliding window epoching (MUST match EEG windows exactly)
    duration = config["pipeline"]["epoch_duration"]
    overlap = config["pipeline"]["epoch_overlap"]
    step = duration - overlap
    sfreq = raw_hbo.info["sfreq"]
    
    data = raw_hbo.get_data()
    n_channels, n_times = data.shape
    epoch_samples = int(duration * sfreq)
    step_samples = int(step * sfreq)
    
    if epoch_samples < 1:
        raise ValueError(
            f"Epoch duration {duration}s is shorter than one sample at sfreq={sfreq}Hz"
        )
        
    epochs_list = []
    start = 0
    while start + epoch_samples <= n_times:
        epoch = data[:, start : start + epoch_samples]
        epochs_list.append(epoch)
        start += step_samples
        
    if not epochs_list:
        raise ValueError("No fNIRS epochs could be extracted — recording too short. Try adjusting epoch duration.")
        
    result = np.stack(epochs_list, axis=0).astype(np.float32)
    log.info(f"fNIRS Pipeline STEP 6: Result shape {result.shape}")
    
    return result
