# pyright: reportGeneralTypeIssues=false
# ============================================================
# FILE: bci_pipeline/pipelines/eeg_pipeline.py
# ============================================================
"""Pipeline for preprocessing EEG data."""

import numpy as np
import mne
from typing import Any, cast
from utils.logger import get_logger

log = get_logger(__name__)

def _create_ica(**kwargs: Any) -> Any:
    return mne.preprocessing.ICA(**kwargs)


def run_eeg_pipeline(raw: Any, config: dict) -> mne.Epochs:
    """Runs the full EEG preprocessing pipeline.
    
    Args:
        raw: The raw continuous EEG data.
        config: Configuration dictionary.
        
    Returns:
        mne.Epochs: Epoched and preprocessed EEG data.
        
    Raises:
        ValueError: If fewer than 3 target channels are available or epoch params are invalid.
    """
    # STEP 1 — Channel selection
    target_channels = config["pipeline"]["eeg"]["channels"]
    available = [ch for ch in target_channels if ch in raw.ch_names]
    if len(available) < 3:
        raise ValueError(f"Too few target channels found: {available}. Cannot proceed with spatial filtering.")
    
    raw = raw.copy().pick_channels(available, verbose=False)
    log.info(f"EEG Pipeline STEP 1: Channels selected: {available}")
    
    # STEP 2 — Resample
    target_sfreq = config["pipeline"]["eeg"]["resample_hz"]
    if raw.info["sfreq"] != target_sfreq:
        raw.resample(target_sfreq, verbose=False)
        log.info(f"EEG Pipeline STEP 2: Resampled to {target_sfreq}Hz")
        
    # STEP 3 — Bandpass filter
    raw.filter(
        l_freq=config["pipeline"]["eeg"]["bandpass_low"],
        h_freq=config["pipeline"]["eeg"]["bandpass_high"],
        method="iir",
        iir_params={"order": 5, "ftype": "butter"},
        picks="eeg",
        verbose=False
    )
    log.info("EEG Pipeline STEP 3: Applied bandpass filter")
    
    # STEP 4 — Notch filter
    raw.notch_filter(
        freqs=config["pipeline"]["eeg"]["notch_freq"],
        picks="eeg",
        method="iir",
        verbose=False
    )
    log.info("EEG Pipeline STEP 4: Applied notch filter")
    
    # STEP 5 — ICA artifact removal
    n_channels_available = len(raw.ch_names)
    if n_channels_available < 2:
        raise ValueError(f"Insufficient channels for ICA. Minimum 2 required, but only {n_channels_available} found.")
        
    n_components = min(
        config["pipeline"]["eeg"]["max_ica_components"],
        n_channels_available - 1
    )
    
    ica = _create_ica(
        n_components=n_components,
        method="fastica",
        random_state=42,
        max_iter=400,
        verbose=False
    )
    
    try:
        ica.fit(raw, picks="eeg", verbose=False)
        
        eog_indices = []
        for ch in ["Fp1", "Fp2"]:
            if ch in raw.ch_names:
                indices, _ = ica.find_bads_eog(raw, ch_name=ch, threshold=3.0, verbose=False)
                eog_indices.extend(indices)
                
        if not eog_indices:
            eog_indices = [0]
            
        ica.exclude = list(set(eog_indices))[:min(2, n_components)]
        ica.apply(raw, verbose=False)
        log.info(f"EEG Pipeline STEP 5: ICA applied, n_components used: {n_components}, components excluded: {ica.exclude}")
    except Exception as e:
        log.warning(f"EEG Pipeline STEP 5: ICA fitting or apply failed: {e}. Gracefully continuing with uncleaned raw.")
    
    # STEP 6 — Re-reference to average
    raw.set_eeg_reference("average", projection=False, verbose=False)
    log.info("EEG Pipeline STEP 6: Re-referenced to average")
    
    # STEP 7 — Sliding window epoching
    duration = config["pipeline"]["epoch_duration"]
    overlap = config["pipeline"]["epoch_overlap"]
    step = duration - overlap
    sfreq = raw.info["sfreq"]
    
    total_samples = raw.n_times
    epoch_samples = int(duration * sfreq)
    step_samples = int(step * sfreq)
    
    if epoch_samples < 1:
        raise ValueError(f"Epoch duration {duration}s is shorter than one sample at sfreq={sfreq}Hz.")
        
    onset_samples = np.arange(0, total_samples - epoch_samples + 1, step_samples)
    
    events = np.column_stack([
        onset_samples.astype(int),
        np.zeros(len(onset_samples), dtype=int),
        np.ones(len(onset_samples), dtype=int)
    ])
    
    epochs = mne.Epochs(
        raw,
        events=events,
        event_id={"window": 1},
        tmin=0.0,
        tmax=duration - (1.0 / sfreq),
        baseline=None,
        preload=True,
        reject=None,
        verbose=False
    )
    
    log.info(f"EEG Pipeline STEP 7: Epoched data into {len(epochs)} windows, shape {epochs.get_data().shape}")
    
    return epochs
