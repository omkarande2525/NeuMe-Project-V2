# ============================================================
# FILE: bci_pipeline/loaders/physionet_loader.py
# ============================================================
"""Loader for the PhysioNet EEG Motor Movement/Imagery Dataset."""

import mne
from tqdm import tqdm
from utils.logger import get_logger

log = get_logger(__name__)

def load_physionet(subject_id: int, config: dict) -> mne.io.Raw:
    """Loads PhysioNet EEG data for a specific subject and concatenates runs.
    
    Args:
        subject_id: Integer ID of the subject (1 to 109).
        config: Configuration dictionary.
        
    Returns:
        mne.io.Raw: Concatenated MNE Raw object with standard 10-20 montage.
        
    Raises:
        ValueError: If subject_id is out of range.
    """
    if not (1 <= subject_id <= 109):
        raise ValueError(f"Subject {subject_id:03d} failed: Subject ID out of range [1, 109]. Check subject configuration.")
        
    tasks = config["datasets"]["physionet"]["tasks"]
    log.info(f"Downloading/loading PhysioNet data for subject {subject_id:03d}")
    
    try:
        raws = []
        for task in tqdm(tasks, desc=f"Loading runs for subject {subject_id:03d}"):
            paths = mne.datasets.eegbci.load_data(subject_id, task, update_path=True, verbose=False)
            for path in paths:
                raw_run = mne.io.read_raw_edf(path, preload=True, verbose=False)
                mne.datasets.eegbci.standardize(raw_run)
                raws.append(raw_run)
                
        raw = mne.concatenate_raws(raws, verbose=False)
        raw.set_montage("standard_1020", on_missing="warn", verbose=False)
        
        duration = raw.times[-1]
        log.info(f"Subject {subject_id:03d}: loaded {len(raws)} runs, total duration {duration:.2f}s")
        return raw
    except Exception as e:
        raise RuntimeError(f"Subject {subject_id:03d} failed during PhysioNet loading: Try checking network connection or clearing mne_data cache.") from e
