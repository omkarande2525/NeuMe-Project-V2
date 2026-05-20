# pyright: reportGeneralTypeIssues=false
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "bci_pipeline"))

import mne
import numpy as np
from typing import Any, cast
from pipelines.fnirs_pipeline import run_fnirs_pipeline
from utils.logger import get_logger

log = get_logger("scratch")

def _create_info(**kwargs: Any) -> Any:
    return mne.create_info(**kwargs)


def generate_synthetic_fnirs(subject_id: int):
    sfreq = 10.0
    duration = 300
    n_times = int(duration * sfreq)
    
    n_channels = 8
    ch_names = []
    for i in range(n_channels // 2):
        ch_names.append(f"S1_D{i+1} 760")
        ch_names.append(f"S1_D{i+1} 850")

    info = _create_info(ch_names=ch_names, sfreq=sfreq, ch_types="fnirs_cw_amplitude")
    data = 1.0 + 0.05 * np.random.randn(n_channels, n_times)
    raw = mne.io.RawArray(data, info, verbose=False)
    
    for ch_name, ch in zip(raw.ch_names, raw.info['chs']):
        if '760' in ch_name:
            ch['loc'][9] = 760.0
        elif '850' in ch_name:
            ch['loc'][9] = 850.0
            
    return raw

raw = generate_synthetic_fnirs(1)
config = {
    "pipeline": {
        "fnirs": {"bandpass_low": 0.01, "bandpass_high": 0.1},
        "epoch_duration": 1.0,
        "epoch_overlap": 0.5
    }
}
epochs = run_fnirs_pipeline(raw, config)  # type: ignore[call-arg]
print(epochs.shape)
