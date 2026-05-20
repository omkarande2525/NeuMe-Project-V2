# ============================================================
# FILE: bci_pipeline/README.md
# ============================================================
# BCI Complete Data Pipeline

**Role: Member A — Data Pipeline**

This repository contains the data ingestion, preprocessing, and validating pipeline for our multimodal Brain-Computer Interface (BCI). The resulting clean and aligned data arrays are used directly by members B, C, and D.

## 1. Description
This module processes continuous EEG (PhysioNet & SEED-IV) and fNIRS (ds003831) data, applying standard cleaning protocols, performing physiological transformations (e.g. Beer-Lambert law), sliding window epoching (1.0s durations with 0.5s overlap), and returning synchronised pickled data files.

## 2. Installation Setup

Requires Python 3.11+. Standard install:

```bash
python -m venv venv
source venv/bin/activate  # or venv/Scripts/activate on Windows
pip install -r requirements.txt
```

## 3. Dataset Instructions

All raw datasets should be placed according to the `paths` defined in `config.yaml`:

### EEG (PhysioNet)
- **Automatic**: Included `loaders/physionet_loader.py` downloads data dynamically.
- Uses `mne.datasets.eegbci`.

### fNIRS (OpenNeuro ds003831)
- **Path**: `data/raw/fnirs/ds003831/`
- **Download Link**: [https://openneuro.org/datasets/ds003831](https://openneuro.org/datasets/ds003831)
- Install using datalad: `datalad install https://github.com/OpenNeuroDatasets/ds003831.git`

### EEG (SEED-IV)
- **Path**: `data/raw/seed_iv/eeg_raw_data/`
- **Download Link**: [https://bcmi.sjtu.edu.cn/home/seed/seed-iv.html](https://bcmi.sjtu.edu.cn/home/seed/seed-iv.html) (Requires Access Request)

## 4. Quick Start

Run the entire multimodal extraction sequence for the first 3 subjects:

```bash
python orchestrator.py --subjects 1 2 3 --modality multimodal
```

PhysioNet and SEED-IV are treated as completely independent datasets.
Available modalities: `physionet`, `seed`, `fnirs`, `multimodal` (PhysioNet + fNIRS).

## 5. Output Format Specifications

The processed files are pickled using `protocol=5` for fastest I/O speed.

1. **EEG output**: `data/processed/eeg_epochs_subj[id].pkl` containing `mne.Epochs`.
2. **fNIRS output**: `data/processed/fnirs_epochs_subj[id].pkl` containing `numpy.ndarray`.

*Alignment Trimming Exception*: If simultaneous multimodal epoch counts mismatch slightly natively, the pipeline automatically trims off the trailing epochs to exactly align counts before saving.

## 6. Interface Contract (for Members B/C/D)

| File | Format | Shape | Downstream |
|---|---|---|---|
| `eeg_epochs_subj001.pkl` | MNE Epochs | `(N, 9, 256)` | Members B/C/D |
| `fnirs_epochs_subj001.pkl` | np.ndarray | `(N, n_ch, 10)` | Members B/C/D |

**Note for Member B (Subject ID Mapping):**
While Member A provides the processed epochs per subject, Member B is responsible for generating the `subjects.pkl` file required by Member C for leave-one-subject-out (LOSO) cross-validation. 
Please map the subjects exactly as follows when building the feature matrix:
- `eeg_epochs_subj001.pkl` / `fnirs_epochs_subj001.pkl` -> Subject ID `1`
- `eeg_epochs_subj002.pkl` / `fnirs_epochs_subj002.pkl` -> Subject ID `2`
- ...and so on for all 10 subjects.

## 7. Configuration Reference (config.yaml)

- `pipeline.epoch_duration`: Length of analysis window in seconds (1.0).
- `pipeline.epoch_overlap`: Overlap of adjacent windows in seconds (0.5).
- `pipeline.eeg.channels`: List of EEG standard 10-20 channels to pick.
- `pipeline.eeg.resample_hz`: Target downsample rate (256).
- `pipeline.eeg.bandpass_low`/`bandpass_high`: Frequency for filtering.
- `pipeline.eeg.notch_freq`: Powerline interference filter point.
- `pipeline.eeg.max_ica_components`: Limit of fastICA artifact removal components.
- `datasets.physionet.tasks`: The list of paradigms to extract.
- `datasets.fnirs.base_path`: Directory housing subject OpenNeuro folders.
- `datasets.seed.base_path`: Directory containing MAT extraction.

## 8. Running Tests

To ensure the integrity of the data pipeline, run the whole testing suite using Pytest:

```bash
pytest tests/ -v
```
