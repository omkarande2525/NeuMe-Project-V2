# NuMe-Project

## Overview

`NuMe-Project` is a Brain-Computer Interface (BCI) preprocessing pipeline for EEG and fNIRS datasets. The repository provides dataset loaders, preprocessing pipelines, validation logic, and an orchestrator for running modality-specific or multimodal workflows.

## Repository structure

- `bci_pipeline/`
  - `config.yaml` - default pipeline and dataset configuration.
  - `orchestrator.py` - command-line entry point for dataset loading, preprocessing, validation, and saving processed outputs.
  - `validators.py` - runtime checks for EEG epochs, fNIRS arrays, and multimodal alignment.
  - `loaders/` - dataset-specific loaders for PhysioNet EEG, SEED-IV EEG, and fNIRS SNIRF.
  - `pipelines/` - core preprocessing pipelines for EEG and fNIRS.
  - `tests/` - pytest suite covering loaders, pipeline output, and validation logic.
  - `utils/` - helper modules for logging and file I/O.
- `data/` - placeholder directories for raw and processed data.
- `logs/` - log output directory created at runtime.

## Key functionality

- EEG preprocessing: channel selection, resampling, filtering, ICA artifact reduction, re-referencing, and sliding-window epoching.
- fNIRS preprocessing: optical density conversion, scalp coupling quality filtering, Beer-Lambert conversion, HbO selection, filtering, and sliding-window epoching.
- Validation: ensures EEG and fNIRS outputs are structurally correct and aligned before persisting.
- Orchestration: supports `physionet`, `seed`, `fnirs`, or `multimodal` processing modes.

## Installation

Install the Python dependencies using the active Python 3.13 environment.

```powershell
C:/Users/LENOVO/AppData/Local/Programs/Python/Python313/python.exe -m pip install -r bci_pipeline/requirements.txt
```

> If `requirements.txt` is missing, install at least `mne`, `numpy`, `scipy`, `pytest`, `PyYAML`, and `tqdm`.

## Configuration

The default pipeline configuration is stored in `bci_pipeline/config.yaml`.

Important settings include:

- `pipeline.epoch_duration` and `pipeline.epoch_overlap`
- EEG channel selection and filter parameters under `pipeline.eeg`
- fNIRS filter parameters under `pipeline.fnirs`
- dataset paths for `physionet`, `fnirs`, and `seed`
- output paths under `paths.processed`
- optional fNIRS synthetic fallback control under `datasets.fnirs.allow_synthetic`

## Running the pipeline

From the repository root:

```powershell
C:/Users/LENOVO/AppData/Local/Programs/Python/Python313/python.exe bci_pipeline/orchestrator.py --config bci_pipeline/config.yaml --modality multimodal
```

Example mode flags:

- `--modality physionet`
- `--modality seed`
- `--modality fnirs`
- `--modality multimodal`

Use `--skip-existing` to avoid reprocessing already saved subject files.

## Running tests

```powershell
C:/Users/LENOVO/AppData/Local/Programs/Python/Python313/python.exe -m pytest -q
```

## Notes

- The fNIRS loader can optionally generate synthetic fallback data if `datasets.fnirs.allow_synthetic` is enabled.
- Processed outputs are saved as pickle files under the configured `paths.processed` directory.
