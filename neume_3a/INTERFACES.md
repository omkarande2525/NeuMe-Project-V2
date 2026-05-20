# Interface Contracts — Module 3A

## Features → Classification
- `data/processed/z_matrix.pkl`
  - type: `numpy` float32, shape `(N_epochs, 15)`
  - 15 features = 8 EEG features + 4 fNIRS features + 3 combined features
- `data/processed/labels.pkl`
  - type: `list` of `str`, length `N_epochs`
  - values: exactly "FATIGUE", "FOCUSED", or "FLOW"
- `data/processed/subjects.pkl`
  - type: `list` of `str`, length `N_epochs`
  - values: Subject IDs for Stratified/Group splitting (e.g. "1", "2", "3", "4")
- `data/processed/normalizer.pkl`
  - type: `dict` `{"mean": np.array(15,), "std": np.array(15,)}`

## Features → Engine
- `features/feature_extractor.py`
  - class: `FeatureExtractor`
  - method: `.extract(eeg_epoch: np.ndarray, fnirs_epoch: np.ndarray) -> np.array(15,)`
  - method: `.load(path: str) -> None`
- `data/processed/llm_training_data.jsonl`
  - format: alpaca JSONL, each line: `{"instruction":..., "input":"", "output":...}`

## Classification → Engine
- `classification/predictor.py`
  - class: `CognitiveStatePredictor`
  - method: `.predict(z: np.array(15,)) -> tuple[str, float]`
    - returns: `("FATIGUE"|"FOCUSED"|"FLOW", confidence_0_to_1)`
  - method: `.load(model_path: str, encoder_path: str) -> None`
- `data/processed/eegnet_trained.pt`
- `data/processed/rf_baseline.pkl`
- `data/processed/label_encoder.pkl`
