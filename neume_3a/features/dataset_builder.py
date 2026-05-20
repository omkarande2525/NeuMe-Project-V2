import time
import numpy as np
import yaml
import json
from pathlib import Path
from shared.logger import get_logger
from shared.io_utils import save_pickle, save_jsonl
from features.data_loader import TensorDataLoader
from features.feature_extractor import FeatureExtractor
from features.label_assigner import LabelAssigner
from features.augmentation import DataAugmentor

class DatasetBuilder:
    """
    Master builder that produces all Features deliverables.
    Extracts features from the real multimodal epoch tensors.
    """

    def __init__(self, config_path: str = 'config/config.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        self.logger = get_logger("DatasetBuilder")
        self.logger.info("Initializing DatasetBuilder components")
        
        self.data_loader = TensorDataLoader(self.config)
        self.extractor = FeatureExtractor()
        self.assigner = LabelAssigner(self.config)
        self.augmentor = DataAugmentor(self.config)

    def build_feature_matrix(self) -> tuple[np.ndarray, list, list]:
        """
        1. Load all 4 subjects
        2. Fit LabelAssigner
        3. Extract features into Z matrix
        4. Fit normalizer on Z
        5. Re-extract normalized Z
        """
        self.logger.info("--- Phase 1: Building Feature Matrix ---")
        
        all_eeg = []
        all_fnirs = []
        all_subjects = []
        
        # Load the 4 available subjects
        for subj_id in range(1, 5):
            try:
                eeg, fnirs = self.data_loader.load_subject(subj_id)
                all_eeg.extend(eeg)
                all_fnirs.extend(fnirs)
                all_subjects.extend([str(subj_id)] * len(eeg))
            except Exception as e:
                self.logger.error(f"Failed to load subject {subj_id}: {e}")
                
        self.logger.info(f"Loaded total {len(all_eeg)} epochs.")
        
        # Fit LabelAssigner
        self.assigner.fit(all_eeg)
        labels = self.assigner.assign_batch(all_eeg)
        
        # Extract raw features (NO NORMALIZATION YET to prevent leakage)
        z_matrix = np.array([self.extractor.extract(eeg, fnirs) for eeg, fnirs in zip(all_eeg, all_fnirs)])
        
        self.logger.info(f"Feature matrix shape: {z_matrix.shape}")
        return z_matrix, labels, all_subjects

    def augment_dataset(self, z_matrix: np.ndarray, labels: list, subjects: list) -> tuple[np.ndarray, list, list]:
        """
        Apply DataAugmentor to balance classes.
        Subjects are replicated to match the augmentations.
        """
        self.logger.info("--- Phase 2: Augmenting Dataset ---")
        z_aug, labels_aug = self.augmentor.balance_classes(z_matrix, labels, target_per_class=1000)
        
        # To maintain subject mapping, we just repeat the subject list proportionally
        # (This is a simplified assumption for the augmented data, real stratification uses original data)
        # For prototype engineering, we just append "aug" as the subject ID for synthetic data
        n_added = len(labels_aug) - len(labels)
        subjects_aug = subjects + ["aug"] * n_added
        
        # Shuffle
        idx = np.random.permutation(len(z_aug))
        z_aug = z_aug[idx]
        labels_aug = [labels_aug[i] for i in idx]
        subjects_aug = [subjects_aug[i] for i in idx]
        
        return z_aug, labels_aug, subjects_aug

    def build_llm_training_data(self, z_matrix: np.ndarray, labels: list) -> list[dict]:
        """Convert each (z_vector, label) pair into an Alpaca-format dict."""
        self.logger.info("--- Phase 3: Building LLM Training Data ---")
        
        records = []
        for z, label in zip(z_matrix, labels):
            energy = z[3]
            slope = z[11]
            modality_ratio = z[12]
            
            # Base templates
            templates = {
                "FATIGUE": {"lanes": 2, "speed": 0.5, "difficulty_adjustment": -0.15, "environment_theme": "calm", "narrative_tone": "encouraging", "distractor_density": 0.25},
                "FOCUSED": {"lanes": 3, "speed": 1.0, "difficulty_adjustment": 0.0, "environment_theme": "focus", "narrative_tone": "neutral", "distractor_density": 0.5},
                "FLOW": {"lanes": 4, "speed": 1.6, "difficulty_adjustment": 0.15, "environment_theme": "neural", "narrative_tone": "challenging", "distractor_density": 0.75}
            }
            
            base_params = templates[label]
            
            # Add variation
            params = {}
            for k, v in base_params.items():
                if isinstance(v, (int, float)):
                    variation = v * 0.1
                    new_val = v + np.random.uniform(-variation, variation)
                    params[k] = round(new_val, 3) if isinstance(v, float) else int(round(new_val))
                else:
                    params[k] = v
            
            instruction = f"You are the AI Game Director for Neume cognitive rehabilitation. A patient's real-time cognitive state has been classified. Output ONLY a valid JSON object with game parameters. Patient state: {label}. Feature summary: signal_energy={energy:.3f}, fnirs_slope={slope:.4f}, modality_ratio={modality_ratio:.3f}."
            
            records.append({
                "instruction": instruction,
                "input": "",
                "output": json.dumps(params)
            })
            
        self.logger.info(f"Generated {len(records)} LLM training records.")
        return records

    def run_all(self) -> None:
        """Executes full pipeline."""
        t0 = time.perf_counter()
        
        # 1. Build RAW feature matrix
        z_matrix, labels, subjects = self.build_feature_matrix()
        
        # 2. Build LLM Training Data (from raw features or representative samples)
        # Note: We generate LLM data here before any augmentation to ensure realistic feature distributions
        jsonl_records = self.build_llm_training_data(z_matrix, labels)
        
        # 3. Save
        self.logger.info("--- Phase 4: Saving Deliverables ---")
        save_pickle(z_matrix, "data/processed/z_matrix.pkl")
        save_pickle(labels, "data/processed/labels.pkl")
        save_pickle(subjects, "data/processed/subjects.pkl")
        save_jsonl(jsonl_records, "data/processed/llm_training_data.jsonl")
        
        elapsed = time.perf_counter() - t0
        self.logger.info(f"Pipeline completed in {elapsed:.2f}s. Outputs saved to data/processed/")

if __name__ == "__main__":
    builder = DatasetBuilder()
    builder.run_all()
