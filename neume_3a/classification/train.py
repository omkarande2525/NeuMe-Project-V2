import argparse
import time
import numpy as np
import yaml
from pathlib import Path
from sklearn.preprocessing import StandardScaler

from shared.logger import get_logger
from shared.io_utils import load_pickle, save_pickle
from features.augmentation import DataAugmentor
from classification.models.random_forest_baseline import RFBaseline

logger = get_logger("train")

def load_raw_data():
    """Loads the unnormalized, unaugmented z_matrix, labels, and subjects."""
    z_matrix = load_pickle("data/processed/z_matrix.pkl")
    labels = load_pickle("data/processed/labels.pkl")
    subjects = load_pickle("data/processed/subjects.pkl")
    return z_matrix, np.array(labels), np.array(subjects)

def strict_subject_split(X: np.ndarray, y: np.ndarray, groups: np.ndarray, val_group: str = "4"):
    """
    Splits the dataset strictly by subject ID to prevent leakage.
    Default: Subjects 1, 2, 3 -> Train, Subject 4 -> Val
    """
    train_idx = np.where(groups != val_group)[0]
    val_idx = np.where(groups == val_group)[0]
    
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    
    logger.info(f"Subject Split: Train on {len(np.unique(groups[train_idx]))} subjects, Val on {len(np.unique(groups[val_idx]))} subject")
    logger.info(f"Train size: {len(X_train)}, Val size: {len(X_val)}")
    
    return X_train, X_val, y_train.tolist(), y_val.tolist()

def train_random_forest(config: dict):
    logger.info("--- Starting Strict RF Training Pipeline ---")
    
    # 1. Load RAW data
    X, y, groups = load_raw_data()
    
    # 2. Strict Subject Split
    X_train, X_val, y_train, y_val = strict_subject_split(X, y, groups, val_group="4")
    
    # 3. Fit Normalizer ONLY on train
    scaler = StandardScaler()
    X_train_norm = scaler.fit_transform(X_train)
    X_val_norm = scaler.transform(X_val)
    
    # 4. Augment ONLY train
    augmentor = DataAugmentor(config)
    X_train_aug, y_train_aug = augmentor.balance_classes(X_train_norm, y_train, target_per_class=1000)
    
    # Shuffle train
    idx = np.random.permutation(len(X_train_aug))
    X_train_aug = X_train_aug[idx]
    y_train_aug = [y_train_aug[i] for i in idx]
    
    # 5. Train RF
    rf_model = RFBaseline(config)
    rf_model.fit(X_train_aug, y_train_aug)
    
    # 6. Evaluate on pure validation set
    logger.info("\n--- Evaluating on Validation Subject 4 ---")
    results = rf_model.evaluate(X_val_norm, y_val)
    
    # Save the strictly fitted models
    rf_model.save("data/processed/rf_baseline.pkl")
    
    # Create the normalizer dict for FeatureExtractor compatibility
    normalizer_state = {
        "mean": scaler.mean_.astype(np.float32),
        "std": scaler.scale_.astype(np.float32),
        "n_features": 15
    }
    save_pickle(normalizer_state, "data/processed/normalizer.pkl")
    logger.info("Saved strictly fitted rf_baseline.pkl and normalizer.pkl")
    
    return rf_model, results

if __name__ == "__main__":
    with open("config/config.yaml", 'r') as f:
        config = yaml.safe_load(f)
        
    train_random_forest(config)
