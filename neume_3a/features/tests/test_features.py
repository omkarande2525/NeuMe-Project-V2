import pytest
import numpy as np
import pandas as pd
from features.feature_extractor import FeatureExtractor
from features.label_assigner import LabelAssigner
from features.augmentation import DataAugmentor

def test_feature_extractor_output_shape():
    extractor = FeatureExtractor()
    eeg_epoch = np.random.randn(9, 256).astype(np.float32)
    fnirs_epoch = np.random.randn(8, 10).astype(np.float32)
    
    features = extractor.extract(eeg_epoch, fnirs_epoch)
    
    assert features.shape == (15,)
    assert features.dtype == np.float32

def test_feature_extractor_no_nan():
    extractor = FeatureExtractor()
    eeg_epoch = np.random.randn(9, 256).astype(np.float32)
    fnirs_epoch = np.random.randn(8, 10).astype(np.float32)
    
    # Introduce NaNs to test robustness
    eeg_epoch[0, 0] = np.nan
    fnirs_epoch[0, 0] = np.nan
    
    # Wait, the extractor uses np.mean which returns NaN if NaNs are present.
    # The nan_to_num at the end should catch it.
    features = extractor.extract(eeg_epoch, fnirs_epoch)
    assert not np.isnan(features).any()

def test_label_assigner_three_classes():
    assigner = LabelAssigner({})
    
    # Create epochs with distinct energy levels
    low_energy = np.ones((9, 256)) * 0.1
    mid_energy = np.ones((9, 256)) * 1.0
    high_energy = np.ones((9, 256)) * 10.0
    
    epochs = [low_energy, mid_energy, high_energy]
    
    assigner.fit(epochs)
    
    assert assigner.assign(low_energy) == "FATIGUE"
    assert assigner.assign(mid_energy) == "FOCUSED"
    assert assigner.assign(high_energy) == "FLOW"

def test_augmentation_increases_count():
    augmentor = DataAugmentor({})
    X = np.random.randn(50, 15)
    
    X_aug = augmentor.gaussian_noise(X, n_copies=3)
    assert X_aug.shape == (150, 15)
    
def test_augmentation_balance():
    augmentor = DataAugmentor({})
    X = np.random.randn(10, 15)
    y = ["FATIGUE"]*2 + ["FOCUSED"]*3 + ["FLOW"]*5
    
    X_bal, y_bal = augmentor.balance_classes(X, y, target_per_class=20)
    
    assert len(y_bal) == 60
    assert y_bal.count("FATIGUE") == 20
    assert y_bal.count("FOCUSED") == 20
    assert y_bal.count("FLOW") == 20
