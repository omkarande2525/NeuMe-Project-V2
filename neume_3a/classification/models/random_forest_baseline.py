import time
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from shared.logger import get_logger
from shared.io_utils import save_pickle, load_pickle

class RFBaseline:
    """
    Scikit-learn Random Forest baseline for 3-class cognitive state classification.
    Provides interpretable feature importance.
    """

    def __init__(self, config: dict = None):
        self.config = config or {}
        self.logger = get_logger("RFBaseline")
        self.clf = RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            class_weight='balanced',
            n_jobs=-1,
            random_state=42
        )
        self.label_encoder = LabelEncoder()
        self.feature_names = [
            'F3_mean', 'C3_mean', 'P3_mean', 'eeg_energy',
            'Fz_mean', 'F3_diff_std', 'eeg_all_mean', 'eeg_legacy_avg',
            'fnirs_ch1_mean', 'fnirs_ch2_mean', 'fnirs_ch3_mean', 'fnirs_slope',
            'modality_ratio', 'interaction_term', 'differential_baseline'
        ]

    def fit(self, X_train: np.ndarray, y_train: list) -> None:
        t0 = time.perf_counter()
        self.logger.debug("[RFBaseline.fit] starting")
        
        y_encoded = self.label_encoder.fit_transform(y_train)
        self.clf.fit(X_train, y_encoded)
        
        # Log feature importance
        importances = self.clf.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        self.logger.info("Top 5 Feature Importances:")
        for i in range(5):
            idx = indices[i]
            self.logger.info(f"  {i+1}. {self.feature_names[idx]} ({importances[idx]:.4f})")
            
        elapsed = time.perf_counter() - t0
        self.logger.debug(f"[RFBaseline.fit] done in {elapsed:.2f}s")

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        preds_encoded = self.clf.predict(X)
        probs = self.clf.predict_proba(X)
        preds_str = self.label_encoder.inverse_transform(preds_encoded)
        return preds_str, probs

    def evaluate(self, X_test: np.ndarray, y_test: list) -> dict:
        t0 = time.perf_counter()
        self.logger.debug("[RFBaseline.evaluate] starting")
        
        preds_str, probs = self.predict(X_test)
        
        acc = accuracy_score(y_test, preds_str)
        cm = confusion_matrix(y_test, preds_str, labels=self.label_encoder.classes_)
        
        self.logger.info(f"\nClassification Report:\n{classification_report(y_test, preds_str, digits=4)}")
        
        # Get per-class F1 for consistency
        report_dict = classification_report(y_test, preds_str, output_dict=True)
        f1_scores = {k: v['f1-score'] for k, v in report_dict.items() if k in self.label_encoder.classes_}
        
        results = {
            "accuracy": acc,
            "per_class_f1": f1_scores,
            "confusion_matrix": cm.tolist()
        }
        
        elapsed = time.perf_counter() - t0
        self.logger.debug(f"[RFBaseline.evaluate] done in {elapsed:.2f}s")
        return results

    def save(self, path: str) -> None:
        t0 = time.perf_counter()
        self.logger.debug("[RFBaseline.save] starting")
        state = {
            "clf": self.clf,
            "label_encoder": self.label_encoder
        }
        save_pickle(state, path)
        elapsed = time.perf_counter() - t0
        self.logger.debug(f"[RFBaseline.save] done in {elapsed:.2f}s")

    def load(self, path: str) -> None:
        t0 = time.perf_counter()
        self.logger.debug("[RFBaseline.load] starting")
        state = load_pickle(path)
        self.clf = state["clf"]
        self.label_encoder = state["label_encoder"]
        elapsed = time.perf_counter() - t0
        self.logger.debug(f"[RFBaseline.load] done in {elapsed:.2f}s")
