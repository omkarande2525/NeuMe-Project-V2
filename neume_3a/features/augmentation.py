import time
import numpy as np
from shared.logger import get_logger

class DataAugmentor:
    """
    Applies data augmentation to feature vectors (Z matrix) to improve 
    model robustness and balance classes.
    """

    def __init__(self, config: dict, random_seed: int = 42):
        self.config = config
        self.logger = get_logger("DataAugmentor")
        np.random.seed(random_seed)

    def gaussian_noise(self, X: np.ndarray, n_copies: int = 5, noise_std: float = 0.02) -> np.ndarray:
        """
        Add Gaussian noise to the feature vectors.
        Returns array of shape (n_copies * len(X), n_features).
        """
        t0 = time.perf_counter()
        self.logger.debug("[DataAugmentor.gaussian_noise] starting")
        
        augmented_list = []
        for _ in range(n_copies):
            noise = np.random.normal(0, noise_std, X.shape)
            augmented_list.append(X + noise)
            
        result = np.vstack(augmented_list)
        
        elapsed = time.perf_counter() - t0
        self.logger.debug(f"[DataAugmentor.gaussian_noise] done in {elapsed:.2f}s")
        return result

    def time_shift(self, X_epochs: np.ndarray, n_copies: int = 3, max_shift: int = 10) -> np.ndarray:
        """
        Note: This was originally designed for raw epochs.
        Since we now augment the Z matrix (features directly), time_shift is less applicable.
        We provide a simplified feature-level perturbation as proxy.
        """
        # Time shift at feature level is tricky. We'll add slightly correlated noise as proxy
        return self.gaussian_noise(X_epochs, n_copies, noise_std=0.01)

    def amplitude_scale(self, X: np.ndarray, n_copies: int = 3, scale_range: tuple = (0.85, 1.15)) -> np.ndarray:
        """
        Multiply feature vectors by random scalars within scale_range.
        """
        t0 = time.perf_counter()
        self.logger.debug("[DataAugmentor.amplitude_scale] starting")
        
        augmented_list = []
        for _ in range(n_copies):
            scales = np.random.uniform(scale_range[0], scale_range[1], size=(X.shape[0], 1))
            augmented_list.append(X * scales)
            
        result = np.vstack(augmented_list)
        
        elapsed = time.perf_counter() - t0
        self.logger.debug(f"[DataAugmentor.amplitude_scale] done in {elapsed:.2f}s")
        return result

    def balance_classes(self, X: np.ndarray, y: list, target_per_class: int = 1000) -> tuple[np.ndarray, list]:
        """
        Applies gaussian_noise augmentation to minority classes until
        each class has exactly target_per_class samples.
        """
        t0 = time.perf_counter()
        self.logger.debug("[DataAugmentor.balance_classes] starting")
        
        y_np = np.array(y)
        unique_classes, counts = np.unique(y_np, return_counts=True)
        
        self.logger.info(f"Class distribution before balancing: {dict(zip(unique_classes, counts))}")
        
        X_balanced = []
        y_balanced = []
        
        for cls in unique_classes:
            idx = np.where(y_np == cls)[0]
            X_cls = X[idx]
            n_samples = len(X_cls)
            
            if n_samples >= target_per_class:
                # Downsample or just take target_per_class
                selected_idx = np.random.choice(n_samples, target_per_class, replace=False)
                X_balanced.append(X_cls[selected_idx])
                y_balanced.extend([cls] * target_per_class)
            else:
                # Augment
                X_balanced.append(X_cls)
                y_balanced.extend([cls] * n_samples)
                
                n_needed = target_per_class - n_samples
                while n_needed > 0:
                    n_to_generate = min(n_needed, n_samples)
                    idx_to_augment = np.random.choice(n_samples, n_to_generate, replace=False)
                    X_aug = self.gaussian_noise(X_cls[idx_to_augment], n_copies=1, noise_std=0.03)
                    X_balanced.append(X_aug)
                    y_balanced.extend([cls] * n_to_generate)
                    n_needed -= n_to_generate
                    
        X_final = np.vstack(X_balanced)
        
        unique_classes_new, counts_new = np.unique(y_balanced, return_counts=True)
        self.logger.info(f"Class distribution after balancing: {dict(zip(unique_classes_new, counts_new))}")
        
        elapsed = time.perf_counter() - t0
        self.logger.debug(f"[DataAugmentor.balance_classes] done in {elapsed:.2f}s")
        return X_final, y_balanced
