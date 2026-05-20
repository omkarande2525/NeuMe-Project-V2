# ============================================================
# FILE: bci_pipeline/utils/io_utils.py
# ============================================================
"""I/O utilities for saving and loading data."""

import pickle
from pathlib import Path
from typing import Any
from .logger import get_logger

log = get_logger(__name__)

def save_pickle(obj: Any, path: Path | str) -> None:
    """Saves an object to a pickle file.
    
    Args:
        obj: The object to serialize.
        path: The file path to save to.
        
    Raises:
        IOError: If saving the file fails.
    """
    filepath = Path(path)
    try:
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump(obj, f, protocol=5)
        log.info(f"Saved {type(obj).__name__} to {filepath}")
    except Exception as e:
        err_msg = f"Failed to save pickle file at {filepath}: {e}"
        log.error(err_msg)
        raise IOError(err_msg) from e

def load_pickle(path: Path | str) -> Any:
    """Loads an object from a pickle file.
    
    Args:
        path: The file path to load from.
        
    Returns:
        The deserialized object.
        
    Raises:
        FileNotFoundError: If the file does not exist.
        IOError: If loading the file fails.
    """
    filepath = Path(path)
    if not filepath.exists():
        err_msg = f"Pickle file not found at {filepath}"
        log.error(err_msg)
        raise FileNotFoundError(err_msg)
        
    try:
        with open(filepath, "rb") as f:
            obj = pickle.load(f)
        log.info(f"Loaded {type(obj).__name__} from {filepath}")
        return obj
    except Exception as e:
        err_msg = f"Failed to load pickle file at {filepath}: {e}"
        log.error(err_msg)
        raise IOError(err_msg) from e
