import json
import joblib
from pathlib import Path
from .logger import get_logger

logger = get_logger("io_utils")

def save_pickle(obj, path: str) -> None:
    """Save an object using joblib with compression."""
    try:
        ensure_dir(path)
        joblib.dump(obj, path, compress=3)
        file_size = Path(path).stat().st_size / (1024 * 1024)
        logger.debug(f"Saved {path} ({file_size:.2f} MB)")
    except Exception as e:
        logger.error(f"Failed to save pickle to {path}: {e}", exc_info=True)
        raise

def load_pickle(path: str):
    """Load an object using joblib."""
    try:
        if not Path(path).exists():
            raise FileNotFoundError(f"Pickle file not found: {path}")
        logger.debug(f"Loading {path}")
        return joblib.load(path)
    except Exception as e:
        logger.error(f"Failed to load pickle from {path}: {e}", exc_info=True)
        raise

def save_jsonl(records: list, path: str) -> None:
    """Save list of dicts to JSONL."""
    try:
        ensure_dir(path)
        with open(path, 'w', encoding='utf-8') as f:
            for rec in records:
                f.write(json.dumps(rec) + "\n")
        logger.debug(f"Saved {len(records)} records to {path}")
    except Exception as e:
        logger.error(f"Failed to save JSONL to {path}: {e}", exc_info=True)
        raise

def load_jsonl(path: str) -> list:
    """Load JSONL into list of dicts."""
    records = []
    try:
        if not Path(path).exists():
            raise FileNotFoundError(f"JSONL file not found: {path}")
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
        logger.debug(f"Loaded {len(records)} records from {path}")
        return records
    except Exception as e:
        logger.error(f"Failed to load JSONL from {path}: {e}", exc_info=True)
        raise

def ensure_dir(path: str) -> None:
    """Ensure parent directory exists."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
