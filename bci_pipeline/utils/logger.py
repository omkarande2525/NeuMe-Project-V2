# ============================================================
# FILE: bci_pipeline/utils/logger.py
# ============================================================
"""Module-level logger factory."""

import logging
import logging.handlers
from pathlib import Path
import os

def get_logger(name: str) -> logging.Logger:
    """Creates or retrieves a logger with console and file handlers.
    
    Args:
        name: The name of the logger.
        
    Returns:
        logging.Logger: The configured logger instance.
    """
    logger = logging.getLogger(name)
    
    if logger.handlers:
        return logger
        
    logger.setLevel(logging.DEBUG)
    
    log_dir = Path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "pipeline.log"
    
    formatter = logging.Formatter("[%(asctime)s] %(levelname)-8s %(name)s — %(message)s")
    
    file_handler = logging.handlers.RotatingFileHandler(
        log_path, maxBytes=5 * 1024 * 1024, backupCount=3
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger
