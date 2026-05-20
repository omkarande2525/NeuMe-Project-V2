import logging
import sys
from pathlib import Path

# ANSI color codes
COLORS = {
    'DEBUG': '\033[36m',     # Cyan
    'INFO': '\033[32m',      # Green
    'WARNING': '\033[33m',   # Yellow
    'ERROR': '\033[31m',     # Red
    'CRITICAL': '\033[1;31m',# Bold Red
    'RESET': '\033[0m'
}

class ColorFormatter(logging.Formatter):
    def format(self, record):
        color = COLORS.get(record.levelname, COLORS['RESET'])
        record.levelname = f"{color}{record.levelname}{COLORS['RESET']}"
        record.name = f"\033[34m{record.name}{COLORS['RESET']}"
        return super().format(record)

def get_logger(name: str, log_file: str = None) -> logging.Logger:
    """
    Returns a configured thread-safe singleton logger.
    Format: "[HH:MM:SS] [LEVEL] [module] — message"
    """
    logger = logging.getLogger(name)
    
    # If already configured, return it
    if logger.handlers:
        return logger
        
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    
    fmt_str = "[%(asctime)s] [%(levelname)s] [%(name)s] — %(message)s"
    date_fmt = "%H:%M:%S"
    
    # Console Handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(ColorFormatter(fmt_str, datefmt=date_fmt))
    logger.addHandler(ch)
    
    # File Handler
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        from logging.handlers import RotatingFileHandler
        fh = RotatingFileHandler(str(log_path), maxBytes=5*1024*1024, backupCount=3)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter(fmt_str, datefmt=date_fmt))
        logger.addHandler(fh)
        
    return logger
