import logging
import sys
from pathlib import Path
# To avoid circular dependency if Config tries to log during init,
# make Config an optional argument or handle its absence.
# For simplicity here, assume Config can be instantiated without logging.
from .config_loader import Config 

# Store loggers to prevent duplicate handlers
_loggers = {}

def setup_logger(name: str = "SmartSupplyRL", config_instance: Config = None):
    if name in _loggers:
        return _loggers[name]

    if config_instance is None:
        try:
            config_instance = Config()
        except Exception as e: # Fallback if config fails during very early init
            print(f"Error initializing Config for logger: {e}. Using basic console logging.")
            logger = logging.getLogger(name)
            logger.setLevel(logging.INFO)
            ch = logging.StreamHandler(sys.stdout)
            ch.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
            logger.addHandler(ch)
            _loggers[name] = logger
            return logger


    log_level_str = config_instance.main_config.get("log_level", "INFO").upper()
    log_file_name = config_instance.main_config.get("log_file_name", "app.log")
    log_file_path = config_instance.get_path('output_dir_abs', 'logs', log_file_name)
    
    log_level = getattr(logging, log_level_str, logging.INFO)

    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    logger.propagate = False # Prevents log duplication if root logger also has handlers

    # Ensure log directory exists
    log_file_path.parent.mkdir(parents=True, exist_ok=True)

    # File Handler
    fh = logging.FileHandler(log_file_path)
    fh.setLevel(log_level)

    # Console Handler
    ch = logging.StreamHandler(sys.stdout)
    # You might want a different level for console, e.g., INFO, while file is DEBUG
    # ch.setLevel(logging.INFO) 
    ch.setLevel(log_level)


    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(module)s:%(lineno)d - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)
    
    _loggers[name] = logger
    return logger

# Usage:
# from smart_supply_rl.utils.logger import setup_logger
# logger = setup_logger(__name__) # Use module name for better context
# logger.info("This is an info message from my_module.")