import os
import logging
import sys
from datetime import datetime


#################################################### LOGGING CONFIGURATION ####################################################

def setup_logging():
    """Configure structured logging for the training pipeline"""
    
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    
    log_format = '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'

    logger = logging.getLogger('fraud_detection_training')
    logger.setLevel(logging.INFO)
    
    logger.handlers.clear()
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter(log_format)
    console_handler.setFormatter(console_formatter)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_handler = logging.FileHandler(f'{log_dir}/training_{timestamp}.log')
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(log_format)
    file_handler.setFormatter(file_formatter)
    
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger
