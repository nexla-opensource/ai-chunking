import logging
import sys
from logging.handlers import RotatingFileHandler

def setup_logger(name='app', log_file='app.log', level=logging.INFO):
    """Configure and return a logger instance"""
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Create handlers
    console_handler = logging.StreamHandler(sys.stdout)
    file_handler = RotatingFileHandler(log_file, maxBytes=1024*1024, backupCount=5)

    # Create formatters
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Add formatters to handlers
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    # Add handlers to logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger

# Create default logger instance
logger = setup_logger() 