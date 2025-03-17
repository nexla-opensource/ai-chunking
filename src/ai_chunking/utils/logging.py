import logging
import sys

def setup_logger(name="ai_chunking", level=logging.INFO):
    """Set up and configure logger for the package.
    
    Args:
        name (str): Name of the logger
        level (int): Logging level (default: logging.INFO)
        
    Returns:
        logging.Logger: Configured logger instance
    """
    logger = logging.getLogger(name)
    
    if not logger.handlers:  # Only add handler if the logger doesn't have one
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    logger.setLevel(level)
    logger.propagate = False
    
    return logger

# Create default logger instance
logger = setup_logger() 