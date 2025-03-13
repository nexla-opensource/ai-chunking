import logging
import os
from logging.handlers import RotatingFileHandler

from .processor import DocumentProcessor
from ai_chunking.llm.base import StructuredLLMClient, LLMConfig, LLMError
from ai_chunking.llm.factory import LLMFactory, LLMProvider
from ai_chunking.llm.providers.openai import OpenAIStructuredClient
from ai_chunking.chunkers.auto_chunking.models.document import (ContentType, EnrichedPageData, Heading, HeadingLevel,
                          Page, ProcessedChunk, TableData)

__version__ = "0.1.0"

# Configure logging
def setup_logging():
    # Create logs directory if needed
    os.makedirs('logs', exist_ok=True)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    
    # Configure package logger
    logger = logging.getLogger('ai_chunking')
    logger.setLevel(logging.DEBUG)
    logger.propagate = False  # Prevent propagation to root logger
    
    # Only set up handlers if none exist
    if not logger.handlers:
        # Console handler - INFO level with simple format
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter('%(levelname)s: %(message)s')
        console_handler.setFormatter(console_formatter)
        
        # File handler - DEBUG level with detailed format
        file_handler = RotatingFileHandler(
            'logs/ai_chunking.log',
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        
        # Add handlers
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)
        
        # Set specific logger levels
        logging.getLogger('ai_chunking.llm').setLevel(logging.DEBUG)
        # logging.getLogger('httpx').setLevel(logging.WARNING)  # Suppress httpx logs

# Set up logging when package is imported
setup_logging()

__all__ = [
    # Main processor
    "DocumentProcessor",
    
    # LLM components
    "StructuredLLMClient",
    "LLMConfig",
    "LLMError",
    "LLMFactory",
    "LLMProvider",
    "OpenAIStructuredClient",
    
    # Data models
    "ContentType",
    "EnrichedPageData",
    "Heading",
    "HeadingLevel",
    "Page",
    "ProcessedChunk",
    "TableData",
] 