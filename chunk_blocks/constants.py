"""
Constants for PDF content chunking.
"""

# Chunking parameters
MIN_CHUNK_SIZE = 150  # Minimum chunk size in tokens
MAX_CHUNK_SIZE = 1000  # Maximum chunk size in tokens
SIMILARITY_THRESHOLD = 0.85  # Threshold for semantic similarity merging

# OpenAI configuration
MODEL_NAME = "text-embedding-3-small"  # OpenAI embedding model
ENCODING_NAME = "cl100k_base"  # Tiktoken encoding name for token counting

# Block types to skip when extracting content
SKIP_BLOCK_TYPES = ["Document", "Page", "ListGroup"]  # Skip these high-level container blocks

# Sentence endings for detecting incomplete sentences across pages
SENTENCE_ENDINGS = [
    ".", "!", "?", ":", ";",
    ".\n", "!\n", "?\n", ":\n", ";\n"
]

# List item beginnings for detecting continued lists across pages
LIST_ITEM_BEGINNINGS = [
    "• ", "- ", "* ", "1. ", "2. ", "3. ", "a. ", "b. ", "c. ",
    "i. ", "ii. ", "iii. ", "I. ", "II. ", "III. "
] 