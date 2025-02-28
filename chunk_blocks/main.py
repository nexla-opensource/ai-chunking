"""
Main entry point for PDF content chunking.
"""

import argparse
import json
import os
import sys
from typing import Dict, Any, List
from dotenv import load_dotenv

load_dotenv()

from constants import MIN_CHUNK_SIZE, MAX_CHUNK_SIZE, SIMILARITY_THRESHOLD
from pdf_chunker import PDFChunker


def main():
    """Main entry point."""
    input_path = "/Users/sakshammittal/Documents/Nexla OpenSource GitHub/ai-chunking/temp/input/oaktree/Brochure_Oaktree Performing Credit Platform_2024_06_30/Brochure_Oaktree Performing Credit Platform_2024_06_30.json"
    output_path = "/Users/sakshammittal/Documents/Nexla OpenSource GitHub/ai-chunking/temp/output/oaktree/block-chunking/Brochure_Oaktree Performing Credit Platform_2024_06_30_chunked.json"

    # Check if input file exists
    if not os.path.exists(input_path):
        print(f"Error: Input file '{input_path}' not found")
        sys.exit(1)
    
    # Check if OpenAI API key is set
    if not os.environ.get("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable is not set")
        sys.exit(1)
    
    try:
        # Load JSON content
        with open(input_path, 'r') as f:
            document = json.load(f)
        
        print(f"Processing document from {input_path}")
        
        # Extract source information from input path
        file_name = os.path.basename(input_path)
        base_name, ext = os.path.splitext(file_name)
        
        # Add source information to document metadata
        if "metadata" not in document:
            document["metadata"] = {}
        document["metadata"]["file_name"] = file_name
        document["metadata"]["source_name"] = base_name
        document["metadata"]["source_type"] = ext.lstrip(".") or "pdf"  # Use pdf as default if no extension
        
        # Initialize chunker
        chunker = PDFChunker(
            min_chunk_size=MIN_CHUNK_SIZE,
            max_chunk_size=MAX_CHUNK_SIZE,
            similarity_threshold=SIMILARITY_THRESHOLD
        )
        
        # Process document
        chunks = chunker.process_document(document)
        
        print(f"Created {len(chunks)} chunks")
        
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(output_path)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Created directory: {output_dir}")
        
        # Save output
        with open(output_path, 'w') as f:
            json.dump(chunks, f, indent=2)
        
        print(f"Saved chunked output to {output_path}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 