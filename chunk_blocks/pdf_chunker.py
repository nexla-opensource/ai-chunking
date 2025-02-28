"""
PDFChunker class for chunking PDF content represented as a JSON structure.
"""

import json
import os
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from openai import OpenAI

from constants import (
    MIN_CHUNK_SIZE, 
    MAX_CHUNK_SIZE, 
    SIMILARITY_THRESHOLD, 
    MODEL_NAME,
    SKIP_BLOCK_TYPES,
    SENTENCE_ENDINGS,
    LIST_ITEM_BEGINNINGS
)
from utils import (
    extract_text_from_html, 
    cosine_similarity, 
    count_tokens,
    find_block_by_id,
    extract_page_id
)
from models import BlockInfo, Chunk, SectionInfo

class PDFChunker:
    """
    Class to process PDF content JSON and create chunks with preserved context.
    """
    
    def __init__(self, min_chunk_size: int = MIN_CHUNK_SIZE, 
                 max_chunk_size: int = MAX_CHUNK_SIZE,
                 similarity_threshold: float = SIMILARITY_THRESHOLD):
        """
        Initialize the PDFChunker.
        
        Args:
            min_chunk_size: Minimum chunk size in tokens
            max_chunk_size: Maximum chunk size in tokens
            similarity_threshold: Threshold for semantic similarity merging
        """
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.similarity_threshold = similarity_threshold
        self.client = OpenAI()
        
    def get_block_content(self, block: Dict[str, Any]) -> str:
        """
        Get the text content from a block.
        
        Args:
            block: Block dictionary
            
        Returns:
            Extracted text content
        """
        content = ""
        
        # Extract text based on block type
        if block.get("block_type") == "Text":
            content = extract_text_from_html(block.get("html", ""))
        elif block.get("block_type") == "SectionHeader":
            content = extract_text_from_html(block.get("html", ""))
        elif block.get("block_type") == "Table":
            content = extract_text_from_html(block.get("html", ""))
        elif block.get("block_type") == "ListItem":
            content = extract_text_from_html(block.get("html", ""))
        elif block.get("block_type") == "TableCell":
            content = extract_text_from_html(block.get("html", ""))
        
        return content.strip()
    
    def get_section_path(self, block: Dict[str, Any]) -> List[str]:
        """
        Get the section path for a block.
        
        Args:
            block: Block dictionary
            
        Returns:
            List of section IDs in hierarchical order
        """
        section_path = []
        if "sectionPath" in block and block["sectionPath"]:
            section_path = block["sectionPath"]
        return section_path
    
    def get_section_headers(self, document: Dict[str, Any], section_path: List[str]) -> Dict[str, str]:
        """
        Get the section header text for each section ID.
        
        Args:
            document: Full document JSON
            section_path: List of section IDs
            
        Returns:
            Dictionary mapping section IDs to their header text
        """
        section_headers = {}
        
        for section_id in section_path:
            section_block = find_block_by_id(document, section_id)
            if section_block and section_block.get("type") == "SectionHeader":
                header_text = self.get_block_content(section_block)
                section_headers[section_id] = header_text
        
        return section_headers
    
    def get_embedding(self, text: str) -> List[float]:
        """
        Get the embedding vector for a text.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
        """
        response = self.client.embeddings.create(
            input=text,
            model=MODEL_NAME
        )
        return response.data[0].embedding
    
    def should_merge_blocks(self, block1: BlockInfo, block2: BlockInfo) -> bool:
        """
        Determine if two blocks should be merged based on size and similarity.
        
        Args:
            block1: First block info
            block2: Second block info
            
        Returns:
            Boolean indicating whether blocks should be merged
        """
        # Small block merging
        block1_tokens = count_tokens(block1["content"])
        if block1_tokens < self.min_chunk_size:
            return True
            
        # Check if blocks are from consecutive pages in the same section
        if (block1["section_path"] == block2["section_path"] and 
            block1["page_id"] is not None and 
            block2["page_id"] is not None):
            
            # Check if they're consecutive pages
            try:
                page1 = int(block1["page_id"])
                page2 = int(block2["page_id"])
                if page2 - page1 == 1:
                    # Check if first block ends with a sentence break
                    # and second block starts with a new paragraph or list item
                    last_chars = block1["content"][-5:] if len(block1["content"]) > 5 else block1["content"]
                    first_chars = block2["content"][:20] if len(block2["content"]) > 20 else block2["content"]
                    
                    # Check if first block doesn't end with a sentence ending
                    if not any(last_chars.endswith(ending) for ending in SENTENCE_ENDINGS):
                        return True
                    
                    # Check if second block starts with a list item or appears to be a continuation
                    if any(first_chars.startswith(begin) for begin in LIST_ITEM_BEGINNINGS):
                        return True
            except ValueError:
                pass
        
        # Semantic similarity merging
        try:
            embedding1 = self.get_embedding(block1["content"])
            embedding2 = self.get_embedding(block2["content"])
            similarity = cosine_similarity(embedding1, embedding2)
            return similarity >= self.similarity_threshold
        except Exception:
            # If embeddings fail, fall back to just size-based merging
            return block1_tokens < self.min_chunk_size

    def process_document(self, document: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Process the document to create chunks with preserved context.
        
        Args:
            document: Document JSON structure
            
        Returns:
            List of chunks with context information
        """
        blocks: List[BlockInfo] = []
        
        # Get source information from document metadata
        source_name = document.get("metadata", {}).get("source_name", "")
        source_type = document.get("metadata", {}).get("source_type", "pdf")  # Default to pdf if not set
        
        # Extract blocks and metadata recursively
        def extract_blocks(node, current_path=None):
            if current_path is None:
                current_path = []
                
            # Skip specific block types
            if node.get("block_type") in SKIP_BLOCK_TYPES:
                # Process children of skipped blocks
                if "children" in node and node["children"]:
                    for child in node["children"]:
                        extract_blocks(child, current_path)
                return
                
            content = self.get_block_content(node)
            if not content.strip():
                # Process children of empty blocks
                if "children" in node and node["children"]:
                    for child in node["children"]:
                        extract_blocks(child, current_path)
                return
                
            section_path = []
            if "section_hierarchy" in node:
                # Convert section hierarchy to ordered list
                hierarchy = node["section_hierarchy"]
                ordered_keys = sorted(hierarchy.keys())
                section_path = [hierarchy[k] for k in ordered_keys]
            
            section_headers = {}
            for section_id in section_path:
                section_block = find_block_by_id(document, section_id)
                if section_block and section_block.get("block_type") == "SectionHeader":
                    header_text = self.get_block_content(section_block)
                    section_headers[section_id] = header_text
            
            page_id = extract_page_id(node.get("id", ""))
            
            block_info: BlockInfo = {
                "id": node.get("id", ""),
                "type": node.get("block_type", ""),
                "content": content,
                "section_path": section_path,
                "section_headers": section_headers,
                "page_id": page_id
            }
            
            blocks.append(block_info)
            
            # Process children
            if "children" in node and node["children"]:
                for child in node["children"]:
                    extract_blocks(child, current_path + [node.get("id", "")])
        
        # Start extraction from the root
        extract_blocks(document)
        
        if not blocks:
            print("Warning: No blocks were extracted from the document")
            return []
            
        # Create chunks
        chunks: List[Chunk] = []
        current_chunk: Optional[Chunk] = None
        
        for i, block in enumerate(blocks):
            if not current_chunk:
                # Start a new chunk
                current_chunk = {
                    "chunk_id": f"chunk_{len(chunks)}",
                    "text": block["content"],
                    "metadata": {
                        "source_type": source_type,
                        "source_name": source_name,
                        "types": [block["type"]],
                        "token_count": count_tokens(block["content"])
                    },
                    "block_ids": [block["id"]],
                    "section_path": block["section_path"],
                    "section_headers": block["section_headers"],
                    "page_ids": [block["page_id"]] if block["page_id"] else []
                }
            else:
                # Check if we should merge this block with the current chunk
                if i < len(blocks) - 1 and self.should_merge_blocks(block, blocks[i + 1]):
                    # Add to current chunk since next block should be merged with this one
                    new_text = current_chunk["text"] + "\n\n" + block["content"]
                    current_chunk["text"] = new_text
                    current_chunk["metadata"]["token_count"] = count_tokens(new_text)
                    current_chunk["metadata"]["types"].append(block["type"])
                    current_chunk["block_ids"].append(block["id"])
                    if block["page_id"] and block["page_id"] not in current_chunk["page_ids"]:
                        current_chunk["page_ids"].append(block["page_id"])
                    
                    # Update section path with deepest common path
                    if block["section_path"]:
                        if not current_chunk["section_path"]:
                            current_chunk["section_path"] = block["section_path"]
                        else:
                            # Find common section path
                            common_path = []
                            for i, section_id in enumerate(current_chunk["section_path"]):
                                if i < len(block["section_path"]) and section_id == block["section_path"][i]:
                                    common_path.append(section_id)
                                else:
                                    break
                            current_chunk["section_path"] = common_path
                    
                    # Update section headers
                    current_chunk["section_headers"].update(block["section_headers"])
                else:
                    # Finalize current chunk and start a new one
                    chunks.append(current_chunk)
                    current_chunk = {
                        "chunk_id": f"chunk_{len(chunks)}",
                        "text": block["content"],
                        "metadata": {
                            "source_type": source_type,
                            "source_name": source_name,
                            "types": [block["type"]],
                            "token_count": count_tokens(block["content"])
                        },
                        "block_ids": [block["id"]],
                        "section_path": block["section_path"],
                        "section_headers": block["section_headers"],
                        "page_ids": [block["page_id"]] if block["page_id"] else []
                    }
        
        # Add the last chunk if it exists
        if current_chunk:
            chunks.append(current_chunk)
            
        if not chunks:
            print("Warning: No chunks were created from the blocks")
            
        return chunks 