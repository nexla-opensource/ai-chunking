import os
import json
from typing import Dict, Any, List


def restructure_chunks(document_obj: Dict[str, Any]) -> List[Dict[str, Any]]:
    restructured_chunks = []
    all_summaries = []
    
    idx = 0
    for section in document_obj["sections"]:
        for chunk in section["chunks"]:
            metadata_attributes = [f"{key.replace('chunk_', '').replace('_', ' ')}: {value}" for key, value in chunk["metadata"].items()]
            metadata_text = '\n'.join(metadata_attributes)
            text = f"""
                {chunk["text"]}
                {chunk["global_context"]}

                {metadata_text}

                Source Name: {document_obj["name"]}
                Source Type: PDF
            """
            
            restructured_chunk = {
                "chunk_id": f"{idx}",
                "text": text,
                "metadata": {
                    "source_name": document_obj["name"],
                    "source_type": "pdf",
                    "total_pages": document_obj["total_pages"],
                    "document_summary": document_obj["document_summary"],
                    "text": chunk["text"],
                    "tokens_count": chunk["tokens_count"],
                    "page_numbers": chunk["page_numbers"],
                    "summary": chunk["summary"],
                    "global_context": chunk["global_context"],
                    # Placeholder values that will be updated later
                    "previous_summary": "",
                    "next_summary": "",
                    "questions": chunk["questions_this_excerpt_can_answer"],
                    "other_metadata": chunk["metadata"]
                }
            }
            
            restructured_chunks.append(restructured_chunk)
            all_summaries.append(chunk["summary"])
            idx += 1
    
    # Now map previous and next summaries
    for i in range(len(restructured_chunks)):
        if i > 0:
            restructured_chunks[i]["metadata"]["previous_summary"] = all_summaries[i-1]
        if i < len(restructured_chunks) - 1:
            restructured_chunks[i]["metadata"]["next_summary"] = all_summaries[i+1]
    
    return restructured_chunks


input_dir = "/Users/sakshammittal/Documents/Nexla OpenSource GitHub/ai-chunking/temp/output/financebench"
output_dir = "/Users/sakshammittal/Documents/Nexla OpenSource GitHub/ai-chunking/temp/output/financebench/section-based-chunks"

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Get all JSON files in the input directory
files = [f for f in os.listdir(input_dir) if f.endswith('.json')]
for file in files:
    filename = file.split(".")[0]
    with open(os.path.join(input_dir, file), "r") as f:
        document_obj = json.load(f)
        restructured_chunks = restructure_chunks(document_obj)
        json.dump(restructured_chunks, open(os.path.join(output_dir, f"{filename}.json"), "w"), indent=4)

