# PDF Content Chunking for Vector Database Ingestion

A modular Python tool for processing JSON-structured PDF content and creating optimized chunks for vector database ingestion. The system intelligently chunks content while preserving hierarchical context (headers, sections) and merges small blocks based on size, semantic similarity, and continuity across pages.

## Features

- Extracts blocks from structured PDF JSON representation
- Preserves hierarchical context (section headers, paths)
- Intelligently merges small blocks based on:
  - Block size (token count)
  - Semantic similarity (using OpenAI embeddings)
  - Cross-page continuity detection
- Configurable chunk sizes and similarity thresholds
- Rich metadata for context preservation

## Code Structure

The system is organized into the following modules:

- `main.py` - Command-line entry point and orchestration
- `pdf_chunker.py` - Core chunking logic with the PDFChunker class
- `constants.py` - Configuration constants and parameters
- `utils.py` - Utility functions for text processing and embeddings
- `models.py` - Type definitions for better code readability

## Requirements

- Python 3.7+
- OpenAI API key (set as environment variable `OPENAI_API_KEY`)
- Required packages (see `requirements.txt`):
  - openai
  - numpy
  - tiktoken

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/pdf-content-chunking.git
   cd pdf-content-chunking
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Set your OpenAI API key:
   ```
   export OPENAI_API_KEY="your-api-key-here"
   ```

## Usage

### Basic Usage

```bash
python main.py -i input_pdf.json -o chunked_output.json
```

### Advanced Options

```bash
python main.py \
  -i input_pdf.json \
  -o chunked_output.json \
  --min-chunk-size 200 \
  --max-chunk-size 800 \
  --similarity-threshold 0.8
```

### Parameters

- `-i, --input`: Input JSON file (required)
- `-o, --output`: Output JSON file (default: chunked_output.json)
- `--min-chunk-size`: Minimum chunk size in tokens (default: 150)
- `--max-chunk-size`: Maximum chunk size in tokens (default: 1000)
- `--similarity-threshold`: Threshold for semantic similarity (default: 0.85)

## Testing

You can test the system with a sample JSON generator:

```bash
python sample_json.py  # Creates sample_pdf.json
python main.py -i sample_pdf.json -o chunked_output.json
```

## Output Format

The output is a JSON file containing an array of chunks, each with:

```json
[
  {
    "chunk_id": "chunk_0",
    "text": "The actual content of the chunk...",
    "metadata": {
      "types": ["Text", "SectionHeader"],
      "section_headers": {
        "section-id-1": "Section Title",
        "section-id-2": "Subsection Title"
      },
      "section_path": ["section-id-1", "section-id-2"],
      "page_ids": ["0", "1"],
      "token_count": 245
    },
    "block_ids": ["block-id-1", "block-id-2"],
    "section_path": ["section-id-1", "section-id-2"],
    "section_headers": {
      "section-id-1": "Section Title",
      "section-id-2": "Subsection Title"
    },
    "page_ids": ["0", "1"]
  }
]
```

## How It Works

1. **Content Extraction**: The system extracts text from HTML blocks in the PDF JSON.
2. **Context Preservation**: It tracks section headers and paths to maintain document hierarchy.
3. **Block Analysis**: Each block is analyzed for size (token count) and content.
4. **Semantic Similarity**: OpenAI embeddings are used to determine if blocks are semantically related.
5. **Cross-Page Detection**: The system detects incomplete sentences or list items that continue across pages.
6. **Chunk Formation**: Blocks are intelligently merged into chunks based on the analysis.

## Chunking Rules

Blocks are merged based on the following rules:

1. If a block is smaller than the minimum chunk size, it is merged with adjacent blocks.
2. If two blocks have semantic similarity above the threshold, they are merged.
3. If content appears to continue across pages (incomplete sentences, list items), blocks are merged.

## Using the Chunks with Vector Databases

The output chunks are ready for ingestion into vector databases like:

- Pinecone
- Weaviate
- Qdrant
- ChromaDB

Each chunk contains rich metadata that can be used for filtering and relevance scoring when querying.

## License

[MIT License](LICENSE) 