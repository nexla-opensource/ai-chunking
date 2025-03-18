# AI Chunking

A powerful Python library for semantic document chunking and enrichment using AI. This library provides intelligent document chunking capabilities with various strategies to split text while preserving semantic meaning, particularly useful for processing markdown documentation.

## Features

- Multiple chunking strategies:
  - Recursive Text Splitting: Hierarchical document splitting with configurable chunk sizes
  - Section-based Semantic Chunking: Structure-aware semantic splitting using section markers
  - Base Chunking: Extensible base implementation for custom chunking strategies

- Key Benefits:
  - Preserve semantic meaning across chunks
  - Configurable chunk sizes and overlap
  - Support for various text formats
  - Easy to extend with custom chunking strategies

## Chunking Strategies

### RecursiveTextSplitter
Uses a hierarchical approach to split documents at natural breakpoints while maintaining context. It works by recursively splitting text using a list of separators in order of priority (headings, paragraphs, sentences, words).

- **Use when**: You have structured text with clear headings and sections and want simple, reliable chunking.
- **Benefits**: Fast, deterministic, maintains hierarchical structure.

### SemanticTextChunker
Uses embeddings to find natural semantic breakpoints in text, creating chunks that preserve meaning regardless of formatting or explicit structure markers.

- **Use when**: Working with documents without clear section markers or when meaning preservation is critical.
- **Benefits**: Creates cohesive chunks based on semantic similarity rather than arbitrary length.

### SectionBasedSemanticChunker
Processes documents by first identifying logical sections, then creating semantically meaningful chunks within each section. Enriches chunks with section and document summaries, providing rich context.

- **Use when**: You need deeper semantic understanding of document structure and relationships between sections.
- **Benefits**: Preserves document structure, adds rich metadata including section summaries and contextual relationships.

### AutoAIChunker
The most advanced chunker that uses LLMs to intelligently analyze document structure, creating optimal chunks with rich metadata and summaries.

- **Use when**: You need maximum semantic understanding and are willing to trade processing time for quality.
- **Benefits**: Superior chunk quality, rich metadata generation, deep semantic understanding.

## Installation

```bash
pip install ai-chunking
```

## Environment Setup

To use the semantic chunking capabilities, you need to set up the following environment variables:

### Required Environment Variables

```bash
# Required for semantic chunkers (SemanticTextChunker, SectionBasedSemanticChunker, AutoAIChunker)
export OPENAI_API_KEY=your_openai_api_key_here
```


You can also set these environment variables in your Python code:

```python
import os
os.environ["OPENAI_API_KEY"] = "your_openai_api_key_here"
```

## Usage Examples

### Processing Multiple Documents

Each chunker type supports processing multiple documents together:

```python
from ai_chunking import RecursiveTextSplitter

chunker = RecursiveTextSplitter(
    chunk_size=1000,
    chunk_overlap=100
)

# List of markdown files to process
files = ['document1.md', 'document2.md', 'document3.md']

# Process all documents at once
all_chunks = chunker.chunk_documents(files)

print(f"Processed {len(files)} documents into {len(all_chunks)} chunks")
```

### Processing Multiple Documents with Semantic Chunkers

```python
from ai_chunking import SectionBasedSemanticChunker, SemanticTextChunker

# Choose the chunker based on your needs
chunker = SemanticTextChunker(
    chunk_size=1000,
    similarity_threshold=0.85
)

# Or use the more advanced section-based chunker
# chunker = SectionBasedSemanticChunker(
#     min_chunk_size=100,
#     max_chunk_size=1000
# )

# List of files to process
markdown_files = ['user_guide.md', 'api_reference.md', 'tutorials.md']

# Process all documents, getting enriched chunks with metadata
all_chunks = chunker.chunk_documents(markdown_files)

# Access metadata for each chunk
for i, chunk in enumerate(all_chunks):
    print(f"Chunk {i}:")
    print(f"  Source: {chunk.metadata.get('filename')}")
    print(f"  Tokens: {chunk.metadata.get('tokens_count')}")
    if 'summary' in chunk.metadata:
        print(f"  Summary: {chunk.metadata.get('summary')}")
    print(f"  Text snippet: {chunk.text[:100]}...\n")
```

### Custom Chunking Strategy

You can create your own markdown-specific chunking strategy:

```python
from ai_chunking import BaseChunker
import re

class MarkdownChunker(BaseChunker):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.heading_pattern = re.compile(r'^#{1,6}\s+', re.MULTILINE)

    def chunk_documents(self, files: list[str]) -> list[str]:
        chunks = []
        for file in files:
            with open(file, 'r') as f:
                content = f.read()
                # Split on markdown headings while preserving them
                sections = self.heading_pattern.split(content)
                # Remove empty sections and trim whitespace
                chunks.extend([section.strip() for section in sections if section.strip()])
        return chunks

# Usage
chunker = MarkdownChunker()
chunks = chunker.chunk_documents(['README.md', 'CONTRIBUTING.md'])
```

## Configuration

Each chunker accepts different configuration parameters:

### RecursiveTextSplitter
- `chunk_size`: Maximum size of each chunk (default: 500)
- `chunk_overlap`: Number of characters to overlap between chunks (default: 50)
- `separators`: List of separators to use for splitting (default: ["\n\n", "\n", " ", ""])

### SemanticTextChunker
- `chunk_size`: Maximum chunk size in tokens (default: 1024)
- `min_chunk_size`: Minimum chunk size in tokens (default: 128)
- `similarity_threshold`: Threshold for semantic similarity when determining breakpoints (default: 0.9)

The chunker uses OpenAI's text-embedding-3-small model for embeddings to determine semantic breakpoints in the text. If chunks are too large, they are further split using recursive semantic chunking. If chunks are too small, they are merged with adjacent chunks when possible.

Each chunk includes metadata:
- Source filename
- Token count
- Source path

### SectionBasedSemanticChunker
- `min_section_chunk_size`: Minimum size for section chunks in tokens (default: 5000)
- `max_section_chunk_size`: Maximum size for section chunks in tokens (default: 50000)
- `min_chunk_size`: Minimum size for individual chunks in tokens (default: 128)
- `max_chunk_size`: Maximum size for individual chunks in tokens (default: 1000)
- `page_content_similarity_threshold`: Threshold for content similarity between pages (default: 0.8)

The chunker uses OpenAI's text-embedding-3-small model for embeddings with dynamic threshold settings:
- For chunks ≤ 500 tokens: 80% threshold (more aggressive splitting)
- For chunks 501-1000 tokens: 85% threshold (moderate splitting)
- For chunks > 1000 tokens: 90% threshold (less aggressive splitting)

Each chunk includes rich metadata:
- Unique chunk ID
- Text content and token count
- Page numbers
- Summary with metadata
- Global context
- Section and document summaries
- Previous/Next chunk summaries
- List of questions the excerpt can answer

## Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create a new branch for your feature
3. Write tests for your changes
4. Submit a pull request

For more details, see our [Contributing Guidelines](CONTRIBUTING.md).

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

- Issue Tracker: [GitHub Issues](https://github.com/nexla-opensource/ai-chunking/issues)
- Documentation: Coming soon

## Citation

If you use this software in your research, please cite:

```bibtex
@software{ai_chunking2024,
  title = {AI Chunking: A Python Library for Semantic Document Processing},
  author = {Desai, Amey},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/nexla-opensource/ai-chunking}
}
```
