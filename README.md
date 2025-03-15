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

## Installation

```bash
pip install ai-chunking
```

## Quick Start

```python
from ai_chunking import RecursiveTextSplitter

# Initialize a recursive text splitter
chunker = RecursiveTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

# Read and process a markdown file
with open('documentation.md', 'r') as f:
    markdown_content = f.read()

chunks = chunker.split_text(markdown_content)

# Access the chunks
for i, chunk in enumerate(chunks, 1):
    print(f"Chunk {i}:\n{chunk}\n---")
```

## Usage Examples

### Recursive Text Splitting

The `RecursiveTextSplitter` splits markdown content into chunks while preserving markdown structure:

```python
from ai_chunking import RecursiveTextSplitter

splitter = RecursiveTextSplitter(
    chunk_size=1000,  # Maximum size of each chunk
    chunk_overlap=100,  # Overlap between chunks
    separators=["\n## ", "\n### ", "\n\n", "\n", " "]  # Markdown-aware separators
)

# Process a large markdown documentation file
with open('large_documentation.md', 'r') as f:
    markdown_content = f.read()

chunks = splitter.split_text(markdown_content)

# Save chunks to separate files for processing
for i, chunk in enumerate(chunks, 1):
    with open(f'chunk_{i}.md', 'w') as f:
        f.write(chunk)
```

### Section-based Semantic Chunking

The `SectionBasedSemanticChunker` is particularly well-suited for markdown files as it respects heading hierarchy:

```python
from ai_chunking import SectionBasedSemanticChunker

chunker = SectionBasedSemanticChunker(
    section_markers=["# ", "## ", "### "],  # Markdown heading levels
    min_chunk_size=100,
    max_chunk_size=1000
)

# Process markdown documentation
with open('api_docs.md', 'r') as f:
    markdown_content = f.read()

chunks = chunker.split_text(markdown_content)

# Print sections with their headings
for i, chunk in enumerate(chunks, 1):
    print(f"Section {i}:\n{chunk}\n---")
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

    def split_text(self, text: str) -> list[str]:
        # Split on markdown headings while preserving them
        sections = self.heading_pattern.split(text)
        # Remove empty sections and trim whitespace
        return [section.strip() for section in sections if section.strip()]

# Usage
chunker = MarkdownChunker()
with open('README.md', 'r') as f:
    chunks = chunker.split_text(f.read())
```

## Configuration

Each chunker accepts different configuration parameters:

### RecursiveTextSplitter
- `chunk_size`: Maximum size of each chunk (default: 500)
- `chunk_overlap`: Number of characters to overlap between chunks (default: 50)
- `separators`: List of separators to use for splitting (default: ["\n\n", "\n", " ", ""])

### SectionBasedSemanticChunker
- `section_markers`: List of strings that indicate section boundaries
- `min_chunk_size`: Minimum size of a chunk (default: 100)
- `max_chunk_size`: Maximum size of a chunk (default: 1000)

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
