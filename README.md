# AI Chunking

A powerful Python library for semantic document chunking and enrichment using AI. This library provides intelligent document chunking capabilities for PDF, Word, and Markdown files using various chunking strategies.

## Features

- Multiple chunking strategies:
  - Semantic Chunking: AI-powered content-aware chunking
  - Recursive Chunking: Hierarchical document splitting
  - Section-based Semantic Chunking: Structure-aware semantic splitting
  - Custom AI Chunking: Configurable AI-based chunking

- Supported file formats:
  - PDF documents
  - Microsoft Word documents
  - Markdown files

## Installation

```bash
pip install ai-chunking
```

## Quick Start

```python
from ai_chunking import SemanticChunker

# Initialize a semantic chunker
chunker = SemanticChunker(
    chunk_size=500,
    chunk_overlap=50
)

# Process a document
chunks = chunker.chunk_document("path/to/your/document.pdf")

# Access the chunks
for chunk in chunks:
    print(f"Chunk content: {chunk.text}")
    print(f"Chunk metadata: {chunk.metadata}")
```

## Usage Examples

### Semantic Chunking

```python
from ai_chunking import SemanticChunker

chunker = SemanticChunker(
    chunk_size=500,
    chunk_overlap=50,
    semantic_similarity_threshold=0.7
)
```

### Recursive Chunking

```python
from ai_chunking import RecursiveChunker

chunker = RecursiveChunker(
    max_chunk_size=1000,
    min_chunk_size=100,
    overlap=50
)
```

### Section-based Semantic Chunking

```python
from ai_chunking import SectionBasedChunker

chunker = SectionBasedChunker(
    section_markers=["##", "###"],
    semantic_threshold=0.8
)
```

### Custom AI Chunking

```python
from ai_chunking import CustomAIChunker

chunker = CustomAIChunker(
    model="gpt-4",
    chunk_strategy="custom",
    config={
        "max_tokens": 500,
        "custom_rules": ["your_rules_here"]
    }
)
```

## Configuration

Each chunker type accepts different configuration parameters. Please refer to the documentation for detailed configuration options.

## Contributing

We welcome contributions! Please see our contributing guidelines for more details.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

- Documentation: [https://docs.ai-chunking.org](https://docs.ai-chunking.org)
- Issue Tracker: [GitHub Issues](https://github.com/yourusername/ai-chunking/issues)

## Citation

If you use this software in your research, please cite:

```bibtex
@software{ai_chunking2024,
  title = {AI Chunking: A Python Library for Semantic Document Processing},
  author = {Desai, Amey},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/yourusername/ai-chunking}
}
```
