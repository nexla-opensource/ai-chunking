# Contributing to AI-Chunking

Thank you for your interest in contributing to AI-Chunking! This document provides guidelines and instructions for contributing to the project.

## Table of Contents
- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Making Contributions](#making-contributions)
- [Pull Request Process](#pull-request-process)
- [Coding Standards](#coding-standards)
- [Testing Guidelines](#testing-guidelines)
- [Documentation](#documentation)

## Code of Conduct

We are committed to providing a welcoming and inclusive environment for all contributors. Please be respectful and professional in all interactions.

## Getting Started

1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/nexla-opensource/ai-chunking.git
   cd ai-chunking
   ```
3. Set up your development environment (see [Development Setup](#development-setup))

## Development Setup

1. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install development dependencies:
   ```bash
   pip install -e ".[dev]"
   ```

3. Install pre-commit hooks:
   ```bash
   pre-commit install
   ```

## Making Contributions

### Types of Contributions

We welcome various types of contributions:

1. Bug fixes
2. Feature enhancements
3. Documentation improvements
4. Performance optimizations
5. New chunking strategies
6. Test coverage improvements

### Project Structure

The project is organized as follows:

```
ai-chunking/
├── src/
│   └── ai_chunking/
│       ├── chunkers/           # Different chunking implementations
│       │   ├── auto_ai_chunker/
│       │   ├── recursive_text_splitting_chunker/
│       │   ├── section_based_semantic_chunker/
│       │   └── semantic_chunker/
│       ├── models/            # Data models and types
│       ├── utils/             # Utility functions
│       └── llm/              # LLM integration
├── tests/                    # Test files
├── docs/                     # Documentation
└── examples/                 # Example usage
```

## Pull Request Process

1. Create a new branch for your feature/fix:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes, following our [Coding Standards](#coding-standards)

3. Write/update tests as needed

4. Run the test suite:
   ```bash
   pytest
   ```

5. Commit your changes:
   ```bash
   git add .
   git commit -m "feat: your descriptive commit message"
   ```

6. Push to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```

7. Create a Pull Request through GitHub

## Coding Standards

We follow these coding standards:

1. Use Python type hints
2. Follow PEP 8 style guide
3. Document classes and functions using docstrings
4. Keep functions focused and single-purpose
5. Use meaningful variable and function names
6. Add comments for complex logic

Example:
```python
from typing import List, Optional

def process_chunks(
    chunks: List[str],
    min_size: int,
    max_size: Optional[int] = None
) -> List[str]:
    """
    Process text chunks ensuring they meet size constraints.

    Args:
        chunks: List of text chunks to process
        min_size: Minimum chunk size in characters
        max_size: Optional maximum chunk size in tokens

    Returns:
        List of processed chunks
    """
    # Implementation here
    pass
```

## Testing Guidelines

1. Write unit tests for new features
2. Ensure tests are deterministic
3. Use meaningful test names and descriptions
4. Include both positive and negative test cases
5. Mock external dependencies appropriately

Example test structure:
```python
def test_chunk_size_constraints():
    """Test that chunks respect size constraints."""
    chunker = SemanticChunker(min_size=100, max_size=1000)
    text = "..." # Test text
    chunks = chunker.split_text(text)
    
    assert all(len(chunk) >= 100 for chunk in chunks)
    assert all(count_tokens(chunk) <= 1000 for chunk in chunks)
```

## Documentation

1. Keep README.md updated with new features
2. Document API changes in docstrings
3. Update example code as needed
4. Include docstrings for all public functions
5. Add inline comments for complex logic

For significant changes:
1. Update the documentation in the `docs/` directory
2. Add example usage in the `examples/` directory
3. Update the changelog

## Questions or Need Help?

Feel free to:
1. Open an issue for questions
2. Join our community discussions
3. Reach out to maintainers

Thank you for contributing to AI-Chunking!
