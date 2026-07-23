import sys
from pathlib import Path

# Make sibling test helper modules (e.g. cassette.py) importable as plain
# top-level imports ("import cassette") from any test in this directory.
sys.path.insert(0, str(Path(__file__).parent))
