"""
conftest.py
───────────
Adds the project root (auditor/) to sys.path so tests can import
src.* modules without installation.
"""
import sys
from pathlib import Path

# Insert the auditor/ directory (this file's parent) at the front of sys.path
sys.path.insert(0, str(Path(__file__).parent))
