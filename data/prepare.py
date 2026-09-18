"""Repository-facing data entrypoint.

Usage: python data/prepare.py --dataset Beauty --download
"""
import sys
from pathlib import Path

# Running a script inside data/ makes Python's initial import path `data/`.
# Add the repository root so the reusable `genrec.data` package is found.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from genrec.data.prepare import main

if __name__ == "__main__": main()
