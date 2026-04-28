"""
Entry point for running CannotMax via `python -m src.cannotmax`
"""
import sys
import os

# Add parent directory to path to allow imports from root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from main import main

if __name__ == "__main__":
    main()
