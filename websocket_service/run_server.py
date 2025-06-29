#!/usr/bin/env python3
"""
Simple script to run the MuseTalk WebSocket server.
"""
import os
import sys
from pathlib import Path

# Add src to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

# Import and run server
if __name__ == "__main__":
    from server import main
    main()