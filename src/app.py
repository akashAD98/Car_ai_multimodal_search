"""
Car AI Search Engine - Main Application Entry Point

This script imports and runs the Streamlit UI for the Car AI Search Engine.
It provides a convenient way to start the application from the root directory.

Usage:
    streamlit run src/app.py

Author: Akash Desai
"""

import os
import sys
from pathlib import Path

# Add the src directory to the Python path
current_dir = Path(__file__).parent
sys.path.append(str(current_dir.parent))

# Import main function from the UI module
from src.ui.car_search_ui import main

# Run the Streamlit app
if __name__ == "__main__":
    print("Starting Car AI Search Engine...")
    print("UI will open in your browser shortly...")
    main()  # Call the main function directly 