"""
Arthashila Application Launcher

This script launches the Arthashila application using the current Python
interpreter, avoiding environment path issues with Streamlit CLI.
"""

import os
import sys
import subprocess

def main():
    """
    Launch the Arthashila application by directly invoking
    the Streamlit module with the current Python interpreter.
    """
    try:
        print("Starting Arthashila Application...")
        # Use the current Python interpreter to run Streamlit
        import streamlit.web.cli as stcli
        
        # Get the absolute path to main.py
        dir_path = os.path.dirname(os.path.realpath(__file__))
        main_script = os.path.join(dir_path, "main.py")
        
        # Configure Streamlit arguments
        sys.argv = ["streamlit", "run", main_script, "--server.headless", "true"]
        
        # Run Streamlit
        sys.exit(stcli.main())
    except Exception as e:
        print(f"Error launching Arthashila: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 