#!/usr/bin/env python3
"""
ElectroML Launcher
Starts the web application and opens it in the default browser
"""

import os
import sys
import webbrowser
import threading
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

# Configurable port via environment variable
PORT = int(os.environ.get('PORT', 5005))

def open_browser():
    """Open the web browser after a short delay."""
    time.sleep(2)  # Wait for server to start
    webbrowser.open(f'http://localhost:{PORT}')

if __name__ == '__main__':
    print(f"""
    ╔══════════════════════════════════════════════════════╗
    ║                                                      ║
    ║                    ElectroML v2.1                    ║
    ║     Electrochemical Data Analysis Web Application    ║
    ║                                                      ║
    ╚══════════════════════════════════════════════════════╝

    Starting ElectroML server on port {PORT}...
    The application will open in your browser automatically.

    Press Ctrl+C to stop the server.
    """)

    # Start browser opening in a separate thread
    browser_thread = threading.Thread(target=open_browser)
    browser_thread.daemon = True
    browser_thread.start()

    # Import and run the refactored Flask app
    from app_refactored import app

    # Run the application
    app.run(host='127.0.0.1', port=PORT, debug=False)
