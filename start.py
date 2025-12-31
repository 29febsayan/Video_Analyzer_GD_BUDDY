#!/usr/bin/env python3
"""
Startup script for Railway deployment
Reads PORT from environment and starts uvicorn
"""
import os
import sys

# Get PORT from environment, default to 8000
port = int(os.environ.get("PORT", 8000))

# Start uvicorn
# Use os.execvp to replace current process
os.execvp("uvicorn", [
    "uvicorn",
    "main:app",
    "--host", "0.0.0.0",
    "--port", str(port)
])

