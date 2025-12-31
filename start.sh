#!/bin/sh
# Startup script for Railway deployment
# Properly handles PORT environment variable

# Get PORT from environment, default to 8000
PORT="${PORT:-8000}"

# Debug: Print PORT value (remove in production if needed)
# echo "Starting with PORT=$PORT"

# Start uvicorn with the correct port
exec uvicorn main:app --host 0.0.0.0 --port "$PORT"

