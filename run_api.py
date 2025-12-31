#!/usr/bin/env python3
"""
FastAPI Server Launcher

Launches the Visual Behavior Analysis API server.
"""

import uvicorn
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    import os
    from pathlib import Path
    
    # Check if API keys exist
    keys_file = Path("api_keys.json")
    if not keys_file.exists() or not Path("api_keys.json").stat().st_size:
        print("=" * 70)
        print("[WARNING] No API keys found!")
        print("=" * 70)
        print("\nGenerating default API key...")
        from api_keys import APIKeyManager
        manager = APIKeyManager()
        api_key = manager.generate_key(name="default_key", expires_days=None)
        print(f"\n[SUCCESS] API Key Generated: {api_key}")
        print(f"\n[IMPORTANT] Save this key! Use it in requests: X-API-Key: {api_key}")
        print("\n" + "=" * 70 + "\n")
    
    print("=" * 70)
    print("Visual Behavior Analysis API Server")
    print("=" * 70)
    print("\nStarting server on http://0.0.0.0:8000")
    print("API Documentation: http://localhost:8000/docs")
    print("Alternative docs: http://localhost:8000/redoc")
    print("\n[INFO] API Key Authentication: Enabled")
    print("   All /analyze/* endpoints require X-API-Key header")
    print("\nPress Ctrl+C to stop the server\n")
    
    import os
    port = int(os.environ.get("PORT", 8000))
    reload = os.environ.get("ENV", "development") == "development"
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=reload,  # Auto-reload only in development
        log_level="info"
    )

