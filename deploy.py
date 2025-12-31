#!/usr/bin/env python3
"""
Deployment Script

Sets up and deploys the Visual Behavior Analysis API system.
"""

import os
import sys
import subprocess
from pathlib import Path
from api_keys import APIKeyManager


def check_dependencies():
    """Check if all required dependencies are installed."""
    print("Checking dependencies...")
    required = [
        'fastapi',
        'uvicorn',
        'opencv-python',
        'numpy',
        'mediapipe',
        'pydantic',
        'Pillow'
    ]
    
    missing = []
    for package in required:
        try:
            __import__(package.replace('-', '_'))
            print(f"  [OK] {package}")
        except ImportError:
            print(f"  [MISSING] {package}")
            missing.append(package)
    
    if missing:
        print(f"\nMissing dependencies: {', '.join(missing)}")
        print("Installing missing dependencies...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install"] + missing)
            print("Dependencies installed successfully!")
        except Exception as e:
            print(f"Error installing dependencies: {e}")
            return False
    
    return True


def generate_api_key():
    """Generate a default API key if none exists."""
    manager = APIKeyManager()
    
    # Check if keys file exists and has keys
    if manager.keys_file.exists() and manager.keys:
        print(f"\nAPI keys file exists: {manager.keys_file}")
        print("Existing keys found. Skipping key generation.")
        print("\nTo generate a new key, run: python api_keys.py")
        return None
    
    # Generate default key
    print("\nGenerating default API key...")
    api_key = manager.generate_key(name="default_key", expires_days=None)
    
    print("=" * 70)
    print("API KEY GENERATED")
    print("=" * 70)
    print(f"\nAPI Key: {api_key}")
    print(f"\n[IMPORTANT] Save this key securely!")
    print(f"   This key will be required for all API requests.")
    print(f"   Include it in the header: X-API-Key: {api_key}")
    print(f"\nKeys file: {manager.keys_file}")
    print("=" * 70)
    
    return api_key


def create_env_file(api_key: str = None):
    """Create .env file with configuration."""
    env_file = Path(".env")
    
    if env_file.exists():
        print("\n.env file already exists. Skipping creation.")
        return
    
    content = f"""# Visual Behavior Analysis API Configuration

# API Settings
API_HOST=0.0.0.0
API_PORT=8000
API_RELOAD=true

# API Key (if not set, generate one using: python api_keys.py)
# DEFAULT_API_KEY={api_key if api_key else "generate_using_api_keys.py"}

# CORS Settings
CORS_ORIGINS=*

# Logging
LOG_LEVEL=info
"""
    
    with open(env_file, 'w') as f:
        f.write(content)
    
    print(f"\nCreated .env file: {env_file}")


def main():
    """Main deployment function."""
    print("=" * 70)
    print("Visual Behavior Analysis API - Deployment")
    print("=" * 70)
    
    # Check dependencies
    if not check_dependencies():
        print("\n[ERROR] Dependency check failed. Please install dependencies manually.")
        return 1
    
    # Generate API key
    api_key = generate_api_key()
    
    # Create .env file
    create_env_file(api_key)
    
    print("\n" + "=" * 70)
    print("DEPLOYMENT COMPLETE")
    print("=" * 70)
    print("\nNext steps:")
    print("1. Start the API server: python run_api.py")
    print("2. Access API docs: http://localhost:8000/docs")
    print("3. Use your API key in requests: X-API-Key: <your_key>")
    print("\n" + "=" * 70)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

