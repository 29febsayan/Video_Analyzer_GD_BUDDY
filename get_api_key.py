#!/usr/bin/env python3
"""
Get API Key Script

Displays the current API key for use in other applications.
"""

import json
from pathlib import Path

def get_api_key():
    """Get the first active API key."""
    keys_file = Path("api_keys.json")
    
    if not keys_file.exists():
        print("No API keys file found. Run: python deploy.py")
        return None
    
    try:
        with open(keys_file, 'r') as f:
            keys = json.load(f)
        
        # Find first active key
        for key, data in keys.items():
            if data.get("active", False):
                return key, data
        
        print("No active API keys found.")
        return None
    except Exception as e:
        print(f"Error reading API keys: {e}")
        return None


if __name__ == "__main__":
    print("=" * 70)
    print("API KEY INFORMATION")
    print("=" * 70)
    
    result = get_api_key()
    
    if result:
        api_key, key_data = result
        print(f"\nAPI Key: {api_key}")
        print(f"\nKey Details:")
        print(f"  Name: {key_data.get('name', 'N/A')}")
        print(f"  Created: {key_data.get('created', 'N/A')}")
        print(f"  Usage Count: {key_data.get('usage_count', 0)}")
        print(f"  Last Used: {key_data.get('last_used', 'Never')}")
        print(f"  Expires: {key_data.get('expires', 'Never')}")
        print(f"\n" + "=" * 70)
        print("USE THIS KEY IN YOUR REQUESTS:")
        print("=" * 70)
        print(f"Header: X-API-Key: {api_key}")
        print("=" * 70)
    else:
        print("\nNo API key available. Run: python deploy.py")

