#!/usr/bin/env python3
"""
API Key Management System

Generates and manages API keys for authentication.
"""

import secrets
import json
import os
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime, timedelta


class APIKeyManager:
    """Manages API keys for authentication."""
    
    def __init__(self, keys_file: str = "api_keys.json"):
        self.keys_file = Path(keys_file)
        self.keys = self._load_keys()
    
    def _load_keys(self) -> Dict:
        """Load API keys from file."""
        if self.keys_file.exists():
            try:
                with open(self.keys_file, 'r') as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}
    
    def _save_keys(self):
        """Save API keys to file."""
        try:
            with open(self.keys_file, 'w') as f:
                json.dump(self.keys, f, indent=2)
        except Exception as e:
            print(f"Error saving API keys: {e}")
    
    def generate_key(self, name: str = "default", expires_days: Optional[int] = None) -> str:
        """
        Generate a new API key.
        
        Args:
            name: Name/identifier for the key
            expires_days: Number of days until expiration (None = never expires)
        
        Returns:
            The generated API key
        """
        # Generate a secure random key
        api_key = f"vb_analysis_{secrets.token_urlsafe(32)}"
        
        # Create key metadata
        key_data = {
            "name": name,
            "created": datetime.now().isoformat(),
            "active": True,
            "usage_count": 0,
            "last_used": None
        }
        
        if expires_days:
            key_data["expires"] = (datetime.now() + timedelta(days=expires_days)).isoformat()
        else:
            key_data["expires"] = None
        
        # Store key
        self.keys[api_key] = key_data
        self._save_keys()
        
        return api_key
    
    def validate_key(self, api_key: str) -> bool:
        """
        Validate an API key.
        
        Args:
            api_key: The API key to validate
        
        Returns:
            True if key is valid and active, False otherwise
        """
        if api_key not in self.keys:
            return False
        
        key_data = self.keys[api_key]
        
        # Check if key is active
        if not key_data.get("active", False):
            return False
        
        # Check expiration
        expires = key_data.get("expires")
        if expires:
            try:
                expires_date = datetime.fromisoformat(expires)
                if datetime.now() > expires_date:
                    return False
            except Exception:
                pass
        
        # Update usage stats
        key_data["usage_count"] = key_data.get("usage_count", 0) + 1
        key_data["last_used"] = datetime.now().isoformat()
        self._save_keys()
        
        return True
    
    def revoke_key(self, api_key: str) -> bool:
        """Revoke an API key."""
        if api_key in self.keys:
            self.keys[api_key]["active"] = False
            self._save_keys()
            return True
        return False
    
    def list_keys(self) -> List[Dict]:
        """List all API keys (without showing the actual keys)."""
        result = []
        for key, data in self.keys.items():
            result.append({
                "key_prefix": key[:20] + "...",
                "name": data.get("name", "unknown"),
                "created": data.get("created"),
                "active": data.get("active", False),
                "usage_count": data.get("usage_count", 0),
                "last_used": data.get("last_used"),
                "expires": data.get("expires")
            })
        return result
    
    def get_key_info(self, api_key: str) -> Optional[Dict]:
        """Get information about a specific API key."""
        if api_key in self.keys:
            data = self.keys[api_key].copy()
            return data
        return None


# Global instance
_key_manager = None

def get_key_manager() -> APIKeyManager:
    """Get the global API key manager instance."""
    global _key_manager
    if _key_manager is None:
        _key_manager = APIKeyManager()
    return _key_manager


if __name__ == "__main__":
    # Generate a default API key
    manager = APIKeyManager()
    api_key = manager.generate_key(name="default_key", expires_days=None)
    print("=" * 70)
    print("API KEY GENERATED")
    print("=" * 70)
    print(f"\nAPI Key: {api_key}")
    print(f"\nSave this key securely. It will be required for API access.")
    print(f"\nKeys file: {manager.keys_file}")
    print("=" * 70)

