#!/usr/bin/env python3
"""
Test script for Visual Behavior Analysis API

Tests the API endpoints with sample requests.
"""

import requests
import base64
import cv2
import numpy as np
from pathlib import Path
import json

API_BASE_URL = "http://localhost:8000"

# Load API key from file or use default
API_KEY = None
keys_file = Path("api_keys.json")
if keys_file.exists():
    try:
        with open(keys_file, 'r') as f:
            keys = json.load(f)
            # Get the first active key
            for key, data in keys.items():
                if data.get("active", False):
                    API_KEY = key
                    break
    except Exception:
        pass

if not API_KEY:
    print("[WARNING] No API key found!")
    print("Please run: python deploy.py or python api_keys.py")
    print("Then update API_KEY in this script or set it as environment variable.\n")


def test_health():
    """Test health check endpoint."""
    print("Testing /health endpoint...")
    try:
        response = requests.get(f"{API_BASE_URL}/health")
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}\n")
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}\n")
        return False


def test_analyze_frame_file(image_path: str):
    """Test frame analysis with file upload."""
    if not API_KEY:
        print("Skipping test - API key required\n")
        return False
    
    print(f"Testing /analyze/frame endpoint with file: {image_path}...")
    try:
        headers = {"X-API-Key": API_KEY}
        with open(image_path, "rb") as f:
            files = {"file": f}
            response = requests.post(f"{API_BASE_URL}/analyze/frame", files=files, headers=headers)
        
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"Frame: {data.get('frame')}")
            print(f"Frame Confidence: {data.get('frame_confidence')} [{data.get('confidence_status')}]")
            print(f"Attention: {data.get('attention')}%")
            print(f"Head Movement: {data.get('head_movement')}")
            print(f"Shoulder Tilt: {data.get('shoulder_tilt')}°")
            print(f"Hand Activity: {data.get('hand_activity')} ({data.get('hands_detected')} hands)")
            print(f"Success: {data.get('success')}")
            if data.get('warnings'):
                print(f"Warnings: {data.get('warnings')}")
            print()
            return True
        else:
            print(f"Error: {response.text}\n")
            return False
    except Exception as e:
        print(f"Error: {e}\n")
        return False


def test_analyze_frame_base64(image_path: str):
    """Test frame analysis with base64 encoding."""
    if not API_KEY:
        print("Skipping test - API key required\n")
        return False
    
    print(f"Testing /analyze/base64 endpoint with file: {image_path}...")
    try:
        # Read and encode image
        with open(image_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode()
        
        # Send request
        headers = {"X-API-Key": API_KEY}
        response = requests.post(
            f"{API_BASE_URL}/analyze/base64",
            json={"image": image_data},
            headers=headers
        )
        
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"Frame: {data.get('frame')}")
            print(f"Frame Confidence: {data.get('frame_confidence')} [{data.get('confidence_status')}]")
            print(f"Attention: {data.get('attention')}%")
            print(f"Head Movement: {data.get('head_movement')}")
            print(f"Shoulder Tilt: {data.get('shoulder_tilt')}°")
            print(f"Hand Activity: {data.get('hand_activity')} ({data.get('hands_detected')} hands)")
            print()
            return True
        else:
            print(f"Error: {response.text}\n")
            return False
    except Exception as e:
        print(f"Error: {e}\n")
        return False


def create_test_frame():
    """Create a simple test frame using OpenCV."""
    # Create a simple colored frame
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[:] = (50, 50, 50)  # Dark gray background
    
    # Add some text
    cv2.putText(frame, "Test Frame", (200, 240), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # Save frame
    test_frame_path = "test_frame.jpg"
    cv2.imwrite(test_frame_path, frame)
    return test_frame_path


def main():
    """Run all tests."""
    print("=" * 70)
    print("Visual Behavior Analysis API Test Suite")
    print("=" * 70)
    print()
    
    # Test health check
    if not test_health():
        print("API server is not running. Please start it with: python run_api.py")
        return
    
    # Check for test video or create test frame
    test_video = Path("WIN_20251230_00_17_22_Pro.mp4")
    test_frame = None
    
    if test_video.exists():
        # Extract first frame from video
        print("Extracting frame from test video...")
        cap = cv2.VideoCapture(str(test_video))
        ret, frame = cap.read()
        if ret:
            test_frame = "test_frame_from_video.jpg"
            cv2.imwrite(test_frame, frame)
            print(f"Saved test frame: {test_frame}\n")
        cap.release()
    else:
        # Create a simple test frame
        print("Creating test frame...")
        test_frame = create_test_frame()
        print(f"Created test frame: {test_frame}\n")
    
    if test_frame and Path(test_frame).exists():
        # Test file upload
        test_analyze_frame_file(test_frame)
        
        # Test base64
        test_analyze_frame_base64(test_frame)
    else:
        print("No test frame available. Please provide an image file.")
    
    print("=" * 70)
    print("Tests completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()

