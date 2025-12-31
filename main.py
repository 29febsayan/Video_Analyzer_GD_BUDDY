#!/usr/bin/env python3
"""
FastAPI Application for Visual Behavior Analysis System

Provides REST API endpoints for real-time behavior analysis from video frames.
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Header, Depends, Security
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
from typing import Optional
import cv2
import numpy as np
import base64
import io
from PIL import Image
import time

from production_ready_analyzer import ProductionBehaviorAnalyzer, ProductionConfig
from api_keys import get_key_manager, APIKeyManager

# Initialize FastAPI app
app = FastAPI(
    title="Visual Behavior Analysis API",
    description="Real-time behavioral metrics analysis from video frames",
    version="1.0.0"
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize analyzer
config = ProductionConfig()
analyzer = ProductionBehaviorAnalyzer(config)

# Initialize API key manager
key_manager = get_key_manager()

# API Key security
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(api_key: Optional[str] = Security(api_key_header)):
    """
    Verify API key from request header.
    
    API key should be sent in header: X-API-Key: your_api_key_here
    """
    if not api_key:
        raise HTTPException(
            status_code=401,
            detail="API key required. Please provide X-API-Key header."
        )
    
    if not key_manager.validate_key(api_key):
        raise HTTPException(
            status_code=401,
            detail="Invalid or expired API key."
        )
    
    return api_key


class FrameAnalysisResponse(BaseModel):
    """Response model for frame analysis."""
    frame: int
    frame_confidence: float
    confidence_status: str
    attention: Optional[float] = None
    head_movement: Optional[float] = None
    shoulder_tilt: Optional[float] = None
    hand_activity: Optional[float] = None
    hands_detected: int = 0
    timestamp: float
    success: bool
    warnings: list = []


class HealthResponse(BaseModel):
    """Health check response model."""
    status: str
    analyzer_ready: bool
    timestamp: float


def decode_image(image_data: bytes) -> np.ndarray:
    """Decode image bytes to numpy array."""
    try:
        # Try to decode as image
        image = Image.open(io.BytesIO(image_data))
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        # Convert to numpy array
        frame = np.array(image)
        # Convert RGB to BGR for OpenCV
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        return frame
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image format: {str(e)}")


def decode_base64_image(base64_string: str) -> np.ndarray:
    """Decode base64 string to numpy array."""
    try:
        # Remove data URL prefix if present
        if ',' in base64_string:
            base64_string = base64_string.split(',')[1]
        
        # Decode base64
        image_data = base64.b64decode(base64_string)
        return decode_image(image_data)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid base64 image: {str(e)}")


@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint - health check."""
    return {
        "status": "healthy",
        "analyzer_ready": analyzer is not None,
        "timestamp": time.time()
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint (no API key required)."""
    return {
        "status": "healthy",
        "analyzer_ready": analyzer is not None,
        "timestamp": time.time()
    }


@app.post("/api-key/generate")
async def generate_api_key(
    name: str = "default",
    expires_days: Optional[int] = None
):
    """
    Generate a new API key.
    
    Note: In production, this should be protected with admin authentication.
    For now, it's open but should be secured in production deployments.
    """
    try:
        api_key = key_manager.generate_key(name=name, expires_days=expires_days)
        return {
            "success": True,
            "api_key": api_key,
            "message": "Save this API key securely. It will not be shown again.",
            "name": name,
            "expires_days": expires_days
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating API key: {str(e)}")


@app.get("/api-key/info")
async def get_api_key_info(api_key: str = Depends(verify_api_key)):
    """Get information about the current API key."""
    key_info = key_manager.get_key_info(api_key)
    if key_info:
        # Don't expose the full key
        key_info["key_prefix"] = api_key[:20] + "..."
        return key_info
    raise HTTPException(status_code=404, detail="API key not found")


@app.post("/analyze/frame", response_model=FrameAnalysisResponse)
async def analyze_frame(
    file: UploadFile = File(...),
    api_key: str = Depends(verify_api_key)
):
    """
    Analyze a single frame from uploaded image file.
    
    Accepts: JPEG, PNG, or other image formats
    Returns: Behavioral metrics including attention, head movement, shoulder tilt, and hand activity
    """
    try:
        # Read image file
        image_data = await file.read()
        
        # Decode image
        frame = decode_image(image_data)
        
        # Process frame
        result = analyzer.process_frame(frame)
        
        # Extract metrics
        metrics = result.get('metrics', {})
        frame_confidence = result.get('frame_confidence', 0.0)
        
        # Determine confidence status
        confidence_status = "PASS" if frame_confidence >= 0.3 else "FAIL"
        
        # Get hand count
        hands_detected = result.get('detection_results', {}).get('hands_detected_count', 0)
        
        # Build response
        response = FrameAnalysisResponse(
            frame=result.get('frame_count', 0),
            frame_confidence=round(frame_confidence, 3),
            confidence_status=confidence_status,
            attention=round(metrics.get('attention_percent'), 1) if metrics.get('attention_percent') is not None else None,
            head_movement=round(metrics.get('head_movement_normalized'), 4) if metrics.get('head_movement_normalized') is not None else None,
            shoulder_tilt=round(metrics.get('shoulder_tilt_deg'), 1) if metrics.get('shoulder_tilt_deg') is not None else None,
            hand_activity=round(metrics.get('hand_activity_normalized'), 4) if metrics.get('hand_activity_normalized') is not None else None,
            hands_detected=hands_detected,
            timestamp=result.get('timestamp', time.time()),
            success=True,
            warnings=result.get('warnings', [])
        )
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing frame: {str(e)}")


@app.post("/analyze/base64", response_model=FrameAnalysisResponse)
async def analyze_base64_frame(
    request: dict,
    api_key: str = Depends(verify_api_key)
):
    """
    Analyze a single frame from base64 encoded image.
    
    Request body: {"image": "base64_string"}
    Returns: Behavioral metrics
    """
    try:
        # Get base64 string from request
        base64_string = request.get('image')
        if not base64_string:
            raise HTTPException(status_code=400, detail="Missing 'image' field in request")
        
        # Decode base64 image
        frame = decode_base64_image(base64_string)
        
        # Process frame
        result = analyzer.process_frame(frame)
        
        # Extract metrics
        metrics = result.get('metrics', {})
        frame_confidence = result.get('frame_confidence', 0.0)
        
        # Determine confidence status
        confidence_status = "PASS" if frame_confidence >= 0.3 else "FAIL"
        
        # Get hand count
        hands_detected = result.get('detection_results', {}).get('hands_detected_count', 0)
        
        # Build response
        response = FrameAnalysisResponse(
            frame=result.get('frame_count', 0),
            frame_confidence=round(frame_confidence, 3),
            confidence_status=confidence_status,
            attention=round(metrics.get('attention_percent'), 1) if metrics.get('attention_percent') is not None else None,
            head_movement=round(metrics.get('head_movement_normalized'), 4) if metrics.get('head_movement_normalized') is not None else None,
            shoulder_tilt=round(metrics.get('shoulder_tilt_deg'), 1) if metrics.get('shoulder_tilt_deg') is not None else None,
            hand_activity=round(metrics.get('hand_activity_normalized'), 4) if metrics.get('hand_activity_normalized') is not None else None,
            hands_detected=hands_detected,
            timestamp=result.get('timestamp', time.time()),
            success=True,
            warnings=result.get('warnings', [])
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing frame: {str(e)}")


@app.get("/analyze/detailed")
async def get_detailed_analysis():
    """
    Get detailed analysis information including explanations.
    Note: This requires processing a frame first via /analyze/frame or /analyze/base64
    """
    return {
        "message": "Use /analyze/frame or /analyze/base64 to process frames",
        "endpoints": {
            "analyze_frame": "/analyze/frame - POST with image file",
            "analyze_base64": "/analyze/base64 - POST with base64 image",
            "health": "/health - GET health check"
        }
    }


if __name__ == "__main__":
    import uvicorn
    import os
    import sys
    
    # Get PORT from environment, default to 8000
    # This handles Railway's PORT variable correctly
    try:
        port = int(os.environ.get("PORT", 8000))
    except (ValueError, TypeError):
        port = 8000
    
    # Start uvicorn
    uvicorn.run(app, host="0.0.0.0", port=port)

