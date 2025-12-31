# FastAPI Usage Guide

## Quick Start

### 1. Deploy the System

```bash
python deploy.py
```

This will:
- Check and install dependencies
- Generate an API key
- Set up configuration files

### 2. Get Your API Key

After deployment, your API key will be displayed. You can also generate a new one:

```bash
python api_keys.py
```

**⚠️ IMPORTANT**: Save your API key securely! You'll need it for all API requests.

### 3. Start the API Server

```bash
python run_api.py
```

Or directly:

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### 4. Access API Documentation

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## API Key Authentication

**All `/analyze/*` endpoints require API key authentication.**

Include your API key in the request header:

```
X-API-Key: your_api_key_here
```

## API Endpoints

### 1. Health Check

**GET** `/health`

Check if the API is running and analyzer is ready.

**Response:**
```json
{
  "status": "healthy",
  "analyzer_ready": true,
  "timestamp": 1704067200.0
}
```

### 2. Analyze Frame (File Upload)

**POST** `/analyze/frame`

Analyze a frame from an uploaded image file.

**Request:**
- Content-Type: `multipart/form-data`
- Body: `file` (image file: JPEG, PNG, etc.)

**Response:**
```json
{
  "frame": 344,
  "frame_confidence": 0.490,
  "confidence_status": "PASS",
  "attention": 79.5,
  "head_movement": 0.0023,
  "shoulder_tilt": 0.2,
  "hand_activity": 0.0018,
  "hands_detected": 2,
  "timestamp": 1704067200.0,
  "success": true,
  "warnings": []
}
```

### 3. Analyze Frame (Base64)

**POST** `/analyze/base64`

Analyze a frame from a base64-encoded image.

**Request:**
```json
{
  "image": "data:image/jpeg;base64,/9j/4AAQSkZJRg..."
}
```

**Response:** Same as `/analyze/frame`

## Example Usage

### Python Example

```python
import requests
import base64

# Your API key (get it from deploy.py or api_keys.py)
API_KEY = "your_api_key_here"
API_BASE_URL = "http://localhost:8000"

headers = {
    "X-API-Key": API_KEY
}

# Method 1: File upload
with open("frame.jpg", "rb") as f:
    response = requests.post(
        f"{API_BASE_URL}/analyze/frame",
        files={"file": f},
        headers=headers
    )
    print(response.json())

# Method 2: Base64
with open("frame.jpg", "rb") as f:
    image_data = base64.b64encode(f.read()).decode()
    response = requests.post(
        f"{API_BASE_URL}/analyze/base64",
        json={"image": image_data},
        headers=headers
    )
    print(response.json())
```

### cURL Example

```bash
# Set your API key
API_KEY="your_api_key_here"

# File upload
curl -X POST "http://localhost:8000/analyze/frame" \
  -H "accept: application/json" \
  -H "X-API-Key: $API_KEY" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@frame.jpg"

# Base64
curl -X POST "http://localhost:8000/analyze/base64" \
  -H "accept: application/json" \
  -H "X-API-Key: $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"image": "base64_string_here"}'
```

### JavaScript/TypeScript Example

```javascript
// Your API key
const API_KEY = "your_api_key_here";
const API_BASE_URL = "http://localhost:8000";

// File upload
const formData = new FormData();
formData.append('file', fileInput.files[0]);

fetch(`${API_BASE_URL}/analyze/frame`, {
  method: 'POST',
  headers: {
    'X-API-Key': API_KEY
  },
  body: formData
})
.then(response => response.json())
.then(data => console.log(data));

// Base64
const base64Image = canvas.toDataURL('image/jpeg');

fetch(`${API_BASE_URL}/analyze/base64`, {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
    'X-API-Key': API_KEY
  },
  body: JSON.stringify({ image: base64Image })
})
.then(response => response.json())
.then(data => console.log(data));
```

## Response Fields

- **frame**: Frame number (sequential counter)
- **frame_confidence**: Confidence score (0.0-1.0)
- **confidence_status**: "PASS" if confidence >= 0.3, else "FAIL"
- **attention**: Attention percentage (0-100) or null
- **head_movement**: Normalized head movement value or null
- **shoulder_tilt**: Shoulder tilt in degrees or null
- **hand_activity**: Normalized hand activity value or null
- **hands_detected**: Number of hands detected (0, 1, or 2)
- **timestamp**: Unix timestamp of analysis
- **success**: Boolean indicating successful processing
- **warnings**: List of any warnings generated during analysis

## Error Handling

All endpoints return appropriate HTTP status codes:

- **200**: Success
- **400**: Bad Request (invalid image format, missing data)
- **500**: Internal Server Error (processing failure)

Error response format:
```json
{
  "detail": "Error message here"
}
```

## Notes

- The analyzer maintains state across frames (for smoothing and calibration)
- Frame numbers are sequential and reset when the analyzer is restarted
- All metrics are validated and may be `null` if detection fails or confidence is too low
- The API is stateless - each request processes a single frame independently

