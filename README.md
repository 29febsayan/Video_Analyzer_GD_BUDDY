# Visual Behavior Analysis System

A real-time visual behavior analysis system that processes webcam video to analyze objective behavioral metrics using MediaPipe pretrained models and OpenCV.

## Features

- **Real-time Analysis**: Live webcam processing with immediate feedback
- **Objective Metrics**: Geometric and motion-based behavioral measurements
- **Multi-modal Detection**: Face, pose, and hand landmark analysis
- **Temporal Smoothing**: Stable metrics through moving average filtering
- **Cross-platform**: Works on Windows, Linux, and macOS
- **Local Processing**: No cloud dependencies or data uploads

## Behavioral Metrics

### Head Pose Analysis

- **Forward Attention Ratio**: Percentage of time facing forward (±15° yaw, ±10° pitch)
- **Head Movement Stability**: Variance in head position over time

### Posture Analysis

- **Shoulder Alignment**: Symmetry between left and right shoulder positions

### Hand Movement Analysis

- **Hand Movement Speed**: Frame-to-frame displacement of wrist landmarks
- **Gesture Variance**: Stability of hand movements over time

## Requirements

- Python 3.9 or 3.10
- Webcam (built-in or USB)
- Windows, Linux, or macOS

## Installation

### 1. Clone or Download

Download the project files to your local machine.

### 2. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 3. Verify Installation

```bash
python production_ready_analyzer.py
```

## Usage

### Basic Usage

1. **Start the Production System**:

   ```bash
   python production_ready_analyzer.py
   ```

2. **Or Launch Web Interface**:

   ```bash
   python run_streamlit.py
   ```

3. **Position Yourself**: Sit in front of your webcam with good lighting

4. **View Real-time Analysis**: The system will display:

   - Live video feed with detected landmarks
   - Current behavioral metrics and categories
   - Real-time feedback overlays

5. **Exit**: Press the ESC key to stop the application

### System Requirements

- **Camera Access**: Ensure your webcam is not being used by other applications
- **Lighting**: Good lighting improves landmark detection accuracy
- **Distance**: Sit 2-3 feet from the camera for optimal detection
- **Background**: Plain backgrounds work best for pose detection

## Troubleshooting

### Common Issues

**Camera Not Found**:

- Ensure webcam is connected and not in use by other applications
- Try running with administrator/sudo privileges if needed

**Poor Detection**:

- Improve lighting conditions
- Ensure face is clearly visible and not obscured
- Check camera focus and cleanliness

**Performance Issues**:

- Close other resource-intensive applications
- Ensure Python dependencies are properly installed
- Check system meets minimum requirements

### Error Messages

- `Camera initialization failed`: Check webcam connection and permissions
- `MediaPipe model loading failed`: Verify mediapipe installation
- `Frame processing error`: Check lighting and camera positioning

## Technical Details

### Dependencies

- **MediaPipe**: Pretrained landmark detection models
- **OpenCV**: Video capture and image processing
- **NumPy**: Numerical computations and array operations

### Architecture

The system uses a modular architecture with the following components:

1. **Video Capture Layer**: OpenCV webcam interface
2. **Detection Layer**: MediaPipe model inference
3. **Analysis Layer**: Geometric and motion calculations
4. **Smoothing Layer**: Temporal filtering using moving averages
5. **Visualization Layer**: Real-time overlay rendering

### Privacy and Security

- **Local Processing**: All analysis performed on your local machine
- **No Data Storage**: No video or behavioral data is saved
- **No Network Communication**: System operates entirely offline
- **User Control**: Complete control over when analysis starts and stops

## Development

### Project Structure

```
video-analysis/
├── production_ready_analyzer.py  # Main production system (Core)
├── main.py                       # FastAPI application
├── run_api.py                    # API server launcher
├── resource_monitor.py           # System monitoring (Optional)
├── requirements.txt              # Python dependencies
├── API_USAGE.md                  # API documentation
├── README.md                     # Main documentation
└── WIN_20251230_00_17_22_Pro.mp4 # Test video
```

### Quick Start

**FastAPI Server (Recommended):**

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Start the API server:
```bash
python run_api.py
```

3. Access API documentation:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

4. Test the API:
```bash
# Health check
curl http://localhost:8000/health

# Analyze frame
curl -X POST "http://localhost:8000/analyze/frame" -F "file=@frame.jpg"
```

See `API_USAGE.md` for detailed API documentation and examples.

**Direct Usage (Standalone):**

```bash
python production_ready_analyzer.py
```

## License

MIT License - See project files for details.

## Support

For technical issues or questions, refer to the troubleshooting section above or check the project documentation in the `.kiro/specs/` directory.
