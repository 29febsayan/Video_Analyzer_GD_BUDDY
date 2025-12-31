# Design Document

## Overview

The Visual Behavior Analysis System is a real-time computer vision application that analyzes objective behavioral metrics from webcam video using MediaPipe pretrained models. The system processes live video frames to extract facial landmarks, pose landmarks, and hand landmarks, then computes geometric and motion-based metrics with temporal smoothing for stable analysis.

## Architecture

### High-Level Architecture

```
Webcam Input → Frame Capture → MediaPipe Processing → Metric Calculation → Temporal Smoothing → Visualization → Display Output
```

### Component Flow

1. **Video Capture Layer**: OpenCV webcam interface
2. **Detection Layer**: MediaPipe model inference
3. **Analysis Layer**: Geometric and motion calculations
4. **Smoothing Layer**: Temporal filtering using moving averages
5. **Visualization Layer**: Real-time overlay rendering
6. **Control Layer**: User interaction and system management

### System Dependencies

- **Python 3.9/3.10**: Core runtime environment
- **OpenCV**: Video capture and image processing
- **MediaPipe**: Pretrained landmark detection models
- **NumPy**: Numerical computations and array operations

## Components and Interfaces

### 1. TemporalSmoother Class

**Purpose**: Provides moving average smoothing for time-series data

**Interface**:
```python
class TemporalSmoother:
    def __init__(self, window_size: int = 7)
    def update(self, value: float) -> float
    def get_variance(self) -> float
    def reset(self) -> None
```

**Implementation Details**:
- Maintains circular buffer of configurable size (default: 7 frames)
- Computes rolling average for smooth metric values
- Calculates variance for stability assessment
- Thread-safe for real-time processing

### 2. LandmarkDetector Class

**Purpose**: Manages MediaPipe model initialization and inference

**Interface**:
```python
class LandmarkDetector:
    def __init__(self)
    def detect_face(self, frame: np.ndarray) -> Optional[List]
    def detect_pose(self, frame: np.ndarray) -> Optional[List]
    def detect_hands(self, frame: np.ndarray) -> Optional[List]
    def cleanup(self) -> None
```

**Implementation Details**:
- Initializes MediaPipe Face Mesh, Pose, and Hands models
- Processes RGB frames and returns normalized landmark coordinates
- Handles model lifecycle and resource management
- Provides graceful degradation when landmarks not detected

### 3. MetricCalculator Class

**Purpose**: Computes behavioral metrics from landmark data

**Interface**:
```python
class MetricCalculator:
    def __init__(self)
    def calculate_head_pose(self, face_landmarks: List) -> Tuple[float, float]
    def calculate_head_movement(self, nose_position: Tuple[float, float]) -> float
    def calculate_shoulder_alignment(self, pose_landmarks: List) -> float
    def calculate_hand_speed(self, hand_landmarks: List, prev_landmarks: List) -> float
```

**Implementation Details**:
- Uses geometric calculations for head pose estimation
- Computes frame-to-frame displacement for movement analysis
- Handles coordinate normalization and scaling
- Provides robust error handling for missing landmarks

### 4. BehaviorAnalyzer Class

**Purpose**: Main system orchestrator and analysis engine

**Interface**:
```python
class BehaviorAnalyzer:
    def __init__(self)
    def process_frame(self, frame: np.ndarray) -> Dict[str, Any]
    def get_current_metrics(self) -> Dict[str, str]
    def draw_overlays(self, frame: np.ndarray, landmarks: Dict) -> np.ndarray
    def run(self) -> None
```

**Implementation Details**:
- Coordinates all system components
- Maintains temporal smoothers for each metric
- Manages metric interpretation and categorization
- Handles real-time visualization and user interaction

## Data Models

### Landmark Data Structure

```python
@dataclass
class LandmarkData:
    face_landmarks: Optional[List[Tuple[float, float, float]]]
    pose_landmarks: Optional[List[Tuple[float, float, float]]]
    hand_landmarks: Optional[List[List[Tuple[float, float, float]]]]
    timestamp: float
```

### Metric Data Structure

```python
@dataclass
class BehaviorMetrics:
    forward_attention_ratio: float
    head_movement_variance: float
    shoulder_alignment_error: float
    hand_movement_speed: float
    gesture_variance: float
    frame_count: int
    timestamp: float
```

### Configuration Structure

```python
@dataclass
class SystemConfig:
    # Head pose thresholds
    yaw_threshold: float = 15.0  # degrees
    pitch_threshold: float = 10.0  # degrees
    
    # Smoothing parameters
    smoother_window_size: int = 7
    attention_window_size: int = 30
    
    # Shoulder alignment thresholds
    alignment_good: float = 0.02
    alignment_fair: float = 0.05
    
    # Visualization settings
    landmark_color: Tuple[int, int, int] = (0, 255, 0)
    text_color: Tuple[int, int, int] = (255, 255, 255)
    font_scale: float = 0.6
```

## Error Handling

### Graceful Degradation Strategy

1. **Missing Face Landmarks**: Continue with other metrics, display "Face not detected"
2. **Missing Pose Landmarks**: Skip shoulder analysis, maintain other metrics
3. **Missing Hand Landmarks**: Set hand metrics to zero, continue analysis
4. **Camera Access Issues**: Display clear error message and exit gracefully
5. **Model Loading Failures**: Provide specific error messages for troubleshooting

### Error Recovery Mechanisms

- **Landmark Smoothing**: Use previous valid landmarks when current detection fails
- **Metric Interpolation**: Maintain metric continuity during brief detection gaps
- **System Resilience**: Continue operation with partial landmark data
- **Resource Management**: Proper cleanup of MediaPipe resources on exit

## Testing Strategy

### Unit Testing Approach

1. **TemporalSmoother Tests**:
   - Verify moving average calculations
   - Test variance computation accuracy
   - Validate buffer management

2. **MetricCalculator Tests**:
   - Test head pose angle calculations with known landmark positions
   - Verify shoulder alignment computations
   - Validate hand speed calculations

3. **LandmarkDetector Tests**:
   - Mock MediaPipe responses for consistent testing
   - Test error handling with invalid inputs
   - Verify resource cleanup

### Integration Testing

1. **End-to-End Pipeline**: Test complete frame processing workflow
2. **Real-time Performance**: Validate frame rate and latency requirements
3. **Resource Usage**: Monitor memory and CPU consumption
4. **Cross-Platform**: Test on Windows, Linux, and macOS

### Manual Testing Protocol

1. **Webcam Functionality**: Verify camera access and frame capture
2. **Landmark Detection**: Test with various lighting conditions and angles
3. **Metric Accuracy**: Validate behavioral measurements against known scenarios
4. **User Interface**: Test overlay rendering and user controls
5. **System Stability**: Extended runtime testing for memory leaks

## Performance Considerations

### Optimization Strategies

1. **Frame Processing**: Resize frames to optimal resolution (640x480) for balance of accuracy and speed
2. **Model Efficiency**: Use MediaPipe's optimized inference pipeline
3. **Memory Management**: Reuse frame buffers and minimize allocations
4. **Computational Load**: Prioritize essential calculations, defer non-critical operations

### Real-time Requirements

- **Target Frame Rate**: 15-30 FPS depending on hardware capabilities
- **Latency**: < 100ms from capture to display
- **CPU Usage**: < 50% on modern hardware
- **Memory Usage**: < 500MB total system memory

### Scalability Considerations

- **Multi-person Detection**: Current design supports single-person analysis
- **Extended Metrics**: Modular design allows easy addition of new behavioral metrics
- **Hardware Adaptation**: Automatic quality adjustment based on system performance
- **Configuration Flexibility**: Runtime parameter adjustment without code changes

## Security and Privacy

### Data Protection

- **Local Processing**: All analysis performed on local machine
- **No Data Storage**: No video or metric data persisted to disk
- **No Network Communication**: System operates entirely offline
- **Memory Security**: Sensitive data cleared from memory on exit

### Ethical Considerations

- **Objective Metrics Only**: No subjective behavioral interpretations
- **Transparent Calculations**: All metric computations clearly documented
- **User Control**: Clear start/stop controls and exit mechanisms
- **Consent Awareness**: System designed for voluntary use with user awareness