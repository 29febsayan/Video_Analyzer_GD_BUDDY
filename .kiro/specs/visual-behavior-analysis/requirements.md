# Requirements Document

## Introduction

A production-quality, real-time visual behavior analysis system that runs locally on the user's machine using system webcam input. The system analyzes objective, measurable visual behavior metrics based on geometry and motion using MediaPipe pretrained models and OpenCV for webcam capture.

## Glossary

- **Visual_Behavior_System**: The complete real-time analysis system that processes webcam video
- **MediaPipe_Models**: Pretrained landmark detection models (Face Mesh, Pose, Hands)
- **Temporal_Smoother**: Moving average buffer component for smoothing frame-level metrics
- **Landmark_Coordinates**: Normalized 2D/3D coordinate points detected by MediaPipe models
- **Forward_Attention_Ratio**: Percentage of frames where head pose indicates forward-facing attention over time window
- **Head_Movement_Variance**: Statistical variance of nose landmark displacement after smoothing
- **Shoulder_Alignment_Error**: Absolute difference between left and right shoulder Y coordinates
- **Hand_Movement_Speed**: Frame-to-frame displacement velocity of wrist landmarks
- **Gesture_Variance**: Statistical variance of smoothed hand movement speed

## Requirements

### Requirement 1

**User Story:** As a behavior analyst, I want to capture real-time video from my system webcam, so that I can analyze visual behavior without external dependencies.

#### Acceptance Criteria

1. THE Visual_Behavior_System SHALL capture video frames using cv2.VideoCapture(0)
2. THE Visual_Behavior_System SHALL process frames continuously in real-time
3. THE Visual_Behavior_System SHALL run locally on Windows, Linux, and macOS platforms
4. THE Visual_Behavior_System SHALL NOT require cloud services or video uploads
5. THE Visual_Behavior_System SHALL exit cleanly when ESC key is pressed

### Requirement 2

**User Story:** As a behavior analyst, I want to estimate head orientation as a proxy for attention direction, so that I can measure objective attention metrics using head pose analysis.

#### Acceptance Criteria

1. THE Visual_Behavior_System SHALL use MediaPipe Face Mesh model for facial landmark detection
2. THE Visual_Behavior_System SHALL extract nose tip, left eye corner, and right eye corner landmarks for head pose calculation
3. THE Visual_Behavior_System SHALL compute head pose yaw angle using the horizontal relationship between eye corners and nose
4. THE Visual_Behavior_System SHALL compute head pose pitch angle using the vertical position of nose relative to eye line
5. WHEN yaw angle is between -15 and +15 degrees AND pitch angle is between -10 and +10 degrees, THE Visual_Behavior_System SHALL register forward attention as true for that frame
6. THE Visual_Behavior_System SHALL calculate Forward_Attention_Ratio as percentage of forward-facing frames over rolling 30-frame window
7. THE Visual_Behavior_System SHALL categorize Forward_Attention_Ratio as Excellent (≥80%), Good (50-79%), Fair (30-49%), or Poor (<30%)

### Requirement 3

**User Story:** As a behavior analyst, I want to track head movement stability, so that I can assess presentation composure objectively.

#### Acceptance Criteria

1. THE Visual_Behavior_System SHALL track nose landmark position frame-to-frame
2. THE Visual_Behavior_System SHALL compute displacement between consecutive nose positions
3. THE Visual_Behavior_System SHALL apply Temporal_Smoother with 7-frame window to head movement data
4. THE Visual_Behavior_System SHALL calculate Head_Movement_Variance from smoothed displacement values
5. THE Visual_Behavior_System SHALL categorize variance as Low (Stable), Medium (Acceptable), or High (Needs Improvement)

### Requirement 4

**User Story:** As a behavior analyst, I want to measure shoulder alignment, so that I can assess posture objectively.

#### Acceptance Criteria

1. THE Visual_Behavior_System SHALL use MediaPipe Pose model to detect shoulder landmarks
2. THE Visual_Behavior_System SHALL compute absolute difference between left and right shoulder Y coordinates
3. THE Visual_Behavior_System SHALL apply Temporal_Smoother to shoulder alignment measurements
4. THE Visual_Behavior_System SHALL categorize Shoulder_Alignment_Error as Aligned (≤0.02), Slight Misalignment (0.02-0.05), or Needs Improvement (>0.05)
5. WHEN shoulder landmarks are not detected, THE Visual_Behavior_System SHALL handle missing data gracefully

### Requirement 5

**User Story:** As a behavior analyst, I want to analyze hand movement and gesture patterns, so that I can measure expressiveness objectively.

#### Acceptance Criteria

1. THE Visual_Behavior_System SHALL use MediaPipe Hands model to detect hand landmarks
2. THE Visual_Behavior_System SHALL compute Hand_Movement_Speed using wrist landmark displacement
3. THE Visual_Behavior_System SHALL average speed across detected hands per frame
4. THE Visual_Behavior_System SHALL apply Temporal_Smoother to hand movement speed data
5. THE Visual_Behavior_System SHALL calculate Gesture_Variance from smoothed hand speed values
6. THE Visual_Behavior_System SHALL categorize hand movement as Under-expressive (Low), Balanced (Moderate), or Over-expressive (High)
7. THE Visual_Behavior_System SHALL categorize gesture variance as Controlled (Low), Natural (Medium), or Erratic (High)

### Requirement 6

**User Story:** As a behavior analyst, I want to see live visual feedback with detected landmarks and metrics, so that I can monitor analysis results in real-time.

#### Acceptance Criteria

1. THE Visual_Behavior_System SHALL overlay detected landmarks on live video feed
2. THE Visual_Behavior_System SHALL display current metric values as text overlay
3. THE Visual_Behavior_System SHALL draw face mesh landmarks when detected
4. THE Visual_Behavior_System SHALL draw pose landmarks for shoulders when detected
5. THE Visual_Behavior_System SHALL draw hand landmarks when detected
6. THE Visual_Behavior_System SHALL maintain smooth and stable visual output

### Requirement 7

**User Story:** As a developer, I want a modular and maintainable codebase, so that I can extend and modify the system easily.

#### Acceptance Criteria

1. THE Visual_Behavior_System SHALL implement reusable Temporal_Smoother class with configurable window size
2. THE Visual_Behavior_System SHALL separate metric calculation logic into distinct functions
3. THE Visual_Behavior_System SHALL handle missing landmarks gracefully without system crashes
4. THE Visual_Behavior_System SHALL use clear variable names and comprehensive code comments
5. THE Visual_Behavior_System SHALL avoid hard-coded magic numbers where possible

### Requirement 8

**User Story:** As a user, I want simple installation and execution, so that I can run the system immediately on my local machine.

#### Acceptance Criteria

1. THE Visual_Behavior_System SHALL require only Python 3.9 or 3.10
2. THE Visual_Behavior_System SHALL use only mediapipe, opencv-python, and numpy libraries
3. THE Visual_Behavior_System SHALL provide complete requirements.txt file
4. THE Visual_Behavior_System SHALL include clear installation and execution instructions
5. THE Visual_Behavior_System SHALL run as a single Python script without additional configuration