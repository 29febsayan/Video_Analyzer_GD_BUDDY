# Implementation Plan

- [x] 1. Set up project structure and dependencies
  - Create requirements.txt with mediapipe, opencv-python, and numpy dependencies
  - Create basic project structure with main script
  - Create installation and usage instructions in README
  - Add SystemConfig dataclass with configuration parameters
  - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5_

- [x] 2. Implement TemporalSmoother class for metric smoothing





  - Create reusable TemporalSmoother class with configurable window size
  - Implement moving average calculation with circular buffer
  - Add variance calculation method for stability assessment
  - Add reset functionality for smoother state management
  - _Requirements: 3.3, 5.4, 7.1, 7.5_

- [x] 2.1 Write unit tests for TemporalSmoother





  - Test moving average calculations with known input sequences
  - Verify variance computation accuracy
  - Test buffer overflow and reset functionality
  - _Requirements: 7.1_

- [x] 3. Implement LandmarkDetector class for MediaPipe integration





  - Initialize MediaPipe Face Mesh, Pose, and Hands models
  - Create detection methods for each model type (face, pose, hands)
  - Implement proper resource cleanup and model lifecycle management
  - Add error handling for model initialization failures
  - _Requirements: 2.1, 4.1, 5.1, 7.3_

- [x] 3.1 Write unit tests for LandmarkDetector






  - Mock MediaPipe responses for consistent testing
  - Test error handling with invalid frame inputs
  - Verify proper resource cleanup on model destruction
  - _Requirements: 7.3_

- [x] 4. Implement MetricCalculator class for behavioral analysis





- [x] 4.1 Create head pose calculation method


  - Extract nose tip, left eye corner, and right eye corner landmarks
  - Compute yaw angle using horizontal eye-to-nose relationships
  - Compute pitch angle using vertical nose position relative to eye line
  - _Requirements: 2.2, 2.3, 2.4_

- [x] 4.2 Create head movement tracking method

  - Track nose landmark position frame-to-frame
  - Compute displacement between consecutive nose positions
  - Return movement magnitude for temporal smoothing
  - _Requirements: 3.1, 3.2_

- [x] 4.3 Create shoulder alignment calculation method

  - Extract left and right shoulder landmarks from pose data
  - Compute absolute difference between shoulder Y coordinates
  - Handle missing shoulder landmarks gracefully
  - _Requirements: 4.2, 4.5_

- [x] 4.4 Create hand movement analysis methods

  - Extract wrist landmarks (index 0) from hand detection results
  - Compute frame-to-frame displacement for each detected hand
  - Calculate average speed across all detected hands
  - _Requirements: 5.2, 5.3_

- [x] 4.5 Write unit tests for MetricCalculator






  - Test head pose calculations with known landmark coordinates
  - Verify shoulder alignment computations with test data
  - Validate hand speed calculations with mock movement sequences
  - _Requirements: 7.3_

- [x] 5. Implement BehaviorAnalyzer main orchestration class





- [x] 5.1 Create frame processing pipeline


  - Integrate LandmarkDetector for multi-model inference
  - Process detected landmarks through MetricCalculator
  - Apply temporal smoothing to all computed metrics
  - _Requirements: 1.2, 6.6, 7.2_

- [x] 5.2 Implement metric interpretation and categorization


  - Apply forward attention ratio thresholds (Excellent ≥80%, Good 50-79%, Fair 30-49%, Poor <30%)
  - Categorize head movement variance (Low/Medium/High)
  - Apply shoulder alignment error thresholds (≤0.02 Aligned, 0.02-0.05 Slight, >0.05 Needs Improvement)
  - Categorize hand movement and gesture variance levels
  - _Requirements: 2.7, 3.5, 4.4, 5.6, 5.7_

- [x] 5.3 Create real-time visualization system


  - Draw detected face mesh landmarks on video frame
  - Draw pose landmarks for shoulders when available
  - Draw hand landmarks for all detected hands
  - Overlay current metric values and categories as text
  - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5_

- [x] 6. Implement webcam capture and main application loop





- [x] 6.1 Create video capture initialization


  - Initialize cv2.VideoCapture(0) for system webcam access
  - Configure frame capture parameters for optimal performance
  - Add error handling for camera access failures
  - _Requirements: 1.1, 7.4_

- [x] 6.2 Implement main processing loop


  - Capture frames continuously from webcam
  - Process each frame through BehaviorAnalyzer pipeline
  - Display processed frames with overlays in real-time
  - Handle ESC key press for clean system exit
  - _Requirements: 1.2, 1.5, 6.6_

- [x] 6.3 Write integration tests for main application






  - Test complete frame processing pipeline end-to-end
  - Verify webcam initialization and frame capture
  - Test real-time performance and frame rate stability
  - _Requirements: 1.1, 1.2, 6.6_

- [-] 7. Final integration and system validation


- [x] 7.1 Integrate all components into single executable script



  - Combine all classes into main application file or organize as modules
  - Ensure proper import structure and dependencies
  - Verify single-script execution capability
  - _Requirements: 8.5_

- [x] 7.2 Perform cross-platform compatibility testing






  - Test on Windows, Linux, and macOS systems
  - Verify webcam access across different platforms
  - Validate MediaPipe model loading on all systems
  - _Requirements: 1.3_

- [x] 7.3 Optimize performance and resource usage






  - Profile frame processing performance and identify bottlenecks
  - Optimize memory usage and prevent memory leaks
  - Ensure stable frame rate under various conditions
  - _Requirements: 6.6, 7.3_