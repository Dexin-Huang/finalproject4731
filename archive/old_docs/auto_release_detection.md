# Auto Release Frame Detector

Automatically detects the release frame in basketball free throw videos using pose estimation and computer vision.

## Overview

This module identifies the exact frame when a basketball player releases the ball during a free throw. It supports two camera angles:

- **Side View**: Standard broadcast angle from the side of the court
- **Broadcast View**: Behind-the-basket angle looking at the shooter

## How It Works

### Side View Detection
1. **Pose Estimation**: Uses YOLOv8 to detect all people and their body keypoints
2. **Shooter Identification**: Finds the most *isolated* person (free throw shooter stands apart from rebounders)
3. **Release Detection**: Looks for shooting pose characteristics:
   - Arm angle between 100-145° (elbow extension)
   - Wrist elevated above midpoint of frame
   - Valid body proportions and position

### Broadcast View Detection  
1. **Pose Estimation**: Same YOLOv8 detection
2. **Shooter Identification**: Finds the most *centered* person (shooter is in middle of frame)
3. **Ball Detection**: Confirms release by detecting basketball near shooter's hands
4. **Release Detection**: Similar pose analysis with adjusted thresholds for frontal view

### Confidence Classification

Each detection is classified into one of three categories:

| Status | Description |
|--------|-------------|
| `PERFECT` | High confidence - automated pipeline can trust this |
| `LIKELY_CORRECT` | Medium confidence - probably correct but verify if critical |
| `NEEDS_REVIEW` | Low confidence - manual review recommended |

## Installation

```bash
# Install dependencies
pip install ultralytics opencv-python numpy

# Models are downloaded automatically on first run
```

## Usage

### Command Line

```bash
# Basic usage
python detect_release.py --input videos/ --output output/

# Without saving frames (faster, less disk space)
python detect_release.py --input videos/ --output output/ --no-frames
```

### Python API

```python
from detect_release import run_detection, process_video

# Process all videos in a directory
summary = run_detection(
    input_dir='videos/',
    output_dir='output/',
    save_frames=True
)

# Process a single video
from ultralytics import YOLO
pose_model = YOLO('yolov8m-pose.pt')
ball_model = YOLO('yolov8n.pt')

result, camera_angle = process_video(
    'video.mp4', 
    pose_model, 
    ball_model, 
    'output/'
)
```

## Output Format

### Directory Structure
```
output/
├── candidates.json      # All detection results
├── perfect.json         # High-confidence detections only
├── likely_correct.json  # Medium-confidence detections
├── needs_review.json    # Low-confidence detections
└── frames/
    └── video_name/
        ├── frame_0247.jpg
        ├── frame_0248.jpg
        ├── frame_0249.jpg
        ├── frame_0250.jpg      # Release frame
        ├── frame_0251.jpg
        ├── frame_0252.jpg
        ├── release_0250.jpg    # Clean release frame
        └── visualization.jpg   # Annotated detection
```

### JSON Schema

```json
{
  "video_file": "example_video.mp4",
  "camera_angle": "side",
  "release_frame": 250,
  "fps": 60.0,
  "frame_width": 1280,
  "frame_height": 720,
  "status": "PERFECT",
  "confidence": "high_hd",
  "metrics": {
    "score": 0.658,
    "isolation_ratio": 0.125,
    "isolation_px": 184.0,
    "angle": 118.5,
    "num_people": 23,
    "is_hd": true
  },
  "ball_location": {
    "x": 640.5,
    "y": 198.3,
    "confidence": 0.85
  },
  "shooter": {
    "wrist": {"x": 650.2, "y": 215.8},
    "elbow": {"x": 628.4, "y": 278.1},
    "shoulder": {"x": 612.0, "y": 348.5},
    "arm_angle": 118.5,
    "isolation_distance": 184.0,
    "isolation_ratio": 0.125,
    "keypoints_2d": [[x, y], ...],
    "keypoints_confidence": [0.95, 0.92, ...]
  },
  "sequence_frame_indices": [247, 248, 249, 250, 251, 252]
}
```

### Field Descriptions

| Field | Description |
|-------|-------------|
| `video_file` | Source video filename |
| `camera_angle` | Detected camera angle (`side` or `broadcast`) |
| `release_frame` | Frame index of detected release |
| `fps` | Video frame rate |
| `status` | Confidence classification |
| `metrics.score` | Overall detection score (0-1) |
| `metrics.isolation_ratio` | Distance to nearest person / frame diagonal |
| `metrics.angle` | Arm angle at elbow in degrees |
| `metrics.num_people` | Number of people detected in frame |
| `ball_location` | Detected basketball position |
| `shooter.keypoints_2d` | 17 COCO keypoints (x, y) |
| `sequence_frame_indices` | Frames around release (t-3 to t+2) |

## Integration with Pipeline

### Replacing existing candidates.json

The output `candidates.json` can directly replace your existing file:

```bash
# Run detection
python detect_release.py --input raw_videos/ --output data/

# Output is saved to data/candidates.json
```

### Using with extract_sequences.py

The `sequence_frame_indices` field provides the frames needed for feature extraction:

```python
import json

with open('data/candidates.json') as f:
    candidates = json.load(f)

for c in candidates:
    video = c['video_file']
    frames = c['sequence_frame_indices']  # [t-3, t-2, t-1, t, t+1, t+2]
    release = c['release_frame']
    
    # Extract features from these frames
    ...
```

### Filtering by Confidence

```python
# Only use high-confidence detections for training
perfect = [c for c in candidates if c['status'] == 'PERFECT']

# Or load directly
with open('data/perfect.json') as f:
    perfect = json.load(f)
```

## Algorithm Details

### Side View Scoring

```
total_score = (
    isolation_score * 0.50 +    # Distance from other players
    angle_score * 0.20 +        # Arm extension angle
    wrist_height_score * 0.15 + # How high is the wrist
    pose_confidence * 0.10 +    # Keypoint detection confidence
    people_score * 0.05         # Fewer people = clearer scene
)
```

### Broadcast View Scoring

```
total_score = (
    center_score * 0.50 +       # How centered is the shooter
    angle_score * 0.25 +        # Arm extension angle
    wrist_height_score * 0.15 + # How high is the wrist
    pose_confidence * 0.10      # Keypoint detection confidence
)

# Bonus for ball detection near hands
if ball_near_wrist:
    total_score += 0.15
```

### Camera Angle Detection

The detector automatically classifies camera angle based on:

| Feature | Side View | Broadcast View |
|---------|-----------|----------------|
| People count | 12+ | 5-9 |
| Shooter position | Isolated to one side | Centered in frame |
| Shooter size | ~20-40% of frame | ~30-50% of frame |
| Player distribution | Clustered near basket | Balanced left/right |

## Limitations

- Requires clear view of the shooter's upper body
- May struggle with heavily occluded scenes
- Optimized for NBA broadcast footage (1280x720 HD)
- Ball detection is supplementary, not required

## Dependencies

- `ultralytics>=8.0.0` - YOLOv8 for pose estimation and object detection
- `opencv-python>=4.8.0` - Video processing and visualization
- `numpy` - Numerical operations

## References

- YOLOv8 Pose: https://docs.ultralytics.com/tasks/pose/
- COCO Keypoints: https://cocodataset.org/#keypoints-2020
