> **Note**: This describes an intermediate pipeline (Temporal CNN, 77.7% accuracy).
> For the latest results (KeyJointNet, **91.95% accuracy**), see `session_dec18/`.

# Free Throw Prediction System

A machine learning system that predicts basketball free throw outcomes (make/miss) using pose estimation and temporal analysis.

## Table of Contents

1. [Overview](#overview)
2. [System Architecture](#system-architecture)
3. [Installation](#installation)
4. [Quick Start](#quick-start)
5. [Pipeline Details](#pipeline-details)
6. [File Reference](#file-reference)
7. [Usage Examples](#usage-examples)
8. [Model Performance](#model-performance)
9. [Troubleshooting](#troubleshooting)

---

## Overview

This system analyzes basketball free throw videos to predict whether a shot will go in (MAKE) or miss (MISS) based on the shooter's body pose at the moment of ball release.

### Key Features

- Automatic release frame detection from video
- 3D pose estimation using SAM3D Body (70 joints)
- Fallback to 2D pose estimation (17 COCO joints)
- Temporal analysis of shooting motion (4 frames around release)
- Real-time prediction app with Streamlit

### Pipeline Overview

```
Raw Videos
    |
    v
[auto_release_detection.py] -----> candidates.json (release frames + 2D poses)
    |
    v
[extract_sequences.py] --> features.json (SAM3D 3D poses, optional)
    |
    v
[build_features.py] -----> enhanced_all.json (velocity, acceleration)
    |
    v
[train_key_joints.py] ---> best_key_joints_model.pth
    |
    v
[app.py] ----------------> Real-time predictions
```

---

## System Architecture

### Detection Pipeline

```
                                    VIDEO INPUT
                                         |
                                         v
                        +--------------------------------+
                        |     Release Frame Detection    |
                        |   (auto_release_detection.py)  |
                        |                                |
                        |  - YOLOv8 pose estimation      |
                        |  - Shooter identification      |
                        |  - Arm angle analysis          |
                        |  - Ball detection              |
                        +--------------------------------+
                                         |
                                         v
                              candidates.json
                        +--------------------------------+
                        | - video_file                   |
                        | - release_frame                |
                        | - shooter.keypoints_2d (17)    |
                        | - shooter.arm_angle            |
                        | - ball_location                |
                        | - status (PERFECT/LIKELY/NEEDS)|
                        | - label (0=miss, 1=make)       |
                        +--------------------------------+
```

### Feature Extraction Pipeline

```
candidates.json
       |
       +----------------------+----------------------+
       |                                             |
       v                                             v
+------------------+                    +------------------------+
|  2D Mode (Fast)  |                    |   SAM3D Mode (Accurate)|
| build_features.py|                    | extract_sequences.py   |
|                  |                    |                        |
| - Uses existing  |                    | - Loads SAM3 + SAM3D   |
|   2D keypoints   |                    | - Extracts 70-joint    |
| - Simulates      |                    |   3D poses per frame   |
|   temporal motion|                    | - Ball tracking        |
| - 17 joints      |                    | - Shooter tracking     |
+------------------+                    +------------------------+
       |                                             |
       v                                             v
       +---------------------------------------------+
                              |
                              v
                    enhanced_all.json
              +--------------------------------+
              | - keypoints_3d: (4, V, 3)      |
              | - velocity: (4, V, 3)          |
              | - acceleration: (4, V, 3)      |
              | - label: 0/1                   |
              | - metadata (arm_angle, etc.)   |
              +--------------------------------+
```

### Training Pipeline

```
enhanced_all.json
       |
       v
+--------------------------------+
|     KeyJointNet Training       |
|     (train_key_joints.py)      |
|                                |
|  - 5-fold cross validation     |
|  - Focal loss for imbalance    |
|  - Joint attention mechanism   |
|  - Temporal convolution        |
+--------------------------------+
       |
       v
best_key_joints_model.pth
+--------------------------------+
| - model_state_dict             |
| - model_config                 |
| - mean_accuracy                |
| - overall_auc                  |
+--------------------------------+
```

---

## Installation

### Requirements

```bash
# Core dependencies
pip install -r requirements.txt

# For SAM3D extraction (optional, GPU required)
# Follow SAM3D Body installation instructions
```

### Directory Structure

```
project/
|-- data/
|   |-- release_detection/
|   |   |-- candidates.json      
|   |-- features/
|   |   |-- features.json        
|   |   |-- enhanced_all.json    
|   |   |-- enhanced_labeled.json
|   |-- hq_videos/
|       |-- ft0/                 
|       |-- ft1/                 
|-- models/
|   |-- best_key_joints_model.pth
|-- src/
|   |-- detect_release.py
|   |-- extract_sequences.py
|   |-- build_features.py
|   |-- train_key_joints.py
|-- run_new_pipeline.py
|-- app.py
|-- README.md
```

---

## Quick Start

### Option 1: Full Pipeline with SAM3D (Recommended)

```bash
# Step 1: Detect release frames (if not already done)
python src/auto_release_detection.py --input data/hq_videos --output data/release_detection

# Step 2: Run training pipeline with SAM3D
python run_new_pipeline.py \
    --candidates data/release_detection/candidates.json \
    --use-sam3d \
    --video-dir data/hq_videos

# Step 3: Run the app
streamlit run app.py
```

### Option 2: Fast Pipeline (don't do this - poor performance)

```bash
# Run training pipeline without SAM3D (lower accuracy)
python run_new_pipeline.py \
    --candidates data/release_detection/candidates.json

# Run the app
streamlit run app.py
```

### Option 3: Train on Existing Features

```bash
# If you already have enhanced_all.json
python run_new_pipeline.py \
    --skip-build \
    --data data/features/enhanced_all.json
```

---

## Pipeline Details

### 1. Release Frame Detection (auto_release_detection.py)

Automatically detects the release frame in free throw videos.

**How it works:**
- Uses YOLOv8 for pose estimation
- Identifies shooter by isolation (side view) or centering (broadcast view)
- Detects release by arm angle (100-145 degrees) and wrist height
- Outputs confidence classification

**Confidence Levels:**

| Status | Description |
|--------|-------------|
| `PERFECT` | High confidence, automated pipeline can trust |
| `LIKELY_CORRECT` | Medium confidence, verify if critical |
| `NEEDS_REVIEW` | Low confidence, manual review needed |

**Usage:**
```bash
python src/auto_release_detection.py \
    --input data/hq_videos/ \
    --output data/release_detection/ \
    --no-frames  # Skip saving frame images
```

**Output (candidates.json):**
```json
{
  "video_file": "video.mp4",
  "release_frame": 250,
  "camera_angle": "side",
  "status": "PERFECT",
  "confidence": "high_hd",
  "shooter": {
    "keypoints_2d": [[x, y], ...],
    "keypoints_confidence": [0.95, ...],
    "arm_angle": 118.5,
    "wrist": {"x": 650, "y": 215},
    "elbow": {"x": 628, "y": 278},
    "shoulder": {"x": 612, "y": 348}
  },
  "ball_location": {"x": 640, "y": 200, "confidence": 0.85},
  "sequence_frame_indices": [247, 248, 249, 250, 251, 252],
  "metrics": {
    "score": 0.658,
    "isolation_ratio": 0.125,
    "num_people": 23
  },
  "label": 1
}
```

### 2. SAM3D Feature Extraction (extract_sequences.py)

Extracts 70-joint 3D poses using SAM3D Body.

**Requirements:**
- CUDA GPU
- SAM3 and SAM3D Body installed
- Videos accessible

**Usage:**
```bash
python src/extract_sequences.py \
    --input data/release_detection/candidates.json \
    --output data/features/features.json \
    --video-dir data/hq_videos \
    --labeled-only \
    --device cuda
```

**What it does:**
1. Reads candidates.json to get video paths and release frames
2. Uses shooter.keypoints_2d bounding box to track shooter across frames
3. Extracts 4 frames around release (t-2, t-1, t, t+1)
4. Runs SAM3 for person segmentation
5. Runs SAM3D Body for 70-joint 3D pose estimation
6. Outputs features.json with 3D keypoints per frame

### 3. Feature Building (build_features.py)

Builds training features from candidates.json (2D mode) or features.json (SAM3D mode).

**Usage:**
```bash
# 2D mode (from candidates.json directly)
python src/build_features.py \
    --input data/release_detection/candidates.json \
    --output-dir data/features

# Outputs: enhanced_all.json, enhanced_labeled.json
```

**Output (enhanced_all.json):**
```json
{
  "video_id": "video",
  "label": 1,
  "keypoints_3d": [[[x, y, z], ...], ...],  // (4, V, 3)
  "velocity": [...],                         // (4, V, 3)
  "acceleration": [...],                     // (4, V, 3)
  "arm_angle": 118.5,
  "status": "PERFECT",
  "num_joints": 17,
  "has_keypoints": true
}
```

### 4. Model Training (train_key_joints.py)

Trains the KeyJointNet model with 5-fold cross-validation.

**Architecture:**
- Joint attention mechanism (learns important joints)
- Temporal convolution (captures motion patterns)
- Focal loss (handles class imbalance)

**Usage:**
```bash
python src/train_key_joints.py \
    --data data/features/enhanced_all.json \
    --output models/best_key_joints_model.pth
```

**Model checkpoint contents:**
```python
{
    'model_state_dict': ...,
    'model_class': 'KeyJointNet',
    'model_config': {
        'num_joints': 17,  # or 70 for SAM3D
        'in_channels': 9,
        'hidden_dim': 64
    },
    'key_joints': [0, 1, 2, ...],
    'mean_accuracy': 0.716,
    'overall_auc': 0.752,
    'num_samples': 872,
    'timestamp': '2024-...'
}
```

### 5. Prediction App (app.py)

Streamlit app for real-time predictions.

**Usage:**
```bash
streamlit run app.py
```

**Features:**
- Upload video and play in real-time
- Press SPACEBAR at release moment
- Shows prediction with confidence
- Displays make/miss probabilities
- Shows ground truth if video is in dataset

---

## File Reference

### Source Files

| File | Purpose |
|------|---------|
| `src/detect_release.py` | Detect release frames from videos using YOLOv8 |
| `src/extract_sequences.py` | Extract SAM3D 3D poses from videos |
| `src/build_features.py` | Build training features from 2D keypoints |
| `src/train_key_joints.py` | Train KeyJointNet prediction model |
| `run_new_pipeline.py` | Orchestrate full pipeline |
| `app.py` | Streamlit prediction app |

### Data Files

| File | Description |
|------|-------------|
| `candidates.json` | Release detection output with 2D poses |
| `features.json` | SAM3D extraction output with 3D poses |
| `enhanced_all.json` | Training features (all samples) |
| `enhanced_labeled.json` | Training features (labeled only) |

---

## Usage Examples

### Run Full Pipeline

```bash
# Full SAM3D pipeline
python run_new_pipeline.py \
    --candidates data/release_detection/candidates.json \
    --use-sam3d \
    --video-dir data/hq_videos/ft0 data/hq_videos/ft1

# Only high-quality samples
python run_new_pipeline.py \
    --candidates data/release_detection/candidates.json \
    --use-sam3d \
    --filter-quality PERFECT

# Test with limited samples
python run_new_pipeline.py \
    --candidates data/release_detection/candidates.json \
    --use-sam3d \
    --limit 50
```

### Pipeline Arguments

| Argument | Description |
|----------|-------------|
| `--candidates` | Path to candidates.json |
| `--features-dir` | Output directory for features |
| `--model-output` | Path to save model |
| `--use-sam3d` | Enable SAM3D 3D pose extraction |
| `--video-dir` | Directory containing videos |
| `--skip-build` | Skip feature building |
| `--skip-train` | Skip training |
| `--filter-quality` | Filter by PERFECT or high |
| `--limit` | Limit number of samples |

---

## Model Performance

### Expected Accuracy

| Mode | Joints | Accuracy | AUC | Notes |
|------|--------|----------|-----|-------|
| SAM3D (3D) | 70 | ~71% | ~0.75 | Recommended |
| 2D Only | 17 | ~27% | ~0.51 | Not recommended |

**Why 2D mode has poor performance:**
- Only has keypoints for release frame
- Simulates temporal motion with tiny offsets
- No real motion information between frames

### Confidence-Based Accuracy

| Prediction | Threshold | Accuracy | Use Case |
|------------|-----------|----------|----------|
| High Miss | prob < 0.35 | ~87.5% | Bet on miss |
| High Make | prob > 0.65 | ~75.0% | Bet on make |
| Uncertain | 0.35-0.65 | ~60% | No bet |

---

## Troubleshooting

### "No labeled data found"

Your data file has no samples with `label` field.

```bash
# Check labels
python -c "
import json
d = json.load(open('data/features/enhanced_all.json'))
labeled = [x for x in d if x.get('label', -1) != -1]
print(f'{len(labeled)} labeled out of {len(d)} total')
"
```

### "Index out of bounds" during training

Joint count mismatch between model and data.

**Fix:** The updated `train_key_joints.py` auto-detects joint count. Make sure you're using the latest version.

### "Model not found" in app

Train the model first:

```bash
python src/train_key_joints.py --data data/features/enhanced_all.json
```

### SAM3D extraction fails

1. Check CUDA: `python -c "import torch; print(torch.cuda.is_available())"`
2. Verify SAM3D installation
3. Check video paths exist

### Poor accuracy (~27%)

You're using 2D-only mode. Use SAM3D:

```bash
python run_new_pipeline.py --use-sam3d --video-dir data/hq_videos
```

### "Using mock pose data" in app

The uploaded video doesn't match any entry in `enhanced_all.json`. The app needs pre-extracted poses for real predictions.

---

## Algorithm Details

### Release Detection Scoring (Side View)

```
total_score = (
    isolation_score * 0.50 +    # Distance from other players
    angle_score * 0.20 +        # Arm extension (100-145 deg)
    wrist_height_score * 0.15 + # Wrist above midpoint
    pose_confidence * 0.10 +    # Keypoint detection quality
    people_score * 0.05         # Fewer people = clearer
)
```

### KeyJointNet Architecture

```
Input: (N, 9, 4, V)
       N = batch size
       9 = channels (xyz + velocity + acceleration)
       4 = temporal frames
       V = joints (17 or 70)

Joint Attention -> Temporal Conv -> Pool -> Classifier

Output: (N, 2) [miss_prob, make_prob]
```

---

## License

MIT License