> **Note**: Model file updated to `models/best_merged_calibrated.pth` (91.95% accuracy with KeyJointNet).
> Some references in this doc may refer to the older ST-GCN model.

# Free Throw Prediction App

A real-time basketball free throw prediction system using Streamlit. Upload a video, watch it play, hit SPACEBAR at the release moment, and get an instant make/miss prediction.

## Overview

This application predicts whether a basketball free throw will go in (MAKE) or miss (MISS) based on the shooter's body pose at the moment of ball release.

### How It Works

```
┌─────────────────────────────────────────────────────────────────┐
│                      STREAMLIT APP                              │
│                    (app_realtime.py)                            │
│                                                                 │
│  ┌──────────────────────────┐  ┌───────────────────────────┐   │
│  │      VIDEO PLAYER        │  │     PREDICTION PANEL      │   │
│  │                          │  │                           │   │
│  │  - Upload video          │  │  - MAKE / MISS result     │   │
│  │  - Play in real-time     │  │  - Probability bars       │   │
│  │  - SPACEBAR to capture   │  │  - Confidence score       │   │
│  │  - Shows release frame   │  │  - Ground truth (if known)│   │
│  └──────────────────────────┘  └───────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                      PREDICTION PIPELINE                        │
│                                                                 │
│  1. Video Upload                                                │
│       │                                                         │
│       ▼                                                         │
│  2. Match video filename to enhanced_all.json                   │
│       │                                                         │
│       ▼                                                         │
│  3. Load pre-extracted pose data (keypoints_3d, velocity, etc.) │
│       │                                                         │
│       ▼                                                         │
│  4. Run model inference                                  │
│       │                                                         │
│       ▼                                                         │
│  5. Return prediction: {make_prob, miss_prob, confidence}       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Files

| File | Purpose |
|------|---------|
| `app.py` | Streamlit application with video player and prediction UI |
| `models/best_key_joints_model.pth` | Trained model weights |
| `data/features/enhanced_all.json` | Pre-extracted pose features for videos |
| `src/models/stgcn.py` | ST-GCN model architecture |

## Quick Start

### 1. Install Dependencies

```bash
pip install streamlit torch numpy opencv-python
```

### 2. Train the Model (if not already done)

```bash
python src/train_key_joints.py --data data/features/enhanced_all.json
```

This creates `models/best_key_joints_model.pth`.

### 3. Run the App

```bash
streamlit run app_realtime.py
```

The app opens in your browser at `http://localhost:8501`.

### 4. Use the App

1. Upload a free throw video (MP4, MOV, or AVI)
2. Click PLAY or press SPACEBAR to start playback
3. Watch the video play in real-time
4. Press SPACEBAR at the exact moment of ball release
5. View the prediction result on the right panel

## User Interface

### Layout

The app uses a side-by-side layout:

- **Left panel**: Video player with playback controls
- **Right panel**: Prediction result display

### Controls

| Control | Action |
|---------|--------|
| PLAY button | Start/pause video playback |
| RESET button | Reset video to beginning, clear prediction |
| SPACEBAR | Start playback (if paused) or capture frame (if playing) |

### Prediction Display

- **MAKE** (green): Model predicts the shot will go in
- **MISS** (red): Model predicts the shot will miss
- **NO BET** (gray): Low confidence, prediction unreliable
- **NO MODEL** (warning): Model file not found

### Information Shown

- Make/Miss probability percentages
- Confidence score (0-100%)
- Captured frame timestamp
- Ground truth label (if video is in the dataset)

## Technical Details

### Model Architecture

The app uses an ST-GCN (Spatial-Temporal Graph Convolutional Network) that analyzes body pose as a graph:

- **Nodes**: Body joints (25 upper body joints)
- **Edges**: Skeletal connections between joints
- **Features**: 9 channels per joint
  - Position: x, y, z
  - Velocity: vx, vy, vz
  - Acceleration: ax, ay, az

### Input Format

```
Pose sequence shape: (4, 25, 9)
                      │   │   └── Features per joint
                      │   └────── Number of joints
                      └────────── Temporal frames (around release)
```

### Model Loading

The app loads the model on startup using Streamlit's caching:

```python
@st.cache_resource
def load_model():
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model
```

### Pose Data Matching

The app matches uploaded videos to pre-extracted poses:

1. Extract video filename (e.g., `ft0_v112_010910_x264.mp4`)
2. Search `enhanced_all.json` for matching `video_id`
3. Load `keypoints_3d`, `velocity`, `acceleration` arrays
4. Feed to model for prediction

If no match is found, the app uses mock data (random values) and displays a warning.

## Configuration

### Model Path

Default: `models/best_key_joints_model.pth`

To change, modify the `MODEL_PATH` variable in `app_realtime.py`:

```python
MODEL_PATH = "path/to/your/model.pth"
```

### Pose Data Path

Default search paths:
- `data/features/enhanced_all.json`
- `data/features/enhanced_clean.json`

### Device Selection

The app automatically selects GPU if available:

```python
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

## Limitations

### Pose Extraction

The current implementation does NOT extract poses from video frames in real-time. Instead, it:

1. Matches the video filename to pre-extracted data in `enhanced_all.json`
2. Uses the pre-computed pose features for prediction

For videos not in the dataset, predictions will be based on random mock data and will not be meaningful.

### Supported Videos

For accurate predictions, upload videos that:
- Have a matching entry in `enhanced_all.json`
- Filename contains the `video_id` from the dataset

### Single Prediction

The prediction is computed once when the video is uploaded, not at the moment SPACEBAR is pressed. The SPACEBAR interaction is for demonstration purposes to simulate a live betting experience.

## Troubleshooting

### "Model Not Found"

The model file does not exist. Train it first:

```bash
python src/train_key_joints.py --data data/features/enhanced_all.json
```

### "Using mock pose data"

The uploaded video filename does not match any entry in `enhanced_all.json`. Either:
- Upload a video from your dataset
- Add pose data for your video to `enhanced_all.json`

### Import Errors

Make sure the `src/models/stgcn.py` file exists and contains the `STGCN` class.

### Streamlit Not Found

Install Streamlit:

```bash
pip install streamlit
```
