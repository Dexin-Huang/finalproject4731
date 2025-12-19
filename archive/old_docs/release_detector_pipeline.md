# Release Frame Detector Pipeline

## Overview

This document describes the LSTM-based release frame detector that identifies the exact video frame where the basketball leaves the shooter's hands during a free throw.

## Why This Matters

Accurate release frame detection is critical for:
- Extracting the correct 4-frame sequence around the release moment
- Capturing actual shooting mechanics (not estimated)
- Improving downstream pose-based prediction accuracy

## Pipeline Comparison

### Before (Heuristic Approach)
```
Video → SAM3 Mask Separation → Distance Threshold → Estimated Release Frame
                                      ↓
                          Often wrong (30% fallback rate)
                          Fixed thresholds don't generalize
```

### After (Learned Approach)
```
Video → 30-Frame Window → 18 Features/Frame → LSTM Model → Predicted Release Frame
                                                  ↓
                                    ±0.16 frames accuracy (5ms at 30fps)
                                    100% within 1 frame of ground truth
```

---

## Architecture

### Training Pipeline (One-Time)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           TRAINING PIPELINE                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Manual Labels              Feature Extraction              Model Training  │
│  (200 videos)               (SAM3 + SAM3D Body)            (LSTM)          │
│       │                            │                            │          │
│       ▼                            ▼                            ▼          │
│  labels/                   data/release_features/      models/             │
│  release_frames.json       features.json               release_detector_   │
│                                                        lstm.pt             │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Inference Pipeline (For New Videos)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          INFERENCE PIPELINE                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  New Video → Extract 30-Frame Window → Compute 18 Features → LSTM Model    │
│                      │                        │                   │        │
│                      ▼                        ▼                   ▼        │
│              Centered on estimate      Per-frame features    Softmax over  │
│              from candidates.json      (ball, hand, arm)     frame indices │
│                                                                   │        │
│                                                                   ▼        │
│                                                          Predicted Release │
│                                                          Frame + Confidence│
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Full End-to-End Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        FULL PREDICTION PIPELINE                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Video → Release Detection → 4-Frame Sequence → SAM3D Pose → ST-GCN → Made │
│               │                    │                │           │     Miss │
│               ▼                    ▼                ▼           ▼          │
│         Priority:            [t-2, t-1, t, t+1]  3D Joints   Temporal      │
│         1. Manual labels     around release      (70 joints) Graph Conv    │
│         2. LSTM detector                                                   │
│         3. candidates.json                                                 │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Feature Extraction

### The 18 Features Per Frame

| Category | Features | Description |
|----------|----------|-------------|
| **Ball Position** | `ball_x`, `ball_y` | Normalized ball centroid coordinates |
| **Ball Size** | `ball_area` | Ball mask area in pixels |
| **Ball Velocity** | `ball_velocity_x`, `ball_velocity_y` | Frame-to-frame displacement |
| **Ball Dynamics** | `ball_speed`, `ball_acceleration` | Magnitude and rate of change |
| **Hand Distance** | `dist_left_wrist`, `dist_right_wrist`, `min_hand_dist` | Ball-to-wrist distances |
| **Arm Angles** | `left_arm_angle`, `right_arm_angle` | Elbow joint angles |
| **Arm Extension** | `left_arm_extension`, `right_arm_extension` | Wrist height relative to shoulder |
| **Overlap** | `ball_shooter_overlap` | Ball-shooter mask IoU |
| **Hand Velocity** | `left_wrist_velocity`, `right_wrist_velocity` | Wrist movement speed |
| **Detection Flag** | `ball_detected` | Binary flag for ball visibility |

### Feature Extraction Dependencies

- **SAM3**: Ball and shooter mask segmentation (text-prompted)
- **SAM3D Body**: 3D pose estimation (70 joints including hands)
- **OpenCV**: Video frame extraction
- **NumPy**: Feature computation

---

## Model Architecture

### LSTM Release Detector

```python
ReleaseFrameDetectorLSTM(
    num_features=18,      # Input features per frame
    hidden_size=64,       # LSTM hidden dimension
    num_layers=2,         # Bi-LSTM layers
    dropout=0.2           # Regularization
)
```

**Architecture:**
```
Input: (batch, 30, 18)           # 30 frames, 18 features
    │
    ▼
Linear(18 → 64)                  # Input projection
    │
    ▼
Bi-LSTM(64, 2 layers)            # Temporal modeling
    │
    ▼
Linear(128 → 1)                  # Per-frame output
    │
    ▼
Output: (batch, 30)              # Release probability per frame
    │
    ▼
Softmax → argmax                 # Predicted frame index
```

### Alternative Architectures

| Model | Parameters | MAE | Within-1 | Within-3 |
|-------|------------|-----|----------|----------|
| CNN | 298K | 1.26 | 63.2% | 89.5% |
| Transformer | 155K | 0.53 | 100% | 100% |
| **LSTM** | **176K** | **0.16** | **100%** | **100%** |

---

## Training

### Data Preparation

```bash
# 1. Label release frames using the labeling tool
cd labeling-tool && npm run dev
# Navigate to http://localhost:3000/release-labeler

# 2. Extract features from labeled videos (requires GPU)
python src/extract_release_features.py \
    --labels labels/release_frames.json \
    --videos "data/Basketball_51 dataset" \
    --output data/release_features/features.json
```

### Model Training

```bash
# Train LSTM model with cross-validation
python src/train_release_detector.py \
    --features data/release_features/features.json \
    --model lstm \
    --output models/release_detector_lstm.pt \
    --epochs 50

# Alternative: Train CNN or Transformer
python src/train_release_detector.py --model cnn --output models/release_detector_cnn.pt
python src/train_release_detector.py --model transformer --output models/release_detector_transformer.pt
```

### Training Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--epochs` | 50 | Maximum training epochs |
| `--batch_size` | 32 | Batch size |
| `--lr` | 0.001 | Learning rate |
| `--seq_len` | 30 | Sequence length (frames) |
| `--folds` | 5 | Cross-validation folds |

### Loss Function

**Gaussian Focal Loss**: Soft labels with Gaussian distribution around the true release frame, combined with focal weighting to focus on hard examples.

```python
# Soft labels: Gaussian centered on release frame
labels = gaussian(mean=release_idx, sigma=1.5)

# Focal loss: Down-weight easy examples
loss = focal_weight * cross_entropy(predictions, labels)
```

---

## Usage

### Sequence Extraction with Release Detection

```bash
# Use manual labels (highest accuracy)
python src/extract_sequences.py --release-labels labels/release_frames.json

# Use trained detector (for new videos)
python src/extract_sequences.py --use-detector --detector-model models/release_detector_lstm.pt

# Combined: labels first, detector fallback
python src/extract_sequences.py \
    --release-labels labels/release_frames.json \
    --use-detector \
    --detector-model models/release_detector_lstm.pt
```

### Release Detection Priority

1. **Manual labels** (`labels/release_frames.json`) - Highest quality, used when available
2. **Learned detector** - LSTM model prediction if confident (>0.5)
3. **candidates.json** - Fallback to heuristic estimates

### Standalone Detection

```bash
# Detect release frame in a single video
python src/detect_release.py video.mp4 --model models/release_detector_lstm.pt
```

```python
# Programmatic usage
from src.detect_release import ReleaseDetector

detector = ReleaseDetector(model_path="models/release_detector_lstm.pt")
result = detector.detect("video.mp4")

print(f"Release frame: {result.frame}")
print(f"Confidence: {result.confidence:.3f}")
print(f"Method: {result.method}")
```

---

## File Structure

```
Project/
├── labels/
│   └── release_frames.json          # Manual release frame labels (200 videos)
│
├── data/
│   └── release_features/
│       └── features.json            # Extracted features for training
│
├── models/
│   ├── release_detector.pt          # Trained CNN model
│   ├── release_detector_lstm.pt     # Trained LSTM model (best)
│   └── release_detector_transformer.pt
│
├── src/
│   ├── extract_release_features.py  # Feature extraction (SAM3 + SAM3D)
│   ├── train_release_detector.py    # Model training script
│   ├── detect_release.py            # Unified detector interface
│   ├── extract_sequences.py         # Pipeline integration
│   └── models/
│       └── release_detector.py      # Model architectures
│
└── labeling-tool/
    └── app/
        └── release-labeler/
            └── page.tsx             # Frame-by-frame labeling UI
```

---

## Performance

### Model Accuracy

| Metric | Value | Description |
|--------|-------|-------------|
| **MAE** | 0.16 frames | Mean Absolute Error (~5ms at 30fps) |
| **Within-1** | 100% | Predictions within ±1 frame |
| **Within-3** | 100% | Predictions within ±3 frames |

### Processing Time

| Stage | Time | Hardware |
|-------|------|----------|
| Feature extraction | ~65 sec/video | GPU (RTX 3090) |
| Model inference | ~10 ms/video | GPU |
| Full pipeline (200 videos) | ~3.5 hours | GPU |

### Ball Detection Rate

- **Overall**: 77.2% of frames have successful ball detection
- **Impact**: Missing ball detections are handled with zero-padding

---

## Troubleshooting

### Common Issues

**1. SAM3/SAM3D not found**
```bash
export PYTHONPATH="/workspace/sam3:/workspace/sam-3d-body:$PYTHONPATH"
```

**2. Missing dependencies**
```bash
pip install pycocotools braceexpand scikit-learn
```

**3. CUDA out of memory**
- Reduce batch size in training
- Process fewer frames at a time

**4. Low ball detection rate**
- Check video quality
- Verify SAM3 prompts ("basketball", "person holding basketball")

---

## Future Improvements

1. **Data augmentation**: Temporal shifts, feature noise
2. **Ensemble**: Combine CNN + LSTM + Transformer predictions
3. **Confidence calibration**: Better uncertainty estimates
4. **Active learning**: Prioritize labeling videos where detector is uncertain
