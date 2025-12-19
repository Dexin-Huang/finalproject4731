# SWISH

**Skeletal Wrist Inference for Shot prediction in Hoops**

*COMS4731 Computer Vision - Columbia University - Fall 2025*

Predicting basketball free throw outcomes (make/miss) from 3D body pose at the moment of release using deep learning on skeletal graphs.

## Results

| Model | Accuracy | AUC |
|-------|----------|-----|
| **KeyJointNet** | **91.95%** | **0.97** |

### Asymmetric Performance

| Confidence | Prediction | Accuracy | Samples |
|------------|------------|----------|---------|
| P(make) < 0.40 | MISS | **100%** | 88 |
| P(make) > 0.70 | MAKE | 87.9% | 33 |

The model achieves **100% accuracy** on high-confidence miss predictions.

## Installation

### Prerequisites

- Python 3.8+
- CUDA 11.8+ (for GPU acceleration)
- 8GB+ GPU memory (for SAM3D extraction)

### Setup

```bash
# Clone repository
git clone https://github.com/Dexin-Huang/SWISH.git
cd SWISH

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### SAM3D Body (for feature extraction)

Feature extraction requires [SAM3D Body](https://github.com/facebookresearch/sam-3d-body). Pre-extracted features are included in `data/features/`, so this is only needed if processing new videos.

## Usage

### Run Demo App

```bash
streamlit run app.py
```

Upload a video, and the app predicts make/miss at the release frame.

### Training Pipeline

```bash
# Step 1: Train model (uses pre-extracted features)
python src/train_best_merged.py \
    --orig data/features/enhanced_all.json \
    --output models/my_model.pth

# Step 2: Optimize threshold and calibrate
python src/optimize_threshold.py \
    --model models/my_model.pth \
    --data data/features/optimal_merged.json \
    --output models/my_model_calibrated.pth
```

### Feature Extraction (RunPod GPU)

For processing new videos, run on a GPU instance:

```bash
# Extract SAM3D poses from videos
python src/extract_sequences_v2.py \
    --input data/release_detection/final_perfect.json \
    --output data/features/features.json \
    --video-dir data/hq_videos

# Build enhanced features with velocity/acceleration
python src/build_features_mhr70.py \
    --input data/features/features.json \
    --output data/features/enhanced_mhr70.json
```

## Data Format

### Training Data (`data/features/enhanced_all.json`)

```json
{
  "video_id": "ft0_v108_002649_x264",
  "label": 1,                          // 0=miss, 1=make
  "release_frame": 250,
  "keypoints_3d": [[[x,y,z], ...], ...],  // Shape: (4, 70, 3)
  "velocity": [[[vx,vy,vz], ...], ...],   // Shape: (4, 70, 3)
  "acceleration": [[[ax,ay,az], ...], ...]  // Shape: (4, 70, 3)
}
```

### Data Files

| File | Description | Samples |
|------|-------------|---------|
| `enhanced_all.json` | Full training set | 174 |
| `enhanced_clean.json` | High-quality subset | 108 |
| `features.json` | Raw SAM3D extraction | 340 |

## Repository Structure

```
├── app.py                       # Streamlit demo app
├── docs/
│   └── final_report.tex         # LaTeX paper
├── src/
│   ├── train_best_merged.py     # Main training script
│   ├── optimize_threshold.py    # Threshold optimization
│   ├── extract_sequences_v2.py  # SAM3D feature extraction
│   ├── build_features_mhr70.py  # Feature engineering
│   └── models/
│       └── stgcn.py             # Model architectures
├── models/
│   ├── best_merged_calibrated.pth   # Trained model (91.95%)
│   └── model_info.json          # Model metadata
├── data/
│   ├── features/                # Extracted pose features
│   └── release_detection/       # Auto-detection results
├── figures/                     # Paper figures
└── archive/                     # Historical experiments
```

## Method

1. **Release Detection**: YOLOv8-pose identifies ball release frame
2. **Pose Extraction**: SAM3D Body extracts 70-joint 3D skeleton
3. **Feature Engineering**: 4-frame sequence (t-2, t-1, t, t+1) with velocity/acceleration
4. **Classification**: KeyJointNet with 15 key upper body joints
5. **Calibration**: Platt scaling + threshold optimization (0.64)

## Technical Details

- **Architecture**: KeyJointNet - joint attention + temporal CNN (72K params)
- **Key Joints**: 15 upper body (shoulders, elbows, wrists, fingertips)
- **Loss**: Focal Loss (gamma=2.0) for class imbalance
- **Training**: 5-fold stratified cross-validation
- **Dataset**: 174 samples (102 curated + 72 MHR70 misses)

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Model not found | Ensure `models/best_merged_calibrated.pth` exists |
| CUDA out of memory | Reduce batch size or use CPU |
| SAM3D extraction fails | Use RunPod with GPU; see script comments |
| Import errors | Run `pip install -r requirements.txt` |

## Authors

- Wali Ahmed (wa2294@columbia.edu)
- Dexin Huang (dh3172@columbia.edu)
- Irene Nam (yn2334@columbia.edu)

Columbia University COMS4731 - Fall 2025

