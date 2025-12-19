"""
Build enhanced features from SAM3D Body extraction (MHR70 format).

Takes features.json (raw 3D poses) and computes:
- Normalized keypoints_3d (4, 70, 3)
- Velocity (4, 70, 3)
- Acceleration (4, 70, 3)

Usage:
    python src/build_features_mhr70.py --input data/features/features.json
"""
import json
import numpy as np
import argparse
from pathlib import Path


# MHR70 key joint groups for basketball shooting
MHR70_JOINT_GROUPS = {
    "right_arm": [6, 8, 41, 64, 66, 68],  # shoulder, elbow, wrist, olecranon, cubital_fossa, acromion
    "right_hand": list(range(21, 42)),     # all right hand joints including wrist
    "left_arm": [5, 7, 62, 63, 65, 67],
    "left_hand": list(range(42, 63)),
    "body_core": [0, 9, 10, 69],           # nose, hips, neck
    "legs": [11, 12, 13, 14],              # knees, ankles
    "head": [0, 1, 2, 3, 4],               # nose, eyes, ears
}

# Key joints for shooting analysis (subset for attention)
KEY_SHOOTING_JOINTS = (
    MHR70_JOINT_GROUPS["right_arm"] +
    MHR70_JOINT_GROUPS["right_hand"] +
    MHR70_JOINT_GROUPS["body_core"] +
    MHR70_JOINT_GROUPS["legs"]
)


def build_sequence_features(sample):
    """
    Build enhanced features for a single sequence.

    Returns dict with:
        - keypoints_3d: (4, 70, 3) normalized poses
        - velocity: (4, 70, 3)
        - acceleration: (4, 70, 3)
        - label: 0/1
        - metadata
    """
    frames = sample.get("frames", [])
    label = sample.get("label", -1)

    # Skip unlabeled or invalid samples
    if label == -1:
        return None

    # Check we have 4 valid frames
    valid_frames = [f for f in frames if f.get("pose_valid") and "keypoints_3d" in f]
    if len(valid_frames) < 4:
        return None

    # Extract keypoints_3d for each frame: (4, 70, 3)
    keypoints = []
    for f in frames:
        if f.get("pose_valid") and "keypoints_3d" in f:
            kp = np.array(f["keypoints_3d"])  # (70, 3)
            keypoints.append(kp)
        else:
            # Use previous frame if invalid (shouldn't happen with 100% valid)
            keypoints.append(keypoints[-1] if keypoints else np.zeros((70, 3)))

    keypoints = np.array(keypoints)  # (4, 70, 3)

    # Normalize: center on hip midpoint, scale by max distance
    hip_center = (keypoints[:, 9:10, :] + keypoints[:, 10:11, :]) / 2  # (4, 1, 3)
    keypoints_centered = keypoints - hip_center

    # Scale by max distance from center
    dists = np.linalg.norm(keypoints_centered, axis=-1)  # (4, 70)
    max_dist = dists.max()
    if max_dist > 1e-6:
        keypoints_norm = keypoints_centered / max_dist
    else:
        keypoints_norm = keypoints_centered

    # Compute velocity (frame-to-frame difference)
    velocity = np.zeros_like(keypoints_norm)
    velocity[1:] = keypoints_norm[1:] - keypoints_norm[:-1]
    velocity[0] = velocity[1]  # Copy first frame

    # Compute acceleration (velocity difference)
    acceleration = np.zeros_like(velocity)
    acceleration[1:] = velocity[1:] - velocity[:-1]
    acceleration[0] = acceleration[1]

    return {
        "video_id": sample.get("video_id", ""),
        "video_file": sample.get("video_file", ""),
        "label": label,
        "keypoints_3d": keypoints_norm.tolist(),
        "velocity": velocity.tolist(),
        "acceleration": acceleration.tolist(),
        "release_frame": sample.get("release_frame"),
        "bbox": sample.get("bbox"),
        "ball_pos": sample.get("ball_pos"),
        "num_joints": 70,
        "joint_format": "MHR70",
    }


def main():
    parser = argparse.ArgumentParser(description='Build enhanced features from SAM3D extraction')
    parser.add_argument('--input', type=str, default='data/features/features.json',
                        help='Input features.json from extraction')
    parser.add_argument('--output', type=str, default='data/features/enhanced_mhr70.json',
                        help='Output enhanced features')
    args = parser.parse_args()

    print(f"Loading features from {args.input}...")
    with open(args.input) as f:
        raw_features = json.load(f)
    print(f"Loaded {len(raw_features)} samples")

    # Build enhanced features
    enhanced = []
    skipped = 0

    for sample in raw_features:
        result = build_sequence_features(sample)
        if result:
            enhanced.append(result)
        else:
            skipped += 1

    print(f"Built {len(enhanced)} enhanced samples ({skipped} skipped)")

    # Stats
    makes = sum(1 for s in enhanced if s["label"] == 1)
    misses = sum(1 for s in enhanced if s["label"] == 0)
    print(f"Labels: {makes} makes, {misses} misses")

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(enhanced, f)

    print(f"Saved to {args.output}")

    # Also save labeled-only version
    labeled_path = output_path.parent / "enhanced_mhr70_labeled.json"
    labeled = [s for s in enhanced if s["label"] in [0, 1]]
    with open(labeled_path, "w") as f:
        json.dump(labeled, f)
    print(f"Saved labeled subset ({len(labeled)}) to {labeled_path}")


if __name__ == "__main__":
    main()
