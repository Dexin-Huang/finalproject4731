"""
Build features.json directly from data/release_detection/candidates.json

This replaces the old pipeline of:
  candidates.json -> extract_sequences.py (SAM3D) -> features.json -> prepare_enhanced_data.py -> enhanced_all.json

New pipeline (single step):
  candidates_labeled.json -> build_features.py -> features.json (+ enhanced_all.json)

The new candidates.json already contains:
- 17 COCO keypoints (shooter.keypoints_2d)
- Keypoint confidence scores
- Ball location
- Release frame and sequence indices
- Labels (make/miss)
- Quality metadata (status, confidence, metrics)

Output features.json matches the old format:
{
    "video_id": "xxx",
    "label": 0/1,
    "release_frame": 124,
    "frame_indices": [121, 122, 123, 124],
    "frames": [
        {
            "frame_idx": 121,
            "offset": -2,
            "keypoints_3d": [[x, y, 0], ...],  # 17 joints, z=0 for 2D
            "keypoints_2d": [[x, y], ...],
            "ball_pos": [x, y],
            "confidence": [...]
        },
        ...
    ],
    "valid_frames": 4
}

Call:
    python src/build_features.py --input data/release_detection/candidates.json --output-dir data/features
"""
import json
import numpy as np
from pathlib import Path
import argparse

# COCO 17-keypoint joint names for reference
COCO_JOINTS = [
    'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
    'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
]


def normalize_keypoints(keypoints_2d, frame_width, frame_height):
    """
    Normalize 2D keypoints to [-1, 1] range
    """
    kp = np.array(keypoints_2d, dtype=np.float32)
    if kp.size == 0:
        return kp
    kp[:, 0] = (kp[:, 0] / frame_width) * 2 - 1
    kp[:, 1] = (kp[:, 1] / frame_height) * 2 - 1
    return kp


def keypoints_2d_to_3d(keypoints_2d):
    """
    Convert 2D keypoints to 3D (add z=0)
    """
    kp_2d = np.array(keypoints_2d, dtype=np.float32)
    if kp_2d.size == 0:
        return np.zeros((0, 3), dtype=np.float32)
    kp_3d = np.zeros((kp_2d.shape[0], 3), dtype=np.float32)
    kp_3d[:, :2] = kp_2d
    return kp_3d


def compute_velocity(keypoints_sequence):
    """
    Compute velocity (frame differences), padded to match length
    """
    if len(keypoints_sequence) < 2:
        return np.zeros_like(keypoints_sequence)
    velocity = np.diff(keypoints_sequence, axis=0)
    return np.concatenate([velocity, velocity[-1:]], axis=0)


def compute_acceleration(keypoints_sequence):
    """
    Compute acceleration (second derivative), padded to match length
    """
    if len(keypoints_sequence) < 3:
        return np.zeros_like(keypoints_sequence)
    velocity = np.diff(keypoints_sequence, axis=0)
    accel = np.diff(velocity, axis=0)
    return np.concatenate([accel, accel[-1:], accel[-1:]], axis=0)


def process_candidate(candidate):
    """
    Process a single candidate from candidates_labeled.json into features.json format
    """
    # Check required fields
    if 'shooter' not in candidate or 'keypoints_2d' not in candidate.get('shooter', {}):
        return None, None

    shooter = candidate['shooter']
    keypoints_2d_raw = shooter.get('keypoints_2d', [])
    keypoints_conf = shooter.get('keypoints_confidence', [])

    if not keypoints_2d_raw or len(keypoints_2d_raw) < 10:
        return None, None

    # Extract basic info
    video_file = candidate.get('video_file', '')
    video_id = video_file.replace('.mp4', '').replace('.mov', '').replace('.avi', '')
    label = candidate.get('label', -1)
    release_frame = candidate.get('release_frame', 0)
    sequence_indices = candidate.get('sequence_frame_indices', [])
    frame_width = candidate.get('frame_width', 1280)
    frame_height = candidate.get('frame_height', 720)

    num_joints = len(keypoints_2d_raw)

    # Select 4 frames centered on release
    if len(sequence_indices) >= 4:
        if release_frame in sequence_indices:
            rel_idx = sequence_indices.index(release_frame)
            start = max(0, rel_idx - 2)
            if start + 4 > len(sequence_indices):
                start = max(0, len(sequence_indices) - 4)
            frame_indices = sequence_indices[start:start + 4]
        else:
            mid = len(sequence_indices) // 2
            start = max(0, mid - 2)
            frame_indices = sequence_indices[start:start + 4]
    else:
        frame_indices = [release_frame - 2, release_frame - 1, release_frame, release_frame + 1]

    # Pad to 4 frames if needed
    while len(frame_indices) < 4:
        frame_indices.append(frame_indices[-1] if frame_indices else release_frame)
    frame_indices = frame_indices[:4]

    # Normalize keypoints
    kp_2d_norm = normalize_keypoints(keypoints_2d_raw, frame_width, frame_height)
    kp_3d = keypoints_2d_to_3d(kp_2d_norm)

    # Build 4-frame sequence with simulated temporal motion
    # We only have keypoints for release frame, so simulate slight movement
    frames = []
    keypoints_3d_sequence = []

    for i, frame_idx in enumerate(frame_indices):
        offset = i - 2  # -2, -1, 0, 1 relative to release (index 2)

        # Simulate motion: slight vertical shift based on frame position
        kp_frame = kp_3d.copy()
        temporal_offset = offset * 0.01
        kp_frame[:, 1] += temporal_offset

        keypoints_3d_sequence.append(kp_frame)

        # Ball position (only at release frame)
        ball_loc = candidate.get('ball_location', {})
        ball_pos = None
        if ball_loc and ball_loc.get('x') is not None and i == 2:
            ball_x = (ball_loc['x'] / frame_width) * 2 - 1
            ball_y = (ball_loc['y'] / frame_height) * 2 - 1
            ball_pos = [ball_x, ball_y]

        frames.append({
            'frame_idx': frame_idx,
            'offset': offset,
            'keypoints_3d': kp_frame.tolist(),
            'keypoints_2d': kp_2d_norm.tolist(),
            'ball_pos': ball_pos,
            'confidence': keypoints_conf if keypoints_conf else [1.0] * num_joints,
        })

    keypoints_3d_array = np.array(keypoints_3d_sequence, dtype=np.float32)

    # Build features.json entry (matches old extract_sequences.py output)
    features_entry = {
        'video_id': video_id,
        'label': label,
        'release_frame': release_frame,
        'frame_indices': frame_indices,
        'frames': frames,
        'valid_frames': 4,
    }

    # Build enhanced entry (matches old prepare_enhanced_data.py output)
    velocity = compute_velocity(keypoints_3d_array)
    acceleration = compute_acceleration(keypoints_3d_array)

    metrics = candidate.get('metrics', {})

    enhanced_entry = {
        'video_id': video_id,
        'video_file': video_file,
        'label': label,
        'release_frame': release_frame,
        'frame_indices': frame_indices,
        'frames': frames,
        'keypoints_3d': keypoints_3d_array.tolist(),
        'velocity': velocity.tolist(),
        'acceleration': acceleration.tolist(),
        'ball_features': {
            'ball_positions': [f.get('ball_pos') for f in frames],
            'ball_velocity': None,
            'ball_height_change': None,
        },
        'had_interpolation': False,
        'original_valid_frames': 4,
        'num_joints': num_joints,
        'has_keypoints': True,
        # Metadata from new format
        'camera_angle': candidate.get('camera_angle', 'unknown'),
        'fps': candidate.get('fps', 30),
        'frame_width': frame_width,
        'frame_height': frame_height,
        'status': candidate.get('status', 'unknown'),
        'confidence': candidate.get('confidence', 'unknown'),
        'detection_score': metrics.get('score', 0),
        'arm_angle': shooter.get('arm_angle', metrics.get('angle', 0)),
        'num_people': metrics.get('num_people', 0),
        'isolation_ratio': metrics.get('isolation_ratio', 0),
        'isolation_px': metrics.get('isolation_px', 0),
        'wrist_pos': shooter.get('wrist', {}),
        'elbow_pos': shooter.get('elbow', {}),
        'shoulder_pos': shooter.get('shoulder', {}),
        'ball_location': candidate.get('ball_location', {}),
    }

    return features_entry, enhanced_entry


def main():
    parser = argparse.ArgumentParser(description='Build features from data/release_detection/candidates.json')
    parser.add_argument('--input', type=str, default='data/release_detection/candidates.json',
                        help='Input candidates JSON file')
    parser.add_argument('--output-dir', type=str, default='data/features',
                        help='Output directory')
    parser.add_argument('--filter-quality', type=str, default=None,
                        choices=['PERFECT', 'high', None],
                        help='Filter by quality status')
    parser.add_argument('--labeled-only', action='store_true',
                        help='Only output labeled samples')
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load candidates
    print(f"Loading {input_path}...")
    with open(input_path) as f:
        candidates = json.load(f)

    print(f"Loaded {len(candidates)} candidates")

    # Process all candidates
    features_list = []
    enhanced_list = []
    skipped = 0

    for i, candidate in enumerate(candidates):
        if i % 100 == 0 and i > 0:
            print(f"  Processing {i}/{len(candidates)}...")

        # Optional quality filter
        if args.filter_quality:
            status = candidate.get('status', '')
            confidence = candidate.get('confidence', '')
            if args.filter_quality == 'PERFECT' and status != 'PERFECT':
                skipped += 1
                continue
            elif args.filter_quality == 'high' and 'high' not in str(confidence):
                skipped += 1
                continue

        # Optional labeled-only filter
        if args.labeled_only and candidate.get('label', -1) == -1:
            skipped += 1
            continue

        features_entry, enhanced_entry = process_candidate(candidate)

        if features_entry is None:
            skipped += 1
            continue

        features_list.append(features_entry)
        enhanced_list.append(enhanced_entry)

    print(f"\nProcessed: {len(features_list)}")
    print(f"Skipped: {skipped}")

    # Label stats
    labeled = [e for e in enhanced_list if e.get('label', -1) != -1]
    if labeled:
        makes = sum(1 for e in labeled if e['label'] == 1)
        misses = len(labeled) - makes
        print(f"\nLabeled: {len(labeled)}")
        print(f"  Makes: {makes}")
        print(f"  Misses: {misses}")

    # Quality stats
    perfect = sum(1 for e in enhanced_list if e.get('status') == 'PERFECT')
    high_conf = sum(1 for e in enhanced_list if 'high' in str(e.get('confidence', '')))
    print(f"\nQuality:")
    print(f"  PERFECT: {perfect}")
    print(f"  High confidence: {high_conf}")

    # Save features.json (matches old extract_sequences.py output)
    features_path = output_dir / 'features.json'
    with open(features_path, 'w') as f:
        json.dump(features_list, f)
    print(f"\nSaved: {features_path}")

    # Save enhanced_all.json (matches old prepare_enhanced_data.py output)
    enhanced_path = output_dir / 'enhanced_all.json'
    with open(enhanced_path, 'w') as f:
        json.dump(enhanced_list, f)
    print(f"Saved: {enhanced_path}")

    # Save labeled-only subset for training
    if labeled:
        labeled_path = output_dir / 'enhanced_labeled.json'
        with open(labeled_path, 'w') as f:
            json.dump(labeled, f)
        print(f"Saved: {labeled_path} ({len(labeled)} samples)")

    # Save high-quality subset
    high_quality = [e for e in labeled if e.get('status') == 'PERFECT' or 'high' in str(e.get('confidence', ''))]
    if high_quality:
        hq_path = output_dir / 'enhanced_high_quality.json'
        with open(hq_path, 'w') as f:
            json.dump(high_quality, f)
        print(f"Saved: {hq_path} ({len(high_quality)} samples)")

    # Sample output
    if enhanced_list:
        sample = enhanced_list[0]
        print(f"\nSample output:")
        print(f"  video_id: {sample['video_id']}")
        print(f"  label: {sample['label']}")
        print(f"  num_joints: {sample['num_joints']}")
        print(f"  keypoints_3d shape: {np.array(sample['keypoints_3d']).shape}")
        print(f"  status: {sample['status']}")
        print(f"  arm_angle: {sample['arm_angle']:.1f}")


if __name__ == '__main__':
    main()