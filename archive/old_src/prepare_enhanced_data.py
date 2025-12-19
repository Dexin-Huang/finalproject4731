"""
Prepare enhanced dataset with:
1. Interpolated sequences (3/4 valid -> 4/4)
2. Velocity and acceleration features
3. Ball trajectory features
4. Joint angle features
"""
import json
import numpy as np
from pathlib import Path


def interpolate_missing_frame(frames, missing_idx):
    """
    Linear interpolation for a single missing frame.

    Args:
        frames: list of frame dicts (some may be None or missing keypoints_3d)
        missing_idx: index of missing frame
    """
    # Find nearest valid frames
    prev_idx = None
    next_idx = None

    for i in range(missing_idx - 1, -1, -1):
        if frames[i] and 'keypoints_3d' in frames[i]:
            prev_idx = i
            break

    for i in range(missing_idx + 1, len(frames)):
        if frames[i] and 'keypoints_3d' in frames[i]:
            next_idx = i
            break

    if prev_idx is None and next_idx is None:
        return None

    # Interpolate
    if prev_idx is None:
        # Use next frame
        return frames[next_idx].copy()
    elif next_idx is None:
        # Use previous frame
        return frames[prev_idx].copy()
    else:
        # Linear interpolation
        t = (missing_idx - prev_idx) / (next_idx - prev_idx)

        interpolated = {
            'frame_idx': frames[missing_idx]['frame_idx'] if frames[missing_idx] else missing_idx,
            'offset': [-2, -1, 0, 1][missing_idx],
            'interpolated': True
        }

        # Interpolate keypoints_3d
        kp_prev = np.array(frames[prev_idx]['keypoints_3d'])
        kp_next = np.array(frames[next_idx]['keypoints_3d'])
        interpolated['keypoints_3d'] = (kp_prev * (1 - t) + kp_next * t).tolist()

        # Interpolate keypoints_2d if available
        if 'keypoints_2d' in frames[prev_idx] and 'keypoints_2d' in frames[next_idx]:
            kp2d_prev = np.array(frames[prev_idx]['keypoints_2d'])
            kp2d_next = np.array(frames[next_idx]['keypoints_2d'])
            interpolated['keypoints_2d'] = (kp2d_prev * (1 - t) + kp2d_next * t).tolist()

        # Interpolate ball_pos if available
        if 'ball_pos' in frames[prev_idx] and 'ball_pos' in frames[next_idx]:
            bp_prev = np.array(frames[prev_idx]['ball_pos'])
            bp_next = np.array(frames[next_idx]['ball_pos'])
            interpolated['ball_pos'] = (bp_prev * (1 - t) + bp_next * t).tolist()

        # Interpolate bbox if available
        if 'bbox' in frames[prev_idx] and 'bbox' in frames[next_idx]:
            bb_prev = np.array(frames[prev_idx]['bbox'])
            bb_next = np.array(frames[next_idx]['bbox'])
            interpolated['bbox'] = (bb_prev * (1 - t) + bb_next * t).tolist()

        return interpolated


def compute_velocity(keypoints_sequence):
    """
    Compute velocity (frame-to-frame differences) for keypoints.

    Args:
        keypoints_sequence: (T, V, C) array
    Returns:
        velocity: (T-1, V, C) array
    """
    return np.diff(keypoints_sequence, axis=0)


def compute_acceleration(keypoints_sequence):
    """
    Compute acceleration (second derivative).

    Args:
        keypoints_sequence: (T, V, C) array
    Returns:
        acceleration: (T-2, V, C) array
    """
    velocity = compute_velocity(keypoints_sequence)
    return np.diff(velocity, axis=0)


def compute_joint_angles(keypoints):
    """
    Compute key joint angles for shooting analysis.

    SAM3D Body uses ~70 joints. Common joint indices (may vary):
    - Pelvis: 0
    - Spine: 1-3
    - Shoulders: ~16, 17
    - Elbows: ~18, 19
    - Wrists: ~20, 21

    We'll compute angles for elbow and shoulder joints.
    """
    # These indices are approximate - SAM3D may use different ordering
    # We'll use relative positions instead

    angles = {}

    # For now, compute distance-based features that are skeleton-agnostic
    # Distance from hands to head region
    # Spread of arms
    # Height of hands relative to body center

    return angles


def compute_ball_features(frames):
    """
    Compute ball trajectory features.

    Returns:
        dict with ball velocity, acceleration, height
    """
    ball_positions = []
    for f in frames:
        if f and 'ball_pos' in f:
            ball_positions.append(f['ball_pos'])
        else:
            ball_positions.append(None)

    features = {
        'ball_positions': ball_positions,
        'ball_velocity': None,
        'ball_height_change': None
    }

    # Compute velocity if we have enough points
    valid_pos = [(i, p) for i, p in enumerate(ball_positions) if p is not None]
    if len(valid_pos) >= 2:
        velocities = []
        for i in range(len(valid_pos) - 1):
            idx1, p1 = valid_pos[i]
            idx2, p2 = valid_pos[i + 1]
            dt = idx2 - idx1
            if dt > 0:
                vx = (p2[0] - p1[0]) / dt
                vy = (p2[1] - p1[1]) / dt
                velocities.append([vx, vy])

        if velocities:
            features['ball_velocity'] = velocities
            # Height change (y decreases = ball going up in image coords)
            features['ball_height_change'] = valid_pos[-1][1][1] - valid_pos[0][1][1]

    return features


def process_sequence(seq, min_valid=3):
    """
    Process a single sequence: interpolate if needed, compute features.

    Args:
        seq: sequence dict
        min_valid: minimum valid frames required

    Returns:
        processed sequence dict or None if not enough valid frames
    """
    frames = seq['frames']

    # Count valid frames
    valid_indices = []
    invalid_indices = []
    for i, f in enumerate(frames):
        if f and 'keypoints_3d' in f:
            valid_indices.append(i)
        else:
            invalid_indices.append(i)

    if len(valid_indices) < min_valid:
        return None

    # Interpolate missing frames
    new_frames = frames.copy()
    for idx in invalid_indices:
        interpolated = interpolate_missing_frame(frames, idx)
        if interpolated:
            new_frames[idx] = interpolated

    # Check if all frames now have keypoints
    all_valid = all(f and 'keypoints_3d' in f for f in new_frames)
    if not all_valid:
        return None

    # Extract keypoints sequence
    keypoints_3d = np.array([f['keypoints_3d'] for f in new_frames])  # (4, 70, 3)

    # Compute velocity (between consecutive frames)
    velocity = compute_velocity(keypoints_3d)  # (3, 70, 3)

    # Pad velocity to match temporal dimension
    velocity_padded = np.concatenate([velocity, velocity[-1:]], axis=0)  # (4, 70, 3)

    # Compute acceleration
    accel = compute_acceleration(keypoints_3d)  # (2, 70, 3)
    accel_padded = np.concatenate([accel, accel[-1:], accel[-1:]], axis=0)  # (4, 70, 3)

    # Ball features
    ball_features = compute_ball_features(new_frames)

    # Create enhanced sequence
    enhanced = {
        'video_id': seq['video_id'],
        'label': seq['label'],
        'release_frame': seq['release_frame'],
        'frame_indices': seq['frame_indices'],
        'frames': new_frames,
        'keypoints_3d': keypoints_3d.tolist(),  # (4, 70, 3)
        'velocity': velocity_padded.tolist(),    # (4, 70, 3)
        'acceleration': accel_padded.tolist(),   # (4, 70, 3)
        'ball_features': ball_features,
        'had_interpolation': len(invalid_indices) > 0,
        'original_valid_frames': len(valid_indices)
    }

    return enhanced


def main():
    input_path = Path('data/features/features.json')
    output_dir = Path('data/features')

    with open(input_path) as f:
        data = json.load(f)

    print(f"Loaded {len(data)} sequences")

    # Process all sequences
    enhanced_all = []
    enhanced_clean = []  # 4/4 valid only

    for seq in data:
        processed = process_sequence(seq, min_valid=3)
        if processed:
            enhanced_all.append(processed)
            if not processed['had_interpolation']:
                enhanced_clean.append(processed)

    print(f"\nProcessed sequences:")
    print(f"  With interpolation (3+ valid): {len(enhanced_all)}")
    print(f"  Clean only (4/4 valid): {len(enhanced_clean)}")

    # Stats
    made_all = sum(s['label'] for s in enhanced_all)
    made_clean = sum(s['label'] for s in enhanced_clean)
    print(f"\nLabels:")
    print(f"  All: {made_all} made, {len(enhanced_all) - made_all} miss")
    print(f"  Clean: {made_clean} made, {len(enhanced_clean) - made_clean} miss")

    # Save
    with open(output_dir / 'enhanced_all.json', 'w') as f:
        json.dump(enhanced_all, f)

    with open(output_dir / 'enhanced_clean.json', 'w') as f:
        json.dump(enhanced_clean, f)

    print(f"\nSaved to:")
    print(f"  {output_dir / 'enhanced_all.json'}")
    print(f"  {output_dir / 'enhanced_clean.json'}")

    # Sample check
    sample = enhanced_all[0]
    print(f"\nSample features shape:")
    print(f"  keypoints_3d: {np.array(sample['keypoints_3d']).shape}")
    print(f"  velocity: {np.array(sample['velocity']).shape}")
    print(f"  acceleration: {np.array(sample['acceleration']).shape}")


if __name__ == '__main__':
    main()
