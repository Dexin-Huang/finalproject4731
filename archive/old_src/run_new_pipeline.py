"""
run_new_pipeline.py - Full training pipeline for free throw prediction

Pipeline:
  1. (Optional) Extract SAM3D poses from videos -> features.json
  2. Build enhanced features -> enhanced_all.json
  3. Train KeyJointNet model -> models/best_key_joints_model.pth
  4. Verify saved model

Two modes:
  - WITHOUT SAM3D (fast, very low accuracy): Uses 2D keypoints from candidates.json
  - WITH SAM3D (slow, higher accuracy): Extracts 70-joint 3D poses from videos

Usage:
    # Fast mode (2D keypoints only - not recommended)
    python run_new_pipeline.py --candidates data/release_detection/candidates.json

    # Full mode with SAM3D (requires GPU + SAM3D installed)
    python run_new_pipeline.py --candidates data/release_detection/candidates.json --use-sam3d --video-dir data/hq_videos

    # Skip extraction, just train on existing features
    python run_new_pipeline.py --skip-build --data data/features/enhanced_all.json
"""

import argparse
import os
import sys
import json
import subprocess
from pathlib import Path
from datetime import datetime
import numpy as np


def run_command(cmd, description):
    """
    Run a command and return success status
    """
    print(f"\n{'=' * 60}")
    print(f"Step: {description}")
    print(f"Command: {' '.join(cmd)}")
    print('=' * 60)

    try:
        result = subprocess.run(cmd, check=True)
        print(f"[OK] {description}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[FAILED] {description} (exit code {e.returncode})")
        return False
    except FileNotFoundError:
        print(f"[FAILED] Script not found: {cmd[1]}")
        return False


def extract_sam3d_poses(candidates_path, output_path, video_dirs, filter_quality=None,
                        labeled_only=True, device='cuda', limit=None):
    """
    Extract SAM3D 3D poses from videos
    """
    print(f"\n{'=' * 60}")
    print("Step 0: Extracting SAM3D 3D poses from videos")
    print('=' * 60)

    cmd = [
        sys.executable, 'src/extract_sequences.py',
        '--input', str(candidates_path),
        '--output', str(output_path),
        '--device', device,
    ]

    for vdir in video_dirs:
        cmd.extend(['--video-dir', vdir])

    if filter_quality:
        cmd.extend(['--filter-quality', filter_quality])

    if labeled_only:
        cmd.append('--labeled-only')

    if limit:
        cmd.extend(['--limit', str(limit)])

    return run_command(cmd, "Extract SAM3D poses")


def compute_velocity(keypoints_sequence):
    """
    Compute velocity from keypoints
    """
    if len(keypoints_sequence) < 2:
        return np.zeros_like(keypoints_sequence)
    velocity = np.diff(keypoints_sequence, axis=0)
    return np.concatenate([velocity, velocity[-1:]], axis=0)


def compute_acceleration(keypoints_sequence):
    """
    Compute acceleration from keypoints
    """
    if len(keypoints_sequence) < 3:
        return np.zeros_like(keypoints_sequence)
    velocity = np.diff(keypoints_sequence, axis=0)
    accel = np.diff(velocity, axis=0)
    return np.concatenate([accel, accel[-1:], accel[-1:]], axis=0)


def build_features_from_sam3d(features_path, output_dir):
    """
    Build enhanced features from SAM3D-extracted features.json
    """
    print(f"\n{'=' * 60}")
    print("Step 1a: Building enhanced features from SAM3D output")
    print('=' * 60)

    try:
        print(f"Loading {features_path}...")
        with open(features_path) as f:
            sequences = json.load(f)

        print(f"Loaded {len(sequences)} sequences")

        enhanced_list = []
        skipped = 0

        for seq in sequences:
            frames = seq.get('frames', [])
            valid_frames = [f for f in frames if f and 'keypoints_3d' in f]

            if len(valid_frames) < 2:
                skipped += 1
                continue

            keypoints_3d = []
            for f in frames:
                if f and 'keypoints_3d' in f:
                    keypoints_3d.append(np.array(f['keypoints_3d']))
                elif keypoints_3d:
                    keypoints_3d.append(keypoints_3d[-1])

            if len(keypoints_3d) < 4:
                while len(keypoints_3d) < 4:
                    keypoints_3d.append(keypoints_3d[-1] if keypoints_3d else np.zeros((70, 3)))

            keypoints_3d = np.array(keypoints_3d[:4])

            velocity = compute_velocity(keypoints_3d)
            acceleration = compute_acceleration(keypoints_3d)

            enhanced_entry = {
                'video_id': seq.get('video_id', ''),
                'video_file': seq.get('video_file', ''),
                'label': seq.get('label', -1),
                'release_frame': seq.get('release_frame', 0),
                'frame_indices': seq.get('frame_indices', []),
                'frames': frames,
                'keypoints_3d': keypoints_3d.tolist(),
                'velocity': velocity.tolist(),
                'acceleration': acceleration.tolist(),
                'valid_frames': seq.get('valid_frames', 0),
                'num_joints': keypoints_3d.shape[1],
                'has_keypoints': True,
                'camera_angle': seq.get('camera_angle'),
                'fps': seq.get('fps'),
                'status': seq.get('status'),
                'confidence': seq.get('confidence'),
                'arm_angle': seq.get('arm_angle'),
            }

            enhanced_list.append(enhanced_entry)

        print(f"Processed: {len(enhanced_list)}, Skipped: {skipped}")

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        enhanced_path = output_dir / 'enhanced_all.json'
        with open(enhanced_path, 'w') as f:
            json.dump(enhanced_list, f)
        print(f"Saved: {enhanced_path}")

        labeled = [e for e in enhanced_list if e.get('label', -1) != -1]
        if labeled:
            labeled_path = output_dir / 'enhanced_labeled.json'
            with open(labeled_path, 'w') as f:
                json.dump(labeled, f)
            print(f"Saved: {labeled_path}")

            makes = sum(1 for e in labeled if e['label'] == 1)
            misses = len(labeled) - makes
            print(f"Labels: {makes} makes, {misses} misses")

        return True, len(labeled)

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False, 0


def build_features_from_candidates(candidates_path, output_dir, filter_quality=None, labeled_only=True):
    """
    Build features directly from candidates JSON (2D keypoints only)
    """
    print(f"\n{'=' * 60}")
    print("Step 1: Building features from candidates (2D keypoints)")
    print('=' * 60)

    try:
        sys.path.insert(0, 'src')
        sys.path.insert(0, '.')

        from build_features import process_candidate

        print(f"Loading {candidates_path}...")
        with open(candidates_path) as f:
            candidates = json.load(f)

        print(f"Loaded {len(candidates)} candidates")

        features_list = []
        enhanced_list = []
        skipped = 0

        for i, candidate in enumerate(candidates):
            if i % 200 == 0 and i > 0:
                print(f"  Processing {i}/{len(candidates)}...")

            if filter_quality:
                status = candidate.get('status', '')
                confidence = candidate.get('confidence', '')
                if filter_quality == 'PERFECT' and status != 'PERFECT':
                    skipped += 1
                    continue
                elif filter_quality == 'high' and 'high' not in str(confidence):
                    skipped += 1
                    continue

            if labeled_only and candidate.get('label', -1) == -1:
                skipped += 1
                continue

            features_entry, enhanced_entry = process_candidate(candidate)

            if features_entry:
                features_list.append(features_entry)
                enhanced_list.append(enhanced_entry)
            else:
                skipped += 1

        print(f"\nProcessed: {len(features_list)}, Skipped: {skipped}")

        labeled = [e for e in enhanced_list if e.get('label', -1) != -1]
        makes = sum(1 for e in labeled if e['label'] == 1)
        misses = len(labeled) - makes
        print(f"Labels: {makes} makes, {misses} misses")

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        features_path = output_dir / 'features.json'
        with open(features_path, 'w') as f:
            json.dump(features_list, f)
        print(f"Saved: {features_path}")

        enhanced_path = output_dir / 'enhanced_all.json'
        with open(enhanced_path, 'w') as f:
            json.dump(enhanced_list, f)
        print(f"Saved: {enhanced_path}")

        if labeled:
            labeled_path = output_dir / 'enhanced_labeled.json'
            with open(labeled_path, 'w') as f:
                json.dump(labeled, f)
            print(f"Saved: {labeled_path}")

        return True, len(labeled)

    except ImportError as e:
        print(f"Import error: {e}")
        print("Falling back to subprocess...")

        cmd = [sys.executable, 'src/build_features.py',
               '--input', str(candidates_path),
               '--output-dir', str(output_dir)]
        if filter_quality:
            cmd.extend(['--filter-quality', filter_quality])
        if labeled_only:
            cmd.append('--labeled-only')

        success = run_command(cmd, "Build features")

        enhanced_path = Path(output_dir) / 'enhanced_labeled.json'
        if enhanced_path.exists():
            with open(enhanced_path) as f:
                data = json.load(f)
            return success, len(data)

        return success, 0
    except Exception as e:
        print(f"Error: {e}")
        return False, 0


def train_model(data_path, model_output, n_folds=5, epochs=80, patience=15):
    """Train the KeyJointNet model."""
    print(f"\n{'=' * 60}")
    print("Step 2: Training model")
    print('=' * 60)

    cmd = [
        sys.executable, 'src/train_key_joints.py',
        '--data', str(data_path),
        '--output', str(model_output)
    ]

    return run_command(cmd, "Train model")


def verify_model(model_path):
    """Verify the saved model."""
    print(f"\n{'=' * 60}")
    print("Step 3: Verifying saved model")
    print('=' * 60)

    import torch

    if not os.path.exists(model_path):
        print(f"[FAILED] Model not found: {model_path}")
        return False

    try:
        checkpoint = torch.load(model_path, map_location='cpu')
        size_mb = os.path.getsize(model_path) / (1024 * 1024)

        print(f"Model path: {model_path}")
        print(f"File size: {size_mb:.2f} MB")
        print(f"Model class: {checkpoint.get('model_class', 'unknown')}")
        print(f"Num joints: {checkpoint.get('model_config', {}).get('num_joints', 'unknown')}")
        print(f"Mean accuracy: {checkpoint.get('mean_accuracy', 0):.1%}")
        print(f"Overall AUC: {checkpoint.get('overall_auc', 0):.4f}")
        print(f"Num samples: {checkpoint.get('num_samples', 'unknown')}")
        print(f"Timestamp: {checkpoint.get('timestamp', 'unknown')}")

        print(f"\n[OK] Model verified successfully")
        return True

    except Exception as e:
        print(f"[FAILED] Error loading model: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Run free throw prediction training pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--candidates', type=str, default='data/release_detection/candidates.json')
    parser.add_argument('--features-dir', type=str, default='data/features')
    parser.add_argument('--model-output', type=str, default='models/best_key_joints_model.pth')

    parser.add_argument('--use-sam3d', action='store_true')
    parser.add_argument('--video-dir', type=str, nargs='+', default=['data/hq_videos', 'data/videos'])
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--limit', type=int, default=None)

    parser.add_argument('--skip-build', action='store_true')
    parser.add_argument('--skip-train', action='store_true')
    parser.add_argument('--skip-extract', action='store_true')
    parser.add_argument('--data', type=str, default=None)

    parser.add_argument('--filter-quality', type=str, default=None, choices=['PERFECT', 'high'])
    parser.add_argument('--include-unlabeled', action='store_true')

    parser.add_argument('--n-folds', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=80)
    parser.add_argument('--patience', type=int, default=15)

    args = parser.parse_args()

    start_time = datetime.now()

    print("\n" + "=" * 60)
    print("FREE THROW PREDICTION - TRAINING PIPELINE")
    print("=" * 60)
    print(f"Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Candidates: {args.candidates}")
    print(f"Features dir: {args.features_dir}")
    print(f"Model output: {args.model_output}")
    print(f"Mode: {'SAM3D (3D poses)' if args.use_sam3d else '2D keypoints only'}")

    os.makedirs(args.features_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.model_output) or '.', exist_ok=True)

    results = {}

    # Step 0: SAM3D extraction
    if args.use_sam3d and not args.skip_build and not args.skip_extract:
        features_json_path = os.path.join(args.features_dir, 'features.json')
        success = extract_sam3d_poses(
            args.candidates, features_json_path, args.video_dir,
            filter_quality=args.filter_quality,
            labeled_only=not args.include_unlabeled,
            device=args.device, limit=args.limit
        )
        results['extract_sam3d'] = success
        if not success:
            print("\n[FAILED] SAM3D extraction failed")
            sys.exit(1)

    # Step 1: Build features
    if not args.skip_build:
        if args.use_sam3d:
            features_json_path = os.path.join(args.features_dir, 'features.json')
            if os.path.exists(features_json_path):
                success, _ = build_features_from_sam3d(features_json_path, args.features_dir)
            else:
                print(f"\n[FAILED] features.json not found")
                sys.exit(1)
        else:
            success, _ = build_features_from_candidates(
                args.candidates, args.features_dir,
                filter_quality=args.filter_quality,
                labeled_only=not args.include_unlabeled
            )
        results['build_features'] = success
        if not success:
            print("\n[FAILED] Feature building failed")
            sys.exit(1)
    else:
        print("\n[SKIP] Feature building")
        results['build_features'] = None

    # Determine data path
    if args.data:
        data_path = args.data
    else:
        data_path = os.path.join(args.features_dir, 'enhanced_labeled.json')
        if not os.path.exists(data_path):
            data_path = os.path.join(args.features_dir, 'enhanced_all.json')

    # Step 2: Train
    if not args.skip_train:
        if not os.path.exists(data_path):
            print(f"\n[FAILED] Training data not found: {data_path}")
            sys.exit(1)
        success = train_model(data_path, args.model_output)
        results['train_model'] = success
        if not success:
            print("\n[FAILED] Training failed")
            sys.exit(1)
    else:
        print("\n[SKIP] Model training")
        results['train_model'] = None

    # Step 3: Verify
    if not args.skip_train:
        success = verify_model(args.model_output)
        results['verify_model'] = success
    else:
        results['verify_model'] = None

    # Summary
    end_time = datetime.now()
    duration = end_time - start_time

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"Duration: {duration}")
    print(f"\nResults:")
    for step, success in results.items():
        status = "SKIPPED" if success is None else ("OK" if success else "FAILED")
        print(f"  [{status}] {step}")

    if os.path.exists(args.model_output):
        size_mb = os.path.getsize(args.model_output) / (1024 * 1024)
        print(f"\nTrained model: {args.model_output} ({size_mb:.2f} MB)")
        print("\nNext steps:")
        print(f"  streamlit run app.py")

    print("=" * 60 + "\n")


if __name__ == '__main__':
    main()