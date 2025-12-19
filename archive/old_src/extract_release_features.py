"""
Extract features for release frame detection.

For each video, extracts a sequence of features around the estimated release frame
that can be used to train a release frame detector.

Features per frame:
- Ball position (x, y), area, velocity, speed, acceleration
- Ball-hand distances (wrists, fingertips)
- Arm extension angles and ratios
- Ball-shooter mask overlap
- Hand/wrist velocities

Designed to run on GPU server with SAM3 + SAM3D Body.
"""

import os
import sys
import json
import argparse
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple

# Add SAM3 paths for GPU server
sys.path.insert(0, "/workspace/sam3")
sys.path.insert(0, "/workspace/sam-3d-body")

try:
    import torch
    from PIL import Image
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# Try to import SAM3 (GPU server only)
try:
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
    HAS_SAM3 = True
except ImportError:
    HAS_SAM3 = False
    print("SAM3 not available - some features will be limited")

# Try to import SAM3D Body (GPU server only)
try:
    from sam_3d_body import load_sam_3d_body, SAM3DBodyEstimator
    HAS_SAM3D = True
except ImportError:
    HAS_SAM3D = False
    print("SAM3D Body not available - using 2D features only")


# SAM3D Body joint indices for key body parts
JOINT_INDICES = {
    'pelvis': 0,
    'left_hip': 1,
    'right_hip': 2,
    'spine1': 3,
    'left_knee': 4,
    'right_knee': 5,
    'spine2': 6,
    'left_ankle': 7,
    'right_ankle': 8,
    'spine3': 9,
    'left_foot': 10,
    'right_foot': 11,
    'neck': 12,
    'left_collar': 13,
    'right_collar': 14,
    'head': 15,
    'left_shoulder': 16,
    'right_shoulder': 17,
    'left_elbow': 18,
    'right_elbow': 19,
    'left_wrist': 20,
    'right_wrist': 21,
    # Hand joints start at 22 for left hand, 37 for right hand
    'left_hand_root': 22,
    'right_hand_root': 37,
}

# Fingertip indices (approximate - actual indices depend on hand model)
LEFT_FINGERTIPS = [26, 31, 35, 39, 43]  # thumb, index, middle, ring, pinky tips
RIGHT_FINGERTIPS = [47, 52, 56, 60, 64]


def compute_angle(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
    """Compute angle at p2 formed by p1-p2-p3."""
    v1 = p1 - p2
    v2 = p3 - p2

    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
    cos_angle = np.clip(cos_angle, -1, 1)
    angle = np.arccos(cos_angle)

    return np.degrees(angle)


def get_mask_center(mask: np.ndarray) -> Optional[Tuple[float, float]]:
    """Get centroid of binary mask."""
    if mask is None:
        return None
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    return (float(xs.mean()), float(ys.mean()))


def get_mask_area(mask: np.ndarray) -> float:
    """Get area of binary mask."""
    if mask is None:
        return 0.0
    return float(np.sum(mask > 0))


def masks_overlap_ratio(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """Compute overlap ratio of two masks (intersection / mask1 area)."""
    if mask1 is None or mask2 is None:
        return 0.0
    if mask1.shape != mask2.shape:
        return 0.0

    intersection = np.logical_and(mask1 > 0, mask2 > 0).sum()
    area1 = np.sum(mask1 > 0)

    if area1 == 0:
        return 0.0
    return float(intersection / area1)


class ReleaseFeatureExtractor:
    """Extract features for release frame detection."""

    def __init__(self, device: str = "cuda"):
        self.device = device
        self.sam3_processor = None
        self.sam3db_estimator = None

        if HAS_SAM3 and HAS_TORCH:
            print("Loading SAM3...")
            bpe_path = "/workspace/sam3/sam3/assets/bpe_simple_vocab_16e6.txt.gz"
            sam3_model = build_sam3_image_model(device=device, bpe_path=bpe_path)
            self.sam3_processor = Sam3Processor(sam3_model)
            print("SAM3 loaded!")

        if HAS_SAM3D and HAS_TORCH:
            print("Loading SAM3D Body...")
            sam3db_model, sam3db_cfg = load_sam_3d_body(
                "/workspace/sam-3d-body/checkpoints/sam-3d-body-dinov3/model.ckpt",
                device=device,
                mhr_path="/workspace/sam-3d-body/checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt"
            )
            self.sam3db_estimator = SAM3DBodyEstimator(
                sam_3d_body_model=sam3db_model,
                model_cfg=sam3db_cfg,
            )
            print("SAM3D Body loaded!")

    def get_ball_mask(self, frame_rgb: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[Tuple[float, float]], float]:
        """Detect basketball using SAM3."""
        if self.sam3_processor is None:
            return None, None, 0.0

        pil_img = Image.fromarray(frame_rgb)

        with torch.inference_mode():
            state = self.sam3_processor.set_image(pil_img)
            output = self.sam3_processor.set_text_prompt(state=state, prompt="basketball")

        masks = output["masks"]
        scores = output["scores"]

        if len(masks) == 0:
            return None, None, 0.0

        # Get best mask
        best_idx = torch.argmax(torch.tensor([s.item() if torch.is_tensor(s) else s for s in scores]))
        mask = masks[best_idx]
        score = float(scores[best_idx].item() if torch.is_tensor(scores[best_idx]) else scores[best_idx])

        if torch.is_tensor(mask):
            mask = mask.cpu().numpy()
        if mask.ndim == 3:
            mask = mask[0]
        mask = (mask > 0.5).astype(np.uint8)

        center = get_mask_center(mask)

        return mask, center, score

    def get_shooter_mask(self, frame_rgb: np.ndarray) -> Tuple[Optional[np.ndarray], float]:
        """Detect person holding basketball using SAM3."""
        if self.sam3_processor is None:
            return None, 0.0

        pil_img = Image.fromarray(frame_rgb)

        prompts = ["person holding basketball", "person shooting basketball", "basketball player"]
        best_mask = None
        best_score = 0.0

        for prompt in prompts:
            with torch.inference_mode():
                state = self.sam3_processor.set_image(pil_img)
                output = self.sam3_processor.set_text_prompt(state=state, prompt=prompt)

            masks = output["masks"]
            scores = output["scores"]

            if len(masks) > 0:
                best_idx = torch.argmax(torch.tensor([s.item() if torch.is_tensor(s) else s for s in scores]))
                score = float(scores[best_idx].item() if torch.is_tensor(scores[best_idx]) else scores[best_idx])

                if score > best_score:
                    mask = masks[best_idx]
                    if torch.is_tensor(mask):
                        mask = mask.cpu().numpy()
                    if mask.ndim == 3:
                        mask = mask[0]
                    best_mask = (mask > 0.5).astype(np.uint8)
                    best_score = score

        return best_mask, best_score

    def get_3d_pose(self, frame_rgb: np.ndarray, shooter_mask: np.ndarray) -> Optional[np.ndarray]:
        """Get 3D pose using SAM3D Body."""
        if self.sam3db_estimator is None or shooter_mask is None:
            return None

        # Get bounding box from mask
        ys, xs = np.where(shooter_mask > 0)
        if len(xs) == 0:
            return None

        x1, y1, x2, y2 = xs.min(), ys.min(), xs.max(), ys.max()
        bbox = np.array([[x1, y1, x2, y2]])

        try:
            outputs = self.sam3db_estimator.process_one_image(
                frame_rgb,
                bboxes=bbox,
                masks=shooter_mask.astype(np.uint8)[None, :, :],
                use_mask=True,
            )

            if len(outputs) > 0 and "pred_keypoints_3d" in outputs[0]:
                kp3d = outputs[0]["pred_keypoints_3d"]
                if torch.is_tensor(kp3d):
                    kp3d = kp3d.cpu().numpy()
                return kp3d
        except Exception as e:
            print(f"  SAM3D Body error: {e}")

        return None

    def extract_frame_features(
        self,
        frame_rgb: np.ndarray,
        prev_ball_pos: Optional[Tuple[float, float]] = None,
        prev_ball_vel: Optional[Tuple[float, float]] = None,
        prev_keypoints: Optional[np.ndarray] = None,
    ) -> Dict:
        """Extract all features for a single frame."""
        h, w = frame_rgb.shape[:2]

        features = {
            # Ball features
            'ball_x': None,
            'ball_y': None,
            'ball_area': 0.0,
            'ball_velocity_x': 0.0,
            'ball_velocity_y': 0.0,
            'ball_speed': 0.0,
            'ball_acceleration': 0.0,
            'ball_detected': False,

            # Hand-ball distances
            'dist_left_wrist': None,
            'dist_right_wrist': None,
            'dist_left_fingers': None,
            'dist_right_fingers': None,
            'min_hand_dist': None,

            # Arm features
            'left_arm_angle': None,
            'right_arm_angle': None,
            'left_arm_extension': None,
            'right_arm_extension': None,

            # Mask features
            'ball_shooter_overlap': 0.0,
            'shooter_detected': False,

            # Pose velocity features
            'left_wrist_velocity': 0.0,
            'right_wrist_velocity': 0.0,

            # Raw data for debugging
            'ball_pos': None,
            'keypoints_3d': None,
        }

        # Detect ball
        ball_mask, ball_center, ball_score = self.get_ball_mask(frame_rgb)

        if ball_center is not None:
            features['ball_x'] = ball_center[0] / w  # Normalize to [0, 1]
            features['ball_y'] = ball_center[1] / h
            features['ball_area'] = get_mask_area(ball_mask) / (w * h)  # Normalize
            features['ball_detected'] = True
            features['ball_pos'] = ball_center

            # Compute ball velocity
            if prev_ball_pos is not None:
                dx = (ball_center[0] - prev_ball_pos[0]) / w
                dy = (ball_center[1] - prev_ball_pos[1]) / h
                features['ball_velocity_x'] = dx
                features['ball_velocity_y'] = dy
                features['ball_speed'] = np.sqrt(dx**2 + dy**2)

                # Compute acceleration
                if prev_ball_vel is not None:
                    prev_speed = np.sqrt(prev_ball_vel[0]**2 + prev_ball_vel[1]**2)
                    features['ball_acceleration'] = features['ball_speed'] - prev_speed

        # Detect shooter
        shooter_mask, shooter_score = self.get_shooter_mask(frame_rgb)

        if shooter_mask is not None:
            features['shooter_detected'] = True

            # Compute overlap
            if ball_mask is not None:
                features['ball_shooter_overlap'] = masks_overlap_ratio(ball_mask, shooter_mask)

        # Get 3D pose
        keypoints_3d = self.get_3d_pose(frame_rgb, shooter_mask)

        if keypoints_3d is not None and len(keypoints_3d) >= 70:
            features['keypoints_3d'] = keypoints_3d.tolist()

            # Extract key joints
            left_shoulder = keypoints_3d[JOINT_INDICES['left_shoulder']]
            right_shoulder = keypoints_3d[JOINT_INDICES['right_shoulder']]
            left_elbow = keypoints_3d[JOINT_INDICES['left_elbow']]
            right_elbow = keypoints_3d[JOINT_INDICES['right_elbow']]
            left_wrist = keypoints_3d[JOINT_INDICES['left_wrist']]
            right_wrist = keypoints_3d[JOINT_INDICES['right_wrist']]
            pelvis = keypoints_3d[JOINT_INDICES['pelvis']]

            # Compute arm angles (shoulder-elbow-wrist)
            features['left_arm_angle'] = compute_angle(left_shoulder, left_elbow, left_wrist)
            features['right_arm_angle'] = compute_angle(right_shoulder, right_elbow, right_wrist)

            # Compute arm extension (wrist height relative to shoulder)
            shoulder_height = (left_shoulder[1] + right_shoulder[1]) / 2
            features['left_arm_extension'] = (shoulder_height - left_wrist[1]) / (shoulder_height - pelvis[1] + 1e-8)
            features['right_arm_extension'] = (shoulder_height - right_wrist[1]) / (shoulder_height - pelvis[1] + 1e-8)

            # Compute hand-ball distances
            if ball_center is not None:
                # Project 3D wrist to 2D (approximate - use x, y directly if in image coords)
                # Note: This assumes keypoints are in image coordinates or similar scale
                ball_2d = np.array([ball_center[0], ball_center[1]])

                # Scale factor for 3D to 2D projection (approximate)
                scale = w / 2  # Rough estimate

                left_wrist_2d = np.array([left_wrist[0] * scale + w/2, left_wrist[1] * scale + h/2])
                right_wrist_2d = np.array([right_wrist[0] * scale + w/2, right_wrist[1] * scale + h/2])

                features['dist_left_wrist'] = np.linalg.norm(ball_2d - left_wrist_2d) / w
                features['dist_right_wrist'] = np.linalg.norm(ball_2d - right_wrist_2d) / w
                features['min_hand_dist'] = min(features['dist_left_wrist'], features['dist_right_wrist'])

            # Compute wrist velocities
            if prev_keypoints is not None and len(prev_keypoints) >= 70:
                prev_left_wrist = prev_keypoints[JOINT_INDICES['left_wrist']]
                prev_right_wrist = prev_keypoints[JOINT_INDICES['right_wrist']]

                features['left_wrist_velocity'] = np.linalg.norm(left_wrist - prev_left_wrist)
                features['right_wrist_velocity'] = np.linalg.norm(right_wrist - prev_right_wrist)

        return features

    def extract_video_features(
        self,
        video_path: str,
        center_frame: int,
        window_size: int = 30,
    ) -> List[Dict]:
        """Extract features for a window of frames around the center frame."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Failed to open video: {video_path}")
            return []

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Calculate frame range
        start_frame = max(0, center_frame - window_size // 2)
        end_frame = min(total_frames - 1, center_frame + window_size // 2)

        features_sequence = []
        prev_ball_pos = None
        prev_ball_vel = None
        prev_keypoints = None

        for frame_idx in range(start_frame, end_frame + 1):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame_bgr = cap.read()

            if not ret:
                features_sequence.append(None)
                continue

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

            features = self.extract_frame_features(
                frame_rgb,
                prev_ball_pos=prev_ball_pos,
                prev_ball_vel=prev_ball_vel,
                prev_keypoints=prev_keypoints,
            )
            features['frame_idx'] = frame_idx
            features['relative_frame'] = frame_idx - center_frame

            # Update previous values
            if features['ball_pos'] is not None:
                prev_ball_vel = (features['ball_velocity_x'], features['ball_velocity_y'])
                prev_ball_pos = features['ball_pos']

            if features.get('keypoints_3d') is not None:
                prev_keypoints = np.array(features['keypoints_3d'])

            # Remove raw data before storing
            features_clean = {k: v for k, v in features.items() if k not in ['ball_pos', 'keypoints_3d']}
            features_sequence.append(features_clean)

        cap.release()
        return features_sequence


def process_labeled_videos(
    labels_path: str,
    videos_dir: str,
    output_path: str,
    window_size: int = 30,
    limit: Optional[int] = None,
):
    """Process all labeled videos and extract features."""
    # Load release frame labels
    with open(labels_path) as f:
        labels = json.load(f)

    if len(labels) == 0:
        print("No labels found! Please label some release frames first.")
        return

    print(f"Found {len(labels)} labeled videos")

    extractor = ReleaseFeatureExtractor()

    results = []
    video_ids = list(labels.keys())
    if limit:
        video_ids = video_ids[:limit]

    for video_id in tqdm(video_ids, desc="Extracting features"):
        label_info = labels[video_id]
        release_frame = label_info['release_frame']
        confidence = label_info.get('confidence', 'high')

        # Find video file
        prefix = video_id.split('_')[0]
        video_path = os.path.join(videos_dir, prefix, f"{video_id}.mp4")

        if not os.path.exists(video_path):
            print(f"Video not found: {video_path}")
            continue

        features_seq = extractor.extract_video_features(
            video_path,
            center_frame=release_frame,
            window_size=window_size,
        )

        if len(features_seq) == 0:
            print(f"No features extracted for {video_id}")
            continue

        results.append({
            'video_id': video_id,
            'release_frame': release_frame,
            'confidence': confidence,
            'window_start': features_seq[0]['frame_idx'] if features_seq[0] else None,
            'window_size': len(features_seq),
            'features': features_seq,
        })

    # Save results - convert numpy types to Python types for JSON serialization
    def convert_to_serializable(obj):
        if isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(v) for v in obj]
        return obj

    results = convert_to_serializable(results)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved features for {len(results)} videos to {output_path}")

    # Print statistics
    valid_frames = sum(
        sum(1 for f in r['features'] if f is not None and f.get('ball_detected'))
        for r in results
    )
    total_frames = sum(len(r['features']) for r in results)
    print(f"Ball detected in {valid_frames}/{total_frames} frames ({100*valid_frames/total_frames:.1f}%)")


def main():
    parser = argparse.ArgumentParser(description="Extract features for release frame detection")
    parser.add_argument('--labels', type=str, default='labels/release_frames.json',
                        help='Path to release frame labels')
    parser.add_argument('--videos', type=str, default='/workspace/videos',
                        help='Path to videos directory')
    parser.add_argument('--output', type=str, default='data/release_features/features.json',
                        help='Output path for extracted features')
    parser.add_argument('--window', type=int, default=30,
                        help='Window size (frames) around release')
    parser.add_argument('--limit', type=int, default=None,
                        help='Limit number of videos to process')

    args = parser.parse_args()

    process_labeled_videos(
        labels_path=args.labels,
        videos_dir=args.videos,
        output_path=args.output,
        window_size=args.window,
        limit=args.limit,
    )


if __name__ == '__main__':
    main()
