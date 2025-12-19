"""
Release Frame Detection using SAM3 Mask Overlap

Approach:
1. SAM3 segments "basketball" and "person shooting basketball"
2. Track mask overlap over frames
3. Release = when ball mask separates from shooter mask

More robust than keypoint detection on broadcast footage.
"""
import os
import sys
import json
import argparse
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, "/workspace/sam3")

try:
    import torch
    from PIL import Image
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
    HAS_SAM3 = True
except ImportError:
    HAS_SAM3 = False
    print("SAM3 not available - run on GPU server")


class SAM3ReleaseDetector:
    """Detect release using SAM3 mask overlap."""

    def __init__(self):
        if HAS_SAM3:
            print("Loading SAM3...")
            self.model = build_sam3_image_model()
            self.processor = Sam3Processor(self.model)
            print("SAM3 loaded.")
        else:
            self.processor = None

    def get_mask(self, state, prompt):
        """Get binary mask for a text prompt."""
        output = self.processor.set_text_prompt(state=state, prompt=prompt)
        masks = output["masks"]
        scores = output["scores"]

        if len(masks) > 0 and scores[0] > 0.2:
            mask = masks[0]
            if torch.is_tensor(mask):
                mask = mask.cpu().numpy()
            mask = mask.squeeze()
            if mask.ndim > 2:
                mask = mask[0]
            return (mask > 0.5).astype(np.uint8), float(scores[0])

        return None, 0.0

    def get_mask_center(self, mask):
        """Get centroid of mask."""
        if mask is None:
            return None
        M = cv2.moments(mask)
        if M["m00"] > 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            return (cx, cy)
        return None

    def mask_distance(self, ball_mask, shooter_mask):
        """
        Compute distance between ball mask and shooter mask.
        Returns minimum distance between mask edges.
        """
        if ball_mask is None or shooter_mask is None:
            return None

        # Get ball centroid
        ball_center = self.get_mask_center(ball_mask)
        if ball_center is None:
            return None

        # Find closest point on shooter mask to ball center
        shooter_points = np.argwhere(shooter_mask > 0)  # (y, x) format
        if len(shooter_points) == 0:
            return None

        # Compute distance from ball center to all shooter points
        ball_pt = np.array([ball_center[1], ball_center[0]])  # (y, x)
        distances = np.linalg.norm(shooter_points - ball_pt, axis=1)
        min_distance = distances.min()

        return float(min_distance)

    def masks_overlap(self, mask1, mask2, threshold=0.1):
        """Check if two masks overlap significantly."""
        if mask1 is None or mask2 is None:
            return False, 0.0

        if mask1.shape != mask2.shape:
            return False, 0.0

        intersection = np.logical_and(mask1, mask2).sum()
        ball_area = mask1.sum()

        if ball_area == 0:
            return False, 0.0

        overlap_ratio = intersection / ball_area
        return overlap_ratio > threshold, float(overlap_ratio)

    def process_frame(self, frame):
        """Process a single frame and return ball/shooter info."""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_frame = Image.fromarray(frame_rgb)
        state = self.processor.set_image(pil_frame)

        # Detect ball
        ball_mask, ball_score = self.get_mask(state, "basketball")
        ball_center = self.get_mask_center(ball_mask)

        # Detect person holding ball - this tracks when ball is in hands
        shooter_mask = None
        shooter_score = 0
        for prompt in ["person holding ball", "person holding basketball", "hands holding ball"]:
            mask, score = self.get_mask(state, prompt)
            if score > shooter_score:
                shooter_mask = mask
                shooter_score = score

        # Compute distance between ball and shooter masks
        distance = self.mask_distance(ball_mask, shooter_mask)

        # Check overlap
        overlap, overlap_ratio = self.masks_overlap(ball_mask, shooter_mask)

        return {
            'ball_mask': ball_mask,
            'ball_center': ball_center,
            'ball_score': ball_score,
            'shooter_mask': shooter_mask,
            'shooter_score': shooter_score,
            'distance': distance,  # Key metric!
            'overlap': overlap,
            'overlap_ratio': overlap_ratio
        }

    def process_video(self, video_path, sample_rate=2):
        """Process video and detect release frame."""
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return None

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        tracking_data = []
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % sample_rate == 0:
                result = self.process_frame(frame)
                result['frame'] = frame_idx
                tracking_data.append(result)

            frame_idx += 1

        cap.release()

        # Find release frame
        release_frame, confidence = self.find_release_frame(tracking_data)

        return {
            'release_frame': release_frame,
            'confidence': confidence,
            'fps': fps,
            'total_frames': total_frames,
            'tracking_samples': len(tracking_data),
            'ball_detections': sum(1 for t in tracking_data if t['ball_center']),
            'shooter_detections': sum(1 for t in tracking_data if t['shooter_mask'] is not None)
        }

    def find_release_frame(self, tracking_data):
        """Find release = when distance starts increasing significantly."""
        # Get frames with valid distance measurements
        valid_data = [t for t in tracking_data if t['distance'] is not None]

        if len(valid_data) < 3:
            # Fallback
            if tracking_data:
                return int(tracking_data[int(len(tracking_data) * 0.3)]['frame']), 0.3
            return None, 0.0

        # Method 1: Find where distance increases significantly
        # Release = ball starts moving away from shooter
        distances = [t['distance'] for t in valid_data]
        frames = [t['frame'] for t in valid_data]

        for i in range(1, len(distances) - 1):
            prev_dist = distances[i-1]
            curr_dist = distances[i]
            next_dist = distances[i+1]

            # Distance increasing consistently (ball moving away)
            # And distance is now > some threshold (ball has left hand area)
            if (curr_dist > prev_dist + 10 and
                next_dist > curr_dist + 5 and
                curr_dist > 30):  # Ball is away from shooter
                return frames[i], 0.9

        # Method 2: Find transition from overlapping to not
        for i in range(1, len(valid_data)):
            prev = valid_data[i-1]
            curr = valid_data[i]

            if prev['overlap'] and not curr['overlap']:
                return curr['frame'], 0.8

        # Method 3: Find minimum distance point (ball closest to hand = about to release)
        min_dist_idx = np.argmin(distances)
        # Release is right after minimum distance
        if min_dist_idx < len(valid_data) - 1:
            return frames[min_dist_idx + 1], 0.6

        # Fallback
        return int(tracking_data[int(len(tracking_data) * 0.3)]['frame']), 0.3

    def visualize_detection(self, video_path, release_frame, output_path):
        """Create visualization with masks."""
        cap = cv2.VideoCapture(str(video_path))

        frames_to_show = [
            release_frame - 10,
            release_frame - 5,
            release_frame,
            release_frame + 5,
            release_frame + 10
        ]

        images = []
        for target in frames_to_show:
            target = max(0, target)
            cap.set(cv2.CAP_PROP_POS_FRAMES, target)
            ret, frame = cap.read()

            if ret:
                result = self.process_frame(frame)

                # Overlay masks
                overlay = frame.copy()

                if result['shooter_mask'] is not None:
                    # Blue tint for shooter
                    shooter_overlay = np.zeros_like(frame)
                    shooter_overlay[:, :, 0] = 255  # Blue channel
                    mask_3ch = np.stack([result['shooter_mask']] * 3, axis=-1)
                    overlay = np.where(mask_3ch,
                                      cv2.addWeighted(overlay, 0.7, shooter_overlay, 0.3, 0),
                                      overlay)

                if result['ball_mask'] is not None:
                    # Orange tint for ball
                    ball_overlay = np.zeros_like(frame)
                    ball_overlay[:, :, 2] = 255  # Red
                    ball_overlay[:, :, 1] = 165  # Green (orange)
                    mask_3ch = np.stack([result['ball_mask']] * 3, axis=-1)
                    overlay = np.where(mask_3ch,
                                      cv2.addWeighted(overlay, 0.5, ball_overlay, 0.5, 0),
                                      overlay)

                # Draw ball center
                if result['ball_center']:
                    cv2.circle(overlay, result['ball_center'], 10, (0, 255, 0), -1)

                # Labels
                label = "RELEASE" if target == release_frame else f"f{target}"
                color = (0, 0, 255) if target == release_frame else (0, 255, 0)
                cv2.putText(overlay, label, (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                # Show distance
                dist_str = f"d={result['distance']:.0f}" if result['distance'] else "d=N/A"
                cv2.putText(overlay, dist_str, (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                status = "ON" if result['overlap'] else "OFF"
                cv2.putText(overlay, f"Ball {status} shooter", (10, 85),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                images.append(cv2.resize(overlay, (320, 240)))

        cap.release()

        if len(images) >= 5:
            grid = np.hstack(images)
            cv2.imwrite(str(output_path), grid)


def process_batch(videos_dir, output_path, limit=None, viz_dir=None):
    """Process batch of videos."""
    detector = SAM3ReleaseDetector()

    video_files = list(Path(videos_dir).glob("*.mp4"))
    if limit:
        video_files = video_files[:limit]

    print(f"Processing {len(video_files)} videos...")

    results = []
    for video_path in tqdm(video_files):
        result = detector.process_video(video_path)
        if result:
            result['video_id'] = video_path.stem
            results.append(result)

            if viz_dir:
                os.makedirs(viz_dir, exist_ok=True)
                viz_path = Path(viz_dir) / f"{video_path.stem}_release.jpg"
                detector.visualize_detection(video_path, result['release_frame'], viz_path)

    os.makedirs(Path(output_path).parent, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    high_conf = sum(1 for r in results if r.get('confidence', 0) > 0.5)
    print(f"\nResults saved to {output_path}")
    print(f"High-confidence: {high_conf}/{len(results)}")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--videos_dir', type=str, required=True)
    parser.add_argument('--output', type=str, default='sam3_release_labels.json')
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--viz_dir', type=str, default=None)
    parser.add_argument('--sample_rate', type=int, default=2, help='Process every Nth frame')
    args = parser.parse_args()

    if not HAS_SAM3:
        print("ERROR: Run on GPU server with SAM3")
        sys.exit(1)

    process_batch(args.videos_dir, args.output, args.limit, args.viz_dir)


if __name__ == '__main__':
    main()
