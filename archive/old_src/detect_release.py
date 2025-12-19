"""
Unified Release Frame Detector

Provides a single interface for detecting the release frame in basketball
free throw videos, combining learned and heuristic approaches.

Usage:
    from src.detect_release import ReleaseDetector

    detector = ReleaseDetector(model_path="models/release_detector.pt")
    result = detector.detect(video_path)
    print(f"Release frame: {result.frame}, Confidence: {result.confidence}")
"""

import os
import json
import numpy as np
import cv2
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Dict, Tuple

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# Import feature extractor and model
try:
    from extract_release_features import ReleaseFeatureExtractor, JOINT_INDICES
    HAS_EXTRACTOR = True
except ImportError:
    HAS_EXTRACTOR = False

try:
    from models.release_detector import get_model, FEATURE_NAMES
    HAS_MODEL = True
except ImportError:
    HAS_MODEL = False


@dataclass
class ReleaseResult:
    """Result of release frame detection."""
    frame: int
    confidence: float
    method: str  # 'learned', 'heuristic', 'hybrid'
    needs_review: bool = False
    features: Optional[Dict] = None


class HeuristicReleaseDetector:
    """
    Heuristic-based release detection using ball-shooter mask separation.

    Fallback when learned model is not available.
    """

    def __init__(self):
        self.extractor = None
        if HAS_EXTRACTOR:
            try:
                self.extractor = ReleaseFeatureExtractor()
            except Exception as e:
                print(f"Could not initialize feature extractor: {e}")

    def detect(
        self,
        video_path: str,
        estimated_frame: Optional[int] = None,
        window_size: int = 30,
    ) -> ReleaseResult:
        """
        Detect release frame using heuristic approach.

        Looks for:
        1. Ball-hand distance increasing
        2. Ball velocity increasing
        3. Ball-shooter mask separation
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return ReleaseResult(
                frame=0, confidence=0.0, method='heuristic',
                needs_review=True
            )

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30

        # If no estimated frame, start at 30% of video
        if estimated_frame is None:
            estimated_frame = int(total_frames * 0.3)

        # Define search window
        start_frame = max(0, estimated_frame - window_size // 2)
        end_frame = min(total_frames - 1, estimated_frame + window_size // 2)

        # Track ball positions and distances
        ball_positions = []
        distances = []
        overlaps = []

        for frame_idx in range(start_frame, end_frame + 1):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame_bgr = cap.read()

            if not ret:
                ball_positions.append(None)
                distances.append(None)
                overlaps.append(1.0)
                continue

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

            if self.extractor is not None:
                features = self.extractor.extract_frame_features(frame_rgb)
                ball_pos = features.get('ball_pos')
                min_dist = features.get('min_hand_dist')
                overlap = features.get('ball_shooter_overlap', 1.0)

                ball_positions.append(ball_pos)
                distances.append(min_dist)
                overlaps.append(overlap)
            else:
                ball_positions.append(None)
                distances.append(None)
                overlaps.append(1.0)

        cap.release()

        # Find release frame using heuristics
        release_frame, confidence = self._find_release(
            ball_positions, distances, overlaps,
            start_frame, end_frame
        )

        return ReleaseResult(
            frame=release_frame,
            confidence=confidence,
            method='heuristic',
            needs_review=confidence < 0.5,
        )

    def _find_release(
        self,
        ball_positions: List,
        distances: List,
        overlaps: List,
        start_frame: int,
        end_frame: int,
    ) -> Tuple[int, float]:
        """Find release frame from tracking data."""
        n = len(distances)

        # Method 1: Distance increasing
        for i in range(1, n - 1):
            if distances[i] is None or distances[i-1] is None or distances[i+1] is None:
                continue

            if (distances[i] > distances[i-1] + 0.02 and
                distances[i+1] > distances[i] + 0.01 and
                distances[i] > 0.05):
                return start_frame + i, 0.8

        # Method 2: Overlap transition
        for i in range(1, n):
            if overlaps[i-1] > 0.1 and overlaps[i] < 0.1:
                return start_frame + i, 0.7

        # Method 3: Ball velocity spike
        velocities = []
        for i in range(1, n):
            if ball_positions[i] and ball_positions[i-1]:
                dx = ball_positions[i][0] - ball_positions[i-1][0]
                dy = ball_positions[i][1] - ball_positions[i-1][1]
                vel = np.sqrt(dx**2 + dy**2)
                velocities.append((i, vel))

        if velocities:
            # Find frame with max velocity increase
            for i in range(1, len(velocities)):
                if velocities[i][1] > velocities[i-1][1] * 1.5 and velocities[i][1] > 10:
                    return start_frame + velocities[i][0], 0.6

        # Method 4: Minimum distance + 1
        valid_distances = [(i, d) for i, d in enumerate(distances) if d is not None]
        if valid_distances:
            min_idx = min(valid_distances, key=lambda x: x[1])[0]
            if min_idx < n - 1:
                return start_frame + min_idx + 1, 0.5

        # Fallback: return estimated frame
        return (start_frame + end_frame) // 2, 0.3


class LearnedReleaseDetector:
    """
    Learned release detector using trained neural network.
    """

    def __init__(self, model_path: str, device: str = 'cuda'):
        if not HAS_TORCH or not HAS_MODEL:
            raise RuntimeError("PyTorch and model module required for learned detector")

        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        # Load model
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model_type = checkpoint.get('model_type', 'cnn')
        self.num_features = checkpoint.get('num_features', 18)
        self.seq_len = checkpoint.get('seq_len', 30)
        self.feature_names = checkpoint.get('feature_names', FEATURE_NAMES)

        self.model = get_model(self.model_type, num_features=self.num_features)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()

        # Feature extractor
        self.extractor = None
        if HAS_EXTRACTOR:
            try:
                self.extractor = ReleaseFeatureExtractor()
            except Exception as e:
                print(f"Could not initialize feature extractor: {e}")

    def detect(
        self,
        video_path: str,
        estimated_frame: Optional[int] = None,
    ) -> ReleaseResult:
        """Detect release frame using learned model."""
        if self.extractor is None:
            return ReleaseResult(
                frame=estimated_frame or 0,
                confidence=0.0,
                method='learned',
                needs_review=True,
            )

        # Extract features
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        if estimated_frame is None:
            estimated_frame = int(total_frames * 0.3)

        features_seq = self.extractor.extract_video_features(
            video_path,
            center_frame=estimated_frame,
            window_size=self.seq_len,
        )

        if len(features_seq) < self.seq_len:
            return ReleaseResult(
                frame=estimated_frame,
                confidence=0.0,
                method='learned',
                needs_review=True,
            )

        # Prepare input
        X = self._prepare_features(features_seq)
        X = torch.tensor(X, dtype=torch.float32).unsqueeze(0).to(self.device)

        # Predict
        with torch.no_grad():
            frame_idx, confidence = self.model.predict(X)
            frame_idx = frame_idx.item()
            confidence = confidence.item()

        # Convert relative index to absolute frame
        window_start = features_seq[0].get('frame_idx', 0) if features_seq[0] else 0
        release_frame = window_start + frame_idx

        return ReleaseResult(
            frame=release_frame,
            confidence=confidence,
            method='learned',
            needs_review=confidence < 0.5,
        )

    def _prepare_features(self, features_seq: List[Dict]) -> np.ndarray:
        """Prepare feature matrix from sequence."""
        X = []
        for frame_data in features_seq[:self.seq_len]:
            if frame_data is None:
                row = np.zeros(len(self.feature_names))
            else:
                row = []
                for name in self.feature_names:
                    value = frame_data.get(name)
                    if value is None:
                        value = 0.0
                    elif isinstance(value, bool):
                        value = float(value)
                    row.append(float(value))
                row = np.array(row)
            X.append(row)

        # Pad if needed
        while len(X) < self.seq_len:
            X.append(np.zeros(len(self.feature_names)))

        X = np.stack(X)

        # Normalize
        X = np.nan_to_num(X, nan=0.0)
        mean = X.mean(axis=0, keepdims=True)
        std = X.std(axis=0, keepdims=True) + 1e-8
        X = (X - mean) / std

        return X


class ReleaseDetector:
    """
    Unified release detector that combines learned and heuristic approaches.

    Uses learned model when available and confident, falls back to heuristics.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = 'cuda',
        confidence_threshold: float = 0.5,
    ):
        self.confidence_threshold = confidence_threshold
        self.learned_detector = None
        self.heuristic_detector = HeuristicReleaseDetector()

        # Try to load learned model
        if model_path and os.path.exists(model_path):
            try:
                self.learned_detector = LearnedReleaseDetector(model_path, device)
                print(f"Loaded learned detector from {model_path}")
            except Exception as e:
                print(f"Could not load learned detector: {e}")
                print("Falling back to heuristic detector")

    def detect(
        self,
        video_path: str,
        estimated_frame: Optional[int] = None,
        use_hybrid: bool = True,
    ) -> ReleaseResult:
        """
        Detect release frame.

        Args:
            video_path: Path to video file
            estimated_frame: Optional estimated release frame to search around
            use_hybrid: Whether to combine learned and heuristic approaches

        Returns:
            ReleaseResult with frame index, confidence, and method used
        """
        learned_result = None
        heuristic_result = None

        # Try learned detector first
        if self.learned_detector is not None:
            try:
                learned_result = self.learned_detector.detect(video_path, estimated_frame)
            except Exception as e:
                print(f"Learned detector failed: {e}")

        # Run heuristic detector
        heuristic_result = self.heuristic_detector.detect(
            video_path, estimated_frame
        )

        # Decide which result to return
        if learned_result is None:
            return heuristic_result

        if not use_hybrid:
            return learned_result

        # Hybrid approach: use learned if confident, otherwise combine
        if learned_result.confidence >= self.confidence_threshold:
            return learned_result

        # If methods agree (within 3 frames), use learned
        if abs(learned_result.frame - heuristic_result.frame) <= 3:
            return ReleaseResult(
                frame=learned_result.frame,
                confidence=max(learned_result.confidence, heuristic_result.confidence),
                method='hybrid',
                needs_review=False,
            )

        # Methods disagree - flag for review
        return ReleaseResult(
            frame=learned_result.frame,
            confidence=min(learned_result.confidence, 0.5),
            method='hybrid',
            needs_review=True,
        )

    def detect_batch(
        self,
        video_paths: List[str],
        estimated_frames: Optional[List[int]] = None,
        progress: bool = True,
    ) -> List[ReleaseResult]:
        """Detect release frames for multiple videos."""
        results = []

        if estimated_frames is None:
            estimated_frames = [None] * len(video_paths)

        iterator = zip(video_paths, estimated_frames)
        if progress:
            from tqdm import tqdm
            iterator = tqdm(list(iterator), desc="Detecting release frames")

        for video_path, est_frame in iterator:
            result = self.detect(video_path, est_frame)
            results.append(result)

        return results


def main():
    """Command-line interface for release detection."""
    import argparse

    parser = argparse.ArgumentParser(description="Detect release frame in video")
    parser.add_argument('video', type=str, help='Path to video file')
    parser.add_argument('--model', type=str, default='models/release_detector.pt',
                        help='Path to trained model')
    parser.add_argument('--estimated', type=int, default=None,
                        help='Estimated release frame')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda or cpu)')

    args = parser.parse_args()

    detector = ReleaseDetector(
        model_path=args.model if os.path.exists(args.model) else None,
        device=args.device,
    )

    result = detector.detect(args.video, args.estimated)

    print(f"\nRelease Detection Result:")
    print(f"  Frame: {result.frame}")
    print(f"  Confidence: {result.confidence:.3f}")
    print(f"  Method: {result.method}")
    print(f"  Needs Review: {result.needs_review}")


if __name__ == '__main__':
    main()
