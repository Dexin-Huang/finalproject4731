"""
Automatically detects the release frame in basketball free throw videos.
Supports both side-view and broadcast-view (center, close up) camera angles.

Key differences from original manual candidates.json - might need to map:
- `release_frame` → whatever field name the pipeline expects
- `sequence_frame_indices` → the frames to extract for pose estimation
- `shooter.keypoints_2d` → 2D pose keypoints if needed

Import:
    pip install ultralytics opencv-python

Call:
    python detect_release.py --input <video_dir> --output <output_dir>

Outputs:
    - candidates.json: All detection results
    - perfect.json: High-confidence detections only
    - frames/: Extracted frames around release point (for eyeballing)
"""

import cv2
import numpy as np
from ultralytics import YOLO
import json
import os
import glob
import argparse
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


## CAMERA ANGLE DETECTION - standard full view vs. close up behind the basket view

def detect_camera_angle(video_path: str, pose_model: YOLO) -> tuple:
    """
    Detect camera angle: BROADCAST vs SIDE

    BROADCAST (behind basket) - tight framing:
    - Few people visible (5-9) due to zoom
    - Shooter is LARGE (30%+ of frame height) and CENTERED
    - People distributed on BOTH sides (rebounders waiting)

    SIDE (full view) - wide framing:
    - Many people visible (12+) or inconsistent/unbalanced distribution
    with no large centered person in the middle
    """
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    sample_frames = []
    for pct in [0.30, 0.35, 0.40, 0.45, 0.50]:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(total * pct))
        ret, frame = cap.read()
        if ret:
            sample_frames.append(frame)
    cap.release()

    if not sample_frames:
        return 'side', 0.5

    broadcast_scores = []

    for frame in sample_frames:
        results = pose_model(frame, verbose=False)

        if results[0].boxes is None:
            broadcast_scores.append(0)
            continue

        boxes = results[0].boxes.xywh.cpu().numpy()
        num_people = len(boxes)

        # Too many people = side view (wide shot)
        if num_people >= 12:
            broadcast_scores.append(0)
            continue

        # Too few people = can't determine
        if num_people < 4:
            broadcast_scores.append(0)
            continue

        # People count
        if 5 <= num_people <= 9:
            people_score = 1.0
        elif num_people == 4 or num_people == 10 or num_people == 11:
            people_score = 0.5
        else:
            people_score = 0

        # Large centered person
        center_person_score = 0
        for box in boxes:
            bx, by, bw, bh = box
            x_norm = bx / w
            height_ratio = bh / h

            if 0.30 <= x_norm <= 0.70:
                if height_ratio > 0.35:
                    center_person_score = 1.0
                elif height_ratio > 0.28:
                    center_person_score = max(center_person_score, 0.7)

        # Left/right balance
        left_count = sum(1 for box in boxes if box[0] < w * 0.5)
        right_count = num_people - left_count

        if left_count > 0 and right_count > 0:
            balance_ratio = min(left_count, right_count) / max(left_count, right_count)
            balance_score = balance_ratio
        else:
            balance_score = 0

        # Combined score
        frame_broadcast_score = (
                people_score * 0.3 +
                center_person_score * 0.5 +
                balance_score * 0.2
        )

        broadcast_scores.append(frame_broadcast_score)

    if not broadcast_scores:
        return 'side', 0.5

    avg_broadcast_score = sum(broadcast_scores) / len(broadcast_scores)

    if avg_broadcast_score > 0.6:
        return 'broadcast', avg_broadcast_score
    else:
        return 'side', 1.0 - avg_broadcast_score


## UTIL FUNCTIONS

def calculate_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """
    Calculate angle at point b formed by points a-b-c
    """
    a, b, c = np.array(a), np.array(b), np.array(c)
    rad = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(rad * 180.0 / np.pi)
    return 360 - angle if angle > 180 else angle


def calculate_isolation(person_idx: int, all_boxes: np.ndarray) -> tuple:
    """
    Calculate isolation distance for a person from others
    """
    box = all_boxes[person_idx]
    px, py = box[0], box[1]

    distances = []
    for i, other in enumerate(all_boxes):
        if i != person_idx:
            dist = np.sqrt((px - other[0]) ** 2 + (py - other[1]) ** 2)
            distances.append(dist)

    if not distances:
        return 0, 0

    return min(distances), np.mean(distances)


def is_valid_shooter_pose(kpts: np.ndarray, h: int, w: int) -> tuple:
    """
    Validate if keypoints represent a valid shooting pose - pretty manual
    """
    if kpts[5, 2] < 0.4 or kpts[6, 2] < 0.4:
        return False, 0, None, None, None, 0

    right_valid = kpts[6, 2] > 0.4 and kpts[8, 2] > 0.4 and kpts[10, 2] > 0.4
    left_valid = kpts[5, 2] > 0.4 and kpts[7, 2] > 0.4 and kpts[9, 2] > 0.4

    if not right_valid and not left_valid:
        return False, 0, None, None, None, 0

    if right_valid and left_valid:
        use_right = kpts[10, 1] < kpts[9, 1]
    else:
        use_right = right_valid

    if use_right:
        shoulder, elbow, wrist = kpts[6, :2], kpts[8, :2], kpts[10, :2]
        conf = min(kpts[6, 2], kpts[8, 2], kpts[10, 2])
    else:
        shoulder, elbow, wrist = kpts[5, :2], kpts[7, :2], kpts[9, :2]
        conf = min(kpts[5, 2], kpts[7, 2], kpts[9, 2])

    angle = calculate_angle(shoulder, elbow, wrist)
    wrist_height_ratio = wrist[1] / h

    if angle < 100 or angle > 145:
        return False, angle, wrist, elbow, shoulder, conf

    if wrist_height_ratio > 0.70:
        return False, angle, wrist, elbow, shoulder, conf

    return True, angle, wrist, elbow, shoulder, conf


def to_python(val):
    """
    Convert numpy types to Python native types for JSON serialization
    """
    if isinstance(val, (np.floating, np.float32, np.float64)):
        return float(val)
    if isinstance(val, (np.integer, np.int32, np.int64)):
        return int(val)
    if isinstance(val, np.ndarray):
        return val.tolist()
    if isinstance(val, bool):
        return val
    return val


## DETECTOR - SIDE VIEW

def process_side_view(video_path: str, pose_model: YOLO, ball_model: YOLO,
                      output_dir: str, save_frames: bool = True) -> tuple:
    """
    Detect free throw release frame for side view videos using isolation-based
    detection - shooter isolated from group of rebounders near basket.
    """
    filename = os.path.basename(video_path)
    video_name = os.path.splitext(filename)[0]

    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fps = cap.get(cv2.CAP_PROP_FPS)

    frames = []
    while True:
        ret, f = cap.read()
        if not ret:
            break
        frames.append(f)
    cap.release()

    if not frames:
        logger.error(f"No frames in {filename}")
        return None, None, 'error', {}

    logger.info(f"Processing {filename}: {len(frames)} frames, {w}x{h}, {fps:.1f}fps")

    is_hd = w >= 1000
    start_idx = int(total * 0.20)
    end_idx = int(total * 0.75)

    candidates = []

    for frame_idx in range(start_idx, end_idx):
        results = pose_model(frames[frame_idx], verbose=False)

        if results[0].boxes is None or results[0].keypoints is None:
            continue

        boxes = results[0].boxes.xywh.cpu().numpy()
        kpts_list = results[0].keypoints.data.cpu().numpy()

        if len(boxes) < 2:
            continue

        for i in range(len(boxes)):
            box = boxes[i]
            kpts = kpts_list[i]
            bx, by, bw, bh = box

            # Size check
            if bh < h * 0.20 or bh > h * 0.80:
                continue

            # Aspect ratio
            aspect = bh / bw if bw > 0 else 0
            if aspect < 1.5 or aspect > 4.5:
                continue

            # Not at edges
            x_norm = bx / w
            if x_norm < 0.12 or x_norm > 0.88:
                continue

            # Not at top
            if (by - bh / 2) < h * 0.08:
                continue

            # Shooting pose
            is_valid, angle, wrist, elbow, shoulder, pose_conf = is_valid_shooter_pose(kpts, h, w)

            if not is_valid:
                continue

            # Isolation
            min_dist, avg_dist = calculate_isolation(i, boxes)
            frame_diag = np.sqrt(w ** 2 + h ** 2)
            isolation_ratio = min_dist / frame_diag

            # Scoring
            if isolation_ratio > 0.35:
                iso_score = 1.0
            elif isolation_ratio > 0.25:
                iso_score = 0.85
            elif isolation_ratio > 0.18:
                iso_score = 0.7
            elif isolation_ratio > 0.12:
                iso_score = 0.5
            else:
                iso_score = 0.3

            if 105 <= angle <= 125:
                angle_score = 1.0
            elif 100 <= angle <= 135:
                angle_score = 0.7
            else:
                angle_score = 0.4

            wrist_height_score = 1.0 - (wrist[1] / h)
            people_score = 1.0 if len(boxes) <= 4 else 0.8 if len(boxes) <= 8 else 0.5

            total_score = (
                    iso_score * 0.50 +
                    angle_score * 0.20 +
                    wrist_height_score * 0.15 +
                    pose_conf * 0.10 +
                    people_score * 0.05
            )

            candidates.append({
                'frame': frame_idx,
                'person_idx': i,
                'score': total_score,
                'isolation': min_dist,
                'isolation_ratio': isolation_ratio,
                'iso_score': iso_score,
                'angle': angle,
                'angle_score': angle_score,
                'wrist': wrist,
                'elbow': elbow,
                'shoulder': shoulder,
                'pose_conf': pose_conf,
                'box': box,
                'kpts': kpts,
                'all_boxes': boxes,
                'num_people': len(boxes)
            })

    logger.info(f"Found {len(candidates)} candidates")

    if not candidates:
        logger.warning(f"No valid candidates in {filename}")
        return None, None, 'no_candidates', {}

    # Sort by score
    candidates.sort(key=lambda x: x['score'], reverse=True)
    best = candidates[0]

    # Classification
    score = best['score']
    iso_ratio = best['isolation_ratio']

    if iso_ratio >= 0.50:
        status = 'PERFECT'
        confidence = 'very_high'
    elif (is_hd and
          0.12 <= iso_ratio <= 0.15 and
          0.62 <= score <= 0.71):
        if best['num_people'] >= 26:
            status = 'PERFECT'
            confidence = 'high_hd'
        elif best['num_people'] >= 21:
            if iso_ratio <= 0.125:
                status = 'PERFECT'
                confidence = 'high_hd'
            elif score <= 0.66:
                status = 'PERFECT'
                confidence = 'high_hd'
            else:
                status = 'NEEDS_REVIEW'
                confidence = 'low'
        elif best['num_people'] == 17 and 0.65 <= score <= 0.66:
            status = 'PERFECT'
            confidence = 'high_hd'
        else:
            status = 'NEEDS_REVIEW'
            confidence = 'low'
    elif iso_ratio >= 0.30 and best['num_people'] <= 4:
        status = 'LIKELY_CORRECT'
        confidence = 'medium'
    else:
        status = 'NEEDS_REVIEW'
        confidence = 'low'

    metrics = {
        'score': score,
        'isolation_ratio': iso_ratio,
        'isolation_px': best['isolation'],
        'angle': best['angle'],
        'num_people': best['num_people'],
        'is_hd': is_hd,
        'status': status,
        'confidence': confidence
    }

    logger.info(f"Best: frame {best['frame']}, score {score:.3f}, status {status}")

    release_frame = best['frame']
    release_img = frames[release_frame]

    # Ball detection
    ball_pos = None
    ball_conf = 0.0
    wx, wy = int(best['wrist'][0]), int(best['wrist'][1])

    ball_results = ball_model(release_img, verbose=False, conf=0.08)
    if ball_results[0].boxes is not None:
        for box in ball_results[0].boxes.data.tolist():
            x1, y1, x2, y2, conf, cls = box
            if int(cls) != 32:
                continue
            bx, by = (x1 + x2) / 2, (y1 + y2) / 2
            dist = np.sqrt((bx - wx) ** 2 + (by - wy) ** 2)
            if dist < 120 and (ball_pos is None or conf > ball_conf):
                ball_pos = (bx, by)
                ball_conf = conf

    if ball_pos is None:
        ball_pos = (wx, wy - 25)
        ball_conf = 0.0

    # Build output
    candidate_data = {
        'video_file': filename,
        'camera_angle': 'side',
        'release_frame': int(release_frame),
        'fps': float(fps),
        'frame_width': int(w),
        'frame_height': int(h),
        'status': status,
        'confidence': confidence,
        'metrics': {k: to_python(v) for k, v in metrics.items()},
        'ball_location': {
            'x': float(ball_pos[0]),
            'y': float(ball_pos[1]),
            'confidence': float(ball_conf)
        },
        'shooter': {
            'wrist': {'x': float(best['wrist'][0]), 'y': float(best['wrist'][1])},
            'elbow': {'x': float(best['elbow'][0]), 'y': float(best['elbow'][1])},
            'shoulder': {'x': float(best['shoulder'][0]), 'y': float(best['shoulder'][1])},
            'arm_angle': float(best['angle']),
            'isolation_distance': float(best['isolation']),
            'isolation_ratio': float(iso_ratio),
            'keypoints_2d': [[float(x) for x in row] for row in best['kpts'][:, :2].tolist()],
            'keypoints_confidence': [float(x) for x in best['kpts'][:, 2].tolist()]
        },
        'sequence_frame_indices': [
            int(max(0, release_frame - 3)),
            int(max(0, release_frame - 2)),
            int(max(0, release_frame - 1)),
            int(release_frame),
            int(min(len(frames) - 1, release_frame + 1)),
            int(min(len(frames) - 1, release_frame + 2))
        ]
    }

    # Save frames
    if save_frames:
        frames_dir = os.path.join(output_dir, 'frames', video_name)
        os.makedirs(frames_dir, exist_ok=True)

        for idx in candidate_data['sequence_frame_indices']:
            cv2.imwrite(os.path.join(frames_dir, f'frame_{idx:04d}.jpg'), frames[idx])

    # Create visualization for output images for eyeballing
    vis = release_img.copy()

    for j, box in enumerate(best['all_boxes']):
        bx, by, bw, bh = box
        color = (0, 255, 0) if j == best['person_idx'] else (80, 80, 80)
        thickness = 3 if j == best['person_idx'] else 1
        cv2.rectangle(vis, (int(bx - bw / 2), int(by - bh / 2)), (int(bx + bw / 2), int(by + bh / 2)), color, thickness)

    kpts = best['kpts']
    skeleton = [(5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14),
                (14, 16)]
    for p1, p2 in skeleton:
        if kpts[p1, 2] > 0.3 and kpts[p2, 2] > 0.3:
            cv2.line(vis, (int(kpts[p1, 0]), int(kpts[p1, 1])), (int(kpts[p2, 0]), int(kpts[p2, 1])), (0, 255, 0), 2)

    wx, wy = int(best['wrist'][0]), int(best['wrist'][1])
    ex, ey = int(best['elbow'][0]), int(best['elbow'][1])
    sx, sy = int(best['shoulder'][0]), int(best['shoulder'][1])

    cv2.line(vis, (sx, sy), (ex, ey), (0, 165, 255), 3)
    cv2.line(vis, (ex, ey), (wx, wy), (0, 0, 255), 3)
    cv2.circle(vis, (wx, wy), 10, (0, 0, 255), -1)

    bx, by = int(ball_pos[0]), int(ball_pos[1])
    cv2.circle(vis, (bx, by), 12, (0, 255, 255), 3)

    if save_frames:
        cv2.imwrite(os.path.join(frames_dir, 'visualization.jpg'), vis)

    return candidate_data, vis, status, metrics


## DETECTOR - BROADCAST VIEW

def process_broadcast_view(video_path: str, pose_model: YOLO, ball_model: YOLO,
                           output_dir: str, save_frames: bool = True) -> tuple:
    """
    Detect free throw release frame for broadcast (behind basket) view - using
    center-based detection for shooter in the middle of the frame with other
    players (rebounders) on each side
    """
    filename = os.path.basename(video_path)
    video_name = os.path.splitext(filename)[0]

    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fps = cap.get(cv2.CAP_PROP_FPS)

    frames = []
    while True:
        ret, f = cap.read()
        if not ret:
            break
        frames.append(f)
    cap.release()

    if not frames:
        logger.error(f"No frames in {filename}")
        return None, None, 'error', {}

    logger.info(f"Processing {filename} (broadcast): {len(frames)} frames, {w}x{h}, {fps:.1f}fps")

    start_idx = int(total * 0.20)
    end_idx = int(total * 0.60)

    candidates = []

    for frame_idx in range(start_idx, end_idx):
        frame = frames[frame_idx]
        results = pose_model(frame, verbose=False)

        if results[0].boxes is None or results[0].keypoints is None:
            continue

        boxes = results[0].boxes.xywh.cpu().numpy()
        kpts_list = results[0].keypoints.data.cpu().numpy()

        if len(boxes) == 0:
            continue

        center_x = w / 2

        for i in range(len(boxes)):
            box = boxes[i]
            kpts = kpts_list[i]
            bx, by, bw, bh = box

            if bh < h * 0.25:
                continue

            center_dist = abs(bx - center_x)
            center_score = 1.0 - (center_dist / (w / 2))

            if center_score < 0.4:
                continue

            is_valid, angle, wrist, elbow, shoulder, pose_conf = is_valid_shooter_pose(kpts, h, w)

            if not is_valid:
                continue

            if wrist[1] > h * 0.55:
                continue

            if 100 <= angle <= 125:
                angle_score = 1.0
            elif 95 <= angle <= 135:
                angle_score = 0.7
            else:
                angle_score = 0.4

            wrist_height_score = 1.0 - (wrist[1] / h)

            total_score = (
                    center_score * 0.50 +
                    angle_score * 0.25 +
                    wrist_height_score * 0.15 +
                    pose_conf * 0.10
            )

            candidates.append({
                'frame': frame_idx,
                'person_idx': i,
                'score': total_score,
                'center_score': center_score,
                'angle': angle,
                'angle_score': angle_score,
                'wrist': wrist,
                'elbow': elbow,
                'shoulder': shoulder,
                'pose_conf': pose_conf,
                'box': box,
                'kpts': kpts,
                'all_boxes': boxes,
                'num_people': len(boxes)
            })

    logger.info(f"Found {len(candidates)} candidates")

    if not candidates:
        logger.warning(f"No valid candidates in {filename}")
        return None, None, 'no_candidates', {}

    candidates.sort(key=lambda x: x['score'], reverse=True)

    # Check for ball near wrist
    best = candidates[0]
    for c in candidates[:10]:
        frame = frames[c['frame']]
        wx, wy = c['wrist'][0], c['wrist'][1]

        ball_results = ball_model(frame, verbose=False, conf=0.08)
        if ball_results[0].boxes is not None:
            for box in ball_results[0].boxes.data.tolist():
                x1, y1, x2, y2, conf, cls = box
                if int(cls) != 32:
                    continue
                bx, by = (x1 + x2) / 2, (y1 + y2) / 2
                dist = np.sqrt((bx - wx) ** 2 + (by - wy) ** 2)

                if dist < 80:
                    c['ball_nearby'] = True
                    c['ball_dist'] = dist
                    c['ball_pos'] = (bx, by)
                    c['ball_conf'] = conf
                    c['score'] += 0.15
                    break

    candidates.sort(key=lambda x: x['score'], reverse=True)
    best = candidates[0]

    score = best['score']
    release_frame = best['frame']

    # Classification
    if best.get('ball_nearby') and best['center_score'] > 0.6:
        status = 'PERFECT'
        confidence = 'broadcast_confirmed'
    elif best['center_score'] > 0.7 and score > 0.7:
        status = 'PERFECT'
        confidence = 'broadcast_high'
    elif best['center_score'] > 0.5 and score > 0.6:
        status = 'LIKELY_CORRECT'
        confidence = 'broadcast_medium'
    else:
        status = 'NEEDS_REVIEW'
        confidence = 'low'

    metrics = {
        'score': score,
        'center_score': best['center_score'],
        'angle': best['angle'],
        'num_people': best['num_people'],
        'ball_nearby': best.get('ball_nearby', False),
        'status': status,
        'confidence': confidence
    }

    logger.info(f"Best: frame {best['frame']}, score {score:.3f}, status {status}")

    release_img = frames[release_frame]

    if best.get('ball_pos'):
        ball_pos = best['ball_pos']
        ball_conf = best.get('ball_conf', 0)
    else:
        wx, wy = int(best['wrist'][0]), int(best['wrist'][1])
        ball_pos = (wx, wy - 25)
        ball_conf = 0.0

    # Build output
    candidate_data = {
        'video_file': filename,
        'camera_angle': 'broadcast',
        'release_frame': int(release_frame),
        'fps': float(fps),
        'frame_width': int(w),
        'frame_height': int(h),
        'status': status,
        'confidence': confidence,
        'metrics': {k: to_python(v) for k, v in metrics.items()},
        'ball_location': {
            'x': float(ball_pos[0]),
            'y': float(ball_pos[1]),
            'confidence': float(ball_conf)
        },
        'shooter': {
            'wrist': {'x': float(best['wrist'][0]), 'y': float(best['wrist'][1])},
            'elbow': {'x': float(best['elbow'][0]), 'y': float(best['elbow'][1])},
            'shoulder': {'x': float(best['shoulder'][0]), 'y': float(best['shoulder'][1])},
            'arm_angle': float(best['angle']),
            'center_score': float(best['center_score'])
        },
        'sequence_frame_indices': [
            int(max(0, release_frame - 3)),
            int(max(0, release_frame - 2)),
            int(max(0, release_frame - 1)),
            int(release_frame),
            int(min(len(frames) - 1, release_frame + 1)),
            int(min(len(frames) - 1, release_frame + 2))
        ]
    }

    # Save frames
    if save_frames:
        frames_dir = os.path.join(output_dir, 'frames', video_name)
        os.makedirs(frames_dir, exist_ok=True)

        for idx in candidate_data['sequence_frame_indices']:
            cv2.imwrite(os.path.join(frames_dir, f'frame_{idx:04d}.jpg'), frames[idx])

    # Create visualization for output images for eyeballing
    vis = release_img.copy()

    for j, box in enumerate(best['all_boxes']):
        bx, by, bw, bh = box
        color = (0, 255, 0) if j == best['person_idx'] else (80, 80, 80)
        thickness = 3 if j == best['person_idx'] else 1
        cv2.rectangle(vis, (int(bx - bw / 2), int(by - bh / 2)), (int(bx + bw / 2), int(by + bh / 2)), color, thickness)

    kpts = best['kpts']
    skeleton = [(5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14),
                (14, 16)]
    for p1, p2 in skeleton:
        if kpts[p1, 2] > 0.3 and kpts[p2, 2] > 0.3:
            cv2.line(vis, (int(kpts[p1, 0]), int(kpts[p1, 1])), (int(kpts[p2, 0]), int(kpts[p2, 1])), (0, 255, 0), 2)

    wx, wy = int(best['wrist'][0]), int(best['wrist'][1])
    ex, ey = int(best['elbow'][0]), int(best['elbow'][1])
    sx, sy = int(best['shoulder'][0]), int(best['shoulder'][1])

    cv2.line(vis, (sx, sy), (ex, ey), (0, 165, 255), 3)
    cv2.line(vis, (ex, ey), (wx, wy), (0, 0, 255), 3)
    cv2.circle(vis, (wx, wy), 10, (0, 0, 255), -1)

    bx, by = int(ball_pos[0]), int(ball_pos[1])
    ball_color = (0, 255, 255) if best.get('ball_nearby') else (128, 128, 128)
    cv2.circle(vis, (bx, by), 15, ball_color, 3)

    if save_frames:
        cv2.imwrite(os.path.join(frames_dir, 'visualization.jpg'), vis)

    return candidate_data, vis, status, metrics


# MAIN PROCESSER

def process_video(video_path: str, pose_model: YOLO, ball_model: YOLO,
                  output_dir: str, save_frames: bool = True) -> tuple:
    """
    Process a video to detect free throw release frame - find camera
    angle and directs the video to appropriate detector function
    """
    filename = os.path.basename(video_path)
    logger.info(f"Processing: {filename}")

    # Detect camera angle
    camera_angle, angle_conf = detect_camera_angle(video_path, pose_model)
    logger.info(f"Camera angle: {camera_angle.upper()} (conf: {angle_conf:.2f})")

    # Route to appropriate detector
    if camera_angle == 'broadcast':
        result, vis, status, metrics = process_broadcast_view(
            video_path, pose_model, ball_model, output_dir, save_frames
        )
    else:
        result, vis, status, metrics = process_side_view(
            video_path, pose_model, ball_model, output_dir, save_frames
        )

    return result, camera_angle


def run_detection(input_dir: str, output_dir: str, save_frames: bool = True) -> dict:
    """
    run release frame detection on all videos in input directory
    after videos are processed and divided into different views
    """
    logger.info("Loading models...")
    pose_model = YOLO('yolov8m-pose.pt')
    ball_model = YOLO('yolov8n.pt')
    logger.info("Models loaded")

    os.makedirs(output_dir, exist_ok=True)

    # Find videos
    vids = (glob.glob(os.path.join(input_dir, "*.mp4")) +
            glob.glob(os.path.join(input_dir, "*.avi")) +
            glob.glob(os.path.join(input_dir, "*.mov")))

    if not vids:
        logger.warning(f"No videos found in {input_dir}")
        return {}

    logger.info(f"Found {len(vids)} videos")

    results = {
        'perfect': [],
        'likely_correct': [],
        'needs_review': [],
        'all': []
    }

    side_count = 0
    broadcast_count = 0

    for vp in sorted(vids):
        result, camera_angle = process_video(
            vp, pose_model, ball_model, output_dir, save_frames
        )

        if result:
            results['all'].append(result)

            if result['status'] == 'PERFECT':
                results['perfect'].append(result)
            elif result['status'] == 'LIKELY_CORRECT':
                results['likely_correct'].append(result)
            else:
                results['needs_review'].append(result)

            if camera_angle == 'side':
                side_count += 1
            else:
                broadcast_count += 1

    # Save results
    with open(os.path.join(output_dir, 'candidates.json'), 'w') as f:
        json.dump(results['all'], f, indent=2)

    with open(os.path.join(output_dir, 'perfect.json'), 'w') as f:
        json.dump(results['perfect'], f, indent=2)

    with open(os.path.join(output_dir, 'likely_correct.json'), 'w') as f:
        json.dump(results['likely_correct'], f, indent=2)

    with open(os.path.join(output_dir, 'needs_review.json'), 'w') as f:
        json.dump(results['needs_review'], f, indent=2)

    # Summary
    summary = {
        'total_videos': len(vids),
        'processed': len(results['all']),
        'side_view': side_count,
        'broadcast_view': broadcast_count,
        'perfect': len(results['perfect']),
        'likely_correct': len(results['likely_correct']),
        'needs_review': len(results['needs_review'])
    }

    logger.info("=" * 50)
    logger.info("RESULTS SUMMARY")
    logger.info("=" * 50)
    logger.info(f"Side view: {side_count}")
    logger.info(f"Broadcast view: {broadcast_count}")
    logger.info(f"PERFECT: {len(results['perfect'])}")
    logger.info(f"LIKELY_CORRECT: {len(results['likely_correct'])}")
    logger.info(f"NEEDS_REVIEW: {len(results['needs_review'])}")
    logger.info(f"Total: {len(results['all'])}/{len(vids)}")
    logger.info("=" * 50)

    return summary


def main():
    parser = argparse.ArgumentParser(
        description='Detect free throw release frames in basketball videos'
    )
    parser.add_argument(
        '--input', '-i',
        type=str,
        default='videos',
        help='Input directory containing video files'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='output',
        help='Output directory for results'
    )
    parser.add_argument(
        '--no-frames',
        action='store_true',
        help='Do not save extracted frames'
    )

    args = parser.parse_args()

    run_detection(
        input_dir=args.input,
        output_dir=args.output,
        save_frames=not args.no_frames
    )


if __name__ == "__main__":
    main()
