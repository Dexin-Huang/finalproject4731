"""
Extract 4-frame sequences: t-2, t-1, t (apex), t+1
For each frame: SAM3 masks + SAM3D Body 3D pose
"""
import torch
import numpy as np
import cv2
from pathlib import Path
import json
from PIL import Image
from tqdm import tqdm
import sys
import argparse

# TODO: Paths - adjust these for actual setup
BPE_PATH = "/workspace/sam3/sam3/assets/bpe_simple_vocab_16e6.txt.gz"
SAM3D_CHECKPOINT = "/workspace/sam-3d-body/checkpoints/sam-3d-body-dinov3/model.ckpt"
MHR_PATH = "/workspace/sam-3d-body/checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt"

sys.path.insert(0, "/workspace/sam-3d-body")

def to_list(obj):
    """Convert numpy arrays and tensors to lists."""
    if torch.is_tensor(obj):
        return obj.cpu().numpy().tolist()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

def load_models(device="cuda"):
    """
    Load SAM3 and SAM3D Body models
    """
    print("Loading SAM3...")
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
    sam3_model = build_sam3_image_model(device=device, bpe_path=BPE_PATH)
    sam3_processor = Sam3Processor(sam3_model)
    print("SAM3 loaded!")

    print("Loading SAM3D Body...")
    from sam_3d_body import load_sam_3d_body, SAM3DBodyEstimator
    sam3db_model, sam3db_cfg = load_sam_3d_body(
        SAM3D_CHECKPOINT,
        device=device,
        mhr_path=MHR_PATH
    )
    sam3db_estimator = SAM3DBodyEstimator(
        sam_3d_body_model=sam3db_model,
        model_cfg=sam3db_cfg,
    )
    print("SAM3D Body loaded!")

    return sam3_processor, sam3db_estimator

def find_video_path(video_file, video_dirs):
    """
    Find video file in possible directories
    """
    # Try direct path first
    if Path(video_file).exists():
        return str(video_file)

    # Try each video directory
    for video_dir in video_dirs:
        # Direct in directory
        path = Path(video_dir) / video_file
        if path.exists():
            return str(path)

        # In ft0/ft1 subdirectories
        for subdir in ['ft0', 'ft1']:
            path = Path(video_dir) / subdir / video_file
            if path.exists():
                return str(path)

    return None


def extract_frame_features(frame_rgb, pil_img, sam3_processor, sam3db_estimator, reference_bbox=None):
    """
    Extract 3D pose and ball position from a single frame

    Inputs:
        frame_rgb: numpy array (H, W, 3) RGB
        pil_img: PIL Image
        sam3_processor: SAM3 processor for segmentation
        sam3db_estimator: SAM3D Body estimator
        reference_bbox: Optional bbox to guide shooter detection

    Returns:
        dict with keypoints_3d, keypoints_2d, ball_pos, bbox
    """
    h, w = frame_rgb.shape[:2]
    frame_data = {}

    # SAM3: basketball detection
    try:
        with torch.inference_mode():
            state = sam3_processor.set_image(pil_img)
            ball_output = sam3_processor.set_text_prompt(state=state, prompt="basketball")

        ball_masks = ball_output["masks"]
        if len(ball_masks) > 0:
            scores = [s.item() if torch.is_tensor(s) else s for s in ball_output["scores"]]
            best_ball_idx = np.argmax(scores)
            best_ball = ball_masks[best_ball_idx]
            ball_np = best_ball.cpu().numpy() if torch.is_tensor(best_ball) else best_ball
            if ball_np.ndim == 3:
                ball_np = ball_np[0]
            ys, xs = np.where(ball_np > 0)
            if len(xs) > 0:
                frame_data["ball_pos"] = [float(xs.mean()), float(ys.mean())]
    except Exception as e:
        pass

    # SAM3: shooter detection
    try:
        with torch.inference_mode():
            state = sam3_processor.set_image(pil_img)
            person_output = sam3_processor.set_text_prompt(
                state=state,
                prompt="person shooting basketball"
            )

        person_masks = person_output["masks"]
        if len(person_masks) == 0:
            return frame_data

        # Find best person (highest score, or closest to reference bbox)
        scores = [s.item() if torch.is_tensor(s) else s for s in person_output["scores"]]

        if reference_bbox is not None:
            # Find mask closest to reference
            ref_center = [(reference_bbox[0] + reference_bbox[2]) / 2,
                          (reference_bbox[1] + reference_bbox[3]) / 2]
            best_dist = float('inf')
            best_idx = 0
            for idx, mask in enumerate(person_masks):
                mask_np = mask.cpu().numpy() if torch.is_tensor(mask) else mask
                if mask_np.ndim == 3:
                    mask_np = mask_np[0]
                ys, xs = np.where(mask_np > 0)
                if len(xs) > 0:
                    center = [xs.mean(), ys.mean()]
                    dist = np.sqrt((center[0] - ref_center[0]) ** 2 + (center[1] - ref_center[1]) ** 2)
                    if dist < best_dist:
                        best_dist = dist
                        best_idx = idx
        else:
            best_idx = np.argmax(scores)

        shooter_mask = person_masks[best_idx]
        shooter_mask_np = shooter_mask.cpu().numpy() if torch.is_tensor(shooter_mask) else shooter_mask
        if shooter_mask_np.ndim == 3:
            shooter_mask_np = shooter_mask_np[0]

        ys, xs = np.where(shooter_mask_np > 0)
        if len(xs) == 0:
            return frame_data

        x1, y1, x2, y2 = xs.min(), ys.min(), xs.max(), ys.max()
        bbox = np.array([[x1, y1, x2, y2]])
        frame_data["bbox"] = [int(x1), int(y1), int(x2), int(y2)]

    except Exception as e:
        return frame_data

    # SAM3D Body: 3D pose extraction
    try:
        outputs = sam3db_estimator.process_one_image(
            frame_rgb,
            bboxes=bbox,
            masks=shooter_mask_np.astype(np.uint8)[None, :, :],
            use_mask=True,
        )

        if len(outputs) > 0:
            pose_out = outputs[0]
            if "pred_keypoints_3d" in pose_out:
                frame_data["keypoints_3d"] = to_list(pose_out["pred_keypoints_3d"])
            if "pred_keypoints_2d" in pose_out:
                frame_data["keypoints_2d"] = to_list(pose_out["pred_keypoints_2d"])
            if "pred_joint_coords" in pose_out:
                frame_data["joint_coords"] = to_list(pose_out["pred_joint_coords"])
            if "body_pose_params" in pose_out:
                frame_data["body_pose"] = to_list(pose_out["body_pose_params"])
    except Exception as e:
        pass

    return frame_data


def process_candidate(candidate, sam3_processor, sam3db_estimator, video_dirs):
    """
    Process a single candidate from candidates_labeled.json.

    Returns:
        dict matching features.json format, or None if failed
    """
    video_file = candidate.get('video_file', '')
    video_path = find_video_path(video_file, video_dirs)

    if video_path is None:
        return None

    video_id = video_file.replace('.mp4', '').replace('.mov', '').replace('.avi', '')
    label = candidate.get('label', -1)
    release_frame = candidate.get('release_frame', 0)
    sequence_indices = candidate.get('sequence_frame_indices', [])

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
        # Fallback: create indices around release
        frame_indices = [release_frame + off for off in [-2, -1, 0, 1]]

    # Ensure 4 frames
    while len(frame_indices) < 4:
        frame_indices.append(frame_indices[-1] if frame_indices else release_frame)
    frame_indices = frame_indices[:4]

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Clamp frame indices
    frame_indices = [max(0, min(total_frames - 1, f)) for f in frame_indices]

    # Get reference bbox from candidate's 2D keypoints (for shooter tracking)
    reference_bbox = None
    shooter = candidate.get('shooter', {})
    if 'keypoints_2d' in shooter and shooter['keypoints_2d']:
        kp2d = np.array(shooter['keypoints_2d'])
        x1, y1 = kp2d.min(axis=0)
        x2, y2 = kp2d.max(axis=0)
        # Expand bbox slightly
        pad = 20
        reference_bbox = [x1 - pad, y1 - pad, x2 + pad, y2 + pad]

    sequence = {
        "video_id": video_id,
        "video_file": video_file,
        "label": label,
        "release_frame": release_frame,
        "frame_indices": frame_indices,
        "frames": [],
        # Preserve metadata from candidates
        "camera_angle": candidate.get('camera_angle'),
        "fps": candidate.get('fps'),
        "status": candidate.get('status'),
        "confidence": candidate.get('confidence'),
        "arm_angle": shooter.get('arm_angle'),
    }

    valid_frames = 0

    for fidx, frame_idx in enumerate(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame_bgr = cap.read()

        if not ret:
            sequence["frames"].append({"frame_idx": frame_idx, "offset": fidx - 2})
            continue

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)

        # Extract features
        frame_data = extract_frame_features(
            frame_rgb, pil_img,
            sam3_processor, sam3db_estimator,
            reference_bbox=reference_bbox
        )

        frame_data["frame_idx"] = frame_idx
        frame_data["offset"] = fidx - 2  # -2, -1, 0, 1 relative to release

        if "keypoints_3d" in frame_data:
            valid_frames += 1
            # Update reference bbox for next frame (track shooter)
            if "bbox" in frame_data:
                reference_bbox = frame_data["bbox"]

        sequence["frames"].append(frame_data)

    cap.release()
    sequence["valid_frames"] = valid_frames

    return sequence


def main():
    parser = argparse.ArgumentParser(description='Extract SAM3D poses from candidates')
    parser.add_argument('--input', type=str, default='data/release_detection/candidates.json',
                        help='Input candidates JSON file')
    parser.add_argument('--output', type=str, default='data/features/features.json',
                        help='Output features JSON file')
    parser.add_argument('--video-dir', type=str, nargs='+',
                        default=['data/hq_videos', 'data/videos', '/workspace/videos'],
                        help='Directories to search for videos')
    parser.add_argument('--limit', type=int, default=None,
                        help='Limit number of candidates to process')
    parser.add_argument('--labeled-only', action='store_true',
                        help='Only process labeled candidates')
    parser.add_argument('--filter-quality', type=str, default=None,
                        choices=['PERFECT', 'high'],
                        help='Filter by quality status')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda or cpu)')
    args = parser.parse_args()

    # Load candidates
    print(f"Loading {args.input}...")
    with open(args.input) as f:
        candidates = json.load(f)
    print(f"Loaded {len(candidates)} candidates")

    # Filter
    if args.labeled_only:
        candidates = [c for c in candidates if c.get('label', -1) != -1]
        print(f"After labeled filter: {len(candidates)}")

    if args.filter_quality:
        if args.filter_quality == 'PERFECT':
            candidates = [c for c in candidates if c.get('status') == 'PERFECT']
        elif args.filter_quality == 'high':
            candidates = [c for c in candidates if 'high' in str(c.get('confidence', ''))]
        print(f"After quality filter: {len(candidates)}")

    if args.limit:
        candidates = candidates[:args.limit]
        print(f"Limited to: {len(candidates)}")

    # Load models
    sam3_processor, sam3db_estimator = load_models(device=args.device)

    # Process
    print(f"\nProcessing {len(candidates)} candidates...")
    results = []
    failed = 0

    for candidate in tqdm(candidates, desc="Extracting sequences"):
        try:
            result = process_candidate(
                candidate,
                sam3_processor,
                sam3db_estimator,
                args.video_dir
            )

            if result and result.get('valid_frames', 0) > 0:
                results.append(result)
            else:
                failed += 1

        except Exception as e:
            failed += 1
            continue

        # Progress update
        if len(results) % 50 == 0 and len(results) > 0:
            avg_valid = np.mean([r['valid_frames'] for r in results])
            print(f"  Processed {len(results)}, avg valid frames: {avg_valid:.1f}/4")

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(results, f)

    # Stats
    valid_counts = [r['valid_frames'] for r in results]
    print(f"\nDone!")
    print(f"Successful: {len(results)}")
    print(f"Failed: {failed}")
    print(
        f"Total valid frames: {sum(valid_counts)}/{len(results) * 4} ({100 * sum(valid_counts) / (len(results) * 4):.1f}%)")
    print(f"Avg valid per sequence: {np.mean(valid_counts):.2f}/4")
    print(f"Saved to: {output_path}")

    # Label distribution
    labeled = [r for r in results if r.get('label', -1) != -1]
    if labeled:
        makes = sum(1 for r in labeled if r['label'] == 1)
        misses = len(labeled) - makes
        print(f"\nLabeled: {len(labeled)} ({makes} makes, {misses} misses)")


if __name__ == "__main__":
    main()