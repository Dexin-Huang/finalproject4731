"""
Extract 4-frame sequences: t-2, t-1, t (release), t+1
For each frame: SAM3D Body 3D pose using bbox from auto-detection

Saves progress every 10 samples to avoid data loss.

NOTE: This script is designed to run on RunPod cloud GPU with SAM3D Body installed.
The /workspace/ paths are specific to the RunPod environment.
To run locally, update the paths to your SAM3D installation:
    - sys.path: Point to your sam-3d-body directory
    - checkpoint/mhr_path: Point to your model checkpoints
"""
import torch
import numpy as np
import cv2
from pathlib import Path
import json
from tqdm import tqdm
import sys
import argparse
import time

sys.path.insert(0, "/workspace/sam-3d-body")


def to_list(obj):
    """Convert numpy arrays and tensors to lists."""
    if torch.is_tensor(obj):
        return obj.cpu().numpy().tolist()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def get_shooter_bbox(candidate):
    shooter = candidate.get("shooter", {})
    keypoints_2d = shooter.get("keypoints_2d", [])
    if not keypoints_2d:
        return None
    valid_points = [(x, y) for x, y in keypoints_2d if x > 0 and y > 0]
    if len(valid_points) < 3:
        return None
    xs = [p[0] for p in valid_points]
    ys = [p[1] for p in valid_points]
    x1, y1 = min(xs), min(ys)
    x2, y2 = max(xs), max(ys)
    w, h = x2 - x1, y2 - y1
    pad_x, pad_y = w * 0.2, h * 0.2
    x1, y1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
    x2, y2 = x2 + pad_x, y2 + pad_y
    return [int(x1), int(y1), int(x2), int(y2)]


def extract_pose_from_frame(frame_rgb, bbox, sam3db_estimator):
    frame_data = {}
    bbox_array = np.array([[bbox[0], bbox[1], bbox[2], bbox[3]]])
    frame_data["bbox"] = bbox
    try:
        outputs = sam3db_estimator.process_one_image(
            frame_rgb, bboxes=bbox_array, use_mask=False
        )
        if len(outputs) > 0:
            pose_out = outputs[0]
            if "pred_keypoints_3d" in pose_out:
                frame_data["keypoints_3d"] = to_list(pose_out["pred_keypoints_3d"])
            if "pred_keypoints_2d" in pose_out:
                frame_data["keypoints_2d"] = to_list(pose_out["pred_keypoints_2d"])
            if "pred_joint_coords" in pose_out:
                frame_data["joint_coords"] = to_list(pose_out["pred_joint_coords"])
            frame_data["pose_valid"] = True
    except Exception as e:
        frame_data["pose_error"] = str(e)
    return frame_data


def process_candidate(candidate, video_dir, sam3db_estimator):
    video_file = candidate.get("video_file", "")
    release_frame = candidate.get("release_frame")
    label = candidate.get("label", -1)
    bbox = get_shooter_bbox(candidate)
    if bbox is None:
        return None
    ball_location = candidate.get("ball_location")

    video_path = None
    for subdir in ["ft0", "ft1"]:
        p = video_dir / subdir / video_file
        if p.exists():
            video_path = p
            break
    if video_path is None:
        return None

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_offsets = [-2, -1, 0, 1]
    frame_indices = [max(0, min(total_frames - 1, release_frame + off)) for off in frame_offsets]

    sequence = {
        "video_id": Path(video_file).stem,
        "video_file": video_file,
        "label": label,
        "release_frame": release_frame,
        "frame_indices": frame_indices,
        "bbox": bbox,
        "frames": []
    }
    if ball_location and 'x' in ball_location and 'y' in ball_location:
        sequence["ball_pos"] = [float(ball_location['x']), float(ball_location['y'])]

    valid_frames = 0
    for fidx, frame_idx in enumerate(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame_bgr = cap.read()
        if not ret:
            sequence["frames"].append({
                "frame_idx": frame_idx,
                "offset": frame_offsets[fidx],
                "error": "read_failed"
            })
            continue

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_data = extract_pose_from_frame(frame_rgb, bbox, sam3db_estimator)
        frame_data["frame_idx"] = frame_idx
        frame_data["offset"] = frame_offsets[fidx]
        if frame_data.get("pose_valid"):
            valid_frames += 1
        sequence["frames"].append(frame_data)

    cap.release()
    sequence["valid_frames"] = valid_frames
    return sequence


def main():
    parser = argparse.ArgumentParser(description='Extract SAM3D sequences from videos')
    parser.add_argument('--input', type=str, default='/workspace/final_perfect.json')
    parser.add_argument('--output', type=str, default='/workspace/features.json')
    parser.add_argument('--video-dir', type=str, default='/workspace/hq_videos')
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--resume', action='store_true', help='Resume from existing output')
    args = parser.parse_args()

    device = args.device

    # Load existing results if resuming
    results = []
    processed_videos = set()
    if args.resume and Path(args.output).exists():
        try:
            results = json.load(open(args.output))
            processed_videos = {r["video_file"] for r in results}
            print(f"Resuming: found {len(results)} existing samples")
        except:
            pass

    print("Loading SAM3D Body...")
    from sam_3d_body import load_sam_3d_body, SAM3DBodyEstimator
    sam3db_model, sam3db_cfg = load_sam_3d_body(
        "/workspace/sam-3d-body/checkpoints/sam-3d-body-dinov3/model.ckpt",
        device=device,
        mhr_path="/workspace/sam-3d-body/checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt"
    )
    sam3db_estimator = SAM3DBodyEstimator(
        sam_3d_body_model=sam3db_model,
        model_cfg=sam3db_cfg,
    )
    print("SAM3D Body loaded!")

    print(f"Loading candidates from {args.input}...")
    candidates = json.load(open(args.input))
    print(f"Loaded {len(candidates)} candidates")

    # Filter already processed
    if processed_videos:
        candidates = [c for c in candidates if c.get("video_file") not in processed_videos]
        print(f"Remaining after resume: {len(candidates)}")

    if args.limit > 0:
        candidates = candidates[:args.limit]
        print(f"Limited to {len(candidates)} candidates")

    video_dir = Path(args.video_dir)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    stats = {"total": len(candidates), "processed": 0, "failed_no_bbox": 0, "failed_no_video": 0}
    start_time = time.time()

    for i, candidate in enumerate(tqdm(candidates, desc="Extracting sequences")):
        if get_shooter_bbox(candidate) is None:
            stats["failed_no_bbox"] += 1
            continue

        sequence = process_candidate(candidate, video_dir, sam3db_estimator)
        if sequence is not None:
            results.append(sequence)
            stats["processed"] += 1
        else:
            stats["failed_no_video"] += 1

        # Save every 10 samples
        if len(results) % 10 == 0 and len(results) > 0:
            with open(output_path, "w") as f:
                json.dump(results, f)
            elapsed = time.time() - start_time
            rate = stats["processed"] / elapsed * 60 if elapsed > 0 else 0
            print(f"\n  Saved {len(results)} samples ({rate:.1f}/min)")

    # Final save
    with open(output_path, "w") as f:
        json.dump(results, f)

    print("\n" + "=" * 60)
    print("EXTRACTION COMPLETE")
    print("=" * 60)
    print(f"Total candidates: {stats['total']}")
    print(f"Processed: {stats['processed']}")
    print(f"Failed (no bbox): {stats['failed_no_bbox']}")
    print(f"Failed (no video): {stats['failed_no_video']}")

    if results:
        valid_counts = [r["valid_frames"] for r in results]
        print(f"\nValid frames: {sum(valid_counts)}/{len(results) * 4} ({100 * sum(valid_counts) / (len(results) * 4):.1f}%)")
        makes = sum(1 for r in results if r["label"] == 1)
        misses = sum(1 for r in results if r["label"] == 0)
        print(f"Labels: {makes} makes, {misses} misses")

    print(f"\nOutput saved to: {args.output}")


if __name__ == "__main__":
    main()
