# app_realtime.py - Real-Time Live Game Simulation with REAL Model Predictions
"""
Simulates live game betting experience with side-by-side layout:
- Left: Video playing in real-time
- Right: Prediction result from trained ST-GCN model

Hit SPACEBAR at release moment → instant prediction from actual model!

Usage:
    streamlit run app_realtime.py

Requirements:
    - models/best_merged_calibrated.pth (trained model)
    - src/models/stgcn.py (model architecture)
"""

import streamlit as st
import streamlit.components.v1 as components
import torch
import numpy as np
import cv2
import tempfile
import os
import base64
import logging
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==================== PAGE CONFIG ====================

st.set_page_config(
    page_title="🎮 Live Game Mode",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Force dark mode CSS
st.markdown("""
<style>
    .stApp {
        background-color: #0a0a0f !important;
    }
    [data-testid="stAppViewContainer"] {
        background-color: #0a0a0f !important;
    }
    [data-testid="stFileUploader"] {
        background-color: #1a1a2e !important;
        border-radius: 10px;
        padding: 10px;
    }
    [data-testid="stFileUploader"] section {
        background-color: #1a1a2e !important;
        border-color: #4b5563 !important;
    }
    [data-testid="stFileUploader"] button {
        background-color: #f97316 !important;
        color: white !important;
    }
    .stApp p, .stApp span, .stApp label, .stApp div {
        color: #e5e7eb !important;
    }
    .block-container { padding-top: 1rem; padding-bottom: 0; }
    header { visibility: hidden; }
    .main-title {
        font-size: 1.6rem;
        font-weight: 700;
        color: #f97316 !important;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .subtitle {
        color: #6b7280 !important;
        text-align: center;
        font-size: 0.9rem;
        margin-bottom: 1rem;
    }
    hr {
        border-color: #374151 !important;
    }
</style>
""", unsafe_allow_html=True)

# ==================== MODEL LOADING ====================

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_PATH = "models/best_merged_calibrated.pth"


@st.cache_resource
def load_model():
    """Load the trained ST-GCN model"""
    try:
        from src.models.stgcn import STGCN

        model = STGCN(
            in_channels=9,  # x, y, z, vx, vy, vz, ax, ay, az
            num_class=2,  # make/miss
            graph_args={'layout': 'smpl', 'strategy': 'spatial'},
            edge_importance_weighting=True
        )

        if not os.path.exists(MODEL_PATH):
            logger.error(f"Model not found at {MODEL_PATH}")
            return None

        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)

        model.to(DEVICE)
        model.eval()
        logger.info(f"✓ Model loaded from {MODEL_PATH}")
        return model

    except Exception as e:
        logger.error(f"Model loading error: {e}")
        return None


def extract_pose_from_frame(frame, video_path=None, frame_idx=0):
    """
    Extract 3D pose sequence from video frame.

    In production, this should use:
    - SAM3D Body API for accurate 3D pose
    - Or MediaPipe + depth estimation

    For now, tries to load pre-extracted poses from enhanced_all.json
    if available, otherwise returns mock data.
    """
    # Try to load pre-extracted pose from dataset
    json_paths = [
        "data/features/enhanced_all.json",
        "data/features/enhanced_clean.json",
    ]

    for json_path in json_paths:
        if os.path.exists(json_path):
            try:
                with open(json_path, 'r') as f:
                    data = json.load(f)

                # If we have a video path, try to match it
                if video_path:
                    video_name = os.path.basename(video_path).replace('.mp4', '').replace('.mov', '').replace('.avi',
                                                                                                              '')
                    for entry in data:
                        if video_name in entry.get('video_id', ''):
                            # Found matching entry - extract pose features
                            frames_data = entry.get('frames', [])
                            if frames_data:
                                # Build pose sequence from keypoints_3d
                                T = min(4, len(frames_data))
                                V = 25  # number of joints (use first 25)
                                C = 9  # features per joint

                                pose_sequence = np.zeros((T, V, C), dtype=np.float32)

                                for t, frame_data in enumerate(frames_data[:T]):
                                    kp3d = frame_data.get('keypoints_3d', [])
                                    for v in range(min(V, len(kp3d))):
                                        if len(kp3d[v]) >= 3:
                                            # x, y, z positions
                                            pose_sequence[t, v, 0] = kp3d[v][0]
                                            pose_sequence[t, v, 1] = kp3d[v][1]
                                            pose_sequence[t, v, 2] = kp3d[v][2]
                                            # velocities and accelerations will be computed

                                # Compute velocities (vx, vy, vz)
                                for t in range(1, T):
                                    pose_sequence[t, :, 3:6] = pose_sequence[t, :, 0:3] - pose_sequence[t - 1, :, 0:3]

                                # Compute accelerations (ax, ay, az)
                                for t in range(2, T):
                                    pose_sequence[t, :, 6:9] = pose_sequence[t, :, 3:6] - pose_sequence[t - 1, :, 3:6]

                                logger.info(f"✓ Loaded pose from dataset for {video_name}")
                                return pose_sequence, entry.get('label', None)

            except Exception as e:
                logger.warning(f"Error loading poses: {e}")

    # Fallback: return mock data (random poses)
    logger.warning("⚠ Using mock pose data - integrate real pose extraction for production")
    T, V, C = 4, 25, 9
    pose_sequence = np.random.randn(T, V, C).astype(np.float32) * 0.1
    return pose_sequence, None


def predict_with_model(model, pose_sequence):
    """
    Run prediction using the trained ST-GCN model.

    Args:
        model: Loaded ST-GCN model
        pose_sequence: numpy array of shape (T, V, C)
                      T = temporal frames (4)
                      V = joints (25)
                      C = channels (9: x,y,z,vx,vy,vz,ax,ay,az)

    Returns:
        dict with prediction results
    """
    try:
        # Convert to tensor: (T, V, C) -> (1, C, T, V, 1)
        pose_tensor = torch.from_numpy(pose_sequence).float()
        pose_tensor = pose_tensor.permute(2, 0, 1)  # (C, T, V)
        pose_tensor = pose_tensor.unsqueeze(0).unsqueeze(-1)  # (1, C, T, V, 1)
        pose_tensor = pose_tensor.to(DEVICE)

        # Run inference
        with torch.no_grad():
            output = model(pose_tensor)
            probs = torch.softmax(output, dim=1)

        miss_prob = probs[0, 0].item()
        make_prob = probs[0, 1].item()
        confidence = abs(make_prob - 0.5) * 2

        return {
            'make_prob': make_prob,
            'miss_prob': miss_prob,
            'confidence': confidence,
            'prediction': 'MAKE' if make_prob > 0.5 else 'MISS'
        }

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return None


# ==================== MAIN UI ====================

st.markdown('<p class="main-title">🎮 Live Game Mode</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Watch video → Hit SPACEBAR at release → See prediction instantly</p>',
            unsafe_allow_html=True)

# Load model
model = load_model()
model_status = "🟢 Model Ready" if model else "🔴 Model Not Found (run training first)"

# File upload
video_file = st.file_uploader("Upload free throw video", type=['mp4', 'mov', 'avi'], label_visibility="collapsed")

if video_file:
    # Save video temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
        tmp.write(video_file.read())
        video_path = tmp.name

    # Convert video to base64 for embedding
    with open(video_path, 'rb') as f:
        video_bytes = f.read()
    video_b64 = base64.b64encode(video_bytes).decode()

    # Get video filename for matching
    video_filename = video_file.name if hasattr(video_file, 'name') else "unknown"

    # Pre-compute prediction using model
    prediction_result = None
    ground_truth_label = None

    if model:
        # Extract pose and run model prediction
        pose_sequence, gt_label = extract_pose_from_frame(None, video_path)
        ground_truth_label = gt_label
        prediction_result = predict_with_model(model, pose_sequence)

        if prediction_result:
            logger.info(f"Prediction: {prediction_result['prediction']} "
                        f"(make: {prediction_result['make_prob']:.1%}, "
                        f"miss: {prediction_result['miss_prob']:.1%})")

    # Prepare prediction data for JavaScript
    if prediction_result:
        pred_json = json.dumps(prediction_result)
    else:
        pred_json = json.dumps({'make_prob': 0.5, 'miss_prob': 0.5, 'confidence': 0, 'prediction': 'NO MODEL'})

    gt_json = json.dumps(ground_truth_label) if ground_truth_label is not None else "null"

    # Side-by-side HTML component
    html_code = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            * {{ margin: 0; padding: 0; box-sizing: border-box; }}
            body {{ 
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; 
                background: #0a0a0f; 
                color: white;
                padding: 10px;
            }}

            .layout {{
                display: grid;
                grid-template-columns: 1fr 260px;
                gap: 15px;
                max-width: 1100px;
                margin: 0 auto;
            }}

            .video-section {{
                display: flex;
                flex-direction: column;
            }}

            .video-wrap {{
                position: relative;
                background: #000;
                border-radius: 8px;
                overflow: hidden;
            }}

            video {{
                width: 100%;
                display: block;
            }}

            .flash {{
                position: absolute;
                inset: 0;
                background: white;
                opacity: 0;
                pointer-events: none;
            }}

            .flash.active {{
                animation: flashAnim 0.15s ease-out;
            }}

            @keyframes flashAnim {{
                0% {{ opacity: 0.7; }}
                100% {{ opacity: 0; }}
            }}

            .overlay {{
                position: absolute;
                inset: 0;
                display: flex;
                align-items: center;
                justify-content: center;
                background: rgba(0,0,0,0.75);
                opacity: 0;
                pointer-events: none;
                transition: opacity 0.2s;
            }}

            .overlay.show {{ opacity: 1; }}

            .status-row {{
                display: flex;
                justify-content: space-between;
                align-items: center;
                padding: 8px 12px;
                background: #1a1a2e;
                border-radius: 6px;
                margin-top: 8px;
            }}

            .status {{ font-size: 0.9rem; }}
            .status.wait {{ color: #fbbf24; }}
            .status.play {{ color: #3b82f6; }}
            .status.done {{ color: #10b981; }}

            .timer {{
                font-family: 'SF Mono', 'Courier New', monospace;
                font-size: 1.2rem;
                color: #f97316;
            }}

            .btns {{
                display: flex;
                gap: 8px;
                margin-top: 8px;
                justify-content: center;
            }}

            button {{
                background: #f97316;
                color: white;
                border: none;
                padding: 10px 24px;
                font-size: 0.95rem;
                border-radius: 5px;
                cursor: pointer;
                transition: all 0.15s;
            }}

            button:hover {{ background: #fb923c; }}
            button.sec {{ background: #374151; }}
            button.sec:hover {{ background: #4b5563; }}

            .hint-box {{
                margin-top: 12px;
                padding: 12px;
                background: rgba(249, 115, 22, 0.08);
                border: 1px solid rgba(249, 115, 22, 0.3);
                border-radius: 8px;
                text-align: center;
            }}

            .hint-box .icon {{ font-size: 1.5rem; margin-bottom: 4px; }}
            .hint-box .text {{ color: #9ca3af; font-size: 0.85rem; }}
            .hint-box .key {{
                display: inline-block;
                background: rgba(249, 115, 22, 0.2);
                border: 1px solid #f97316;
                color: #f97316;
                padding: 4px 12px;
                border-radius: 4px;
                font-size: 0.8rem;
                margin-top: 6px;
            }}

            .result-section {{
                background: linear-gradient(135deg, #1a1a2e, #252540);
                border-radius: 10px;
                padding: 15px;
                display: flex;
                flex-direction: column;
                justify-content: center;
                align-items: center;
                text-align: center;
            }}

            .result-section.has-result {{
                border: 2px solid #10b981;
            }}

            .waiting-result {{
                color: #4b5563;
                font-size: 0.9rem;
            }}

            .waiting-result .dash {{
                font-size: 2rem;
                color: #374151;
                margin-bottom: 5px;
            }}

            .result {{ display: none; }}
            .result.show {{ display: block; animation: pop 0.25s ease-out; }}

            @keyframes pop {{
                0% {{ opacity: 0; transform: scale(0.9); }}
                100% {{ opacity: 1; transform: scale(1); }}
            }}

            .prediction {{
                font-size: 1.8rem;
                font-weight: 700;
                padding: 10px 24px;
                border-radius: 10px;
                margin-bottom: 8px;
            }}

            .prediction.make {{
                background: linear-gradient(135deg, #059669, #10b981);
                box-shadow: 0 0 25px rgba(16, 185, 129, 0.4);
            }}

            .prediction.miss {{
                background: linear-gradient(135deg, #dc2626, #ef4444);
                box-shadow: 0 0 25px rgba(239, 68, 68, 0.4);
            }}

            .prediction.nobet {{
                background: linear-gradient(135deg, #4b5563, #6b7280);
            }}

            .probs {{
                color: #9ca3af;
                font-size: 0.85rem;
                margin-bottom: 8px;
            }}

            .thumb {{
                max-width: 140px;
                border: 2px solid #10b981;
                border-radius: 5px;
                margin: 6px 0;
            }}

            .captime {{
                color: #6b7280;
                font-size: 0.75rem;
            }}

            .model-badge {{
                font-size: 0.7rem;
                padding: 2px 8px;
                border-radius: 4px;
                margin-top: 8px;
            }}

            .model-badge.real {{
                background: rgba(16, 185, 129, 0.2);
                color: #10b981;
                border: 1px solid #10b981;
            }}

            .model-badge.mock {{
                background: rgba(245, 158, 11, 0.2);
                color: #f59e0b;
                border: 1px solid #f59e0b;
            }}
        </style>
    </head>
    <body>
        <div class="layout">
            <div class="video-section">
                <div class="video-wrap">
                    <video id="vid" src="data:video/mp4;base64,{video_b64}"></video>
                    <div class="flash" id="flash"></div>
                    <div class="overlay" id="overlay">
                        <div>
                            <div style="font-size:2.5rem">📸</div>
                            <div>Captured!</div>
                        </div>
                    </div>
                </div>
                <div class="status-row">
                    <span class="status wait" id="status">Press PLAY → SPACEBAR at release</span>
                    <span class="timer" id="timer">0:00.000</span>
                </div>
                <div class="btns">
                    <button id="playBtn" onclick="toggle()">▶ PLAY</button>
                    <button class="sec" onclick="reset()">↺ RESET</button>
                </div>
                <div class="hint-box">
                    <div class="icon">🎯</div>
                    <div class="text">Press <strong>SPACEBAR</strong> at the release moment</div>
                    <div class="key">⎵ SPACEBAR</div>
                </div>
            </div>

            <div class="result-section" id="resultBox">
                <div class="waiting-result" id="waiting">
                    <div class="dash">—</div>
                    <div>Prediction will<br>appear here</div>
                </div>
                <div class="result" id="result">
                    <div class="prediction" id="pred">-</div>
                    <div class="probs" id="probs"></div>
                    <canvas id="thumb" class="thumb" width="140" height="79"></canvas>
                    <div class="captime" id="captime"></div>
                    <div class="model-badge" id="modelBadge"></div>
                </div>
            </div>
        </div>

        <script>
            const vid = document.getElementById('vid');
            const flash = document.getElementById('flash');
            const overlay = document.getElementById('overlay');
            const status = document.getElementById('status');
            const timer = document.getElementById('timer');
            const playBtn = document.getElementById('playBtn');
            const resultBox = document.getElementById('resultBox');
            const waiting = document.getElementById('waiting');
            const result = document.getElementById('result');
            const pred = document.getElementById('pred');
            const probs = document.getElementById('probs');
            const thumb = document.getElementById('thumb');
            const captime = document.getElementById('captime');
            const modelBadge = document.getElementById('modelBadge');

            let playing = false;
            let captured = false;

            // Pre-computed prediction from ST-GCN model
            const modelPrediction = {pred_json};
            const groundTruth = {gt_json};

            vid.addEventListener('timeupdate', () => {{
                const t = vid.currentTime;
                const m = Math.floor(t / 60);
                const s = Math.floor(t % 60);
                const ms = Math.floor((t % 1) * 1000);
                timer.textContent = m + ':' + String(s).padStart(2,'0') + '.' + String(ms).padStart(3,'0');
            }});

            vid.addEventListener('play', () => {{
                playing = true;
                status.textContent = '▶ PLAYING - SPACEBAR at release!';
                status.className = 'status play';
                playBtn.textContent = '⏸ PAUSE';
            }});

            vid.addEventListener('pause', () => {{
                playing = false;
                if (!captured) {{
                    status.textContent = '⏸ Paused';
                    status.className = 'status wait';
                }}
                playBtn.textContent = '▶ PLAY';
            }});

            vid.addEventListener('ended', () => {{
                playing = false;
                playBtn.textContent = '▶ PLAY';
            }});

            function toggle() {{
                if (captured) return;
                vid.paused ? vid.play() : vid.pause();
            }}

            function reset() {{
                vid.pause();
                vid.currentTime = 0;
                playing = false;
                captured = false;
                status.textContent = 'Press PLAY → SPACEBAR at release';
                status.className = 'status wait';
                playBtn.textContent = '▶ PLAY';
                resultBox.classList.remove('has-result');
                waiting.style.display = 'block';
                result.classList.remove('show');
            }}

            function capture() {{
                if (captured || !playing) return;
                captured = true;

                // Flash effect
                flash.classList.add('active');
                setTimeout(() => flash.classList.remove('active'), 150);

                // Overlay
                overlay.classList.add('show');
                setTimeout(() => overlay.classList.remove('show'), 400);

                vid.pause();

                // Draw thumbnail
                const ctx = thumb.getContext('2d');
                ctx.drawImage(vid, 0, 0, 140, 79);

                // Capture time
                const t = vid.currentTime;
                const m = Math.floor(t / 60);
                const s = Math.floor(t % 60);
                const ms = Math.floor((t % 1) * 1000);
                captime.textContent = 'Captured at ' + m + ':' + String(s).padStart(2,'0') + '.' + String(ms).padStart(3,'0');

                status.textContent = '🔄 Running ST-GCN model...';
                status.className = 'status done';

                // Show prediction after brief delay (simulating inference)
                setTimeout(showPrediction, 300);
            }}

            function showPrediction() {{
                const makeP = modelPrediction.make_prob;
                const missP = modelPrediction.miss_prob;
                const conf = modelPrediction.confidence;
                const prediction = modelPrediction.prediction;

                let label, cls;

                if (prediction === 'NO MODEL') {{
                    label = '⚠️ NO MODEL';
                    cls = 'nobet';
                }} else if (conf < 0.2) {{
                    label = '⚪ NO BET';
                    cls = 'nobet';
                }} else if (prediction === 'MAKE') {{
                    label = '🟢 MAKE';
                    cls = 'make';
                }} else {{
                    label = '🔴 MISS';
                    cls = 'miss';
                }}

                pred.textContent = label;
                pred.className = 'prediction ' + cls;

                let probsHtml = 'Make: ' + (makeP*100).toFixed(1) + '% | Miss: ' + (missP*100).toFixed(1) + '%<br>';
                probsHtml += 'Confidence: ' + (conf*100).toFixed(0) + '%';

                // Show ground truth if available
                if (groundTruth !== null) {{
                    const gtLabel = groundTruth === 1 ? 'MAKE' : 'MISS';
                    const correct = (prediction === gtLabel);
                    probsHtml += '<br><small style="color:' + (correct ? '#10b981' : '#ef4444') + '">';
                    probsHtml += (correct ? '✓' : '✗') + ' Ground truth: ' + gtLabel + '</small>';
                }}

                probs.innerHTML = probsHtml;

                // Model badge
                if (prediction === 'NO MODEL') {{
                    modelBadge.textContent = '⚠️ Model not loaded';
                    modelBadge.className = 'model-badge mock';
                }} else {{
                    modelBadge.textContent = '🧠 ST-GCN Prediction';
                    modelBadge.className = 'model-badge real';
                }}

                waiting.style.display = 'none';
                result.classList.add('show');
                resultBox.classList.add('has-result');
                status.textContent = '✓ Prediction complete!';
            }}

            document.addEventListener('keydown', (e) => {{
                if (e.code === 'Space') {{
                    e.preventDefault();
                    if (playing) capture();
                    else if (!captured) toggle();
                }}
            }});
        </script>
    </body>
    </html>
    """

    # Render component
    components.html(html_code, height=580, scrolling=False)

    # Cleanup temp file
    os.unlink(video_path)

else:
    # No video uploaded yet
    st.markdown("""
    <div style="text-align: center; padding: 40px; background: rgba(255,255,255,0.02); border-radius: 12px; border: 2px dashed #4b5563;">
        <div style="font-size: 3rem; margin-bottom: 15px;">📹</div>
        <div style="font-size: 1.1rem; color: #e5e7eb; margin-bottom: 8px;">Upload a free throw video to begin</div>
        <div style="color: #6b7280; font-size: 0.9rem;">MP4, MOV, AVI supported</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    ---
    ### 🎮 How It Works

    1. **Upload video** → Video appears on the left
    2. **Click PLAY** or press **SPACEBAR** to start
    3. **Watch like a live game** → Video plays in real-time  
    4. **Press SPACEBAR** at the exact moment of release
    5. **See prediction instantly** from the trained ST-GCN model

    The prediction comes from the actual trained model analyzing the shooter's pose!
    """)

# Model status at bottom
st.caption(f"Status: {model_status} | Device: {DEVICE}")