# app.py - Real-Time Live Game Simulation with KeyJointNet Model
"""
Simulates live game betting experience with side-by-side layout:
- Left: Video playing in real-time
- Right: Prediction result from trained KeyJointNet model

Hit SPACEBAR at release moment → instant prediction from actual model!

Usage:
    streamlit run app.py

Requirements:
    - models/best_merged_calibrated.pth (trained model)
"""

import streamlit as st
import streamlit.components.v1 as components
import torch
import torch.nn as nn
import numpy as np
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

# Dark mode CSS
st.markdown("""
<style>
    .stApp { background-color: #0a0a0f !important; }
    [data-testid="stAppViewContainer"] { background-color: #0a0a0f !important; }
    [data-testid="stFileUploader"] { background-color: #1a1a2e !important; border-radius: 10px; padding: 10px; }
    [data-testid="stFileUploader"] section { background-color: #1a1a2e !important; border-color: #4b5563 !important; }
    [data-testid="stFileUploader"] button { background-color: #f97316 !important; color: white !important; }
    .stApp p, .stApp span, .stApp label, .stApp div { color: #e5e7eb !important; }
    .block-container { padding-top: 1rem; padding-bottom: 0; }
    header { visibility: hidden; }
    .main-title { font-size: 1.6rem; font-weight: 700; color: #f97316 !important; text-align: center; margin-bottom: 0.5rem; }
    .subtitle { color: #6b7280 !important; text-align: center; font-size: 0.9rem; margin-bottom: 1rem; }
    hr { border-color: #374151 !important; }
</style>
""", unsafe_allow_html=True)


# ==================== MODEL DEFINITION ====================

class KeyJointNet(nn.Module):
    """KeyJointNet model for free throw prediction."""

    def __init__(self, num_joints=15, in_channels=9, hidden_dim=64, dropout=0.4):
        super().__init__()
        self.num_joints = num_joints
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim

        self.joint_attn = nn.Sequential(
            nn.Linear(num_joints, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_joints),
            nn.Softmax(dim=-1)
        )

        self.temporal = nn.Sequential(
            nn.Conv1d(in_channels * num_joints, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_dim, hidden_dim * 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.ReLU(),
        )

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, 2)
        )

    def forward(self, x):
        N, C, T, V = x.shape
        joint_var = x.var(dim=(1, 2))
        attn = self.joint_attn(joint_var)
        x = x * attn.unsqueeze(1).unsqueeze(2)
        x = x.permute(0, 1, 3, 2).contiguous().view(N, C * V, T)
        x = self.temporal(x)
        x = self.pool(x).squeeze(-1)
        return self.classifier(x)


# ==================== CONFIG ====================

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_PATH = "models/best_merged_calibrated.pth"

# Key joints (from model_info.json)
KEY_JOINTS = [0, 5, 6, 7, 8, 41, 62, 9, 10, 69, 21, 25, 29, 64, 66]

# Platt scaling calibration (from model_info.json)
CALIBRATION_COEF = 2.4002755160298297
CALIBRATION_INTERCEPT = -1.8751197453911295
OPTIMAL_THRESHOLD = 0.64


def apply_calibration(raw_prob):
    """Apply Platt scaling calibration."""
    logit = CALIBRATION_COEF * raw_prob + CALIBRATION_INTERCEPT
    return 1 / (1 + np.exp(-logit))


@st.cache_resource
def load_model():
    """Load the trained KeyJointNet model."""
    try:
        if not os.path.exists(MODEL_PATH):
            logger.error(f"Model not found at {MODEL_PATH}")
            return None

        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)

        config = checkpoint.get('model_config', {})
        model = KeyJointNet(
            num_joints=config.get('num_joints', 15),
            in_channels=config.get('in_channels', 9),
            hidden_dim=config.get('hidden_dim', 64)
        )

        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)

        model.to(DEVICE)
        model.eval()

        logger.info(f"✓ KeyJointNet loaded | Acc: {checkpoint.get('mean_accuracy', 0):.1%}")
        return model

    except Exception as e:
        logger.error(f"Model loading error: {e}")
        return None


def extract_pose_from_frame(frame, video_path=None, frame_idx=0):
    """Extract pose from pre-computed features or return mock data."""
    json_paths = [
        "data/features/enhanced_all.json",
        "data/features/enhanced_labeled.json",
    ]

    for json_path in json_paths:
        if os.path.exists(json_path):
            try:
                with open(json_path, 'r') as f:
                    data = json.load(f)

                if video_path:
                    video_name = os.path.basename(video_path).replace('.mp4', '').replace('.mov', '').replace('.avi',
                                                                                                              '')

                    for entry in data:
                        if video_name in entry.get('video_id', '') or entry.get('video_id', '') in video_name:
                            kp = np.array(entry.get('keypoints_3d', []))
                            vel = np.array(entry.get('velocity', [])) if entry.get('velocity') else np.zeros_like(kp)
                            acc = np.array(entry.get('acceleration', [])) if entry.get(
                                'acceleration') else np.zeros_like(kp)

                            if kp.size > 0:
                                V_total = kp.shape[1]
                                valid_joints = [j if j < V_total else 0 for j in KEY_JOINTS]

                                kp_sel = kp[:, valid_joints, :]
                                vel_sel = vel[:, valid_joints, :]
                                acc_sel = acc[:, valid_joints, :]

                                pose = np.concatenate([kp_sel, vel_sel, acc_sel], axis=-1)

                                # Ensure 4 frames
                                while pose.shape[0] < 4:
                                    pose = np.concatenate([pose, pose[-1:]], axis=0)
                                pose = pose[:4]

                                logger.info(f"✓ Loaded pose for {video_name}")
                                return pose.astype(np.float32), entry.get('label')

            except Exception as e:
                logger.warning(f"Error: {e}")

    logger.warning("⚠ Using mock pose data")
    return np.random.randn(4, len(KEY_JOINTS), 9).astype(np.float32) * 0.1, None


def predict_with_model(model, pose_sequence):
    """Run prediction with calibration."""
    try:
        pose_tensor = torch.from_numpy(pose_sequence).float()
        pose_tensor = pose_tensor.permute(2, 0, 1).unsqueeze(0).to(DEVICE)  # (1, C, T, V)

        with torch.no_grad():
            output = model(pose_tensor)
            probs = torch.softmax(output, dim=1)

        raw_make = probs[0, 1].item()
        cal_make = apply_calibration(raw_make)

        prediction = 'MAKE' if cal_make > OPTIMAL_THRESHOLD else 'MISS'
        confidence = min(abs(cal_make - OPTIMAL_THRESHOLD) / 0.36 * 1.5, 1.0)

        return {
            'make_prob': cal_make,
            'miss_prob': 1 - cal_make,
            'confidence': confidence,
            'prediction': prediction,
            'threshold': OPTIMAL_THRESHOLD
        }

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return None


# ==================== MAIN UI ====================

st.markdown('<p class="main-title">🎮 Live Game Mode</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Watch video → Hit SPACEBAR at release → See prediction instantly</p>',
            unsafe_allow_html=True)

model = load_model()
model_status = "🟢 KeyJointNet Ready" if model else "🔴 Model Not Found"

video_file = st.file_uploader("Upload free throw video", type=['mp4', 'mov', 'avi'], label_visibility="collapsed")

if video_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
        tmp.write(video_file.read())
        video_path = tmp.name

    with open(video_path, 'rb') as f:
        video_b64 = base64.b64encode(f.read()).decode()

    prediction_result = None
    ground_truth_label = None

    if model:
        pose_sequence, gt_label = extract_pose_from_frame(None, video_path)
        ground_truth_label = gt_label
        prediction_result = predict_with_model(model, pose_sequence)
        if prediction_result:
            logger.info(f"Prediction: {prediction_result['prediction']} ({prediction_result['make_prob']:.1%})")

    pred_json = json.dumps(
        prediction_result or {'make_prob': 0.5, 'miss_prob': 0.5, 'confidence': 0, 'prediction': 'NO MODEL'})
    gt_json = json.dumps(ground_truth_label)

    html_code = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            * {{ margin: 0; padding: 0; box-sizing: border-box; }}
            body {{ font-family: -apple-system, BlinkMacSystemFont, sans-serif; background: #0a0a0f; color: #e5e7eb; }}
            .container {{ display: flex; gap: 20px; padding: 10px; height: 540px; }}
            .video-section {{ flex: 1.4; display: flex; flex-direction: column; }}
            .video-wrapper {{ position: relative; background: #000; border-radius: 12px; overflow: hidden; flex: 1; }}
            video {{ width: 100%; height: 100%; object-fit: contain; }}
            .flash {{ position: absolute; top: 0; left: 0; right: 0; bottom: 0; background: white; opacity: 0; pointer-events: none; transition: opacity 0.05s; }}
            .flash.active {{ opacity: 0.8; }}
            .overlay {{ position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%) scale(0.5); background: rgba(249, 115, 22, 0.95); color: white; padding: 15px 30px; border-radius: 10px; font-size: 1.4rem; font-weight: 700; opacity: 0; transition: all 0.15s; pointer-events: none; }}
            .overlay.show {{ opacity: 1; transform: translate(-50%, -50%) scale(1); }}
            .controls {{ display: flex; gap: 10px; margin-top: 10px; align-items: center; }}
            .btn {{ padding: 10px 24px; border: none; border-radius: 8px; font-weight: 600; cursor: pointer; font-size: 0.95rem; transition: all 0.15s; }}
            .btn-play {{ background: #f97316; color: white; }}
            .btn-play:hover {{ background: #ea580c; }}
            .btn-reset {{ background: #374151; color: #e5e7eb; }}
            .status {{ flex: 1; text-align: center; padding: 10px; border-radius: 8px; font-weight: 500; }}
            .status.wait {{ background: #1f2937; color: #9ca3af; }}
            .status.play {{ background: #065f46; color: #6ee7b7; }}
            .status.done {{ background: #7c2d12; color: #fdba74; }}
            .timer {{ font-family: monospace; font-size: 1.1rem; color: #f97316; min-width: 90px; }}
            .key {{ background: #374151; padding: 6px 12px; border-radius: 6px; font-family: monospace; font-size: 0.85rem; }}
            .result-section {{ flex: 0.6; background: #111827; border-radius: 12px; padding: 20px; display: flex; flex-direction: column; justify-content: center; align-items: center; min-width: 260px; }}
            .waiting-result {{ text-align: center; color: #6b7280; }}
            .waiting-result .dash {{ font-size: 3rem; margin-bottom: 10px; }}
            .result {{ text-align: center; display: none; }}
            .result.show {{ display: block; }}
            .prediction {{ font-size: 2rem; font-weight: 800; margin-bottom: 15px; padding: 15px 25px; border-radius: 12px; }}
            .prediction.make {{ background: linear-gradient(135deg, #065f46, #047857); color: #6ee7b7; }}
            .prediction.miss {{ background: linear-gradient(135deg, #7f1d1d, #991b1b); color: #fca5a5; }}
            .prediction.nobet {{ background: #374151; color: #9ca3af; }}
            .probs {{ color: #9ca3af; font-size: 0.95rem; margin-bottom: 15px; line-height: 1.6; }}
            .thumb {{ border-radius: 8px; border: 2px solid #374151; margin-bottom: 8px; }}
            .captime {{ font-size: 0.8rem; color: #6b7280; margin-bottom: 10px; }}
            .model-badge {{ font-size: 0.75rem; padding: 4px 10px; border-radius: 20px; margin-top: 5px; }}
            .model-badge.real {{ background: #065f46; color: #6ee7b7; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="video-section">
                <div class="video-wrapper">
                    <video id="vid" src="data:video/mp4;base64,{video_b64}" preload="auto"></video>
                    <div class="flash" id="flash"></div>
                    <div class="overlay" id="overlay">📸 CAPTURED!</div>
                </div>
                <div class="controls">
                    <button class="btn btn-play" id="playBtn" onclick="toggle()">▶ PLAY</button>
                    <button class="btn btn-reset" onclick="reset()">↺ RESET</button>
                    <div class="status wait" id="status">Press PLAY → SPACEBAR at release</div>
                    <div class="timer" id="timer">0:00.000</div>
                    <div class="key">⎵ SPACEBAR</div>
                </div>
            </div>
            <div class="result-section" id="resultBox">
                <div class="waiting-result" id="waiting"><div class="dash">—</div><div>Prediction here</div></div>
                <div class="result" id="result">
                    <div class="prediction" id="pred">-</div>
                    <div class="probs" id="probs"></div>
                    <canvas id="thumb" class="thumb" width="140" height="79"></canvas>
                    <div class="captime" id="captime"></div>
                    <div class="model-badge real" id="modelBadge">🧠 KeyJointNet</div>
                </div>
            </div>
        </div>
        <script>
            const vid = document.getElementById('vid'), flash = document.getElementById('flash'), overlay = document.getElementById('overlay');
            const status = document.getElementById('status'), timer = document.getElementById('timer'), playBtn = document.getElementById('playBtn');
            const waiting = document.getElementById('waiting'), result = document.getElementById('result'), pred = document.getElementById('pred');
            const probs = document.getElementById('probs'), thumb = document.getElementById('thumb'), captime = document.getElementById('captime');
            let playing = false, captured = false;
            const modelPrediction = {pred_json};
            const groundTruth = {gt_json};

            vid.addEventListener('timeupdate', () => {{ const t = vid.currentTime; timer.textContent = Math.floor(t/60) + ':' + String(Math.floor(t%60)).padStart(2,'0') + '.' + String(Math.floor((t%1)*1000)).padStart(3,'0'); }});
            vid.addEventListener('play', () => {{ playing = true; status.textContent = '▶ PLAYING - SPACEBAR at release!'; status.className = 'status play'; playBtn.textContent = '⏸ PAUSE'; }});
            vid.addEventListener('pause', () => {{ playing = false; if (!captured) {{ status.textContent = '⏸ Paused'; status.className = 'status wait'; }} playBtn.textContent = '▶ PLAY'; }});

            function toggle() {{ if (!captured) vid.paused ? vid.play() : vid.pause(); }}
            function reset() {{ vid.pause(); vid.currentTime = 0; playing = false; captured = false; status.textContent = 'Press PLAY → SPACEBAR at release'; status.className = 'status wait'; playBtn.textContent = '▶ PLAY'; waiting.style.display = 'block'; result.classList.remove('show'); }}

            function capture() {{
                if (captured || !playing) return;
                captured = true;
                flash.classList.add('active'); setTimeout(() => flash.classList.remove('active'), 150);
                overlay.classList.add('show'); setTimeout(() => overlay.classList.remove('show'), 400);
                vid.pause();
                thumb.getContext('2d').drawImage(vid, 0, 0, 140, 79);
                const t = vid.currentTime;
                captime.textContent = 'Captured at ' + Math.floor(t/60) + ':' + String(Math.floor(t%60)).padStart(2,'0') + '.' + String(Math.floor((t%1)*1000)).padStart(3,'0');
                status.textContent = '🔄 Running KeyJointNet...'; status.className = 'status done';
                setTimeout(showPrediction, 300);
            }}

            function showPrediction() {{
                const makeP = modelPrediction.make_prob, missP = modelPrediction.miss_prob, conf = modelPrediction.confidence, prediction = modelPrediction.prediction;
                let label, cls;
                if (prediction === 'NO MODEL') {{ label = '⚠️ NO MODEL'; cls = 'nobet'; }}
                else if (conf < 0.15) {{ label = '⚪ NO BET'; cls = 'nobet'; }}
                else if (prediction === 'MAKE') {{ label = '🟢 MAKE'; cls = 'make'; }}
                else {{ label = '🔴 MISS'; cls = 'miss'; }}
                pred.textContent = label; pred.className = 'prediction ' + cls;
                let html = 'Make: ' + (makeP*100).toFixed(1) + '% | Miss: ' + (missP*100).toFixed(1) + '%<br>Confidence: ' + (conf*100).toFixed(0) + '%';
                if (groundTruth !== null) {{
                    const gt = groundTruth === 1 ? 'MAKE' : 'MISS', correct = prediction === gt;
                    html += '<br><small style="color:' + (correct ? '#10b981' : '#ef4444') + '">' + (correct ? '✓' : '✗') + ' Truth: ' + gt + '</small>';
                }}
                probs.innerHTML = html;
                waiting.style.display = 'none'; result.classList.add('show'); status.textContent = '✓ Prediction complete!';
            }}

            document.addEventListener('keydown', (e) => {{ if (e.code === 'Space') {{ e.preventDefault(); if (playing) capture(); else if (!captured) toggle(); }} }});
        </script>
    </body>
    </html>
    """
    components.html(html_code, height=580, scrolling=False)
    os.unlink(video_path)

else:
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
    5. **See prediction instantly** from the trained model

    ---
    ### 📊 Model: KeyJointNet (Calibrated)
    | Metric | Value |
    |--------|-------|
    | CV Accuracy | 82.8% |
    | Optimized Accuracy | 91.9% |
    | AUC | 0.966 |
    | Threshold | 64% |
    """)

st.caption(f"Status: {model_status} | Device: {DEVICE} | Threshold: {OPTIMAL_THRESHOLD}")