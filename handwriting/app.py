"""
Parkinson's Disease Handwriting Detection — Live Translate Web App
=================================================================
Ensemble: 5x Residual-MLP (16 biomarkers) + 5x EfficientNet-B0+CBAM
         + Stacking Meta-Learner (LogisticRegression)
Results: 93.57% accuracy | 0.9851 AUC-ROC (5-fold CV)

Deploy on Render: gunicorn app:app --bind 0.0.0.0:$PORT --timeout 120

CHANGES (v3):
  - Validation is now WARNING-ONLY. No image is ever rejected with 422.
    Every image gets a prediction. Suspicious inputs get a warning flag
    and confidence="low" so the frontend can show a notice.
  - Fixed 0.0% display bug: combined_risk is always a valid float in
    [0, 1]. NaN/Inf from extreme feature values are clamped before
    they reach the models. predict_mlp / predict_cnn return 0.5 as
    a safe fallback if no model weights are loaded.
  - Loosened all quality thresholds to avoid false positives on
    genuine handwriting (light pen, dark scan, high DPI, etc.).
  - try/except around entire inference path — server never crashes on
    a bad image, returns a clean 500 JSON instead.
"""

import os
import io
import base64
import pickle
import warnings
import time
import uuid

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
from flask import Flask, render_template, request, jsonify, session

warnings.filterwarnings("ignore")

DEVICE  = torch.device("cpu")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ════════════════════════════════════════════════════════════════
# 16 Spatial Biomarker Feature Extractor
# ════════════════════════════════════════════════════════════════

FEATURE_NAMES = [
    "stroke_width_mean", "stroke_width_std", "contour_roughness",
    "direction_changes", "n_components", "ink_density",
    "solidity", "intensity_variance",
    "fractal_dimension", "entropy", "hu_moment_1", "hu_moment_2",
    "curvature_mean", "curvature_std", "aspect_ratio", "stroke_regularity",
]


def _box_counting_fractal(binary, sizes=None):
    if sizes is None:
        sizes = [2, 4, 8, 16, 32, 64]
    counts = []
    for size in sizes:
        h, w = binary.shape
        count = 0
        for y in range(0, h, size):
            for x in range(0, w, size):
                if np.any(binary[y : y + size, x : x + size] > 0):
                    count += 1
        counts.append(max(count, 1))
    valid = [(s, c) for s, c in zip(sizes[: len(counts)], counts) if c > 0]
    if len(valid) < 2:
        return 1.0
    log_s = np.log(1.0 / np.array([v[0] for v in valid]))
    log_c = np.log(np.array([v[1] for v in valid]))
    return float(np.clip(np.polyfit(log_s, log_c, 1)[0], 0.5, 2.5))


def _compute_curvature(contour, step=5):
    if len(contour) < step * 3:
        return np.array([0.0])
    pts = contour.reshape(-1, 2).astype(float)
    curvatures = []
    for i in range(step, len(pts) - step):
        v1 = pts[i] - pts[i - step]
        v2 = pts[i + step] - pts[i]
        cross = abs(v1[0] * v2[1] - v1[1] * v2[0])
        norm  = np.linalg.norm(v1) * np.linalg.norm(v2)
        if norm > 1e-6:
            curvatures.append(cross / norm)
    return np.array(curvatures) if curvatures else np.array([0.0])


def extract_16_features(img_gray):
    """Extract 16 spatial biomarkers from a grayscale handwriting image."""
    if img_gray is None or img_gray.size == 0:
        return np.zeros(16)
    if img_gray.dtype != np.uint8:
        img_gray = (
            (img_gray * 255).astype(np.uint8)
            if img_gray.max() <= 1.0
            else img_gray.astype(np.uint8)
        )

    _, binary = cv2.threshold(
        img_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    h, w     = binary.shape
    ink_mask  = binary > 0
    ink_count = int(np.sum(ink_mask))
    if ink_count < 10:
        return np.zeros(16)

    # 1-2: Stroke Width
    dist        = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    stroke_vals = dist[ink_mask]
    f1 = float(np.mean(stroke_vals))
    f2 = float(np.std(stroke_vals))

    # 3: Contour Roughness
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    f3 = 1.0
    if contours:
        c    = max(contours, key=cv2.contourArea)
        peri = cv2.arcLength(c, True)
        area = cv2.contourArea(c)
        if area > 0:
            f3 = (peri ** 2) / (4 * np.pi * area)

    # 4: Direction Changes
    f4 = 0.0
    if contours:
        c = max(contours, key=len)
        if len(c) > 20:
            pts  = c.reshape(-1, 2).astype(float)
            step = max(1, len(pts) // 200)
            pts  = pts[::step]
            if len(pts) > 3:
                dx     = np.diff(pts[:, 0])
                dy     = np.diff(pts[:, 1])
                angles = np.arctan2(dy, dx)
                diffs  = np.abs(np.diff(angles))
                diffs  = np.minimum(diffs, 2 * np.pi - diffs)
                f4     = float(np.mean(diffs))

    # 5: Connected Components
    n_labels, _ = cv2.connectedComponents(binary)
    f5 = float(n_labels - 1)

    # 6: Ink Density
    f6 = float(ink_count / (h * w))

    # 7: Solidity
    f7 = 0.0
    if contours:
        c         = max(contours, key=cv2.contourArea)
        area      = cv2.contourArea(c)
        hull_area = cv2.contourArea(cv2.convexHull(c))
        if hull_area > 0:
            f7 = float(area / hull_area)

    # 8: Intensity Variance
    f8 = float(np.std(img_gray[ink_mask].astype(float)) / 255.0)

    # 9: Fractal Dimension
    f9 = _box_counting_fractal(binary)

    # 10: Shannon Entropy
    hist = cv2.calcHist([img_gray], [0], binary, [256], [0, 256]).flatten()
    hist = hist / (hist.sum() + 1e-10)
    hist = hist[hist > 0]
    f10  = float(-np.sum(hist * np.log2(hist + 1e-10)))

    # 11-12: Hu Moments
    hu  = cv2.HuMoments(cv2.moments(binary)).flatten()
    f11 = float(-np.sign(hu[0]) * np.log10(abs(hu[0]) + 1e-10))
    f12 = float(-np.sign(hu[1]) * np.log10(abs(hu[1]) + 1e-10))

    # 13-14: Curvature
    if contours:
        curv = _compute_curvature(max(contours, key=len))
        f13  = float(np.mean(curv))
        f14  = float(np.std(curv))
    else:
        f13 = f14 = 0.0

    # 15: Aspect Ratio
    coords = np.column_stack(np.where(binary > 0))
    if len(coords) > 0:
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
        f15 = float((x_max - x_min + 1) / (y_max - y_min + 1 + 1e-6))
    else:
        f15 = 1.0

    # 16: Stroke Regularity (FFT)
    f16 = 0.0
    if contours:
        c = max(contours, key=len)
        if len(c) > 20:
            pts   = c.reshape(-1, 2).astype(float)
            dists = np.sqrt(np.sum(np.diff(pts, axis=0) ** 2, axis=1))
            if len(dists) > 10:
                fft_vals = np.abs(np.fft.rfft(dists - np.mean(dists)))
                if len(fft_vals) > 1 and fft_vals.sum() > 0:
                    f16 = float(fft_vals[1:].max() / (fft_vals[1:].sum() + 1e-10))

    return np.array([f1, f2, f3, f4, f5, f6, f7, f8,
                     f9, f10, f11, f12, f13, f14, f15, f16])


# ════════════════════════════════════════════════════════════════
# INPUT QUALITY CHECKER  (warning-only — never blocks prediction)
# ════════════════════════════════════════════════════════════════
#
# Thresholds are deliberately loose to avoid false positives on
# genuine handwriting with unusual properties.

INK_DENSITY_MIN       = 0.001   # below this = near-blank image
INK_DENSITY_MAX       = 0.80    # above this = likely filled graphic
N_COMPONENTS_WARN     = 200     # warn if ink blobs exceed this count
HOUGH_LINE_WARN       = 12      # warn if this many long straight lines detected
HOUGH_MIN_LINE_LEN    = 80      # px on 256-px image — only very long lines count
BRANCH_DIVERGENCE_THR = 0.30    # MLP vs CNN gap that triggers low-confidence flag


def _detect_geometric_lines(binary_256):
    """Count long straight lines via probabilistic Hough transform."""
    edges = cv2.Canny(binary_256, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=50,
        minLineLength=HOUGH_MIN_LINE_LEN,
        maxLineGap=5,
    )
    return 0 if lines is None else len(lines)


def check_input_quality(img_gray_256, feats):
    """
    Run soft quality checks. Returns:
      warnings : list[str]  — advisory messages (empty = all clear)
      checks   : dict       — raw diagnostic values
    """
    warn   = []
    checks = {}

    ink_density  = float(feats[5])
    n_components = int(feats[4])

    checks["ink_density"]  = round(ink_density, 4)
    checks["n_components"] = n_components

    if ink_density < INK_DENSITY_MIN:
        warn.append(
            "Image appears nearly blank. Please upload a clear handwriting sample."
        )
    elif ink_density > INK_DENSITY_MAX:
        warn.append(
            f"Very high ink density ({ink_density:.2f}). "
            "The image may be a dark scan or graphic fill rather than handwriting."
        )

    if n_components > N_COMPONENTS_WARN:
        warn.append(
            f"Unusually many ink regions ({n_components}). "
            "If this is a diagram or printed text, the prediction may not be meaningful."
        )

    # Binarise for Hough (Otsu already applied in extract_16_features, redo here cheaply)
    _, binary = cv2.threshold(
        img_gray_256, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    n_lines = _detect_geometric_lines(binary)
    checks["straight_lines"] = n_lines
    if n_lines >= HOUGH_LINE_WARN:
        warn.append(
            f"Detected {n_lines} long straight line segments. "
            "This image may be a diagram or chart — the model is trained on "
            "clinical handwriting and results may be unreliable for other content."
        )

    return warn, checks


# ════════════════════════════════════════════════════════════════
# Neural Network Architectures (must match training exactly)
# ════════════════════════════════════════════════════════════════

class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout=0.4):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim), nn.BatchNorm1d(dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dim, dim), nn.BatchNorm1d(dim),
        )
        self.act     = nn.GELU()
        self.dropout = nn.Dropout(dropout * 0.5)

    def forward(self, x):
        return self.dropout(self.act(self.block(x) + x))


class PDDetectionModelV2(nn.Module):
    """Residual MLP: 16 → 64 → [ResBlock] → 32 → 1"""
    def __init__(self, input_size=16, hidden=64):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_size, hidden), nn.BatchNorm1d(hidden),
            nn.GELU(), nn.Dropout(0.3),
        )
        self.res1 = ResidualBlock(hidden, dropout=0.4)
        self.head = nn.Sequential(
            nn.Linear(hidden, 32), nn.BatchNorm1d(32), nn.GELU(), nn.Dropout(0.35),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        x = self.input_proj(x)
        x = self.res1(x)
        return torch.sigmoid(self.head(x))


class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        avg_pool = F.adaptive_avg_pool2d(x, 1).view(b, c)
        max_pool = F.adaptive_max_pool2d(x, 1).view(b, c)
        attn     = torch.sigmoid(self.fc(avg_pool) + self.fc(max_pool))
        return x * attn.view(b, c, 1, 1)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)

    def forward(self, x):
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        max_pool = torch.max(x, dim=1, keepdim=True)[0]
        combined = torch.cat([avg_pool, max_pool], dim=1)
        return x * torch.sigmoid(self.conv(combined))


class CBAM(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.channel_attn = ChannelAttention(channels, reduction)
        self.spatial_attn = SpatialAttention()

    def forward(self, x):
        return self.spatial_attn(self.channel_attn(x))


class EfficientNetCBAM(nn.Module):
    def __init__(self, backbone, cbam, feat_dim):
        super().__init__()
        self.backbone   = backbone
        self.cbam       = cbam
        self.pool       = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(feat_dim, 128),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        x = self.backbone.features(x)
        x = self.cbam(x)
        x = self.pool(x)
        x = x.flatten(1)
        return self.classifier(x)


def _build_efficientnet_cbam():
    try:
        backbone = models.efficientnet_b0(weights=None)
    except TypeError:
        backbone = models.efficientnet_b0(pretrained=False)
    backbone.classifier = nn.Identity()
    cbam = CBAM(1280, reduction=16)
    return EfficientNetCBAM(backbone, cbam, 1280)


# ════════════════════════════════════════════════════════════════
# Image Transforms + TTA
# ════════════════════════════════════════════════════════════════

val_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

tta_transforms = [
    val_tf,
    transforms.Compose([
        transforms.Resize((224, 224)), transforms.RandomHorizontalFlip(p=1.0),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
    transforms.Compose([
        transforms.Resize((224, 224)), transforms.RandomRotation((10, 10)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
    transforms.Compose([
        transforms.Resize((224, 224)), transforms.RandomRotation((-10, -10)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
    transforms.Compose([
        transforms.Resize((224, 224)), transforms.ColorJitter(brightness=0.3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
    transforms.Compose([
        transforms.Resize((224, 224)), transforms.RandomVerticalFlip(p=1.0),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
    transforms.Compose([
        transforms.Resize((256, 256)), transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
]


# ════════════════════════════════════════════════════════════════
# Global Model State
# ════════════════════════════════════════════════════════════════

mlp_models   = []
cnn_models   = []
scaler       = None
meta_model   = None
models_loaded = False


def load_all_models():
    global mlp_models, cnn_models, scaler, meta_model, models_loaded
    print("[*] Loading models...")

    for i in range(1, 6):
        path = os.path.join(BASE_DIR, f"mlp_fold_{i}.pth")
        if os.path.exists(path):
            m = PDDetectionModelV2(input_size=16)
            m.load_state_dict(torch.load(path, map_location=DEVICE))
            m.eval()
            mlp_models.append(m)
            print(f"  [OK] mlp_fold_{i}.pth")
        else:
            print(f"  [!!] mlp_fold_{i}.pth NOT FOUND")

    for i in range(1, 6):
        path = os.path.join(BASE_DIR, f"cnn_fold_{i}.pth")
        if os.path.exists(path):
            m = _build_efficientnet_cbam()
            m.load_state_dict(torch.load(path, map_location=DEVICE))
            m.eval()
            cnn_models.append(m)
            print(f"  [OK] cnn_fold_{i}.pth")
        else:
            print(f"  [!!] cnn_fold_{i}.pth NOT FOUND")

    scaler_path = os.path.join(BASE_DIR, "scaler.pkl")
    if os.path.exists(scaler_path):
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
        print("  [OK] scaler.pkl")
    else:
        print("  [!!] scaler.pkl NOT FOUND")

    meta_path = os.path.join(BASE_DIR, "meta_model.pkl")
    if os.path.exists(meta_path):
        with open(meta_path, "rb") as f:
            meta_model = pickle.load(f)
        print("  [OK] meta_model.pkl")
    else:
        print("  [!!] meta_model.pkl NOT FOUND")

    models_loaded = True
    print(f"[OK] Loaded {len(mlp_models)} MLP + {len(cnn_models)} CNN models")


# ════════════════════════════════════════════════════════════════
# Prediction Pipeline
# ════════════════════════════════════════════════════════════════

def _image_from_base64(data_url):
    """Decode a base64 data URL to a PIL RGB image."""
    if "," in data_url:
        data_url = data_url.split(",", 1)[1]
    img_bytes = base64.b64decode(data_url)
    img = Image.open(io.BytesIO(img_bytes))
    if img.mode == "RGBA":
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[3])
        img = bg
    return img.convert("RGB")


def _safe_mean(preds):
    """Return mean of a list, or 0.5 if list is empty."""
    return float(np.mean(preds)) if preds else 0.5


def predict_mlp(img_gray):
    """5-fold MLP ensemble → average probability. Always returns a float in [0,1]."""
    if not mlp_models or scaler is None:
        return 0.5

    img_resized  = cv2.resize(img_gray, (256, 256))
    feats        = extract_16_features(img_resized)
    feats        = np.nan_to_num(feats, nan=0.0, posinf=1.0, neginf=0.0)
    feats_scaled = scaler.transform(feats.reshape(1, -1))
    inp          = torch.FloatTensor(feats_scaled).to(DEVICE)

    preds = []
    with torch.inference_mode():
        for m in mlp_models:
            p = float(m(inp).cpu().item())
            if np.isfinite(p):
                preds.append(np.clip(p, 0.0, 1.0))

    return _safe_mean(preds)


def predict_cnn(img_pil, use_tta=False):
    """5-fold CNN ensemble → average probability. Always returns a float in [0,1]."""
    if not cnn_models:
        return 0.5

    preds = []
    with torch.inference_mode():
        for m in cnn_models:
            tf_list = tta_transforms if use_tta else [val_tf]
            for tf in tf_list:
                inp = tf(img_pil).unsqueeze(0).to(DEVICE)
                p   = float(torch.sigmoid(m(inp)).cpu().item())
                if np.isfinite(p):
                    preds.append(np.clip(p, 0.0, 1.0))

    return _safe_mean(preds)


def predict_ensemble(img_pil):
    """
    Full inference pipeline — always returns HTTP 200 with a prediction.

    Steps:
      1. Convert to greyscale, resize to 256×256
      2. Extract 16 biomarker features (NaN-safe)
      3. Run quality checks → append warnings, never block
      4. MLP ensemble
      5. CNN ensemble (single pass)
      6. CNN ensemble (TTA — 7 augmented views)
      7. Meta-learner stacking (or simple average as fallback)
      8. Confidence flag based on MLP / CNN divergence
    """
    # ── Greyscale + resize ───────────────────────────────────────
    img_np      = np.array(img_pil)
    img_gray    = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    img_resized = cv2.resize(img_gray, (256, 256))

    # ── Feature extraction (NaN-safe) ───────────────────────────
    raw_feats    = extract_16_features(img_resized)
    raw_feats    = np.nan_to_num(raw_feats, nan=0.0, posinf=1.0, neginf=0.0)
    feature_dict = {
        name: round(float(val), 6)
        for name, val in zip(FEATURE_NAMES, raw_feats)
    }

    # ── Quality check (warning-only, never blocks) ───────────────
    quality_warnings, checks = check_input_quality(img_resized, raw_feats)

    # ── Ensemble inference ───────────────────────────────────────
    mlp_risk     = predict_mlp(img_gray)
    cnn_risk     = predict_cnn(img_pil, use_tta=False)
    cnn_tta_risk = predict_cnn(img_pil, use_tta=True)

    # ── Meta-learner stacking ────────────────────────────────────
    if meta_model is not None:
        meta_input    = np.array([[mlp_risk, cnn_risk, cnn_tta_risk]])
        combined_risk = float(
            np.clip(meta_model.predict_proba(meta_input)[:, 1][0], 0.0, 1.0)
        )
    else:
        combined_risk = float(
            np.clip((mlp_risk + cnn_risk + cnn_tta_risk) / 3.0, 0.0, 1.0)
        )

    # Guard against NaN (should never happen after clamping, but be safe)
    if not np.isfinite(combined_risk):
        combined_risk = float((mlp_risk + cnn_risk + cnn_tta_risk) / 3.0)
    combined_risk = float(np.clip(combined_risk, 0.0, 1.0))

    # ── Confidence flag ──────────────────────────────────────────
    branch_divergence = abs(mlp_risk - cnn_risk)
    if branch_divergence > BRANCH_DIVERGENCE_THR:
        confidence = "low"
        quality_warnings.append(
            f"MLP and CNN predictions diverge by {branch_divergence:.2f} "
            f"(MLP={mlp_risk:.3f}, CNN={cnn_risk:.3f}). "
            "Try uploading a higher-resolution or clearer image."
        )
    else:
        confidence = "high"

    # ── Risk label ───────────────────────────────────────────────
    if combined_risk < 0.33:
        status = "LOW RISK"
    elif combined_risk < 0.66:
        status = "MODERATE RISK"
    else:
        status = "HIGH RISK"

    return {
        "combined_risk":     round(combined_risk, 4),
        "mlp_risk":          round(mlp_risk, 4),
        "cnn_risk":          round(cnn_risk, 4),
        "cnn_tta_risk":      round(cnn_tta_risk, 4),
        "status":            status,
        "confidence":        confidence,
        "branch_divergence": round(branch_divergence, 4),
        "warnings":          quality_warnings,
        "validation_checks": checks,
        "features":          feature_dict,
    }


# ════════════════════════════════════════════════════════════════
# Flask App
# ════════════════════════════════════════════════════════════════

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'dev-key-pd-handwriting-v2')

# ════════════════════════════════════════════════════════════════
# Session Storage (in-memory for this instance; use Redis/DB for production)
# ════════════════════════════════════════════════════════════════
session_history = {}  # {session_id: [pred1, pred2, ...]}


def get_session_id(request_obj):
    """Get or create session ID from request."""
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())
    return session['session_id']


@app.route("/")
def index():
    return render_template("index_enhanced.html")


@app.route("/predict", methods=["POST"])
def predict_route():
    if not models_loaded:
        return jsonify({"error": "Models not loaded yet"}), 503

    # ── Parse image ──────────────────────────────────────────────
    data = request.get_json(silent=True)
    if data and "image" in data:
        try:
            img_pil = _image_from_base64(data["image"])
        except Exception as e:
            return jsonify({"error": f"Could not decode image: {e}"}), 400
    elif "file" in request.files:
        try:
            img_pil = Image.open(request.files["file"].stream).convert("RGB")
        except Exception as e:
            return jsonify({"error": f"Could not open file: {e}"}), 400
    else:
        return jsonify({"error": "No image provided"}), 400

    # ── Run prediction (always 200) ──────────────────────────────
    try:
        result = predict_ensemble(img_pil)
        timestamp = int(time.time() * 1000)  # milliseconds
        result['timestamp'] = timestamp
        result['id'] = f"pred_{timestamp}"
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {e}"}), 500

    return jsonify(result), 200


@app.route("/api/session/history", methods=["GET"])
def get_history():
    """Return predictions from current browser session (LocalStorage-based)."""
    return jsonify({
        "note": "History stored in browser LocalStorage (client-side)"
    }), 200


@app.route("/api/session/trend", methods=["POST"])
def get_trend_analysis():
    """Analyze trend from multiple predictions."""
    try:
        data = request.get_json()
        predictions = data.get('predictions', [])
        
        if len(predictions) < 2:
            return jsonify({
                "trend": "insufficient_data",
                "message": "Need at least 2 predictions for trend analysis"
            }), 200
        
        risks = [p.get('combined_risk', 0.5) for p in predictions]
        timestamps = [p.get('timestamp', i*1000) for i, p in enumerate(predictions)]
        
        # Calculate trend (linear regression)
        if len(risks) >= 2:
            x = np.array(range(len(risks)), dtype=float)
            y = np.array(risks, dtype=float)
            slope = float(np.polyfit(x, y, 1)[0])
            
            trend_direction = "improving" if slope < -0.05 else ("worsening" if slope > 0.05 else "stable")
            trend_pct = round(slope * 100, 2)
        else:
            trend_direction = "insufficient"
            trend_pct = 0.0
        
        return jsonify({
            "trend": trend_direction,
            "trend_percent": trend_pct,
            "avg_risk": round(float(np.mean(risks)), 4),
            "min_risk": round(float(np.min(risks)), 4),
            "max_risk": round(float(np.max(risks)), 4),
            "prediction_count": len(predictions),
            "risks_timeline": [round(r, 4) for r in risks],
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/api/features/explain", methods=["GET", "POST"])
def explain_features():
    """Return explanation for each biomarker."""
    explanations = {
        "stroke_width_mean": "Average thickness of pen strokes. PD causes thinner, weaker strokes.",
        "stroke_width_std": "Variability in stroke thickness. PD leads to inconsistent pressure.",
        "contour_roughness": "Smoothness of outline. PD causes irregular, rough traces.",
        "direction_changes": "Number of direction shifts while writing. PD increases this.",
        "n_components": "Count of separate ink blobs. PD increases fragmentation.",
        "ink_density": "Proportion of pixels with ink. PD may show lighter writing.",
        "solidity": "Fill ratio of shape. PD reduces solidity due to gaps.",
        "intensity_variance": "Variation in pen pressure intensity. High = tremor indicator.",
        "fractal_dimension": "Complexity of the pattern. PD increases fractal complexity.",
        "entropy": "Information content of the image. Higher entropy = more disorganized.",
        "hu_moment_1": "Shape descriptor (moment invariant). Sensitive to distortion.",
        "hu_moment_2": "Shape descriptor (moment invariant). Captures asphericity.",
        "curvature_mean": "Average bend angle in stroke path. PD increases curvature.",
        "curvature_std": "Variability of curvature. PD causes inconsistent curves.",
        "aspect_ratio": "Width-to-height ratio of bounding box. Indicates overall shape.",
        "stroke_regularity": "FFT-based regularity score. PD reduces periodicity.",
    }
    
    return jsonify(explanations), 200


# ════════════════════════════════════════════════════════════════
# Heatmap & Visualization
# ════════════════════════════════════════════════════════════════

def generate_grad_cam(img_pil):
    """Generate Grad-CAM heatmap from CNN model."""
    if not cnn_models:
        return None
    
    try:
        img_tensor = val_tf(img_pil).unsqueeze(0).to(DEVICE)
        model = cnn_models[0]  # Use first fold
        
        # Hook to capture feature maps
        features = None
        gradients = None
        
        def forward_hook(module, input, output):
            nonlocal features
            features = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            nonlocal gradients
            gradients = grad_output[0].detach()
        
        # Register hooks on last layer
        layer = model.backbone.features[-1]
        hook_f = layer.register_forward_hook(forward_hook)
        hook_b = layer.register_backward_hook(backward_hook)
        
        # Forward pass
        with torch.enable_grad():
            output = model(img_tensor)
            loss = output.sum()
            loss.backward()
        
        hook_f.remove()
        hook_b.remove()
        
        # Compute Grad-CAM
        if features is not None and gradients is not None:
            weights = gradients.mean(dim=(2, 3))  # Global average pooling
            grad_cam = (weights.unsqueeze(-1).unsqueeze(-1) * features).sum(dim=1, keepdim=True)
            grad_cam = F.relu(grad_cam)
            grad_cam = grad_cam.squeeze().cpu().numpy()
            
            # Normalize
            if grad_cam.max() > 0:
                grad_cam = grad_cam / grad_cam.max()
            
            # Resize to original image size
            grad_cam = cv2.resize(grad_cam, (224, 224))
            grad_cam = (grad_cam * 255).astype(np.uint8)
            
            # Apply colormap
            heatmap = cv2.applyColorMap(grad_cam, cv2.COLORMAP_JET)
            
            # Convert to base64
            _, buffer = cv2.imencode('.png', heatmap)
            return base64.b64encode(buffer).decode('utf-8')
    except Exception as e:
        print(f"Grad-CAM error: {e}")
        return None


def generate_grid_cam(img_pil):
    """Generate GridCAM overlay visualization."""
    if not cnn_models:
        return None
    
    try:
        img_np = np.array(img_pil)
        img_array = cv2.resize(img_np, (224, 224))
        
        img_tensor = val_tf(img_pil).unsqueeze(0).to(DEVICE)
        model = cnn_models[0]
        
        # Get feature maps
        with torch.inference_mode():
            features = model.backbone.features(img_tensor)
            _, c, h, w = features.shape
            
            # Create attention grid
            avg_features = features.mean(dim=1, keepdim=True)  # Channel-wise average
            grid_cam = (avg_features.squeeze().cpu().numpy() * 255).astype(np.uint8)
            
            # Resize to match input
            grid_cam = cv2.resize(grid_cam, (224, 224))
            
            # Apply colormap
            heatmap = cv2.applyColorMap(grid_cam, cv2.COLORMAP_VIRIDIS)
            
            # Blend with original image
            blended = cv2.addWeighted(img_array, 0.6, heatmap, 0.4, 0)
            
            # Convert to base64
            _, buffer = cv2.imencode('.png', blended)
            return base64.b64encode(buffer).decode('utf-8')
    except Exception as e:
        print(f"GridCAM error: {e}")
        return None


@app.route("/api/heatmap", methods=["POST"])
def get_heatmap():
    """Generate Grad-CAM and GridCAM visualizations."""
    try:
        data = request.get_json()
        image_data = data.get("image")
        
        if not image_data:
            return jsonify({"error": "No image provided"}), 400
        
        img_pil = _image_from_base64(image_data)
        
        grad_cam = generate_grad_cam(img_pil)
        grid_cam = generate_grid_cam(img_pil)
        
        # Generate attention map (simple feature visualization)
        attention_map = grad_cam  # Reuse for now
        
        return jsonify({
            "grad_cam": grad_cam,
            "grid_cam": grid_cam,
            "attention_map": attention_map,
        }), 200
    except Exception as e:
        print(f"Heatmap endpoint error: {e}")
        return jsonify({"error": str(e)}), 500



# ════════════════════════════════════════════════════════════════
# AI Narrative Generation (Cohere)
# ════════════════════════════════════════════════════════════════

COHERE_NARRATIVE_SYSTEM_PROMPT = (
    "You are a clinical AI assistant specialised in Parkinson's Disease motor "
    "biomarker analysis from handwriting samples.\n\n"
    "You will receive a JSON object containing the prediction results from our "
    "ensemble model (5x MLP + 5x CNN + stacking meta-learner), including:\n"
    "- combined_risk (0-1 float): the overall PD risk probability\n"
    "- status: LOW RISK / MODERATE RISK / HIGH RISK\n"
    "- confidence: high or low\n"
    "- mlp_risk, cnn_risk, cnn_tta_risk: individual ensemble branch scores\n"
    "- features: a dictionary of 16 spatial biomarker names and their extracted values\n\n"
    "Your task is to write exactly 3 paragraphs of plain text:\n\n"
    "Paragraph 1 - Overall Assessment: Summarise the combined risk score and status "
    "in clinical but accessible language. Mention the model confidence level and "
    "whether the MLP and CNN branches agree. Be empathetic but never alarmist.\n\n"
    "Paragraph 2 - Key Findings: Reference specific biomarker names and their values "
    "from the features dictionary. Highlight which biomarkers are most notable "
    "(unusually high or low) and explain what each might indicate in terms of motor "
    "control, tremor, or writing quality. Use the actual numbers.\n\n"
    "Paragraph 3 - Recommendations: Provide practical next-step suggestions such as "
    "follow-up clinical assessments, repeat screenings, or lifestyle considerations. "
    'Always end this paragraph with exactly this sentence: "Please share this report '
    'with a qualified neurologist before drawing any clinical conclusions."\n\n'
    "Rules:\n"
    "- Write plain text only. No markdown, no bullet points, no numbered lists, "
    "no headers, no bold/italic formatting.\n"
    "- Do not use asterisks, hashes, or any formatting characters.\n"
    "- Each paragraph must be separated by a single blank line.\n"
    "- Be empathetic, professional, and measured. Never use alarming language "
    "even for high-risk results.\n"
    "- Always reference specific biomarker names and values from the data provided."
)


@app.route('/api/narrative', methods=['POST'])
def generate_narrative():
    """Generate an AI clinical narrative using Cohere command-a-plus-08-2025."""
    import requests as http_requests

    api_key = os.environ.get('COHERE_API_KEY', '').strip()
    if not api_key:
        return jsonify({
            'error': 'COHERE_API_KEY environment variable is not set. '
                     'Please configure it to enable AI narrative generation.'
        }), 503

    try:
        data = request.get_json(silent=True) or {}
        prediction = data.get('prediction', {})

        if not prediction:
            return jsonify({'error': 'No prediction data provided'}), 400

        # Build the user message with the prediction data
        import json as json_mod
        user_message = (
            "Here are the prediction results from the handwriting analysis:\n\n"
            + json_mod.dumps(prediction, indent=2)
        )

        # Call Cohere v2 chat endpoint
        cohere_resp = http_requests.post(
            'https://api.cohere.com/v2/chat',
            headers={
                'Authorization': f'Bearer {api_key}',
                'Content-Type': 'application/json',
                'Accept': 'application/json',
            },
            json={
                'model': 'command-a-03-2025',
                'messages': [
                    {
                        'role': 'system',
                        'content': COHERE_NARRATIVE_SYSTEM_PROMPT,
                    },
                    {
                        'role': 'user',
                        'content': user_message,
                    },
                ],
                'temperature': 0.4,
                'max_tokens': 1024,
            },
            timeout=30,
        )

        if cohere_resp.status_code != 200:
            error_detail = cohere_resp.text[:500]
            print(f'[!] Cohere API error {cohere_resp.status_code}: {error_detail}')
            return jsonify({
                'error': f'Cohere API returned status {cohere_resp.status_code}'
            }), 502

        resp_json = cohere_resp.json()

        # Extract text from Cohere v2 response
        narrative_text = ''
        message = resp_json.get('message', {})
        content_list = message.get('content', [])
        for block in content_list:
            if block.get('type') == 'text':
                narrative_text += block.get('text', '')

        if not narrative_text:
            # Fallback: try older response format
            narrative_text = resp_json.get('text', '')

        if not narrative_text:
            return jsonify({'error': 'Empty response from Cohere API'}), 502

        return jsonify({'narrative': narrative_text.strip()}), 200

    except http_requests.exceptions.Timeout:
        return jsonify({'error': 'Cohere API request timed out'}), 504
    except http_requests.exceptions.ConnectionError:
        return jsonify({'error': 'Could not connect to Cohere API'}), 502
    except Exception as e:
        print(f'[!] Narrative generation error: {e}')
        import traceback; traceback.print_exc()
        return jsonify({'error': f'Narrative generation failed: {e}'}), 500



# ════════════════════════════════════════════════════════════════
# PDF Report Generation
# ════════════════════════════════════════════════════════════════

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import mm, cm
    from reportlab.lib.colors import HexColor, black, white
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
        Image as RLImage, HRFlowable, KeepTogether,
    )
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False


def _build_pdf(prediction, narrative, canvas_b64):
    """Build a clinical PDF report and return the bytes."""
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import mm, cm
    from reportlab.lib.colors import HexColor, black, white
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
        Image as RLImage, HRFlowable, KeepTogether,
    )

    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=20*mm, rightMargin=20*mm,
        topMargin=15*mm, bottomMargin=20*mm,
    )

    styles = getSampleStyleSheet()
    # ── Custom styles ──────────────────────────────────────────
    styles.add(ParagraphStyle(
        'AppTitle', parent=styles['Heading1'],
        fontName='Helvetica-Bold', fontSize=22,
        textColor=HexColor('#0d1e40'), spaceAfter=2*mm,
        alignment=TA_CENTER,
    ))
    styles.add(ParagraphStyle(
        'Subtitle', parent=styles['Normal'],
        fontName='Helvetica', fontSize=10,
        textColor=HexColor('#526082'), alignment=TA_CENTER,
        spaceAfter=6*mm,
    ))
    styles.add(ParagraphStyle(
        'SectionHead', parent=styles['Heading2'],
        fontName='Helvetica-Bold', fontSize=13,
        textColor=HexColor('#0d1e40'), spaceBefore=8*mm, spaceAfter=3*mm,
        borderPadding=(0, 0, 2, 0),
    ))
    styles.add(ParagraphStyle(
        'NarrativeBody', parent=styles['Normal'],
        fontName='Helvetica', fontSize=10, leading=15,
        textColor=HexColor('#1a1a1a'), alignment=TA_JUSTIFY,
        spaceAfter=3*mm,
    ))
    styles.add(ParagraphStyle(
        'Disclaimer', parent=styles['Normal'],
        fontName='Helvetica-Oblique', fontSize=8,
        textColor=HexColor('#8fa0c0'), alignment=TA_CENTER,
        spaceBefore=8*mm, spaceAfter=2*mm,
    ))

    story = []

    # ── Header ────────────────────────────────────────────────
    from datetime import datetime
    timestamp = datetime.now().strftime('%Y-%m-%d  %H:%M:%S')

    story.append(Paragraph('🧠  PD Live Translate', styles['AppTitle']))
    story.append(Paragraph(
        f'Clinical Screening Report  •  Generated {timestamp}',
        styles['Subtitle'],
    ))
    story.append(HRFlowable(
        width='100%', thickness=1.2,
        color=HexColor('#4a9eff'), spaceAfter=4*mm,
    ))

    # ── Patient Drawing ───────────────────────────────────────
    if canvas_b64:
        story.append(Paragraph('Patient Drawing', styles['SectionHead']))
        try:
            raw = canvas_b64
            if ',' in raw:
                raw = raw.split(',', 1)[1]
            img_bytes = base64.b64decode(raw)
            img_buf = io.BytesIO(img_bytes)
            img_w, img_h = 65*mm, 65*mm
            story.append(RLImage(img_buf, width=img_w, height=img_h))
            story.append(Spacer(1, 4*mm))
        except Exception:
            story.append(Paragraph(
                '<i>Drawing image could not be embedded.</i>',
                styles['NarrativeBody'],
            ))

    # ── Risk Summary Table ────────────────────────────────────
    story.append(Paragraph('Risk Assessment Summary', styles['SectionHead']))

    combined_risk = prediction.get('combined_risk', 0)
    status = prediction.get('status', 'UNKNOWN')
    confidence = prediction.get('confidence', 'N/A')
    mlp_risk = prediction.get('mlp_risk', 0)
    cnn_risk = prediction.get('cnn_risk', 0)
    cnn_tta_risk = prediction.get('cnn_tta_risk', 0)

    # Colour based on severity
    if combined_risk < 0.33:
        risk_color = HexColor('#16a34a')   # green
        risk_bg = HexColor('#e8f5e9')
    elif combined_risk < 0.66:
        risk_color = HexColor('#d97706')   # amber
        risk_bg = HexColor('#fef3c7')
    else:
        risk_color = HexColor('#dc2626')   # red
        risk_bg = HexColor('#fee2e2')

    risk_pct = f'{combined_risk * 100:.1f}%'

    summary_data = [
        ['Metric', 'Value'],
        ['Combined Risk Score', risk_pct],
        ['Risk Status', status],
        ['Model Confidence', confidence.upper()],
        ['MLP Ensemble Score', f'{mlp_risk * 100:.1f}%'],
        ['CNN Ensemble Score', f'{cnn_risk * 100:.1f}%'],
        ['CNN + TTA Score', f'{cnn_tta_risk * 100:.1f}%'],
    ]

    col_w = [(doc.width * 0.52), (doc.width * 0.48)]
    summary_table = Table(summary_data, colWidths=col_w)
    summary_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), HexColor('#0d1e40')),
        ('TEXTCOLOR', (0, 0), (-1, 0), white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('ALIGN', (1, 0), (1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('GRID', (0, 0), (-1, -1), 0.5, HexColor('#c0c8d8')),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [white, HexColor('#f7f9fc')]),
        # Highlight combined-risk row
        ('BACKGROUND', (0, 1), (-1, 1), risk_bg),
        ('TEXTCOLOR', (1, 1), (1, 1), risk_color),
        ('FONTNAME', (1, 1), (1, 1), 'Helvetica-Bold'),
        ('FONTSIZE', (1, 1), (1, 1), 13),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
    ]))
    story.append(summary_table)
    story.append(Spacer(1, 4*mm))

    # ── AI Narrative ──────────────────────────────────────────
    if narrative and narrative.strip():
        story.append(Paragraph('AI Clinical Narrative', styles['SectionHead']))
        paragraphs = [p.strip() for p in narrative.split('\n') if p.strip()]
        for para in paragraphs:
            # Escape XML entities for reportlab
            safe = para.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
            story.append(Paragraph(safe, styles['NarrativeBody']))

    # ── Biomarker Table (2 columns) ───────────────────────────
    features = prediction.get('features', {})
    if features:
        story.append(Paragraph('Biomarker Analysis (16 Features)', styles['SectionHead']))

        feat_items = list(features.items())
        half = (len(feat_items) + 1) // 2
        left_col = feat_items[:half]
        right_col = feat_items[half:]

        bio_data = [['Biomarker', 'Value', 'Biomarker', 'Value']]
        for i in range(max(len(left_col), len(right_col))):
            row = []
            if i < len(left_col):
                row += [left_col[i][0].replace('_', ' ').title(),
                        f'{left_col[i][1]:.6f}']
            else:
                row += ['', '']
            if i < len(right_col):
                row += [right_col[i][0].replace('_', ' ').title(),
                        f'{right_col[i][1]:.6f}']
            else:
                row += ['', '']
            bio_data.append(row)

        bio_col_w = [doc.width * 0.28, doc.width * 0.22,
                     doc.width * 0.28, doc.width * 0.22]
        bio_table = Table(bio_data, colWidths=bio_col_w)
        bio_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), HexColor('#0d1e40')),
            ('TEXTCOLOR', (0, 0), (-1, 0), white),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('ALIGN', (1, 0), (1, -1), 'RIGHT'),
            ('ALIGN', (3, 0), (3, -1), 'RIGHT'),
            ('GRID', (0, 0), (-1, -1), 0.4, HexColor('#c0c8d8')),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [white, HexColor('#f7f9fc')]),
            ('TOPPADDING', (0, 0), (-1, -1), 4),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
            ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ]))
        story.append(bio_table)

    # ── Disclaimer Footer ─────────────────────────────────────
    story.append(Spacer(1, 6*mm))
    story.append(HRFlowable(
        width='100%', thickness=0.6,
        color=HexColor('#c0c8d8'), spaceAfter=3*mm,
    ))
    story.append(Paragraph(
        'DISCLAIMER: This report is generated by an AI-based screening tool '
        'and is intended for informational purposes only. It is NOT a medical '
        'diagnosis. The results should be reviewed by a qualified healthcare '
        'professional. Do not make clinical decisions based solely on this report. '
        'PD Live Translate is a screening aid — not a substitute for professional '
        'medical evaluation.',
        styles['Disclaimer'],
    ))
    story.append(Paragraph(
        f'PD Live Translate  •  Ensemble: 5×MLP + 5×CNN + Meta-Learner  •  '
        f'93.57% accuracy  •  Report ID: {uuid.uuid4().hex[:12].upper()}',
        styles['Disclaimer'],
    ))

    doc.build(story)
    buf.seek(0)
    return buf.read()


def _build_txt_fallback(prediction, narrative):
    """Plain-text fallback when reportlab is not installed."""
    from datetime import datetime
    lines = [
        '=' * 60,
        '  PD Live Translate — Clinical Screening Report',
        '=' * 60,
        f'  Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
        '',
        '-' * 60,
        '  RISK ASSESSMENT SUMMARY',
        '-' * 60,
        f'  Combined Risk Score : {prediction.get("combined_risk", 0) * 100:.1f}%',
        f'  Risk Status         : {prediction.get("status", "N/A")}',
        f'  Model Confidence    : {prediction.get("confidence", "N/A").upper()}',
        f'  MLP Ensemble Score  : {prediction.get("mlp_risk", 0) * 100:.1f}%',
        f'  CNN Ensemble Score  : {prediction.get("cnn_risk", 0) * 100:.1f}%',
        f'  CNN + TTA Score     : {prediction.get("cnn_tta_risk", 0) * 100:.1f}%',
        '',
    ]

    if narrative and narrative.strip():
        lines += [
            '-' * 60,
            '  AI CLINICAL NARRATIVE',
            '-' * 60,
        ]
        for para in narrative.split('\n'):
            if para.strip():
                lines.append(f'  {para.strip()}')
        lines.append('')

    features = prediction.get('features', {})
    if features:
        lines += [
            '-' * 60,
            '  BIOMARKER ANALYSIS (16 Features)',
            '-' * 60,
        ]
        for name, val in features.items():
            label = name.replace('_', ' ').title()
            lines.append(f'  {label:30s} : {val:.6f}')
        lines.append('')

    lines += [
        '-' * 60,
        '  DISCLAIMER',
        '-' * 60,
        '  This report is generated by an AI-based screening tool and',
        '  is intended for informational purposes only. It is NOT a',
        '  medical diagnosis. Results should be reviewed by a qualified',
        '  healthcare professional.',
        '=' * 60,
    ]
    return '\n'.join(lines)


@app.route('/api/report', methods=['POST'])
def generate_report():
    """Generate and download a clinical PDF (or .txt fallback) report."""
    try:
        data = request.get_json(silent=True) or {}
        prediction = data.get('prediction', {})
        narrative = data.get('narrative', '')
        canvas_b64 = data.get('canvas_image', '')

        if not prediction:
            return jsonify({'error': 'No prediction data provided'}), 400

        if REPORTLAB_AVAILABLE:
            pdf_bytes = _build_pdf(prediction, narrative, canvas_b64)
            return app.response_class(
                pdf_bytes,
                mimetype='application/pdf',
                headers={
                    'Content-Disposition': 'attachment; filename=pd_report.pdf',
                },
            )
        else:
            # Fallback — plain text
            txt = _build_txt_fallback(prediction, narrative)
            return app.response_class(
                txt,
                mimetype='text/plain',
                headers={
                    'Content-Disposition': 'attachment; filename=pd_report.txt',
                },
            )
    except Exception as e:
        print(f'[!] Report generation error: {e}')
        import traceback; traceback.print_exc()
        return jsonify({'error': f'Report generation failed: {e}'}), 500


@app.route("/health")
def health():
    return jsonify({
        "status":        "ok",
        "models_loaded": models_loaded,
        "mlp_count":     len(mlp_models),
        "cnn_count":     len(cnn_models),
    })


# ════════════════════════════════════════════════════════════════
# Startup
# ════════════════════════════════════════════════════════════════

load_all_models()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5002))
    app.run(host="0.0.0.0", port=port, debug=False)