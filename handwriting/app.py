"""
Parkinson's Disease Handwriting Detection — Live Translate Web App
=================================================================
Ensemble: 5x Residual-MLP (16 biomarkers) + 5x EfficientNet-B0+CBAM
         + Stacking Meta-Learner (LogisticRegression)
Results: 93.57% accuracy | 0.9851 AUC-ROC (5-fold CV)

Deploy on Render: gunicorn app:app --bind 0.0.0.0:$PORT --timeout 120
"""

import os
import io
import base64
import pickle
import warnings

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
from flask import Flask, render_template, request, jsonify

warnings.filterwarnings("ignore")

DEVICE = torch.device("cpu")
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
        norm = np.linalg.norm(v1) * np.linalg.norm(v2)
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
    h, w = binary.shape
    ink_mask = binary > 0
    ink_count = int(np.sum(ink_mask))
    if ink_count < 10:
        return np.zeros(16)

    # 1-2: Stroke Width (distance transform)
    dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    stroke_vals = dist[ink_mask]
    f1, f2 = float(np.mean(stroke_vals)), float(np.std(stroke_vals))

    # 3: Contour Roughness (isoperimetric quotient)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    f3 = 1.0
    if contours:
        c = max(contours, key=cv2.contourArea)
        peri, area = cv2.arcLength(c, True), cv2.contourArea(c)
        if area > 0:
            f3 = (peri ** 2) / (4 * np.pi * area)

    # 4: Direction Changes
    f4 = 0.0
    if contours:
        c = max(contours, key=len)
        if len(c) > 20:
            pts = c.reshape(-1, 2).astype(float)
            step = max(1, len(pts) // 200)
            pts = pts[::step]
            if len(pts) > 3:
                dx, dy = np.diff(pts[:, 0]), np.diff(pts[:, 1])
                angles = np.arctan2(dy, dx)
                diffs = np.abs(np.diff(angles))
                diffs = np.minimum(diffs, 2 * np.pi - diffs)
                f4 = float(np.mean(diffs))

    # 5: Connected Components
    n_labels, _ = cv2.connectedComponents(binary)
    f5 = float(n_labels - 1)

    # 6: Ink Density
    f6 = float(ink_count / (h * w))

    # 7: Solidity
    f7 = 0.0
    if contours:
        c = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(c)
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
    f10 = float(-np.sum(hist * np.log2(hist + 1e-10)))

    # 11-12: Hu Moments
    hu = cv2.HuMoments(cv2.moments(binary)).flatten()
    f11 = float(-np.sign(hu[0]) * np.log10(abs(hu[0]) + 1e-10))
    f12 = float(-np.sign(hu[1]) * np.log10(abs(hu[1]) + 1e-10))

    # 13-14: Curvature
    if contours:
        curv = _compute_curvature(max(contours, key=len))
        f13, f14 = float(np.mean(curv)), float(np.std(curv))
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
            pts = c.reshape(-1, 2).astype(float)
            dists = np.sqrt(np.sum(np.diff(pts, axis=0) ** 2, axis=1))
            if len(dists) > 10:
                fft_vals = np.abs(np.fft.rfft(dists - np.mean(dists)))
                if len(fft_vals) > 1 and fft_vals.sum() > 0:
                    f16 = float(fft_vals[1:].max() / (fft_vals[1:].sum() + 1e-10))

    return np.array([f1, f2, f3, f4, f5, f6, f7, f8,
                     f9, f10, f11, f12, f13, f14, f15, f16])


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
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout * 0.5)

    def forward(self, x):
        return self.dropout(self.act(self.block(x) + x))


class PDDetectionModelV2(nn.Module):
    """Residual MLP: 16 → 64 → [ResBlock] → 32 → 1"""
    def __init__(self, input_size=16, hidden=64):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_size, hidden), nn.BatchNorm1d(hidden), nn.GELU(), nn.Dropout(0.3),
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
        attn = torch.sigmoid(self.fc(avg_pool) + self.fc(max_pool))
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
        self.backbone = backbone
        self.cbam = cbam
        self.pool = nn.AdaptiveAvgPool2d(1)
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
    """Create EfficientNet-B0 + CBAM architecture (no pretrained weights needed for inference)."""
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
        transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
    transforms.Compose([
        transforms.Resize((224, 224)), transforms.RandomRotation((10, 10)),
        transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
    transforms.Compose([
        transforms.Resize((224, 224)), transforms.RandomRotation((-10, -10)),
        transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
    transforms.Compose([
        transforms.Resize((224, 224)), transforms.ColorJitter(brightness=0.3),
        transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
    transforms.Compose([
        transforms.Resize((224, 224)), transforms.RandomVerticalFlip(p=1.0),
        transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
    transforms.Compose([
        transforms.Resize((256, 256)), transforms.CenterCrop(224),
        transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
]


# ════════════════════════════════════════════════════════════════
# Global Model State
# ════════════════════════════════════════════════════════════════

mlp_models = []
cnn_models = []
scaler = None
meta_model = None
models_loaded = False


def load_all_models():
    """Load all model weights, scaler, and meta-learner at startup."""
    global mlp_models, cnn_models, scaler, meta_model, models_loaded

    print("[*] Loading models...")

    # Load 5 MLP fold models
    for i in range(1, 6):
        path = os.path.join(BASE_DIR, f"mlp_fold_{i}.pth")
        if os.path.exists(path):
            m = PDDetectionModelV2(input_size=16)
            m.load_state_dict(torch.load(path, map_location=DEVICE, weights_only=False))
            m.eval()
            mlp_models.append(m)
            print(f"  [OK] mlp_fold_{i}.pth")
        else:
            print(f"  [!!] mlp_fold_{i}.pth NOT FOUND")

    # Load 5 CNN fold models
    for i in range(1, 6):
        path = os.path.join(BASE_DIR, f"cnn_fold_{i}.pth")
        if os.path.exists(path):
            m = _build_efficientnet_cbam()
            m.load_state_dict(torch.load(path, map_location=DEVICE, weights_only=False))
            m.eval()
            cnn_models.append(m)
            print(f"  [OK] cnn_fold_{i}.pth")
        else:
            print(f"  [!!] cnn_fold_{i}.pth NOT FOUND")

    # Load scaler
    scaler_path = os.path.join(BASE_DIR, "scaler.pkl")
    if os.path.exists(scaler_path):
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
        print("  [OK] scaler.pkl")
    else:
        print("  [!!] scaler.pkl NOT FOUND")

    # Load meta-learner
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
    """Decode a base64 data URL to a PIL RGB image (composites alpha onto white)."""
    if "," in data_url:
        data_url = data_url.split(",", 1)[1]
    img_bytes = base64.b64decode(data_url)
    img = Image.open(io.BytesIO(img_bytes))
    # If RGBA, composite onto white background
    if img.mode == "RGBA":
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[3])
        img = bg
    return img.convert("RGB")


def predict_mlp(img_gray):
    """Run 5-fold MLP ensemble on grayscale image → average probability."""
    if not mlp_models or scaler is None:
        return 0.5

    img_resized = cv2.resize(img_gray, (256, 256))
    feats = extract_16_features(img_resized)
    feats = np.nan_to_num(feats, 0.0).reshape(1, -1)
    feats_scaled = scaler.transform(feats)
    inp = torch.FloatTensor(feats_scaled).to(DEVICE)

    preds = []
    with torch.inference_mode():
        for m in mlp_models:
            preds.append(m(inp).cpu().item())
    return float(np.mean(preds))


def predict_cnn(img_pil, use_tta=False):
    """Run 5-fold CNN ensemble on PIL image → average probability."""
    if not cnn_models:
        return 0.5

    preds = []
    with torch.inference_mode():
        for m in cnn_models:
            if use_tta:
                for tf in tta_transforms:
                    inp = tf(img_pil).unsqueeze(0).to(DEVICE)
                    preds.append(torch.sigmoid(m(inp)).cpu().item())
            else:
                inp = val_tf(img_pil).unsqueeze(0).to(DEVICE)
                preds.append(torch.sigmoid(m(inp)).cpu().item())
    return float(np.mean(preds))


def predict_ensemble(img_pil):
    """Full pipeline: MLP + CNN + CNN-TTA + meta-learner stacking."""
    # Convert to grayscale for MLP
    img_gray = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2GRAY)

    mlp_risk = predict_mlp(img_gray)
    cnn_risk = predict_cnn(img_pil, use_tta=False)
    cnn_tta_risk = predict_cnn(img_pil, use_tta=True)

    # Meta-learner ensemble
    if meta_model is not None:
        meta_input = np.array([[mlp_risk, cnn_risk, cnn_tta_risk]])
        combined_risk = float(meta_model.predict_proba(meta_input)[:, 1][0])
    else:
        combined_risk = (mlp_risk + cnn_risk + cnn_tta_risk) / 3.0

    # Extract feature details for display
    img_resized = cv2.resize(img_gray, (256, 256))
    raw_feats = extract_16_features(img_resized)
    feature_dict = {name: float(val) for name, val in zip(FEATURE_NAMES, raw_feats)}

    if combined_risk < 0.33:
        status = "LOW RISK"
    elif combined_risk < 0.66:
        status = "MODERATE RISK"
    else:
        status = "HIGH RISK"

    return {
        "combined_risk": round(combined_risk, 4),
        "mlp_risk": round(mlp_risk, 4),
        "cnn_risk": round(cnn_risk, 4),
        "cnn_tta_risk": round(cnn_tta_risk, 4),
        "status": status,
        "features": feature_dict,
    }


# ════════════════════════════════════════════════════════════════
# Flask App
# ════════════════════════════════════════════════════════════════

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict_route():
    if not models_loaded:
        return jsonify({"error": "Models not loaded yet"}), 503

    data = request.get_json(silent=True)
    if data and "image" in data:
        # Canvas drawing (base64)
        img_pil = _image_from_base64(data["image"])
    elif "file" in request.files:
        # File upload
        f = request.files["file"]
        img_pil = Image.open(f.stream).convert("RGB")
    else:
        return jsonify({"error": "No image provided"}), 400

    result = predict_ensemble(img_pil)
    return jsonify(result)


@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "models_loaded": models_loaded,
        "mlp_count": len(mlp_models),
        "cnn_count": len(cnn_models),
    })


# ════════════════════════════════════════════════════════════════
# Startup
# ════════════════════════════════════════════════════════════════

load_all_models()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
