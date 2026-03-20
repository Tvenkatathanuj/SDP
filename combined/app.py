"""
Parkinson's Disease Detection — Combined Web App
=================================================
Three modalities:
  1. Handwriting: 5x Residual-MLP + 5x EfficientNet-B0+CBAM + Meta-Learner
  2. Speech: XLS-R 300M + Cross-Attention Fusion + 5-Fold Ensemble
  3. Fusion (CMAFN): EfficientNet-B4+CBAM+SPP + XLS-R+CNN+MFCC+SE
     → Cross-Modal Transformer Attention + GMU + 5-Fold Ensemble

Deploy: gunicorn app:app --bind 0.0.0.0:$PORT --timeout 300
"""

import os
import io
import sys
import base64
import pickle
import random
import warnings
import traceback
from dataclasses import dataclass, field
from typing import List, Tuple
from pathlib import Path

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename

warnings.filterwarnings("ignore")

DEVICE = torch.device("cpu")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
HANDWRITING_DIR = BASE_DIR  # Models now in same directory
SPEECH_DIR = BASE_DIR  # Models now in same directory
FUSION_DIR = BASE_DIR  # Models now in same directory
FUSION_CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoint_fusion")
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# ════════════════════════════════════════════════════════════════
#  HANDWRITING — 16 Spatial Biomarker Feature Extractor
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

    dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    stroke_vals = dist[ink_mask]
    f1, f2 = float(np.mean(stroke_vals)), float(np.std(stroke_vals))

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    f3 = 1.0
    if contours:
        c = max(contours, key=cv2.contourArea)
        peri, area = cv2.arcLength(c, True), cv2.contourArea(c)
        if area > 0:
            f3 = (peri ** 2) / (4 * np.pi * area)

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

    n_labels, _ = cv2.connectedComponents(binary)
    f5 = float(n_labels - 1)
    f6 = float(ink_count / (h * w))

    f7 = 0.0
    if contours:
        c = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(c)
        hull_area = cv2.contourArea(cv2.convexHull(c))
        if hull_area > 0:
            f7 = float(area / hull_area)

    f8 = float(np.std(img_gray[ink_mask].astype(float)) / 255.0)
    f9 = _box_counting_fractal(binary)

    hist = cv2.calcHist([img_gray], [0], binary, [256], [0, 256]).flatten()
    hist = hist / (hist.sum() + 1e-10)
    hist = hist[hist > 0]
    f10 = float(-np.sum(hist * np.log2(hist + 1e-10)))

    hu = cv2.HuMoments(cv2.moments(binary)).flatten()
    f11 = float(-np.sign(hu[0]) * np.log10(abs(hu[0]) + 1e-10))
    f12 = float(-np.sign(hu[1]) * np.log10(abs(hu[1]) + 1e-10))

    if contours:
        curv = _compute_curvature(max(contours, key=len))
        f13, f14 = float(np.mean(curv)), float(np.std(curv))
    else:
        f13 = f14 = 0.0

    coords = np.column_stack(np.where(binary > 0))
    if len(coords) > 0:
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
        f15 = float((x_max - x_min + 1) / (y_max - y_min + 1 + 1e-6))
    else:
        f15 = 1.0

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
#  HANDWRITING — Neural Network Architectures
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
    try:
        backbone = models.efficientnet_b0(weights=None)
    except TypeError:
        backbone = models.efficientnet_b0(pretrained=False)
    backbone.classifier = nn.Identity()
    cbam = CBAM(1280, reduction=16)
    return EfficientNetCBAM(backbone, cbam, 1280)


# ════════════════════════════════════════════════════════════════
#  HANDWRITING — Image Transforms + TTA
# ════════════════════════════════════════════════════════════════

hw_val_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

hw_tta_transforms = [
    hw_val_tf,
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
#  SPEECH — Audio Configuration
# ════════════════════════════════════════════════════════════════

@dataclass
class AudioConfig:
    checkpoint_dir: str = "./checkpoints_audio"
    sample_rate: int = 16000
    max_audio_length: int = 8
    n_mfcc: int = 40
    n_mels: int = 128
    n_fft: int = 2048
    hop_length: int = 512
    use_delta_mfcc: bool = True
    use_spectral_contrast: bool = True
    use_chroma: bool = True
    use_tonnetz: bool = True
    wav2vec2_model_name: str = "facebook/wav2vec2-xls-r-300m"
    wav2vec2_embed_dim: int = 1024
    freeze_wav2vec2: bool = True
    use_attentive_pooling: bool = True
    n_cross_attn_heads: int = 4
    cross_attn_dim: int = 256
    use_bilstm: bool = True
    bilstm_hidden: int = 128
    bilstm_layers: int = 1
    supported_languages: list = field(default_factory=lambda: [
        "italian", "english", "spanish", "turkish", "telugu", "hindi", "german", "french"
    ])
    language_embed_dim: int = 32
    use_language_normalization: bool = True
    use_tta: bool = True
    tta_repeats: int = 5
    gradient_accumulation_steps: int = 2
    ml_pca_dim: int = 512
    hidden_dim: int = 256
    dropout: float = 0.6
    batch_size: int = 16
    num_epochs: int = 50
    learning_rate: float = 3e-5
    weight_decay: float = 0.08
    n_folds: int = 5
    focal_alpha: float = 0.75
    focal_gamma: float = 3.0
    label_smoothing: float = 0.2
    optimize_threshold: bool = True
    patience: int = 10
    use_augmentation: bool = True
    time_stretch_rate: Tuple[float, float] = (0.85, 1.15)
    pitch_shift_steps: int = 3
    noise_factor: float = 0.008
    spec_augment: bool = True
    freq_mask_param: int = 25
    time_mask_param: int = 40
    use_vtlp: bool = True
    vtlp_warp_factor: Tuple[float, float] = (0.9, 1.1)
    use_mixup: bool = True
    mixup_alpha: float = 0.3
    seed: int = 42
    num_workers: int = 0


audio_config = AudioConfig()


# ════════════════════════════════════════════════════════════════
#  SPEECH — Model Architecture
# ════════════════════════════════════════════════════════════════

class MultiHeadCrossAttentionFusion(nn.Module):
    def __init__(self, dims: List[int], n_heads: int = 4, proj_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.n_paths = len(dims)
        self.proj_dim = proj_dim
        self.projections = nn.ModuleList([
            nn.Sequential(nn.Linear(d, proj_dim), nn.LayerNorm(proj_dim), nn.ReLU(), nn.Dropout(dropout))
            for d in dims
        ])
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=proj_dim, num_heads=n_heads, dropout=dropout, batch_first=True
        )
        total_dim = proj_dim * self.n_paths
        self.gate = nn.Sequential(nn.Linear(total_dim, self.n_paths), nn.Softmax(dim=-1))
        self.output_norm = nn.LayerNorm(proj_dim)
        self.output_dropout = nn.Dropout(dropout)
        self.output_dim = proj_dim

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        projected = [proj(f) for proj, f in zip(self.projections, features)]
        stacked = torch.stack(projected, dim=1)
        attended, _ = self.cross_attention(stacked, stacked, stacked)
        attended = attended + stacked
        concat_all = torch.cat(projected, dim=-1)
        gate_weights = self.gate(concat_all)
        fused = (attended * gate_weights.unsqueeze(-1)).sum(dim=1)
        fused = self.output_norm(fused)
        fused = self.output_dropout(fused)
        return fused


class Wav2VecAudioModel(nn.Module):
    def __init__(self, cfg: AudioConfig, n_mfcc_features=None, n_acoustic_features=None):
        super().__init__()
        self.config = cfg
        if n_mfcc_features is None:
            n_mfcc_features = cfg.n_mfcc * 2 + (cfg.n_mfcc * 2 if cfg.use_delta_mfcc else 0)
        if n_acoustic_features is None:
            n_acoustic_features = 14
            if cfg.use_spectral_contrast: n_acoustic_features += 7
            if cfg.use_chroma: n_acoustic_features += 12
            if cfg.use_tonnetz: n_acoustic_features += 6
        self.n_mfcc_features = n_mfcc_features
        self.n_acoustic_features = n_acoustic_features

        w2v_input_dim = cfg.wav2vec2_embed_dim * (2 if cfg.use_attentive_pooling else 1)

        if cfg.use_bilstm:
            self.bilstm = nn.LSTM(
                input_size=cfg.wav2vec2_embed_dim, hidden_size=cfg.bilstm_hidden,
                num_layers=cfg.bilstm_layers, batch_first=True, bidirectional=True,
                dropout=cfg.dropout if cfg.bilstm_layers > 1 else 0
            )
            w2v_mlp_input = w2v_input_dim + cfg.bilstm_hidden * 2
        else:
            self.bilstm = None
            w2v_mlp_input = w2v_input_dim

        self.w2v_encoder = nn.Sequential(
            nn.Linear(w2v_mlp_input, 512), nn.BatchNorm1d(512), nn.GELU(), nn.Dropout(cfg.dropout),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.GELU(), nn.Dropout(cfg.dropout * 0.6)
        )

        channels = [32, 64, 128]
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, channels[0], 3, padding=1), nn.BatchNorm2d(channels[0]),
            nn.GELU(), nn.MaxPool2d(2), nn.Dropout2d(0.25)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(channels[0], channels[1], 3, padding=1), nn.BatchNorm2d(channels[1]),
            nn.GELU(), nn.MaxPool2d(2), nn.Dropout2d(0.25)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(channels[1], channels[2], 3, padding=1), nn.BatchNorm2d(channels[2]),
            nn.GELU(), nn.AdaptiveAvgPool2d((2, 2)), nn.Dropout2d(0.35)
        )
        self.conv_residual = nn.Conv2d(channels[0], channels[2], kernel_size=1)
        self.conv_pool = nn.AdaptiveAvgPool2d((2, 2))

        self.mfcc_encoder = nn.Sequential(
            nn.Linear(n_mfcc_features, 256), nn.BatchNorm1d(256), nn.GELU(), nn.Dropout(cfg.dropout),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.GELU(), nn.Dropout(cfg.dropout * 0.6)
        )
        self.mfcc_skip = nn.Linear(n_mfcc_features, 128)

        self.acoustic_encoder = nn.Sequential(
            nn.Linear(n_acoustic_features, 128), nn.BatchNorm1d(128), nn.GELU(), nn.Dropout(cfg.dropout),
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.GELU()
        )

        pathway_dims = [256, 512, 128, 64]
        self.cross_attn_fusion = MultiHeadCrossAttentionFusion(
            dims=pathway_dims, n_heads=cfg.n_cross_attn_heads,
            proj_dim=cfg.cross_attn_dim, dropout=cfg.dropout * 0.5
        )
        fusion_dim = self.cross_attn_fusion.output_dim

        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, cfg.hidden_dim), nn.BatchNorm1d(cfg.hidden_dim),
            nn.GELU(), nn.Dropout(cfg.dropout), nn.Linear(cfg.hidden_dim, 2)
        )

    def forward(self, mel_spec, mfcc, acoustic, w2v_emb, w2v_seq=None):
        if self.bilstm is not None and w2v_seq is not None:
            _, (h_n, _) = self.bilstm(w2v_seq)
            bilstm_feat = torch.cat([h_n[-2], h_n[-1]], dim=1)
            w2v_combined = torch.cat([w2v_emb, bilstm_feat], dim=1)
        else:
            w2v_combined = w2v_emb
        x_w2v = self.w2v_encoder(w2v_combined)

        x_cnn = self.conv1(mel_spec)
        x_res = x_cnn
        x_cnn = self.conv2(x_cnn)
        x_cnn = self.conv3(x_cnn)
        x_res = self.conv_pool(self.conv_residual(x_res))
        x_cnn = x_cnn + x_res
        x_cnn = x_cnn.view(x_cnn.size(0), -1)

        x_mfcc = self.mfcc_encoder(mfcc) + self.mfcc_skip(mfcc)
        x_acoustic = self.acoustic_encoder(acoustic)

        fused = self.cross_attn_fusion([x_w2v, x_cnn, x_mfcc, x_acoustic])
        return {"logits": self.classifier(fused)}


class AttentiveStatisticalPooling(nn.Module):
    def __init__(self, input_dim, attention_dim=128):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, attention_dim), nn.Tanh(), nn.Linear(attention_dim, 1)
        )

    def forward(self, x):
        attn_weights = F.softmax(self.attention(x), dim=1)
        weighted_mean = (attn_weights * x).sum(dim=1)
        weighted_var = (attn_weights * (x - weighted_mean.unsqueeze(1)) ** 2).sum(dim=1)
        weighted_std = torch.sqrt(weighted_var.clamp(min=1e-8))
        return torch.cat([weighted_mean, weighted_std], dim=1)


# ════════════════════════════════════════════════════════════════
#  SPEECH — Audio Feature Extraction
# ════════════════════════════════════════════════════════════════

def load_audio(filepath, sr=16000, max_length=8):
    import librosa
    waveform, _ = librosa.load(filepath, sr=sr, mono=True)
    if np.max(np.abs(waveform)) > 0:
        waveform = waveform / np.max(np.abs(waveform))
    max_samples = sr * max_length
    if len(waveform) > max_samples:
        waveform = waveform[:max_samples]
    elif len(waveform) < max_samples:
        waveform = np.pad(waveform, (0, max_samples - len(waveform)))
    return waveform


def extract_mfcc(waveform, sr=16000, n_mfcc=40):
    import librosa
    mfcc = librosa.feature.mfcc(y=waveform, sr=sr, n_mfcc=n_mfcc, n_fft=2048, hop_length=512)
    feats = np.concatenate([np.mean(mfcc, axis=1), np.std(mfcc, axis=1)])
    delta = librosa.feature.delta(mfcc, order=1)
    feats = np.concatenate([feats, np.mean(delta, axis=1), np.std(delta, axis=1)])
    return feats


def extract_voice_quality(waveform, sr=16000):
    import librosa
    features = {}
    try:
        import parselmouth
        from parselmouth.praat import call
        sound = parselmouth.Sound(waveform, sampling_frequency=sr)
        pitch = sound.to_pitch(time_step=0.01)
        features["mean_pitch"] = call(pitch, "Get mean", 0, 0, "Hertz")
        features["std_pitch"] = call(pitch, "Get standard deviation", 0, 0, "Hertz")
        pp = call(sound, "To PointProcess (periodic, cc)", 75, 500)
        features["jitter_local"] = call(pp, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
        features["jitter_rap"] = call(pp, "Get jitter (rap)", 0, 0, 0.0001, 0.02, 1.3)
        features["shimmer_local"] = call([sound, pp], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        harmonicity = call(sound, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
        features["hnr"] = call(harmonicity, "Get mean", 0, 0)
        formants = sound.to_formant_burg(time_step=0.01)
        features["f1_mean"] = call(formants, "Get mean", 1, 0, 0, "Hertz")
        features["f2_mean"] = call(formants, "Get mean", 2, 0, 0, "Hertz")
        features["f3_mean"] = call(formants, "Get mean", 3, 0, 0, "Hertz")
    except Exception:
        for k in ["mean_pitch", "std_pitch", "jitter_local", "jitter_rap",
                   "shimmer_local", "hnr", "f1_mean", "f2_mean", "f3_mean"]:
            features[k] = 0.0

    features["spectral_centroid"] = float(np.mean(librosa.feature.spectral_centroid(y=waveform, sr=sr)))
    features["spectral_rolloff"] = float(np.mean(librosa.feature.spectral_rolloff(y=waveform, sr=sr)))
    features["zcr"] = float(np.mean(librosa.feature.zero_crossing_rate(waveform)))
    rms = librosa.feature.rms(y=waveform)
    features["rms_mean"] = float(np.mean(rms))
    features["rms_std"] = float(np.std(rms))
    return features


def extract_extra_features(waveform, sr=16000):
    import librosa
    extras = []
    sc = librosa.feature.spectral_contrast(y=waveform, sr=sr, n_bands=6)
    extras.append(np.mean(sc, axis=1))
    chroma = librosa.feature.chroma_stft(y=waveform, sr=sr)
    extras.append(np.mean(chroma, axis=1))
    try:
        tonnetz = librosa.feature.tonnetz(y=waveform, sr=sr)
        extras.append(np.mean(tonnetz, axis=1))
    except Exception:
        extras.append(np.zeros(6))
    return np.concatenate(extras)


def extract_mel_spectrogram(waveform, sr=16000):
    import torchaudio.transforms as T
    mel_transform = T.MelSpectrogram(sample_rate=sr, n_fft=2048, hop_length=512, n_mels=128, power=2.0)
    waveform_t = torch.from_numpy(waveform).float().unsqueeze(0)
    mel = mel_transform(waveform_t)
    mel = T.AmplitudeToDB()(mel)
    return mel


# ════════════════════════════════════════════════════════════════
#  Global Model State
# ════════════════════════════════════════════════════════════════

# Handwriting
hw_mlp_models = []
hw_cnn_models = []
hw_scaler = None
hw_meta_model = None
hw_models_loaded = False

# Speech
sp_ensemble_models = []
sp_w2v_model = None
sp_w2v_processor = None
sp_asp_pooling = None
sp_models_loaded = False
sp_load_error = None


def load_handwriting_models():
    global hw_mlp_models, hw_cnn_models, hw_scaler, hw_meta_model, hw_models_loaded

    print("[*] Loading handwriting models...")

    for i in range(1, 6):
        path = os.path.join(HANDWRITING_DIR, f"mlp_fold_{i}.pth")
        if os.path.exists(path):
            m = PDDetectionModelV2(input_size=16)
            m.load_state_dict(torch.load(path, map_location=DEVICE, weights_only=False))
            m.eval()
            hw_mlp_models.append(m)
            print(f"  [OK] mlp_fold_{i}.pth")
        else:
            print(f"  [!!] mlp_fold_{i}.pth NOT FOUND")

    for i in range(1, 6):
        path = os.path.join(HANDWRITING_DIR, f"cnn_fold_{i}.pth")
        if os.path.exists(path):
            m = _build_efficientnet_cbam()
            m.load_state_dict(torch.load(path, map_location=DEVICE, weights_only=False))
            m.eval()
            hw_cnn_models.append(m)
            print(f"  [OK] cnn_fold_{i}.pth")
        else:
            print(f"  [!!] cnn_fold_{i}.pth NOT FOUND")

    scaler_path = os.path.join(HANDWRITING_DIR, "scaler.pkl")
    if os.path.exists(scaler_path):
        with open(scaler_path, "rb") as f:
            hw_scaler = pickle.load(f)
        print("  [OK] scaler.pkl")
    else:
        print("  [!!] scaler.pkl NOT FOUND")

    meta_path = os.path.join(HANDWRITING_DIR, "meta_model.pkl")
    if os.path.exists(meta_path):
        with open(meta_path, "rb") as f:
            hw_meta_model = pickle.load(f)
        print("  [OK] meta_model.pkl")
    else:
        print("  [!!] meta_model.pkl NOT FOUND")

    hw_models_loaded = True
    print(f"[OK] Handwriting: {len(hw_mlp_models)} MLP + {len(hw_cnn_models)} CNN models")


def load_speech_models():
    global sp_ensemble_models, sp_w2v_model, sp_w2v_processor, sp_asp_pooling
    global sp_models_loaded, sp_load_error

    print("[*] Loading speech models...")

    try:
        os.environ["USE_TF"] = "0"
        os.environ["TRANSFORMERS_NO_TF"] = "1"
        from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor
        print("  Loading XLS-R 300M...")
        sp_w2v_processor = Wav2Vec2FeatureExtractor.from_pretrained(audio_config.wav2vec2_model_name)
        sp_w2v_model = Wav2Vec2Model.from_pretrained(audio_config.wav2vec2_model_name)
        sp_w2v_model.eval()
        sp_w2v_model.to(DEVICE)
        for param in sp_w2v_model.parameters():
            param.requires_grad = False
        sp_asp_pooling = AttentiveStatisticalPooling(audio_config.wav2vec2_embed_dim, 128)
        sp_asp_pooling.to(DEVICE)
        sp_asp_pooling.eval()
        print(f"  [OK] XLS-R loaded ({sum(p.numel() for p in sp_w2v_model.parameters())/1e6:.0f}M params)")
    except Exception as e:
        sp_load_error = f"Failed to load XLS-R: {e}"
        print(f"  [!!] {sp_load_error}")
        return

    sys.modules["__main__"].AudioConfig = AudioConfig

    fold_files = sorted(Path(SPEECH_DIR).glob("fold_*_model*.pth"))
    if not fold_files:
        sp_load_error = "No fold model .pth files found in speech dir"
        print(f"  [!!] {sp_load_error}")
        return

    for pth_path in fold_files:
        try:
            ckpt = torch.load(str(pth_path), map_location=DEVICE, weights_only=False)
            model = Wav2VecAudioModel(audio_config)
            model.load_state_dict(ckpt["model_state_dict"])
            model.eval()
            model.to(DEVICE)
            val_f1 = ckpt.get("val_f1", 0)
            sp_ensemble_models.append((model, val_f1))
            print(f"  [OK] {pth_path.name} (val F1={val_f1:.3f})")
        except Exception as e:
            print(f"  [!!] {pth_path.name}: {e}")

    if not sp_ensemble_models:
        sp_load_error = "All fold models failed to load"
        print(f"  [!!] {sp_load_error}")
        return

    sp_models_loaded = True
    print(f"[OK] Speech: {len(sp_ensemble_models)} fold models + XLS-R backbone")


# ════════════════════════════════════════════════════════════════
#  HANDWRITING — Prediction Pipeline
# ════════════════════════════════════════════════════════════════

def _image_from_base64(data_url):
    if "," in data_url:
        data_url = data_url.split(",", 1)[1]
    img_bytes = base64.b64decode(data_url)
    img = Image.open(io.BytesIO(img_bytes))
    if img.mode == "RGBA":
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[3])
        img = bg
    return img.convert("RGB")


def predict_hw_mlp(img_gray):
    if not hw_mlp_models or hw_scaler is None:
        return 0.5
    img_resized = cv2.resize(img_gray, (256, 256))
    feats = extract_16_features(img_resized)
    feats = np.nan_to_num(feats, 0.0).reshape(1, -1)
    feats_scaled = hw_scaler.transform(feats)
    inp = torch.FloatTensor(feats_scaled).to(DEVICE)
    preds = []
    with torch.inference_mode():
        for m in hw_mlp_models:
            preds.append(m(inp).cpu().item())
    return float(np.mean(preds))


def predict_hw_cnn(img_pil, use_tta=False):
    if not hw_cnn_models:
        return 0.5
    preds = []
    with torch.inference_mode():
        for m in hw_cnn_models:
            if use_tta:
                for tf in hw_tta_transforms:
                    inp = tf(img_pil).unsqueeze(0).to(DEVICE)
                    preds.append(torch.sigmoid(m(inp)).cpu().item())
            else:
                inp = hw_val_tf(img_pil).unsqueeze(0).to(DEVICE)
                preds.append(torch.sigmoid(m(inp)).cpu().item())
    return float(np.mean(preds))


def predict_handwriting(img_pil):
    img_gray = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2GRAY)
    mlp_risk = predict_hw_mlp(img_gray)
    cnn_risk = predict_hw_cnn(img_pil, use_tta=False)
    cnn_tta_risk = predict_hw_cnn(img_pil, use_tta=True)

    if hw_meta_model is not None:
        meta_input = np.array([[mlp_risk, cnn_risk, cnn_tta_risk]])
        combined_risk = float(hw_meta_model.predict_proba(meta_input)[:, 1][0])
    else:
        combined_risk = (mlp_risk + cnn_risk + cnn_tta_risk) / 3.0

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
#  SPEECH — Prediction Pipeline
# ════════════════════════════════════════════════════════════════

@torch.no_grad()
def extract_xlsr_features(waveform):
    inputs = sp_w2v_processor(waveform, sampling_rate=audio_config.sample_rate, return_tensors="pt", padding=True)
    input_values = inputs.input_values.to(DEVICE)
    outputs = sp_w2v_model(input_values)
    hidden = outputs.last_hidden_state
    pooled = sp_asp_pooling(hidden).squeeze(0).cpu().numpy()
    seq = hidden.squeeze(0).cpu().numpy()
    if seq.shape[0] > 100:
        indices = np.linspace(0, seq.shape[0] - 1, 100, dtype=int)
        seq = seq[indices]
    elif seq.shape[0] < 100:
        seq = np.pad(seq, ((0, 100 - seq.shape[0]), (0, 0)))
    return pooled, seq


@torch.no_grad()
def predict_speech(audio_file):
    waveform = load_audio(audio_file, sr=audio_config.sample_rate, max_length=audio_config.max_audio_length)
    mfcc_feat = extract_mfcc(waveform, audio_config.sample_rate, audio_config.n_mfcc)
    voice_feats = extract_voice_quality(waveform, audio_config.sample_rate)
    extra_feats = extract_extra_features(waveform, audio_config.sample_rate)
    acoustic_feat = np.nan_to_num(np.concatenate([list(voice_feats.values()), extra_feats]), nan=0.0)
    mel_spec = extract_mel_spectrogram(waveform, audio_config.sample_rate)
    w2v_emb, w2v_seq = extract_xlsr_features(waveform)

    mel_t = mel_spec.unsqueeze(0).to(DEVICE)
    mfcc_t = torch.tensor(mfcc_feat, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    acoustic_t = torch.tensor(acoustic_feat, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    w2v_emb_t = torch.tensor(w2v_emb, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    w2v_seq_t = torch.tensor(w2v_seq, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    mfcc_t = torch.nan_to_num(mfcc_t)
    acoustic_t = torch.nan_to_num(acoustic_t)

    fold_probs = []
    for model, val_f1 in sp_ensemble_models:
        model.eval()
        outputs = model(mel_t, mfcc_t, acoustic_t, w2v_emb_t, w2v_seq=w2v_seq_t)
        probs = F.softmax(outputs["logits"], dim=-1)
        pd_prob = probs[0, 1].item()
        fold_probs.append(pd_prob)

    weights = np.array([f1 for _, f1 in sp_ensemble_models])
    if weights.sum() > 0:
        weights = weights / weights.sum()
    else:
        weights = np.ones(len(fold_probs)) / len(fold_probs)
    combined_risk = float(np.average(fold_probs, weights=weights))

    if combined_risk < 0.33:
        status = "LOW RISK"
    elif combined_risk < 0.66:
        status = "MODERATE RISK"
    else:
        status = "HIGH RISK"

    return {
        "combined_risk": round(combined_risk, 4),
        "individual_predictions": [round(p, 4) for p in fold_probs],
        "status": status,
        "confidence": round(1.0 - float(np.std(fold_probs)), 4),
    }


# ════════════════════════════════════════════════════════════════
#  FUSION (CMAFN) — Configuration
# ════════════════════════════════════════════════════════════════

@dataclass
class FusionConfig:
    checkpoint_dir: str = "./checkpoint_fusion"
    hw_image_size: int = 336
    hw_feature_dim: int = 512
    sample_rate: int = 16000
    max_audio_length: int = 8
    n_mfcc: int = 40
    n_mels: int = 128
    n_fft: int = 2048
    hop_length: int = 512
    use_delta_mfcc: bool = True
    use_spectral_contrast: bool = True
    use_chroma: bool = True
    use_tonnetz: bool = True
    wav2vec2_model_name: str = "facebook/wav2vec2-xls-r-300m"
    wav2vec2_embed_dim: int = 1024
    freeze_wav2vec2: bool = True
    audio_feature_dim: int = 512
    fusion_embed_dim: int = 256
    n_attention_heads: int = 8
    n_transformer_layers: int = 2
    gmu_hidden_dim: int = 128
    fusion_dropout: float = 0.5
    modality_dropout: float = 0.25
    batch_size: int = 16
    num_epochs: int = 100
    learning_rate: float = 1e-4
    weight_decay: float = 0.05
    warmup_epochs: int = 5
    patience: int = 10
    focal_alpha: float = 0.5
    focal_gamma: float = 2.0
    label_smoothing: float = 0.1
    contrastive_weight: float = 0.3
    use_mixup: bool = True
    mixup_alpha: float = 0.3
    spec_augment: bool = True
    freq_mask_param: int = 15
    time_mask_param: int = 25
    mc_dropout_samples: int = 10
    seed: int = 42
    num_workers: int = 0
    n_folds: int = 5


fusion_config = FusionConfig()


# Fusion HW encoder paths
_fusion_hw_final = os.path.join(FUSION_DIR, "handwriting_parkinsons_model_final(2).pth")
_fusion_hw_best = os.path.join(FUSION_DIR, "best_handwriting_model(2).pth")
FUSION_HW_ENCODER_PATH = _fusion_hw_final if os.path.exists(_fusion_hw_final) else _fusion_hw_best
FUSION_AUDIO_ENCODER_PATH = os.path.join(FUSION_CHECKPOINT_DIR, "best_audio_model.pth")
FUSION_AFE_WEIGHTS_PATH = os.path.join(FUSION_CHECKPOINT_DIR, "audio_feature_extractor_weights.pth")


# ════════════════════════════════════════════════════════════════
#  FUSION — Handwriting Encoder: EfficientNet-B4 + SPP + CBAM
# ════════════════════════════════════════════════════════════════

class FusionChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.fc(self.avg_pool(x)) + self.fc(self.max_pool(x)))


class FusionSpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        return self.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))


class FusionCBAM(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.channel_attention = FusionChannelAttention(in_channels, reduction)
        self.spatial_attention = FusionSpatialAttention()

    def forward(self, x):
        x = x * self.channel_attention(x)
        x = x * self.spatial_attention(x)
        return x


class SpatialPyramidPooling(nn.Module):
    def __init__(self, pool_sizes=None):
        super().__init__()
        self.pool_sizes = pool_sizes or [1, 2, 4]

    def forward(self, x):
        b = x.size(0)
        pools = []
        for ps in self.pool_sizes:
            pools.append(F.adaptive_avg_pool2d(x, (ps, ps)).view(b, -1))
        return torch.cat(pools, dim=1)


class FusionHWEncoder(nn.Module):
    """EfficientNet-B4 + CBAM + SPP handwriting encoder for CMAFN."""
    def __init__(self, num_classes=2, pretrained=True):
        super().__init__()
        import timm
        self.backbone = timm.create_model(
            "efficientnet_b4", pretrained=pretrained, features_only=True
        )
        with torch.no_grad():
            feat = self.backbone(torch.randn(1, 3, 224, 224))
            fdim = feat[-1].shape[1]
        self.cbam = FusionCBAM(fdim)
        self.spp = SpatialPyramidPooling([1, 2, 4])
        spp_out = fdim * (1 + 4 + 16)
        self.classifier = nn.Sequential(
            nn.Linear(spp_out, 512), nn.BatchNorm1d(512), nn.ReLU(True), nn.Dropout(0.6),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(True), nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )
        self.feature_extractor = nn.Sequential(
            nn.Linear(spp_out, 512), nn.BatchNorm1d(512), nn.ReLU(True),
        )

    def forward(self, x, return_features=False):
        features = self.backbone(x)
        x = self.cbam(features[-1])
        x = self.spp(x).view(x.size(0), -1)
        if return_features:
            return self.feature_extractor(x)
        return self.classifier(x)


# ════════════════════════════════════════════════════════════════
#  FUSION — Audio Encoder: XLS-R + CNN + MFCC + Acoustic + SE
# ════════════════════════════════════════════════════════════════

class FusionAudioEncoder(nn.Module):
    """4-Path Fusion with SE Attention for CMAFN."""
    def __init__(self, n_mfcc_features=160, n_acoustic_features=39,
                 wav2vec2_embed_dim=1024, hidden_dim=256, dropout=0.4):
        super().__init__()
        self.w2v_encoder = nn.Sequential(
            nn.Linear(wav2vec2_embed_dim, 512), nn.BatchNorm1d(512),
            nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(512, 256), nn.BatchNorm1d(256),
            nn.ReLU(), nn.Dropout(dropout * 0.5),
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32),
            nn.ReLU(), nn.MaxPool2d(2), nn.Dropout2d(0.2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64),
            nn.ReLU(), nn.MaxPool2d(2), nn.Dropout2d(0.2),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128),
            nn.ReLU(), nn.AdaptiveAvgPool2d((2, 2)), nn.Dropout2d(0.3),
        )
        cnn_out = 128 * 2 * 2
        self.mfcc_encoder = nn.Sequential(
            nn.Linear(n_mfcc_features, 256), nn.BatchNorm1d(256),
            nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 128), nn.BatchNorm1d(128),
            nn.ReLU(), nn.Dropout(dropout * 0.5),
        )
        self.acoustic_encoder = nn.Sequential(
            nn.Linear(n_acoustic_features, 128), nn.BatchNorm1d(128),
            nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(),
        )
        fusion_dim = 256 + cnn_out + 128 + 64  # 960
        self.se = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim // 8), nn.ReLU(),
            nn.Linear(fusion_dim // 8, fusion_dim), nn.Sigmoid(),
        )
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim), nn.BatchNorm1d(hidden_dim),
            nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, 128), nn.BatchNorm1d(128),
            nn.ReLU(), nn.Dropout(dropout * 0.7),
            nn.Linear(128, 2),
        )
        self.audio_feature_extractor = nn.Sequential(
            nn.Linear(fusion_dim, 512), nn.BatchNorm1d(512), nn.ReLU(),
        )

    def forward(self, mel_spec, mfcc, acoustic, w2v_emb, return_features=False):
        single = mel_spec.size(0) == 1 and self.training
        if single:
            self.eval()
        x_w2v = self.w2v_encoder(w2v_emb)
        x_cnn = self.conv3(self.conv2(self.conv1(mel_spec)))
        x_cnn = x_cnn.view(mel_spec.size(0), -1)
        x_mfcc = self.mfcc_encoder(mfcc)
        x_acoustic = self.acoustic_encoder(acoustic)
        x_fused = torch.cat([x_w2v, x_cnn, x_mfcc, x_acoustic], dim=1)
        x_fused = x_fused * self.se(x_fused)
        if return_features:
            if single:
                self.train()
            return self.audio_feature_extractor(x_fused)
        logits = self.fusion(x_fused)
        if single:
            self.train()
        return {"logits": logits}


# ════════════════════════════════════════════════════════════════
#  FUSION — CMAFN: Cross-Modal Attention Fusion Network
# ════════════════════════════════════════════════════════════════

class ModalityProjection(nn.Module):
    def __init__(self, input_dim, embed_dim, dropout=0.1):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_dim, embed_dim), nn.LayerNorm(embed_dim),
            nn.GELU(), nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim), nn.LayerNorm(embed_dim),
        )

    def forward(self, x):
        return self.proj(x)


class CrossModalTransformerLayer(nn.Module):
    def __init__(self, embed_dim, n_heads, dropout=0.1):
        super().__init__()
        self.hw_cross_attn = nn.MultiheadAttention(
            embed_dim, n_heads, dropout=dropout, batch_first=True)
        self.hw_norm1 = nn.LayerNorm(embed_dim)
        self.hw_ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim), nn.Dropout(dropout))
        self.hw_norm2 = nn.LayerNorm(embed_dim)
        self.audio_cross_attn = nn.MultiheadAttention(
            embed_dim, n_heads, dropout=dropout, batch_first=True)
        self.audio_norm1 = nn.LayerNorm(embed_dim)
        self.audio_ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim), nn.Dropout(dropout))
        self.audio_norm2 = nn.LayerNorm(embed_dim)

    def forward(self, hw_tokens, audio_tokens):
        hw_cross, _ = self.hw_cross_attn(
            query=hw_tokens, key=audio_tokens, value=audio_tokens)
        hw_tokens = self.hw_norm1(hw_tokens + hw_cross)
        hw_tokens = self.hw_norm2(hw_tokens + self.hw_ffn(hw_tokens))
        audio_cross, _ = self.audio_cross_attn(
            query=audio_tokens, key=hw_tokens, value=hw_tokens)
        audio_tokens = self.audio_norm1(audio_tokens + audio_cross)
        audio_tokens = self.audio_norm2(audio_tokens + self.audio_ffn(audio_tokens))
        return hw_tokens, audio_tokens


class GatedMultimodalUnit(nn.Module):
    def __init__(self, dim_hw, dim_audio, hidden_dim):
        super().__init__()
        self.fc_hw = nn.Linear(dim_hw, hidden_dim)
        self.fc_audio = nn.Linear(dim_audio, hidden_dim)
        self.gate = nn.Sequential(
            nn.Linear(dim_hw + dim_audio, hidden_dim), nn.Sigmoid())
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, hw_feat, audio_feat):
        h_hw = torch.tanh(self.fc_hw(hw_feat))
        h_audio = torch.tanh(self.fc_audio(audio_feat))
        z = self.gate(torch.cat([hw_feat, audio_feat], dim=-1))
        return self.norm(z * h_hw + (1 - z) * h_audio)


class CrossModalAttentionFusionNetwork(nn.Module):
    def __init__(self, cfg: FusionConfig):
        super().__init__()
        self.config = cfg
        ed = cfg.fusion_embed_dim
        self.hw_projection = ModalityProjection(
            cfg.hw_feature_dim, ed, cfg.fusion_dropout)
        self.audio_projection = ModalityProjection(
            cfg.audio_feature_dim, ed, cfg.fusion_dropout)
        self.hw_default_token = nn.Parameter(torch.randn(1, 1, ed) * 0.02)
        self.audio_default_token = nn.Parameter(torch.randn(1, 1, ed) * 0.02)
        self.hw_type_embed = nn.Parameter(torch.randn(1, 1, ed) * 0.02)
        self.audio_type_embed = nn.Parameter(torch.randn(1, 1, ed) * 0.02)
        self.cross_modal_layers = nn.ModuleList([
            CrossModalTransformerLayer(ed, cfg.n_attention_heads, cfg.fusion_dropout)
            for _ in range(cfg.n_transformer_layers)])
        self.gmu = GatedMultimodalUnit(ed, ed, cfg.gmu_hidden_dim)
        cls_in = ed * 2 + cfg.gmu_hidden_dim
        self.classifier = nn.Sequential(
            nn.Linear(cls_in, 256), nn.LayerNorm(256), nn.GELU(),
            nn.Dropout(cfg.fusion_dropout),
            nn.Linear(256, 128), nn.LayerNorm(128), nn.GELU(),
            nn.Dropout(cfg.fusion_dropout),
            nn.Linear(128, 2))
        self.hw_head = nn.Sequential(
            nn.Linear(ed, 64), nn.GELU(), nn.Dropout(0.3), nn.Linear(64, 2))
        self.audio_head = nn.Sequential(
            nn.Linear(ed, 64), nn.GELU(), nn.Dropout(0.3), nn.Linear(64, 2))

    def forward(self, hw_features, audio_features,
                hw_mask=None, audio_mask=None):
        B = hw_features.size(0)
        device = hw_features.device
        if hw_mask is None:
            hw_mask = torch.ones(B, dtype=torch.bool, device=device)
        if audio_mask is None:
            audio_mask = torch.ones(B, dtype=torch.bool, device=device)
        if self.training and self.config.modality_dropout > 0:
            for i in range(B):
                r = random.random()
                if r < self.config.modality_dropout / 2:
                    hw_mask[i] = False
                elif r < self.config.modality_dropout:
                    audio_mask[i] = False
        hw_proj = self.hw_projection(hw_features)
        audio_proj = self.audio_projection(audio_features)
        hw_default = self.hw_default_token.expand(B, -1, -1).squeeze(1)
        audio_default = self.audio_default_token.expand(B, -1, -1).squeeze(1)
        hw_proj = torch.where(hw_mask.unsqueeze(-1), hw_proj, hw_default)
        audio_proj = torch.where(audio_mask.unsqueeze(-1), audio_proj, audio_default)
        hw_tokens = (hw_proj + self.hw_type_embed.squeeze(1)).unsqueeze(1)
        audio_tokens = (audio_proj + self.audio_type_embed.squeeze(1)).unsqueeze(1)
        for layer in self.cross_modal_layers:
            hw_tokens, audio_tokens = layer(hw_tokens, audio_tokens)
        hw_attended = hw_tokens.squeeze(1)
        audio_attended = audio_tokens.squeeze(1)
        gmu_out = self.gmu(hw_attended, audio_attended)
        fused = torch.cat([hw_attended, audio_attended, gmu_out], dim=-1)
        return {
            "logits": self.classifier(fused),
            "hw_logits": self.hw_head(hw_attended),
            "audio_logits": self.audio_head(audio_attended),
        }


# ════════════════════════════════════════════════════════════════
#  FUSION — Image Preprocessing (Albumentations)
# ════════════════════════════════════════════════════════════════

import albumentations as A
from albumentations.pytorch import ToTensorV2

fusion_hw_transform = A.Compose([
    A.Resize(fusion_config.hw_image_size, fusion_config.hw_image_size),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])


def fusion_image_from_base64(data_url):
    """Decode a canvas data-URL to a numpy RGB array."""
    if "," in data_url:
        data_url = data_url.split(",", 1)[1]
    img_bytes = base64.b64decode(data_url)
    img = Image.open(io.BytesIO(img_bytes))
    if img.mode == "RGBA":
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[3])
        img = bg
    return np.array(img.convert("RGB"))


# ════════════════════════════════════════════════════════════════
#  FUSION — Audio Feature Extraction
# ════════════════════════════════════════════════════════════════

def fusion_extract_mel_spectrogram(waveform, sr=16000):
    import torchaudio.transforms as T
    mel_tf = T.MelSpectrogram(
        sample_rate=sr, n_fft=fusion_config.n_fft,
        hop_length=fusion_config.hop_length, n_mels=fusion_config.n_mels, power=2.0)
    mel = mel_tf(torch.from_numpy(waveform).float().unsqueeze(0))
    return T.AmplitudeToDB()(mel)


def fusion_extract_mfcc(waveform, sr=16000):
    import librosa
    mfcc = librosa.feature.mfcc(y=waveform, sr=sr, n_mfcc=fusion_config.n_mfcc)
    feats = np.concatenate([np.mean(mfcc, axis=1), np.std(mfcc, axis=1)])
    delta = librosa.feature.delta(mfcc, order=1)
    feats = np.concatenate([feats, np.mean(delta, axis=1), np.std(delta, axis=1)])
    return feats


def fusion_extract_extras(waveform, sr=16000):
    import librosa
    extras = []
    extras.append(np.mean(
        librosa.feature.spectral_contrast(y=waveform, sr=sr, n_bands=6), axis=1))
    extras.append(np.mean(
        librosa.feature.chroma_stft(y=waveform, sr=sr), axis=1))
    try:
        extras.append(np.mean(
            librosa.feature.tonnetz(y=waveform, sr=sr), axis=1))
    except Exception:
        extras.append(np.zeros(6))
    return np.concatenate(extras)


# ════════════════════════════════════════════════════════════════
#  FUSION — Global Model State + Loading
# ════════════════════════════════════════════════════════════════

fu_hw_encoder = None
fu_audio_encoder = None
fu_w2v_model = None
fu_w2v_processor = None
fu_cmafn_models = []
fu_models_loaded = False
fu_load_error = None


def load_fusion_models():
    global fu_hw_encoder, fu_audio_encoder, fu_w2v_model, fu_w2v_processor
    global fu_cmafn_models, fu_models_loaded, fu_load_error, fusion_config

    print("\n[*] Loading CMAFN Fusion Models...")

    # 1. Load CMAFN ensemble
    cmafn_path = os.path.join(FUSION_CHECKPOINT_DIR, "cmafn_final_model.pth")
    if os.path.exists(cmafn_path):
        ckpt = torch.load(cmafn_path, map_location=DEVICE, weights_only=False)
        saved_cfg = ckpt.get("config", {})
        for key in ["hw_feature_dim", "audio_feature_dim", "fusion_embed_dim",
                     "n_attention_heads", "n_transformer_layers", "gmu_hidden_dim",
                     "fusion_dropout", "modality_dropout"]:
            if key in saved_cfg:
                setattr(fusion_config, key, saved_cfg[key])
        fu_cmafn_models.clear()
        for sd in ckpt["ensemble_state_dicts"]:
            m = CrossModalAttentionFusionNetwork(fusion_config).to(DEVICE)
            m.load_state_dict(sd)
            m.eval()
            fu_cmafn_models.append(m)
        print(f"  [OK] {len(fu_cmafn_models)} CMAFN fold models from cmafn_final_model.pth")
    else:
        for i in range(1, 6):
            fp = os.path.join(FUSION_CHECKPOINT_DIR, f"best_fusion_fold_{i}.pth")
            if os.path.exists(fp):
                ckpt = torch.load(fp, map_location=DEVICE, weights_only=False)
                if i == 1 and isinstance(ckpt.get("config"), FusionConfig):
                    fusion_config = ckpt["config"]
                m = CrossModalAttentionFusionNetwork(fusion_config).to(DEVICE)
                m.load_state_dict(ckpt["model_state_dict"])
                m.eval()
                fu_cmafn_models.append(m)
                print(f"  [OK] Fold {i}: bal_acc="
                      f"{ckpt.get('val_balanced_accuracy', 0):.4f}")
        if not fu_cmafn_models:
            fu_load_error = "No CMAFN checkpoints found"
            print(f"  [!!] {fu_load_error}")

    # 2. Load handwriting encoder
    print("  Loading fusion HW encoder...")
    try:
        hw_enc = FusionHWEncoder(num_classes=2, pretrained=True).to(DEVICE)
        if os.path.exists(FUSION_HW_ENCODER_PATH):
            hw_ckpt = torch.load(FUSION_HW_ENCODER_PATH, map_location=DEVICE, weights_only=False)
            if "model_state_dict" in hw_ckpt:
                hw_enc.load_state_dict(hw_ckpt["model_state_dict"], strict=False)
            else:
                hw_enc.load_state_dict(hw_ckpt, strict=False)
            print(f"  [OK] Loaded fusion HW encoder from {os.path.basename(FUSION_HW_ENCODER_PATH)}")
        hw_enc.eval()
        for p in hw_enc.parameters():
            p.requires_grad = False
        globals()["fu_hw_encoder"] = hw_enc
    except Exception as e:
        fu_load_error = f"Fusion HW encoder failed: {e}"
        print(f"  [!!] {fu_load_error}")

    # 3. Load audio encoder
    print("  Loading fusion audio encoder...")
    audio_enc = FusionAudioEncoder().to(DEVICE)
    if os.path.exists(FUSION_AUDIO_ENCODER_PATH):
        audio_ckpt = torch.load(FUSION_AUDIO_ENCODER_PATH, map_location=DEVICE, weights_only=False)
        if "model_state_dict" in audio_ckpt:
            audio_enc.load_state_dict(audio_ckpt["model_state_dict"], strict=False)
        else:
            audio_enc.load_state_dict(audio_ckpt, strict=False)
        print(f"  [OK] Loaded fusion audio encoder")

    if os.path.exists(FUSION_AFE_WEIGHTS_PATH):
        afe_weights = torch.load(FUSION_AFE_WEIGHTS_PATH, map_location=DEVICE, weights_only=True)
        audio_enc.load_state_dict(afe_weights, strict=False)
        print(f"  [OK] Loaded audio_feature_extractor weights")

    audio_enc.eval()
    for p in audio_enc.parameters():
        p.requires_grad = False
    globals()["fu_audio_encoder"] = audio_enc

    # 4. XLS-R — reuse speech model if already loaded, else load fresh
    print("  Loading XLS-R for fusion...")
    try:
        if sp_w2v_model is not None and sp_w2v_processor is not None:
            globals()["fu_w2v_model"] = sp_w2v_model
            globals()["fu_w2v_processor"] = sp_w2v_processor
            print("  [OK] Reusing XLS-R from speech models")
        else:
            os.environ["USE_TF"] = "0"
            os.environ["TRANSFORMERS_NO_TF"] = "1"
            from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor
            globals()["fu_w2v_processor"] = Wav2Vec2FeatureExtractor.from_pretrained(
                fusion_config.wav2vec2_model_name)
            w2v = Wav2Vec2Model.from_pretrained(fusion_config.wav2vec2_model_name)
            w2v.eval().to(DEVICE)
            for p in w2v.parameters():
                p.requires_grad = False
            globals()["fu_w2v_model"] = w2v
            print(f"  [OK] XLS-R loaded for fusion")
    except Exception as e:
        fu_load_error = f"Fusion XLS-R failed: {e}"
        print(f"  [!!] {fu_load_error}")
        return

    fu_models_loaded = True
    print(f"[OK] Fusion: {len(fu_cmafn_models)} CMAFN folds ready")


# ════════════════════════════════════════════════════════════════
#  FUSION — Prediction Pipeline
# ════════════════════════════════════════════════════════════════

@torch.no_grad()
def fusion_extract_xlsr_embedding(waveform):
    inputs = fu_w2v_processor(
        waveform, sampling_rate=16000, return_tensors="pt", padding=True)
    out = fu_w2v_model(inputs.input_values.to(DEVICE))
    return out.last_hidden_state.mean(dim=1).squeeze(0).cpu().numpy()


@torch.no_grad()
def fusion_extract_hw_features(img_np):
    augmented = fusion_hw_transform(image=img_np)
    tensor = augmented['image'].unsqueeze(0).to(DEVICE)
    return fu_hw_encoder(tensor, return_features=True)


@torch.no_grad()
def fusion_extract_audio_features(filepath):
    waveform = load_audio(filepath, sr=fusion_config.sample_rate,
                          max_length=fusion_config.max_audio_length)
    mel = fusion_extract_mel_spectrogram(waveform, fusion_config.sample_rate)
    mfcc = fusion_extract_mfcc(waveform, fusion_config.sample_rate)
    vq = extract_voice_quality(waveform, fusion_config.sample_rate)
    extras = fusion_extract_extras(waveform, fusion_config.sample_rate)
    acoustic = np.concatenate([list(vq.values()), extras])
    w2v_emb = fusion_extract_xlsr_embedding(waveform)

    mel_t = mel.unsqueeze(0).to(DEVICE)
    mfcc_t = torch.tensor(mfcc, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    acoustic_t = torch.tensor(acoustic, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    w2v_t = torch.tensor(w2v_emb, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    mfcc_t = torch.nan_to_num(mfcc_t)
    acoustic_t = torch.nan_to_num(acoustic_t)

    feats = fu_audio_encoder(mel_t, mfcc_t, acoustic_t, w2v_t, return_features=True)
    return feats, vq


@torch.no_grad()
def run_cmafn_ensemble(hw_features, audio_features, hw_mask, audio_mask):
    all_probs, all_hw, all_audio = [], [], []
    for model in fu_cmafn_models:
        model.eval()
        out = model(hw_features, audio_features,
                    hw_mask.to(DEVICE), audio_mask.to(DEVICE))
        probs = F.softmax(out["logits"], dim=-1)
        hw_p = F.softmax(out["hw_logits"], dim=-1)
        au_p = F.softmax(out["audio_logits"], dim=-1)
        all_probs.append(probs[0, 1].item())
        all_hw.append(hw_p[0, 1].item())
        all_audio.append(au_p[0, 1].item())
    return all_probs, all_hw, all_audio


# ════════════════════════════════════════════════════════════════
#  Flask App
# ════════════════════════════════════════════════════════════════

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024

ALLOWED_AUDIO = {"wav", "mp3", "ogg", "flac", "m4a", "webm"}


def allowed_audio(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_AUDIO


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict/handwriting", methods=["POST"])
def predict_handwriting_route():
    if not hw_models_loaded:
        return jsonify({"error": "Handwriting models not loaded yet"}), 503

    data = request.get_json(silent=True)
    if data and "image" in data:
        img_pil = _image_from_base64(data["image"])
    elif "file" in request.files:
        f = request.files["file"]
        img_pil = Image.open(f.stream).convert("RGB")
    else:
        return jsonify({"error": "No image provided"}), 400

    result = predict_handwriting(img_pil)
    return jsonify(result)


@app.route("/predict/speech", methods=["POST"])
def predict_speech_route():
    if not sp_models_loaded:
        msg = sp_load_error or "Speech models not loaded yet"
        return jsonify({"error": msg}), 503

    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400
    if not allowed_audio(file.filename):
        return jsonify({"error": "Invalid file type. Allowed: WAV, MP3, OGG, FLAC, M4A"}), 400

    try:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)

        result = predict_speech(filepath)

        waveform = load_audio(filepath, sr=audio_config.sample_rate, max_length=audio_config.max_audio_length)
        voice_feats = extract_voice_quality(waveform, audio_config.sample_rate)
        duration = len(waveform) / audio_config.sample_rate

        def _sf(v):
            """NaN-safe float conversion."""
            f = float(v) if v is not None else 0.0
            return 0.0 if (f != f) else f  # NaN != NaN

        result["voice_features"] = {
            "duration": round(duration, 2),
            "mean_pitch": round(_sf(voice_feats.get("mean_pitch", 0)), 2),
            "std_pitch": round(_sf(voice_feats.get("std_pitch", 0)), 2),
            "jitter": round(_sf(voice_feats.get("jitter_local", 0)) * 100, 4),
            "shimmer": round(_sf(voice_feats.get("shimmer_local", 0)) * 100, 4),
            "hnr": round(_sf(voice_feats.get("hnr", 0)), 2),
            "spectral_centroid": round(_sf(voice_feats.get("spectral_centroid", 0)), 1),
            "f1_mean": round(_sf(voice_feats.get("f1_mean", 0)), 1),
            "f2_mean": round(_sf(voice_feats.get("f2_mean", 0)), 1),
        }

        try:
            os.remove(filepath)
        except OSError:
            pass

        return jsonify(result)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/predict/fusion", methods=["POST"])
def predict_fusion_route():
    if not fu_models_loaded:
        msg = fu_load_error or "Fusion models not loaded yet"
        return jsonify({"error": msg}), 503

    image_b64 = request.form.get("image_data")
    image_file = request.files.get("image_file")
    audio_file = request.files.get("audio_file")

    has_hw = bool(image_b64) or bool(image_file and image_file.filename)
    has_audio = bool(audio_file and audio_file.filename)

    if not has_hw and not has_audio:
        return jsonify({
            "error": "Provide at least a handwriting image or audio recording"
        }), 400

    # Handwriting features
    hw_feats = torch.zeros(1, fusion_config.hw_feature_dim).to(DEVICE)
    hw_mask = torch.tensor([False])
    if has_hw:
        try:
            if image_b64:
                img_np = fusion_image_from_base64(image_b64)
            else:
                file_bytes = np.frombuffer(image_file.read(), np.uint8)
                img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                img_np = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            hw_feats = fusion_extract_hw_features(img_np)
            hw_mask = torch.tensor([True])
        except Exception as e:
            return jsonify({"error": f"Image processing failed: {e}"}), 400

    # Audio features
    audio_feats = torch.zeros(1, fusion_config.audio_feature_dim).to(DEVICE)
    audio_mask = torch.tensor([False])
    voice_feats = {}
    if has_audio:
        if not allowed_audio(audio_file.filename):
            return jsonify({
                "error": "Invalid audio type. Allowed: WAV, MP3, OGG, FLAC, M4A, WEBM"
            }), 400
        filename = secure_filename(audio_file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        audio_file.save(filepath)
        try:
            audio_feats, voice_feats = fusion_extract_audio_features(filepath)
            audio_mask = torch.tensor([True])
        except Exception as e:
            return jsonify({"error": f"Audio processing failed: {e}"}), 500
        finally:
            try:
                os.remove(filepath)
            except OSError:
                pass

    # Ensemble prediction
    all_probs, all_hw_p, all_audio_p = run_cmafn_ensemble(
        hw_feats, audio_feats, hw_mask, audio_mask)

    combined_risk = float(np.mean(all_probs))
    hw_risk = float(np.mean(all_hw_p))
    audio_risk = float(np.mean(all_audio_p))
    fold_std = float(np.std(all_probs))
    confidence = round(max(0.0, 1.0 - fold_std * 3), 4)

    if combined_risk < 0.33:
        status = "LOW RISK"
    elif combined_risk < 0.66:
        status = "MODERATE RISK"
    else:
        status = "HIGH RISK"

    result = {
        "combined_risk": round(combined_risk, 4),
        "hw_risk": round(hw_risk, 4),
        "audio_risk": round(audio_risk, 4),
        "confidence": confidence,
        "uncertainty": round(fold_std, 4),
        "status": status,
        "modalities_used": [],
        "fold_predictions": [round(p, 4) for p in all_probs],
    }
    if has_hw:
        result["modalities_used"].append("handwriting")
    if has_audio:
        result["modalities_used"].append("audio")
        result["voice_features"] = {
            "mean_pitch": round(float(voice_feats.get("mean_pitch", 0)), 2),
            "std_pitch": round(float(voice_feats.get("std_pitch", 0)), 2),
            "jitter": round(float(voice_feats.get("jitter_local", 0)) * 100, 4),
            "shimmer": round(float(voice_feats.get("shimmer_local", 0)) * 100, 4),
            "hnr": round(float(voice_feats.get("hnr", 0)), 2),
        }

    return jsonify(result)


@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "handwriting": {
            "loaded": hw_models_loaded,
            "mlp_count": len(hw_mlp_models),
            "cnn_count": len(hw_cnn_models),
        },
        "speech": {
            "loaded": sp_models_loaded,
            "model_count": len(sp_ensemble_models),
            "error": sp_load_error,
        },
        "fusion": {
            "loaded": fu_models_loaded,
            "model_count": len(fu_cmafn_models),
            "error": fu_load_error,
        },
    })


# ════════════════════════════════════════════════════════════════
#  Startup
# ════════════════════════════════════════════════════════════════

load_handwriting_models()
load_speech_models()
load_fusion_models()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
