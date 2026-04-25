"""
Cross-Modal Attention Fusion Network (CMAFN) — Web App
======================================================
Multimodal Parkinson's Disease Detection:
  Handwriting (EfficientNet-B4+CBAM+SPP) + Speech (XLS-R+CNN+MFCC+SE)
  → Cross-Modal Transformer Attention + GMU + 5-Fold Ensemble

Deploy: gunicorn app:app --bind 0.0.0.0:$PORT --timeout 300
"""

import os
import io
import sys
import base64
import random
import warnings
from dataclasses import dataclass

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename


warnings.filterwarnings("ignore")

def _safe(v, default=0.0):
    """Convert any float-like to a JSON-safe Python float.
    Replaces NaN / Inf / -Inf with `default` (0.0) so Flask's
    jsonify never emits the bare NaN / Infinity tokens that
    cause JSON.parse to throw in the browser."""
    try:
        f = float(v)
        if f != f or f == float('inf') or f == float('-inf'):
            return default
        return f
    except (TypeError, ValueError):
        return default

DEVICE = torch.device("cpu")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoint_fusion")
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# ════════════════════════════════════════════════════════════════
#  Configuration
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


config = FusionConfig()


@dataclass
class AudioConfig:
    """Needed for unpickling the pre-trained audio model checkpoint."""
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
    hidden_dim: int = 256
    dropout: float = 0.4
    batch_size: int = 16
    num_epochs: int = 100
    learning_rate: float = 1e-4
    weight_decay: float = 0.03
    n_folds: int = 5
    focal_alpha: float = 0.75
    focal_gamma: float = 3.0
    label_smoothing: float = 0.05
    optimize_threshold: bool = True
    patience: int = 20
    use_augmentation: bool = True
    time_stretch_rate: tuple = (0.85, 1.15)
    pitch_shift_steps: int = 3
    noise_factor: float = 0.008
    spec_augment: bool = True
    freq_mask_param: int = 15
    time_mask_param: int = 25
    use_mixup: bool = True
    mixup_alpha: float = 0.4
    seed: int = 42
    num_workers: int = 0


# ════════════════════════════════════════════════════════════════
#  Pre-trained Encoder Weights (local paths)
# ════════════════════════════════════════════════════════════════

# HW encoder: prefer 'final' model, fall back to 'best'
_hw_final = os.path.join(BASE_DIR, "handwriting_parkinsons_model_final(2).pth")
_hw_best = os.path.join(BASE_DIR, "best_handwriting_model(2).pth")
HW_ENCODER_PATH = _hw_final if os.path.exists(_hw_final) else _hw_best

# Audio encoder: in checkpoint_fusion/
AUDIO_ENCODER_PATH = os.path.join(CHECKPOINT_DIR, "best_audio_model.pth")

# Pre-generated audio_feature_extractor weights (from seed reproduction)
AFE_WEIGHTS_PATH = os.path.join(CHECKPOINT_DIR, "audio_feature_extractor_weights.pth")


def _set_seed(seed=42):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ════════════════════════════════════════════════════════════════
#  Handwriting Encoder: EfficientNet-B4 + SPP + CBAM
# ════════════════════════════════════════════════════════════════

class ChannelAttention(nn.Module):
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


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        return self.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))


class CBAM(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction)
        self.spatial_attention = SpatialAttention()

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


class HandwritingParkinsonsModel(nn.Module):
    def __init__(self, num_classes=2, pretrained=True):
        super().__init__()
        import timm
        self.backbone = timm.create_model(
            "efficientnet_b4", pretrained=pretrained, features_only=True
        )
        with torch.no_grad():
            feat = self.backbone(torch.randn(1, 3, 224, 224))
            fdim = feat[-1].shape[1]
        self.cbam = CBAM(fdim)
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
#  Audio Encoder: XLS-R + CNN + MFCC + Acoustic + SE
# ════════════════════════════════════════════════════════════════

class Wav2VecAudioModel(nn.Module):
    """4-Path Fusion with SE Attention (notebook architecture)."""

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
#  CMAFN: Cross-Modal Attention Fusion Network
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
#  Audio Feature Extraction
# ════════════════════════════════════════════════════════════════

def load_audio(filepath, sr=16000, max_length=8):
    import librosa
    waveform, _ = librosa.load(filepath, sr=sr, mono=True)
    target = sr * max_length
    if len(waveform) > target:
        waveform = waveform[:target]
    elif len(waveform) < target:
        waveform = np.pad(waveform, (0, target - len(waveform)))
    return waveform


def extract_mel_spectrogram(waveform, sr=16000):
    import torchaudio.transforms as T
    mel_tf = T.MelSpectrogram(
        sample_rate=sr, n_fft=config.n_fft,
        hop_length=config.hop_length, n_mels=config.n_mels, power=2.0)
    mel = mel_tf(torch.from_numpy(waveform).float().unsqueeze(0))
    return T.AmplitudeToDB()(mel)


def extract_mfcc(waveform, sr=16000):
    import librosa
    mfcc = librosa.feature.mfcc(y=waveform, sr=sr, n_mfcc=config.n_mfcc)
    feats = np.concatenate([np.mean(mfcc, axis=1), np.std(mfcc, axis=1)])
    delta = librosa.feature.delta(mfcc, order=1)
    feats = np.concatenate([feats, np.mean(delta, axis=1), np.std(delta, axis=1)])
    return feats  # 160-d


def extract_voice_quality(waveform, sr=16000):
    import librosa
    features = {}
    try:
        import parselmouth
        from parselmouth.praat import call
        sound = parselmouth.Sound(waveform, sampling_frequency=sr)
        pitch = sound.to_pitch(time_step=0.01)
        features["mean_pitch"] = call(pitch, "Get mean", 0, 0, "Hertz")
        features["std_pitch"] = call(
            pitch, "Get standard deviation", 0, 0, "Hertz")
        pp = call(sound, "To PointProcess (periodic, cc)", 75, 500)
        features["jitter_local"] = call(
            pp, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
        features["jitter_rap"] = call(
            pp, "Get jitter (rap)", 0, 0, 0.0001, 0.02, 1.3)
        features["shimmer_local"] = call(
            [sound, pp], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        harmonicity = call(
            sound, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
        features["hnr"] = call(harmonicity, "Get mean", 0, 0)
        formants = sound.to_formant_burg(time_step=0.01)
        features["f1_mean"] = call(formants, "Get mean", 1, 0, 0, "Hertz")
        features["f2_mean"] = call(formants, "Get mean", 2, 0, 0, "Hertz")
        features["f3_mean"] = call(formants, "Get mean", 3, 0, 0, "Hertz")
    except Exception:
        for k in ["mean_pitch", "std_pitch", "jitter_local", "jitter_rap",
                   "shimmer_local", "hnr", "f1_mean", "f2_mean", "f3_mean"]:
            features[k] = 0.0
    features["spectral_centroid"] = float(
        np.mean(librosa.feature.spectral_centroid(y=waveform, sr=sr)))
    features["spectral_rolloff"] = float(
        np.mean(librosa.feature.spectral_rolloff(y=waveform, sr=sr)))
    features["zcr"] = float(
        np.mean(librosa.feature.zero_crossing_rate(waveform)))
    rms = librosa.feature.rms(y=waveform)
    features["rms_mean"] = float(np.mean(rms))
    features["rms_std"] = float(np.std(rms))
    return features  # 14 values


def extract_extras(waveform, sr=16000):
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
    return np.concatenate(extras)  # 25-d


# ════════════════════════════════════════════════════════════════
#  Image Preprocessing
# ════════════════════════════════════════════════════════════════

# Albumentations transform matching the notebook exactly
hw_transform = A.Compose([
    A.Resize(config.hw_image_size, config.hw_image_size),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])


def image_from_base64(data_url):
    """Decode a canvas data-URL to a numpy RGB array (for albumentations)."""
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
#  Global Model State
# ════════════════════════════════════════════════════════════════

hw_encoder = None
audio_encoder = None
w2v_model = None
w2v_processor = None
fusion_models = []
models_loaded = False
load_error = None


def load_models():
    global hw_encoder, audio_encoder, w2v_model, w2v_processor
    global fusion_models, models_loaded, load_error, config

    print("=" * 60)
    print("  Loading CMAFN Fusion Models")
    print("=" * 60)

    # ── 1. Load CMAFN ensemble ──
    cmafn_path = os.path.join(CHECKPOINT_DIR, "cmafn_final_model.pth")
    if os.path.exists(cmafn_path):
        ckpt = torch.load(cmafn_path, map_location=DEVICE, weights_only=False)
        saved_cfg = ckpt.get("config", {})
        for key in ["hw_feature_dim", "audio_feature_dim", "fusion_embed_dim",
                     "n_attention_heads", "n_transformer_layers", "gmu_hidden_dim",
                     "fusion_dropout", "modality_dropout"]:
            if key in saved_cfg:
                setattr(config, key, saved_cfg[key])
        fusion_models.clear()
        for sd in ckpt["ensemble_state_dicts"]:
            m = CrossModalAttentionFusionNetwork(config).to(DEVICE)
            m.load_state_dict(sd)
            m.eval()
            fusion_models.append(m)
        print(f"  [OK] {len(fusion_models)} CMAFN fold models from cmafn_final_model.pth")
    else:
        for i in range(1, 6):
            fp = os.path.join(CHECKPOINT_DIR, f"best_fusion_fold_{i}.pth")
            if os.path.exists(fp):
                ckpt = torch.load(fp, map_location=DEVICE, weights_only=False)
                if i == 1 and isinstance(ckpt.get("config"), FusionConfig):
                    config = ckpt["config"]
                m = CrossModalAttentionFusionNetwork(config).to(DEVICE)
                m.load_state_dict(ckpt["model_state_dict"])
                m.eval()
                fusion_models.append(m)
                print(f"  [OK] Fold {i}: bal_acc="
                      f"{ckpt.get('val_balanced_accuracy', 0):.4f}")
        if not fusion_models:
            load_error = "No CMAFN checkpoints found"
            print(f"  [!!] {load_error}")

    # ── 2. Load handwriting encoder (with pre-trained PD weights) ──
    print("\n  Loading handwriting encoder...")
    try:
        hw_encoder_obj = HandwritingParkinsonsModel(
            num_classes=2, pretrained=True).to(DEVICE)
        if os.path.exists(HW_ENCODER_PATH):
            hw_ckpt = torch.load(HW_ENCODER_PATH, map_location=DEVICE, weights_only=False)
            if "model_state_dict" in hw_ckpt:
                hw_encoder_obj.load_state_dict(hw_ckpt["model_state_dict"], strict=False)
            else:
                hw_encoder_obj.load_state_dict(hw_ckpt, strict=False)
            size_mb = os.path.getsize(HW_ENCODER_PATH) / (1024 * 1024)
            print(f"  [OK] Loaded HW encoder from {os.path.basename(HW_ENCODER_PATH)} ({size_mb:.0f} MB)")
        else:
            print(f"  [!!] HW weights not found at {HW_ENCODER_PATH}")
        hw_encoder_obj.eval()
        for p in hw_encoder_obj.parameters():
            p.requires_grad = False
        globals()["hw_encoder"] = hw_encoder_obj
    except Exception as e:
        load_error = f"Handwriting encoder failed: {e}"
        print(f"  [!!] {load_error}")

    # ── 3. Load audio encoder (with pre-trained PD weights) ──
    print("\n  Loading audio encoder...")
    audio_encoder_obj = Wav2VecAudioModel().to(DEVICE)
    if os.path.exists(AUDIO_ENCODER_PATH):
        audio_ckpt = torch.load(AUDIO_ENCODER_PATH, map_location=DEVICE, weights_only=False)
        if "model_state_dict" in audio_ckpt:
            audio_encoder_obj.load_state_dict(audio_ckpt["model_state_dict"], strict=False)
        else:
            audio_encoder_obj.load_state_dict(audio_ckpt, strict=False)
        size_mb = os.path.getsize(AUDIO_ENCODER_PATH) / (1024 * 1024)
        print(f"  [OK] Loaded audio encoder from {os.path.basename(AUDIO_ENCODER_PATH)} ({size_mb:.0f} MB)")
    else:
        print(f"  [!!] Audio weights not found at {AUDIO_ENCODER_PATH}")

    # Load pre-generated audio_feature_extractor weights
    if os.path.exists(AFE_WEIGHTS_PATH):
        afe_weights = torch.load(AFE_WEIGHTS_PATH, map_location=DEVICE, weights_only=True)
        audio_encoder_obj.load_state_dict(afe_weights, strict=False)
        print(f"  [OK] Loaded audio_feature_extractor weights from saved file")
    else:
        print(f"  [!!] audio_feature_extractor weights not found — using random init")

    audio_encoder_obj.eval()
    for p in audio_encoder_obj.parameters():
        p.requires_grad = False
    globals()["audio_encoder"] = audio_encoder_obj

    # ── 4. XLS-R for embedding extraction ──
    print("\n  Loading XLS-R 300M (this may download ~1.2 GB on first run)...")
    try:
        os.environ["USE_TF"] = "0"
        os.environ["TRANSFORMERS_NO_TF"] = "1"
        from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor
        globals()["w2v_processor"] = Wav2Vec2FeatureExtractor.from_pretrained(
            config.wav2vec2_model_name)
        w2v = Wav2Vec2Model.from_pretrained(config.wav2vec2_model_name)
        w2v.eval().to(DEVICE)
        for p in w2v.parameters():
            p.requires_grad = False
        globals()["w2v_model"] = w2v
        n_params = sum(p.numel() for p in w2v.parameters()) / 1e6
        print(f"  [OK] XLS-R loaded ({n_params:.0f}M params)")
    except Exception as e:
        load_error = f"XLS-R failed: {e}"
        print(f"  [!!] {load_error}")
        return

    models_loaded = True
    print(f"\n{'=' * 60}")
    print(f"  All models ready — {len(fusion_models)} CMAFN folds")
    print(f"{'=' * 60}")


# ════════════════════════════════════════════════════════════════
#  Prediction Pipeline
# ════════════════════════════════════════════════════════════════

@torch.no_grad()
def extract_xlsr_embedding(waveform):
    inputs = w2v_processor(
        waveform, sampling_rate=16000, return_tensors="pt", padding=True)
    out = w2v_model(inputs.input_values.to(DEVICE))
    return out.last_hidden_state.mean(dim=1).squeeze(0).cpu().numpy()


@torch.no_grad()
def extract_hw_features(img_np):
    """Extract handwriting features from a numpy RGB array.
    Returns (embedding_512d, raw_image_tensor).
    The raw (1,3,H,W) tensor is kept so _run_hw_isolated can call the
    full EfficientNet-B4 classifier path with zero audio involvement."""
    augmented = hw_transform(image=img_np)
    tensor = augmented['image'].unsqueeze(0).to(DEVICE)
    embedding = hw_encoder(tensor, return_features=True)
    return embedding, tensor


@torch.no_grad()
def extract_audio_features(filepath):
    import librosa  # noqa: F811
    waveform = load_audio(filepath, config.sample_rate, config.max_audio_length)
    mel = extract_mel_spectrogram(waveform, config.sample_rate)
    mfcc = extract_mfcc(waveform, config.sample_rate)
    vq = extract_voice_quality(waveform, config.sample_rate)
    extras = extract_extras(waveform, config.sample_rate)
    acoustic = np.concatenate([list(vq.values()), extras])
    w2v_emb = extract_xlsr_embedding(waveform)

    mel_t = mel.unsqueeze(0).to(DEVICE)
    mfcc_t = torch.tensor(mfcc, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    acoustic_t = torch.tensor(acoustic, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    w2v_t = torch.tensor(w2v_emb, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    mfcc_t = torch.nan_to_num(mfcc_t)
    acoustic_t = torch.nan_to_num(acoustic_t)

    feats = audio_encoder(mel_t, mfcc_t, acoustic_t, w2v_t, return_features=True)
    return feats, vq  # (1, 512), dict


@torch.no_grad()
def _run_hw_isolated(hw_image_tensor):
    """Score handwriting using the full EfficientNet-B4 classifier path.
    Receives the raw (1,3,H,W) image tensor — NOT the 512-d embedding —
    so backbone → CBAM → SPP → Linear runs correctly with no audio at all."""
    logits = hw_encoder(hw_image_tensor, return_features=False)  # (1, 2)
    return F.softmax(logits, dim=-1)[0, 1].item()


@torch.no_grad()
def _run_audio_isolated(audio_features):
    """Run audio through each fold model with hw zeroed + masked OFF.
    With hw_mask=False the model replaces hw with its learned default token,
    so cross-attention from audio→hw attends only to that uninformative token.
    The audio_head output is therefore driven almost purely by the audio signal.
    Returns mean score across all 5 folds for stability."""
    dummy_hw  = torch.zeros(1, config.hw_feature_dim, device=DEVICE)
    hw_off    = torch.tensor([False], device=DEVICE)
    audio_on  = torch.tensor([True],  device=DEVICE)
    scores = []
    for model in fusion_models:
        model.eval()
        out = model(dummy_hw, audio_features, hw_off, audio_on)
        scores.append(F.softmax(out["audio_logits"], dim=-1)[0, 1].item())
    return float(np.mean(scores))


@torch.no_grad()
def run_fusion_ensemble(hw_features, audio_features, hw_mask, audio_mask,
                        hw_image_tensor=None):
    """Returns:
      fold_probs  – fused CMAFN scores (all 5 folds), used ONLY for
                    uncertainty / confidence, NOT for combined_risk
      hw_score    – isolated HW score via full EfficientNet-B4 classifier
                    (uses raw image tensor, zero audio involvement)
      audio_score – isolated audio score (audio branch, no hw leakage)
    combined_risk is computed in the predict route as a strict 50/50
    average of hw_score and audio_score when both modalities are present.
    """
    fold_probs = []
    for model in fusion_models:
        model.eval()
        out = model(hw_features, audio_features,
                    hw_mask.to(DEVICE), audio_mask.to(DEVICE))
        fold_probs.append(F.softmax(out["logits"], dim=-1)[0, 1].item())

    # Use raw image tensor for HW — passing the 512-d embedding would
    # crash EfficientNet's conv2d layers expecting a (1,3,H,W) input.
    hw_score    = (_run_hw_isolated(hw_image_tensor)
                   if hw_mask.item() and hw_image_tensor is not None else None)
    audio_score = (_run_audio_isolated(audio_features)
                   if audio_mask.item() else None)

    return fold_probs, hw_score, audio_score


# ════════════════════════════════════════════════════════════════
#  Flask App
# ════════════════════════════════════════════════════════════════

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024

ALLOWED_AUDIO = {"wav", "mp3", "ogg", "flac", "m4a", "webm"}


def _allowed_audio(fn):
    return "." in fn and fn.rsplit(".", 1)[1].lower() in ALLOWED_AUDIO


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if not models_loaded:
        msg = load_error or "Models not loaded yet"
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

    # ── Handwriting features ──
    hw_feats        = torch.zeros(1, config.hw_feature_dim).to(DEVICE)
    hw_image_tensor = None   # raw (1,3,H,W) tensor for isolated HW scoring
    hw_mask         = torch.tensor([False])
    if has_hw:
        try:
            if image_b64:
                img_np = image_from_base64(image_b64)
            else:
                # Read uploaded file via cv2 (match notebook preprocessing)
                file_bytes = np.frombuffer(image_file.read(), np.uint8)
                img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                img_np = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            hw_feats, hw_image_tensor = extract_hw_features(img_np)
            hw_mask = torch.tensor([True])
        except Exception as e:
            return jsonify({"error": f"Image processing failed: {e}"}), 400

    # ── Audio features ──
    audio_feats = torch.zeros(1, config.audio_feature_dim).to(DEVICE)
    audio_mask = torch.tensor([False])
    voice_feats = {}
    if has_audio:
        if not _allowed_audio(audio_file.filename):
            return jsonify({
                "error": "Invalid audio type. Allowed: WAV, MP3, OGG, FLAC, M4A, WEBM"
            }), 400
        filename = secure_filename(audio_file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        audio_file.save(filepath)
        try:
            audio_feats, voice_feats = extract_audio_features(filepath)
            audio_mask = torch.tensor([True])
        except Exception as e:
            return jsonify({"error": f"Audio processing failed: {e}"}), 500
        finally:
            try:
                os.remove(filepath)
            except OSError:
                pass

    # ── Ensemble prediction ──
    try:
        fold_probs, hw_score, audio_score = run_fusion_ensemble(
            hw_feats, audio_feats, hw_mask, audio_mask, hw_image_tensor)
    except Exception as e:
        return jsonify({"error": f"Model inference failed: {e}"}), 500

    # ── Strict 50/50 equal weighting ───────────────────────────────────────
    # hw_score and audio_score come from fully isolated encoders —
    # no cross-modal contamination. Each present modality gets equal weight.
    # Missing modality contributes nothing (excluded, not zeroed).
    active_scores = [s for s in [hw_score, audio_score] if s is not None]
    combined_risk = _safe(np.mean(active_scores))   # 0.5*hw + 0.5*audio when both
    hw_risk       = _safe(hw_score    if hw_score    is not None else 0.0)
    audio_risk    = _safe(audio_score if audio_score is not None else 0.0)
    # ────────────────────────────────────────────────────────────────────────

    # Uncertainty uses the fused fold spread (good proxy for model agreement)
    fold_std   = _safe(np.std(fold_probs))
    confidence = round(max(0.0, 1.0 - fold_std * 3), 4)

    if combined_risk < 0.33:
        status = "LOW RISK"
    elif combined_risk < 0.66:
        status = "MODERATE RISK"
    else:
        status = "HIGH RISK"

    n_active = len(active_scores)
    result = {
        "combined_risk":    round(_safe(combined_risk), 4),
        "hw_risk":          round(_safe(hw_risk),       4),
        "audio_risk":       round(_safe(audio_risk),    4),
        "confidence":       round(_safe(confidence),    4),
        "uncertainty":      round(_safe(fold_std),      4),
        "status":           status,
        "modalities_used":  [],
        "modality_weights": {
            "handwriting": round(1.0 / n_active, 2) if has_hw    else 0.0,
            "audio":       round(1.0 / n_active, 2) if has_audio else 0.0,
        },
        "fold_predictions": [round(_safe(p), 4) for p in fold_probs],
    }
    if has_hw:
        result["modalities_used"].append("handwriting")
    if has_audio:
        result["modalities_used"].append("audio")
        result["voice_features"] = {
            "mean_pitch": round(_safe(voice_feats.get("mean_pitch")), 2),
            "std_pitch":  round(_safe(voice_feats.get("std_pitch")),  2),
            "jitter":     round(_safe(voice_feats.get("jitter_local")) * 100, 4),
            "shimmer":    round(_safe(voice_feats.get("shimmer_local")) * 100, 4),
            "hnr":        round(_safe(voice_feats.get("hnr")), 2),
        }

    return jsonify(result)


@app.route("/health")
def health():
    return jsonify({
        "status": "ok" if models_loaded else "loading",
        "fusion_models": len(fusion_models),
        "hw_encoder": hw_encoder is not None,
        "audio_encoder": audio_encoder is not None,
        "xlsr": w2v_model is not None,
        "error": load_error,
    })


# ════════════════════════════════════════════════════════════════
#  Startup
# ════════════════════════════════════════════════════════════════

print("[*] Starting CMAFN Fusion App...")
load_models()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    app.run(host="0.0.0.0", port=port, debug=False)