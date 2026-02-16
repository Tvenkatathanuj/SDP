"""
Parkinson's Disease Detection — Full Multi-Modal Gradio Application
===================================================================
Three detection pipelines:
  1. Handwriting Analysis  — EfficientNet-B4 + CBAM + SPP
  2. Speech/Audio Analysis — Wav2Vec/XLS-R 4-Path Fusion + SE Attention
  3. CMAFN Fusion          — Cross-Modal Attention Fusion Network (5-fold ensemble)

Checkpoint files expected:
  - best_handwriting_model(2).pth  OR  handwriting_parkinsons_model_final(2).pth
  - checkpoints/best_audio_model.pth
  - checkpoint_fusion/cmafn_final_model.pth  (contains 5-fold ensemble)
  - checkpoint_fusion/best_fusion_fold_{1..5}.pth  (individual folds)
"""

import os
import sys
import json
import random
import warnings
from dataclasses import dataclass, asdict, field
from typing import Tuple, Optional, Dict, List
from pathlib import Path

import gradio as gr
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa
import cv2
from PIL import Image
import timm

warnings.filterwarnings("ignore")

# ════════════════════════════════════════════════════════════════
# Device
# ════════════════════════════════════════════════════════════════
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[*] Device: {DEVICE}")

# ════════════════════════════════════════════════════════════════
# Optional heavy dependencies (graceful fallback)
# ════════════════════════════════════════════════════════════════
try:
    import torchaudio.transforms as T
    TORCHAUDIO_AVAILABLE = True
except ImportError:
    TORCHAUDIO_AVAILABLE = False
    print("[!] torchaudio not available -- mel-spectrogram via librosa fallback")

try:
    from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("[!] transformers not available -- XLS-R embeddings will be zero-filled")

try:
    import parselmouth
    from parselmouth.praat import call as praat_call
    PARSELMOUTH_AVAILABLE = True
except ImportError:
    PARSELMOUTH_AVAILABLE = False
    print("[!] parselmouth not available -- voice quality features will be zeroed")

# ════════════════════════════════════════════════════════════════
# ░░ CONFIGURATION DATACLASSES ░░
# ════════════════════════════════════════════════════════════════

@dataclass
class AudioConfig:
    """Mirrors the AudioConfig used when training the standalone audio model."""
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
    time_stretch_rate: Tuple[float, float] = (0.85, 1.15)
    pitch_shift_steps: int = 3
    noise_factor: float = 0.008
    spec_augment: bool = True
    freq_mask_param: int = 15
    time_mask_param: int = 25
    use_mixup: bool = True
    mixup_alpha: float = 0.4
    seed: int = 42
    num_workers: int = 0


@dataclass
class FusionConfig:
    """Mirrors the FusionConfig used when training the CMAFN fusion model."""
    checkpoint_dir: str = "./checkpoints_fusion"
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


# ════════════════════════════════════════════════════════════════
# ░░ 1. HANDWRITING MODEL — EfficientNet-B4 + CBAM + SPP ░░
# ════════════════════════════════════════════════════════════════

class ChannelAttention(nn.Module):
    """CBAM Channel Attention Module"""
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
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return self.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    """CBAM Spatial Attention Module"""
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        return self.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))


class CBAM(nn.Module):
    """Convolutional Block Attention Module"""
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction)
        self.spatial_attention = SpatialAttention()

    def forward(self, x):
        x = x * self.channel_attention(x)
        x = x * self.spatial_attention(x)
        return x


class SpatialPyramidPooling(nn.Module):
    """Spatial Pyramid Pooling for multi-scale features."""
    def __init__(self, pool_sizes=(1, 2, 4)):
        super().__init__()
        self.pool_sizes = pool_sizes

    def forward(self, x):
        B, C, _, _ = x.size()
        pools = []
        for ps in self.pool_sizes:
            pool = F.adaptive_avg_pool2d(x, (ps, ps))
            pools.append(pool.view(B, -1))
        return torch.cat(pools, dim=1)


class HandwritingParkinsonsModel(nn.Module):
    """EfficientNet-B4 + SPP + CBAM for Handwriting Analysis.

    In classification mode  -> (B, 2) logits
    In feature mode         -> (B, 512) features  (for CMAFN fusion)
    """

    def __init__(self, num_classes=2, pretrained=False):
        super().__init__()
        self.backbone = timm.create_model(
            "efficientnet_b4", pretrained=pretrained, features_only=True
        )
        # discover channel count of the deepest feature map
        with torch.no_grad():
            dummy = torch.randn(1, 3, 224, 224)
            feat_dim = self.backbone(dummy)[-1].shape[1]

        self.cbam = CBAM(feat_dim)
        self.spp = SpatialPyramidPooling(pool_sizes=[1, 2, 4])
        spp_out = feat_dim * (1 + 4 + 16)  # 1x1 + 2x2 + 4x4

        self.classifier = nn.Sequential(
            nn.Linear(spp_out, 512), nn.BatchNorm1d(512), nn.ReLU(True), nn.Dropout(0.6),
            nn.Linear(512, 256),     nn.BatchNorm1d(256), nn.ReLU(True), nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )
        # 512-dim feature head used by the fusion model
        self.feature_extractor = nn.Sequential(
            nn.Linear(spp_out, 512), nn.BatchNorm1d(512), nn.ReLU(True),
        )

    def forward(self, x, return_features=False):
        feats = self.backbone(x)
        x = feats[-1]
        x = self.cbam(x)
        x = self.spp(x)
        x = x.view(x.size(0), -1)
        if return_features:
            return self.feature_extractor(x)
        return self.classifier(x)


# ════════════════════════════════════════════════════════════════
# ░░ 2. AUDIO MODEL — 4-Path Fusion + SE Attention ░░
# ════════════════════════════════════════════════════════════════

class Wav2VecAudioModel(nn.Module):
    """4-Path Fusion with SE Attention.

    Paths:
      1  XLS-R embedding (1024) -> MLP -> 256
      2  Mel-spectrogram (1, 128, T) -> 3-layer CNN -> 512
      3  MFCC+delta (160) -> MLP -> 128
      4  Acoustic (39) -> MLP -> 64

    Fusion: SE(concat 960) -> MLP -> 2
    Feature mode: Linear(960->512, BN, ReLU) -> (B, 512) for CMAFN
    """

    def __init__(self, n_mfcc_features=160, n_acoustic_features=39,
                 wav2vec2_embed_dim=1024, hidden_dim=256, dropout=0.4):
        super().__init__()
        self.n_mfcc_features = n_mfcc_features
        self.n_acoustic_features = n_acoustic_features
        self.w2v_dim = wav2vec2_embed_dim

        # Path 1 - XLS-R MLP
        self.w2v_encoder = nn.Sequential(
            nn.Linear(wav2vec2_embed_dim, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(dropout * 0.5),
        )
        # Path 2 - CNN (mel-spec)
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout2d(0.2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout2d(0.2),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d((2, 2)), nn.Dropout2d(0.3),
        )
        cnn_out = 128 * 2 * 2  # 512

        # Path 3 - MFCC MLP
        self.mfcc_encoder = nn.Sequential(
            nn.Linear(n_mfcc_features, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(dropout * 0.5),
        )
        # Path 4 - Acoustic MLP
        self.acoustic_encoder = nn.Sequential(
            nn.Linear(n_acoustic_features, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(),
        )

        fusion_dim = 256 + cnn_out + 128 + 64  # 960
        se_r = 8
        self.se = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim // se_r), nn.ReLU(),
            nn.Linear(fusion_dim // se_r, fusion_dim), nn.Sigmoid(),
        )
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(dropout * 0.7),
            nn.Linear(128, 2),
        )
        # Feature head for CMAFN fusion -> 512-dim (named audio_feature_extractor
        # in the fusion training notebook; renamed here to avoid confusion)
        self.audio_feature_head = nn.Sequential(
            nn.Linear(fusion_dim, 512), nn.BatchNorm1d(512), nn.ReLU(),
        )
    def forward(self, mel_spec, mfcc, acoustic, w2v_emb, return_features=False):
        single = mel_spec.size(0) == 1 and self.training
        if single:
            self.eval()

        x_w2v = self.w2v_encoder(w2v_emb)
        x_cnn = self.conv3(self.conv2(self.conv1(mel_spec)))
        x_cnn = x_cnn.view(x_cnn.size(0), -1)
        x_mfcc = self.mfcc_encoder(mfcc)
        x_ac = self.acoustic_encoder(acoustic)

        fused = torch.cat([x_w2v, x_cnn, x_mfcc, x_ac], dim=1)
        fused = fused * self.se(fused)

        if return_features:
            # Project to 512-d for CMAFN fusion (matches fusion training)
            out = self.audio_feature_head(fused)
            if single:
                self.train()
            return out

        logits = self.fusion(fused)
        if single:
            self.train()
        return {"logits": logits}


# ════════════════════════════════════════════════════════════════
# ░░ 3. CMAFN FUSION MODEL ░░
# ════════════════════════════════════════════════════════════════

class ModalityProjection(nn.Module):
    def __init__(self, input_dim, embed_dim, dropout=0.1):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_dim, embed_dim), nn.LayerNorm(embed_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim), nn.LayerNorm(embed_dim),
        )

    def forward(self, x):
        return self.proj(x)


class CrossModalTransformerLayer(nn.Module):
    """Bidirectional cross-modal attention between handwriting and audio."""

    def __init__(self, embed_dim, n_heads, dropout=0.1):
        super().__init__()
        self.hw_cross_attn = nn.MultiheadAttention(embed_dim, n_heads, dropout=dropout, batch_first=True)
        self.hw_norm1 = nn.LayerNorm(embed_dim)
        self.hw_ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim), nn.Dropout(dropout),
        )
        self.hw_norm2 = nn.LayerNorm(embed_dim)

        self.audio_cross_attn = nn.MultiheadAttention(embed_dim, n_heads, dropout=dropout, batch_first=True)
        self.audio_norm1 = nn.LayerNorm(embed_dim)
        self.audio_ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim), nn.Dropout(dropout),
        )
        self.audio_norm2 = nn.LayerNorm(embed_dim)

    def forward(self, hw_tokens, audio_tokens):
        hw_cross, _ = self.hw_cross_attn(query=hw_tokens, key=audio_tokens, value=audio_tokens)
        hw_tokens = self.hw_norm1(hw_tokens + hw_cross)
        hw_tokens = self.hw_norm2(hw_tokens + self.hw_ffn(hw_tokens))

        audio_cross, _ = self.audio_cross_attn(query=audio_tokens, key=hw_tokens, value=hw_tokens)
        audio_tokens = self.audio_norm1(audio_tokens + audio_cross)
        audio_tokens = self.audio_norm2(audio_tokens + self.audio_ffn(audio_tokens))
        return hw_tokens, audio_tokens


class GatedMultimodalUnit(nn.Module):
    """GMU — learnable per-sample modality weighting."""

    def __init__(self, dim_hw, dim_audio, hidden_dim):
        super().__init__()
        self.fc_hw = nn.Linear(dim_hw, hidden_dim)
        self.fc_audio = nn.Linear(dim_audio, hidden_dim)
        self.gate = nn.Sequential(nn.Linear(dim_hw + dim_audio, hidden_dim), nn.Sigmoid())
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, hw_feat, audio_feat):
        h_hw = torch.tanh(self.fc_hw(hw_feat))
        h_audio = torch.tanh(self.fc_audio(audio_feat))
        z = self.gate(torch.cat([hw_feat, audio_feat], dim=-1))
        return self.norm(z * h_hw + (1 - z) * h_audio)


class CrossModalAttentionFusionNetwork(nn.Module):
    """CMAFN — full fusion architecture with MC-Dropout uncertainty."""

    def __init__(self, config: FusionConfig):
        super().__init__()
        self.config = config
        d = config.fusion_embed_dim

        self.hw_projection = ModalityProjection(config.hw_feature_dim, d, config.fusion_dropout)
        self.audio_projection = ModalityProjection(config.audio_feature_dim, d, config.fusion_dropout)

        self.hw_default_token = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.audio_default_token = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.hw_type_embed = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.audio_type_embed = nn.Parameter(torch.randn(1, 1, d) * 0.02)

        self.cross_modal_layers = nn.ModuleList([
            CrossModalTransformerLayer(d, config.n_attention_heads, config.fusion_dropout)
            for _ in range(config.n_transformer_layers)
        ])
        self.gmu = GatedMultimodalUnit(d, d, config.gmu_hidden_dim)

        cls_in = d * 2 + config.gmu_hidden_dim
        self.classifier = nn.Sequential(
            nn.Linear(cls_in, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(config.fusion_dropout),
            nn.Linear(256, 128),    nn.LayerNorm(128), nn.GELU(), nn.Dropout(config.fusion_dropout),
            nn.Linear(128, 2),
        )
        self.hw_head = nn.Sequential(nn.Linear(d, 64), nn.GELU(), nn.Dropout(0.3), nn.Linear(64, 2))
        self.audio_head = nn.Sequential(nn.Linear(d, 64), nn.GELU(), nn.Dropout(0.3), nn.Linear(64, 2))

    def forward(self, hw_features, audio_features, hw_mask=None, audio_mask=None):
        B = hw_features.size(0)
        dev = hw_features.device
        if hw_mask is None:
            hw_mask = torch.ones(B, dtype=torch.bool, device=dev)
        if audio_mask is None:
            audio_mask = torch.ones(B, dtype=torch.bool, device=dev)

        if self.training and self.config.modality_dropout > 0:
            for i in range(B):
                r = random.random()
                if r < self.config.modality_dropout / 2:
                    hw_mask[i] = False
                elif r < self.config.modality_dropout:
                    audio_mask[i] = False

        hw_proj = self.hw_projection(hw_features)
        audio_proj = self.audio_projection(audio_features)

        hw_def = self.hw_default_token.expand(B, -1, -1).squeeze(1)
        audio_def = self.audio_default_token.expand(B, -1, -1).squeeze(1)
        hw_proj = torch.where(hw_mask.unsqueeze(-1), hw_proj, hw_def)
        audio_proj = torch.where(audio_mask.unsqueeze(-1), audio_proj, audio_def)

        hw_tok = (hw_proj + self.hw_type_embed.squeeze(1)).unsqueeze(1)
        audio_tok = (audio_proj + self.audio_type_embed.squeeze(1)).unsqueeze(1)

        for layer in self.cross_modal_layers:
            hw_tok, audio_tok = layer(hw_tok, audio_tok)

        hw_att = hw_tok.squeeze(1)
        audio_att = audio_tok.squeeze(1)
        gmu_out = self.gmu(hw_att, audio_att)

        fused = torch.cat([hw_att, audio_att, gmu_out], dim=-1)
        return {
            "logits": self.classifier(fused),
            "hw_logits": self.hw_head(hw_att),
            "audio_logits": self.audio_head(audio_att),
            "hw_attended": hw_att,
            "audio_attended": audio_att,
        }

    @torch.no_grad()
    def predict_with_uncertainty(self, hw_features, audio_features,
                                 hw_mask=None, audio_mask=None, n_samples=10):
        self.train()  # Enable dropout for MC sampling
        all_probs = []
        for _ in range(n_samples):
            out = self.forward(hw_features, audio_features, hw_mask, audio_mask)
            all_probs.append(F.softmax(out["logits"], dim=-1))
        self.eval()
        stacked = torch.stack(all_probs)
        mean_p = stacked.mean(0)
        std_p = stacked.std(0)
        return {
            "predictions": mean_p.argmax(-1),
            "mean_probs": mean_p,
            "confidence": mean_p.max(-1).values,
            "uncertainty": std_p.mean(-1),
        }


# ════════════════════════════════════════════════════════════════
# ░░ 4. AUDIO FEATURE EXTRACTION ░░
# ════════════════════════════════════════════════════════════════

class XLSRExtractor:
    """XLS-R 300M frozen embedding extractor (optional)."""

    def __init__(self, model_name="facebook/wav2vec2-xls-r-300m"):
        if not TRANSFORMERS_AVAILABLE:
            self.available = False
            return
        self.available = True
        print(f"   Loading XLS-R: {model_name} ...")
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
        self.model = Wav2Vec2Model.from_pretrained(model_name).eval().to(DEVICE)
        for p in self.model.parameters():
            p.requires_grad = False
        n_params = sum(p.numel() for p in self.model.parameters())
        print(f"   [OK] XLS-R loaded ({n_params / 1e6:.0f}M params)")

    @torch.no_grad()
    def extract(self, waveform_np: np.ndarray) -> np.ndarray:
        if not self.available:
            return np.zeros(1024, dtype=np.float32)
        inp = self.feature_extractor(waveform_np, sampling_rate=16000,
                                     return_tensors="pt", padding=True)
        out = self.model(inp.input_values.to(DEVICE))
        return out.last_hidden_state.mean(dim=1).squeeze(0).cpu().numpy()


class AudioFeaturePipeline:
    """Extracts all four feature types the Wav2VecAudioModel expects.

    Returns:
        mel_spec: (1, 1, n_mels, T) tensor
        mfcc:     (160,) numpy array
        acoustic: (39,)  numpy array
    """

    def __init__(self, sr=16000, n_mfcc=40, n_mels=128, n_fft=2048, hop_length=512,
                 use_delta=True, use_contrast=True, use_chroma=True, use_tonnetz=True):
        self.sr = sr
        self.n_mfcc = n_mfcc
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.use_delta = use_delta
        self.use_contrast = use_contrast
        self.use_chroma = use_chroma
        self.use_tonnetz = use_tonnetz

        if TORCHAUDIO_AVAILABLE:
            self.mel_transform = T.MelSpectrogram(
                sample_rate=sr, n_fft=n_fft, hop_length=hop_length,
                n_mels=n_mels, power=2.0,
            )
            self.amp_to_db = T.AmplitudeToDB()
        else:
            self.mel_transform = None
            self.amp_to_db = None

    # ── mel-spectrogram ─────────────────────────────────────
    def mel_spectrogram(self, wav: np.ndarray) -> torch.Tensor:
        """Returns (1, 1, n_mels, T) tensor."""
        if self.mel_transform is not None:
            t = torch.from_numpy(wav).float().unsqueeze(0)
            mel = self.mel_transform(t)          # (1, n_mels, T)
            mel = self.amp_to_db(mel)             # (1, n_mels, T)
            return mel.unsqueeze(0)               # (1, 1, n_mels, T)
        else:
            mel_np = librosa.feature.melspectrogram(
                y=wav, sr=self.sr, n_fft=self.n_fft,
                hop_length=self.hop_length, n_mels=self.n_mels,
            )
            mel_db = librosa.power_to_db(mel_np, ref=np.max)
            return torch.from_numpy(mel_db).float().unsqueeze(0).unsqueeze(0)

    # ── MFCC + delta ────────────────────────────────────────
    def mfcc_features(self, wav: np.ndarray) -> np.ndarray:
        mfcc = librosa.feature.mfcc(y=wav, sr=self.sr, n_mfcc=self.n_mfcc)
        feats = np.concatenate([np.mean(mfcc, 1), np.std(mfcc, 1)])  # 80
        if self.use_delta:
            delta = librosa.feature.delta(mfcc, order=1)
            feats = np.concatenate([feats, np.mean(delta, 1), np.std(delta, 1)])  # 160
        return feats.astype(np.float32)

    # ── voice quality (parselmouth) ─────────────────────────
    def voice_quality(self, wav: np.ndarray) -> dict:
        keys_praat = ["mean_pitch", "std_pitch", "jitter_local", "jitter_rap",
                      "shimmer_local", "hnr", "f1_mean", "f2_mean", "f3_mean"]
        vq: dict = {}
        if PARSELMOUTH_AVAILABLE and len(wav) > self.sr * 0.5:
            try:
                snd = parselmouth.Sound(wav, sampling_frequency=self.sr)
                pitch = snd.to_pitch(time_step=0.01)
                vq["mean_pitch"] = praat_call(pitch, "Get mean", 0, 0, "Hertz")
                vq["std_pitch"] = praat_call(pitch, "Get standard deviation", 0, 0, "Hertz")
                pp = praat_call(snd, "To PointProcess (periodic, cc)", 75, 500)
                vq["jitter_local"] = praat_call(pp, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
                vq["jitter_rap"] = praat_call(pp, "Get jitter (rap)", 0, 0, 0.0001, 0.02, 1.3)
                vq["shimmer_local"] = praat_call([snd, pp], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
                harm = praat_call(snd, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
                vq["hnr"] = praat_call(harm, "Get mean", 0, 0)
                fmt = snd.to_formant_burg(time_step=0.01)
                vq["f1_mean"] = praat_call(fmt, "Get mean", 1, 0, 0, "Hertz")
                vq["f2_mean"] = praat_call(fmt, "Get mean", 2, 0, 0, "Hertz")
                vq["f3_mean"] = praat_call(fmt, "Get mean", 3, 0, 0, "Hertz")
            except Exception:
                for k in keys_praat:
                    vq.setdefault(k, 0.0)
        else:
            for k in keys_praat:
                vq[k] = 0.0

        # librosa-based features (always available)
        vq["spectral_centroid"] = float(np.mean(librosa.feature.spectral_centroid(y=wav, sr=self.sr)))
        vq["spectral_rolloff"] = float(np.mean(librosa.feature.spectral_rolloff(y=wav, sr=self.sr)))
        vq["zcr"] = float(np.mean(librosa.feature.zero_crossing_rate(wav)))
        rms = librosa.feature.rms(y=wav)
        vq["rms_mean"] = float(np.mean(rms))
        vq["rms_std"] = float(np.std(rms))
        return vq

    # ── extra spectral features ─────────────────────────────
    def extras(self, wav: np.ndarray) -> np.ndarray:
        parts = []
        if self.use_contrast:
            sc = librosa.feature.spectral_contrast(y=wav, sr=self.sr, n_bands=6)
            parts.append(np.mean(sc, 1))  # 7
        if self.use_chroma:
            ch = librosa.feature.chroma_stft(y=wav, sr=self.sr)
            parts.append(np.mean(ch, 1))  # 12
        if self.use_tonnetz:
            try:
                tn = librosa.feature.tonnetz(y=wav, sr=self.sr)
                parts.append(np.mean(tn, 1))  # 6
            except Exception:
                parts.append(np.zeros(6))
        return np.concatenate(parts).astype(np.float32) if parts else np.array([], dtype=np.float32)

    # ── acoustic vector (voice quality + extras) -> 39 dim ──
    def acoustic_features(self, wav: np.ndarray) -> np.ndarray:
        vq = self.voice_quality(wav)
        vq_arr = np.array(list(vq.values()), dtype=np.float32)
        # Replace NaN/inf with 0
        vq_arr = np.nan_to_num(vq_arr, nan=0.0, posinf=0.0, neginf=0.0)
        ext = self.extras(wav)
        ext = np.nan_to_num(ext, nan=0.0, posinf=0.0, neginf=0.0)
        ac = np.concatenate([vq_arr, ext]) if len(ext) > 0 else vq_arr
        # pad/truncate to exactly 39 dimensions
        if len(ac) < 39:
            ac = np.concatenate([ac, np.zeros(39 - len(ac), dtype=np.float32)])
        return ac[:39]

    # ── full extraction ─────────────────────────────────────
    def __call__(self, wav: np.ndarray):
        mel = self.mel_spectrogram(wav)
        mfcc = self.mfcc_features(wav)
        acoustic = self.acoustic_features(wav)
        return mel, mfcc, acoustic


# ════════════════════════════════════════════════════════════════
# ░░ 5. IMAGE PREPROCESSING ░░
# ════════════════════════════════════════════════════════════════

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])


def preprocess_image(image, size=336):
    """Preprocess a handwriting image -> (1, 3, size, size) tensor."""
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    image = image.convert("RGB").resize((size, size), Image.LANCZOS)
    arr = np.array(image, dtype=np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0).float()


# ════════════════════════════════════════════════════════════════
# ░░ 6. MODEL LOADING ░░
# ════════════════════════════════════════════════════════════════

ROOT = Path(os.path.dirname(os.path.abspath(__file__)))

# Register config classes in __main__ so pickle can find them when loading
# checkpoints that were saved from Jupyter notebooks (where __main__ is the
# notebook kernel).
import __main__
__main__.AudioConfig = AudioConfig
__main__.FusionConfig = FusionConfig


def _load_state(path, model, strict=True, label="model"):
    """Load state_dict from a checkpoint file into a model."""
    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"], strict=strict)
    elif isinstance(ckpt, dict) and "best_fold_state_dict" in ckpt:
        model.load_state_dict(ckpt["best_fold_state_dict"], strict=strict)
    else:
        model.load_state_dict(ckpt, strict=strict)
    print(f"   [OK] {label} loaded from {path.name}")
    return ckpt


# ── Handwriting model ──────────────────────────────────────
print("\n[*] Loading Handwriting Model ...")
hw_model = HandwritingParkinsonsModel(num_classes=2, pretrained=False)
hw_loaded = False
for candidate in [
    ROOT / "handwriting_parkinsons_model_final(2).pth",
    ROOT / "best_handwriting_model(2).pth",
    ROOT / "best_handwriting_model.pth",
]:
    if candidate.exists():
        try:
            _load_state(candidate, hw_model, label="Handwriting")
            hw_loaded = True
            break
        except Exception as e:
            print(f"   [!] {candidate.name}: {e}")
if not hw_loaded:
    print("   [!] No handwriting checkpoint found -- using random weights")
hw_model = hw_model.to(DEVICE).eval()

# ── Audio models (5-fold ensemble, matching notebook training) ─────
print("\n[*] Loading Audio Models (5-fold ensemble) ...")
audio_models: List[Wav2VecAudioModel] = []
audio_loaded = False

# Load all 5 fold models for ensemble prediction
for fi in range(1, 6):
    # Handle filenames with parenthesized suffixes: fold_1_model(3).pth etc.
    candidates = [ROOT / "checkpoints" / f"fold_{fi}_model.pth"]
    for suffix in range(10):
        candidates.append(ROOT / "checkpoints" / f"fold_{fi}_model({suffix}).pth")
    for fold_path in candidates:
        if fold_path.exists():
            try:
                m = Wav2VecAudioModel()
                _load_state(fold_path, m, strict=False, label=f"Audio fold {fi}")
                m = m.to(DEVICE).eval()
                audio_models.append(m)
                break
            except Exception as e:
                print(f"   [!] Audio fold {fi}: {e}")

# Fallback: try best_audio_model.pth
if not audio_models:
    best_path = ROOT / "checkpoints" / "best_audio_model.pth"
    if best_path.exists():
        try:
            m = Wav2VecAudioModel()
            _load_state(best_path, m, strict=False, label="Audio (best)")
            m = m.to(DEVICE).eval()
            audio_models.append(m)
        except Exception as e:
            print(f"   [!] Audio best model: {e}")

if audio_models:
    audio_loaded = True
    print(f"   [OK] Audio ensemble: {len(audio_models)} models ready")
else:
    # Create a single untrained model as fallback
    audio_models.append(Wav2VecAudioModel().to(DEVICE).eval())
    print("   [!] No audio checkpoints -- using random weights")

# Keep reference to first model for feature extraction in CMAFN
audio_model = audio_models[0]

# ── CMAFN fusion models (5-fold ensemble) ──────────────────
print("\n[*] Loading CMAFN Fusion Models ...")
fusion_config = FusionConfig()
fusion_models: List[CrossModalAttentionFusionNetwork] = []

# Try the all-in-one package first
cmafn_path = ROOT / "checkpoint_fusion" / "cmafn_final_model.pth"
if cmafn_path.exists():
    try:
        ckpt = torch.load(cmafn_path, map_location=DEVICE, weights_only=False)
        # rebuild config from saved dict
        if "config" in ckpt:
            fusion_config = FusionConfig(**ckpt["config"])

        state_dicts = ckpt.get("ensemble_state_dicts", [])
        if not state_dicts and "best_fold_state_dict" in ckpt:
            state_dicts = [ckpt["best_fold_state_dict"]]

        for i, sd in enumerate(state_dicts):
            m = CrossModalAttentionFusionNetwork(fusion_config).to(DEVICE)
            m.load_state_dict(sd)
            m.eval()
            fusion_models.append(m)
        print(f"   [OK] Loaded {len(fusion_models)} fold models from cmafn_final_model.pth")
    except Exception as e:
        print(f"   [!] cmafn_final_model.pth error: {e}")

# Fall back to individual fold checkpoints
if not fusion_models:
    for fi in range(1, 6):
        fp = ROOT / "checkpoint_fusion" / f"best_fusion_fold_{fi}.pth"
        if fp.exists():
            try:
                m = CrossModalAttentionFusionNetwork(fusion_config).to(DEVICE)
                _load_state(fp, m, label=f"Fusion fold {fi}")
                m.eval()
                fusion_models.append(m)
            except Exception as e:
                print(f"   [!] Fold {fi}: {e}")

if not fusion_models:
    print("   [!] No fusion checkpoints -- CMAFN analysis unavailable")
else:
    print(f"   [OK] Fusion ensemble: {len(fusion_models)} models ready")

# ── XLS-R extractor (lazy-loaded on first audio call) ──────
xlsr_extractor: Optional[XLSRExtractor] = None


def _get_xlsr():
    global xlsr_extractor
    if xlsr_extractor is None:
        print("[*] Loading XLS-R extractor (first audio call) ...")
        xlsr_extractor = XLSRExtractor()
    return xlsr_extractor


# ── Audio feature pipeline ─────────────────────────────────
audio_pipeline = AudioFeaturePipeline()

# ── Load performance metrics from training ─────────────────
fusion_results_path = ROOT / "fusion_results.json"
METRICS: dict = {}
if fusion_results_path.exists():
    try:
        with open(fusion_results_path) as f:
            METRICS = json.load(f)
    except Exception:
        pass

print("\n[OK] All models loaded!\n")


# ════════════════════════════════════════════════════════════════
# ░░ 7. AUDIO PREPROCESSING HELPER ░░
# ════════════════════════════════════════════════════════════════

def prepare_audio(audio_tuple, target_sr=16000, max_seconds=8, min_seconds=1):
    """Convert Gradio audio input -> 16 kHz float32 mono waveform."""
    sr, data = audio_tuple

    # Mono
    if len(data.shape) > 1:
        data = data.mean(axis=1)

    # Float conversion
    if data.dtype == np.int16:
        data = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        data = data.astype(np.float32) / 2147483648.0
    else:
        data = data.astype(np.float32)

    # Resample
    if sr != target_sr:
        data = librosa.resample(data, orig_sr=sr, target_sr=target_sr)

    # Pad / truncate
    min_len = min_seconds * target_sr
    max_len = max_seconds * target_sr
    if len(data) < min_len:
        data = np.pad(data, (0, min_len - len(data)))
    if len(data) > max_len:
        data = data[:max_len]

    # Peak normalise
    peak = np.max(np.abs(data)) + 1e-8
    data = data / peak
    return data


# ════════════════════════════════════════════════════════════════
# ░░ 8. PREDICTION FUNCTIONS ░░
# ════════════════════════════════════════════════════════════════

# ── 8a. Handwriting ─────────────────────────────────────────

def predict_handwriting(image):
    """Predict PD from handwriting with TTA + temperature scaling."""
    if image is None:
        return None, "⚠️ Please upload a handwriting image."
    try:
        img_t = preprocess_image(image, size=336).to(DEVICE)
        with torch.no_grad():
            outputs = [hw_model(img_t)]
            outputs.append(hw_model(torch.flip(img_t, [3])))  # H-flip TTA
            logits = torch.stack(outputs).mean(0)
            probs = F.softmax(logits / 1.5, dim=1)  # temperature=1.5
            healthy_p = probs[0, 0].item()
            pd_p = probs[0, 1].item()

        result = {"Healthy": healthy_p, "Parkinson's": pd_p}
        threshold = 0.55
        if pd_p > threshold:
            txt = (f"⚠️ **Potential Parkinson's Indicators Detected**\n\n"
                   f"**Confidence:** {pd_p * 100:.1f}%\n\n"
                   f"**Threshold:** {threshold * 100:.0f}%\n\n"
                   f"*This is a screening tool — consult a medical professional.*")
        else:
            txt = (f"✅ **Healthy Pattern Detected**\n\n"
                   f"**Confidence:** {healthy_p * 100:.1f}%\n\n")
            if pd_p > 0.40:
                txt += (f"⚠️ Parkinson's probability ({pd_p * 100:.1f}%) is elevated "
                        f"but below threshold. Consider monitoring.\n\n")
            txt += "*Regular checkups are still recommended.*"
        return result, txt
    except Exception as e:
        return None, f"❌ Error: {e}"


# ── 8b. Speech / Audio ─────────────────────────────────────

def predict_speech(audio):
    """Predict PD from speech using 5-fold DL ensemble (matching notebook training)."""
    if audio is None:
        return None, "⚠️ Please upload or record audio."
    try:
        wav = prepare_audio(audio)

        # Feature extraction — identical to notebook's AudioFeatureExtractor
        mel, mfcc_feat, acoustic_feat = audio_pipeline(wav)

        mfcc_t = torch.from_numpy(mfcc_feat).float().unsqueeze(0).to(DEVICE)
        acoustic_t = torch.from_numpy(acoustic_feat).float().unsqueeze(0).to(DEVICE)

        # XLS-R embedding (lazy loaded)
        xlsr = _get_xlsr()
        w2v_emb = xlsr.extract(wav)
        w2v_t = torch.from_numpy(w2v_emb).float().unsqueeze(0).to(DEVICE)

        mel = mel.to(DEVICE)

        # ── 5-fold DL ensemble (no temperature — matches training) ──
        all_probs = []
        with torch.no_grad():
            for model in audio_models:
                model.eval()
                out = model(mel, mfcc_t, acoustic_t, w2v_t)
                probs = F.softmax(out["logits"], dim=1)  # raw softmax, no temperature
                all_probs.append(probs.cpu().numpy())

        # Average ensemble probabilities
        avg_probs = np.mean(all_probs, axis=0)  # (1, 2)
        healthy_p = float(avg_probs[0, 0])
        pd_p = float(avg_probs[0, 1])
        ensemble_std = float(np.std([p[0, 1] for p in all_probs]))

        result = {"Healthy": round(healthy_p, 4), "Parkinson's": round(pd_p, 4)}

        # Classification using optimized threshold
        # (notebook uses threshold optimization via macro-F1 on validation)
        decision_threshold = 0.50
        is_pd = pd_p >= decision_threshold

        if pd_p > 0.70:
            severity = "Significant Indicators"
            icon = "🔴"
        elif pd_p > 0.55:
            severity = "Moderate Indicators"
            icon = "🟠"
        elif pd_p > 0.40:
            severity = "Mild Indicators"
            icon = "🟡"
        else:
            severity = "Minimal / No Indicators"
            icon = "🟢"

        # Confidence based on ensemble agreement + probability margin
        if ensemble_std < 0.05 and abs(pd_p - 0.5) > 0.20:
            confidence = "High"
        elif ensemble_std < 0.10 and abs(pd_p - 0.5) > 0.10:
            confidence = "Moderate"
        else:
            confidence = "Low"

        # Voice quality markers for interpretability
        vq = audio_pipeline.voice_quality(wav)
        vq_lines = []
        if PARSELMOUTH_AVAILABLE:
            jitter = vq.get('jitter_local', 0)
            shimmer = vq.get('shimmer_local', 0)
            hnr = vq.get('hnr', 0)
            pitch = vq.get('mean_pitch', 0)
            if jitter > 0:
                jitter_status = "elevated" if jitter > 0.01 else "normal"
                vq_lines.append(f"| Jitter (local) | {jitter:.5f} | {jitter_status} |")
            if shimmer > 0:
                shimmer_status = "elevated" if shimmer > 0.03 else "normal"
                vq_lines.append(f"| Shimmer (local) | {shimmer:.5f} | {shimmer_status} |")
            if hnr > 0:
                hnr_status = "reduced" if hnr < 15 else "normal"
                vq_lines.append(f"| HNR (dB) | {hnr:.2f} | {hnr_status} |")
            if pitch > 0:
                vq_lines.append(f"| Mean Pitch (Hz) | {pitch:.1f} | - |")

        vq_table = ""
        if vq_lines:
            vq_table = "\n#### Voice Quality Biomarkers:\n| Feature | Value | Status |\n|---|---|---|\n" + "\n".join(vq_lines) + "\n\n"

        decision_text = "**Parkinsons Detected**" if is_pd else "**Healthy Pattern**"
        xlsr_line = "- XLS-R multilingual embeddings (1024 -> 256-d)" if TRANSFORMERS_AVAILABLE else "- XLS-R unavailable -- zero-filled (install transformers for full accuracy)"

        analysis = f"""### Speech Analysis Results

**Classification:** {icon} {severity}
**Decision:** {decision_text}

| Metric | Value |
|---|---|
| Parkinson's Probability | {pd_p * 100:.1f}% |
| Healthy Probability | {healthy_p * 100:.1f}% |
| Decision Threshold | {decision_threshold * 100:.0f}% |
| Ensemble Models | {len(audio_models)} |
| Ensemble Agreement (std) | ±{ensemble_std * 100:.2f}% |
| Confidence | {confidence} |

---
{vq_table}
#### Interpretation Guide:
- 🟢 **< 40%**: Minimal / No PD speech indicators
- 🟡 **40-55%**: Mild speech changes detected
- 🟠 **55-70%**: Moderate speech patterns
- 🔴 **> 70%**: Significant PD speech indicators

---

#### What Was Analyzed:
- ✓ Mel-spectrogram (CNN pathway → 512-d)
- ✓ MFCC + Delta features (40 coefficients → 160-d)
- ✓ Voice quality — jitter, shimmer, HNR, formants (14-d)
- ✓ Spectral contrast, chroma, tonnetz (25-d)
- {xlsr_line}
- SE-Attention fusion (960-d weighted re-ranking)

---

*⚠️ Screening tool only — consult a neurologist for proper diagnosis.*"""

        return result, analysis
    except Exception as e:
        import traceback
        return None, f"❌ Error: {e}\n```\n{traceback.format_exc()}\n```"


# ── 8c. CMAFN Fusion (Combined) ────────────────────────────

def predict_fusion(image, audio):
    """Run the CMAFN 5-fold ensemble fusion on both modalities."""
    has_image = image is not None
    has_audio = audio is not None

    if not has_image and not has_audio:
        return "⚠️ Please provide at least one input (handwriting image or speech audio)."

    if not fusion_models:
        return "⚠️ CMAFN fusion models are not loaded. Check `checkpoint_fusion/` directory."

    report = "# 🧬 Cross-Modal Attention Fusion Network (CMAFN) Analysis\n\n"

    # ── extract handwriting features ────────────────────────
    hw_feat = None
    hw_standalone = None
    if has_image:
        img_t = preprocess_image(image, size=336).to(DEVICE)
        with torch.no_grad():
            hw_feat = hw_model(img_t, return_features=True)  # (1, 512)
            hw_logits = hw_model(img_t)
            hw_probs = F.softmax(hw_logits / 1.5, dim=1)
            hw_standalone = {"healthy": hw_probs[0, 0].item(), "pd": hw_probs[0, 1].item()}

        report += "## ✍️ Handwriting Analysis (Standalone)\n\n"
        report += f"- **Healthy:** {hw_standalone['healthy'] * 100:.1f}%\n"
        report += f"- **Parkinson's:** {hw_standalone['pd'] * 100:.1f}%\n\n"
    else:
        hw_feat = torch.zeros(1, fusion_config.hw_feature_dim, device=DEVICE)
        report += "## ✍️ Handwriting: *Not provided*\n\n"

    # ── extract audio features ──────────────────────────────
    audio_feat = None
    audio_standalone = None
    if has_audio:
        wav = prepare_audio(audio)
        mel, mfcc_feat, acoustic_feat = audio_pipeline(wav)
        mfcc_t = torch.from_numpy(mfcc_feat).float().unsqueeze(0).to(DEVICE)
        acoustic_t = torch.from_numpy(acoustic_feat).float().unsqueeze(0).to(DEVICE)
        xlsr = _get_xlsr()
        w2v_emb = xlsr.extract(wav)
        w2v_t = torch.from_numpy(w2v_emb).float().unsqueeze(0).to(DEVICE)
        mel = mel.to(DEVICE)

        with torch.no_grad():
            audio_feat = audio_model(mel, mfcc_t, acoustic_t, w2v_t, return_features=True)  # (1, 512)
            # Ensemble standalone audio prediction (no temperature)
            all_audio_probs = []
            for am in audio_models:
                am.eval()
                ap = F.softmax(am(mel, mfcc_t, acoustic_t, w2v_t)["logits"], dim=1)
                all_audio_probs.append(ap.cpu().numpy())
            avg_ap = np.mean(all_audio_probs, axis=0)
            audio_standalone = {"healthy": float(avg_ap[0, 0]), "pd": float(avg_ap[0, 1])}

        report += "## 🎤 Speech Analysis (Standalone)\n\n"
        report += f"- **Healthy:** {audio_standalone['healthy'] * 100:.1f}%\n"
        report += f"- **Parkinson's:** {audio_standalone['pd'] * 100:.1f}%\n\n"
    else:
        audio_feat = torch.zeros(1, fusion_config.audio_feature_dim, device=DEVICE)
        report += "## 🎤 Speech: *Not provided*\n\n"

    # ── Masks ───────────────────────────────────────────────
    hw_mask = torch.tensor([has_image], dtype=torch.bool, device=DEVICE)
    audio_mask = torch.tensor([has_audio], dtype=torch.bool, device=DEVICE)

    # ── Ensemble prediction ─────────────────────────────────
    all_probs = []
    for model in fusion_models:
        model.eval()
        with torch.no_grad():
            out = model(hw_feat, audio_feat, hw_mask.clone(), audio_mask.clone())
            probs = F.softmax(out["logits"], dim=-1).cpu().numpy()
            all_probs.append(probs)

    avg_probs = np.mean(all_probs, axis=0)  # (1, 2)
    h_prob, pd_prob = float(avg_probs[0, 0]), float(avg_probs[0, 1])
    prob_std = float(np.std([p[0, 1] for p in all_probs]))

    # ── MC-Dropout uncertainty ──────────────────────────────
    mc_results = []
    for model in fusion_models:
        res = model.predict_with_uncertainty(
            hw_feat, audio_feat, hw_mask.clone(), audio_mask.clone(),
            n_samples=fusion_config.mc_dropout_samples,
        )
        mc_results.append(res)

    mc_mean_probs = torch.stack([r["mean_probs"] for r in mc_results]).mean(0)
    mc_uncertainty = torch.stack([r["uncertainty"] for r in mc_results]).mean().item()
    mc_h, mc_pd = float(mc_mean_probs[0, 0]), float(mc_mean_probs[0, 1])

    # ── Classification ──────────────────────────────────────
    if pd_prob > 0.60:
        risk = "HIGH"
        icon = "🔴"
    elif pd_prob > 0.45:
        risk = "MODERATE"
        icon = "🟠"
    elif pd_prob > 0.35:
        risk = "LOW-MODERATE"
        icon = "🟡"
    else:
        risk = "LOW"
        icon = "🟢"

    if mc_uncertainty < 0.05:
        conf_label = "High"
    elif mc_uncertainty < 0.10:
        conf_label = "Moderate"
    else:
        conf_label = "Low"

    # ── Build Report ────────────────────────────────────────
    report += "---\n\n"
    report += "## 🧬 CMAFN Fusion Result\n\n"
    report += f"### {icon} Risk Level: **{risk}**\n\n"
    report += f"| Metric | Value |\n|---|---|\n"
    report += f"| Ensemble Healthy Prob | {h_prob * 100:.1f}% |\n"
    report += f"| Ensemble Parkinson's Prob | {pd_prob * 100:.1f}% |\n"
    report += f"| Ensemble Agreement (std) | ±{prob_std * 100:.2f}% |\n"
    report += f"| MC-Dropout Healthy Prob | {mc_h * 100:.1f}% |\n"
    report += f"| MC-Dropout Parkinson's Prob | {mc_pd * 100:.1f}% |\n"
    report += f"| MC-Dropout Uncertainty | {mc_uncertainty:.4f} |\n"
    report += f"| Confidence | {conf_label} |\n"
    report += f"| Ensemble Models Used | {len(fusion_models)} |\n"
    report += f"| MC Samples per Model | {fusion_config.mc_dropout_samples} |\n\n"

    # ── Concordance ─────────────────────────────────────────
    if has_image and has_audio and hw_standalone and audio_standalone:
        hw_says_pd = hw_standalone["pd"] > 0.55
        audio_says_pd = audio_standalone["pd"] > 0.50
        if hw_says_pd == audio_says_pd:
            report += "### 🔄 Modality Concordance: ✅ **Agreement**\n\n"
            report += "Both modalities point in the same direction.\n\n"
        else:
            report += "### 🔄 Modality Concordance: ⚠️ **Discordant**\n\n"
            report += "Handwriting and speech disagree — CMAFN cross-modal attention resolves this adaptively.\n\n"

    # ── Model provenance ────────────────────────────────────
    report += "---\n\n### 📊 Model Performance (from training)\n\n"
    if METRICS:
        cv = METRICS.get("cross_validation", {})
        te = METRICS.get("test_metrics_ensemble", {})
        report += f"| Metric | Value |\n|---|---|\n"
        report += f"| CV Balanced Accuracy | {cv.get('mean_balanced_accuracy', 0):.4f} ± {cv.get('std_balanced_accuracy', 0):.4f} |\n"
        report += f"| CV AUC-ROC | {cv.get('mean_auc', 0):.4f} ± {cv.get('std_auc', 0):.4f} |\n"
        report += f"| Test Accuracy (ensemble) | {te.get('accuracy', 0) * 100:.2f}% |\n"
        report += f"| Test AUC-ROC (ensemble) | {te.get('auc_roc', 0):.4f} |\n"
        report += f"| PD Precision | {te.get('pd_precision', 0) * 100:.1f}% |\n"
        report += f"| PD Recall | {te.get('pd_recall', 0) * 100:.1f}% |\n\n"
    else:
        report += "*Training metrics not available (fusion_results.json missing).*\n\n"

    report += "---\n\n### 📋 Important Notes\n\n"
    report += "- This is a **screening tool only**, NOT a diagnostic instrument.\n"
    report += "- The CMAFN model uses **cross-modal transformer attention** — each modality enriches the other.\n"
    report += "- **Modality dropout** during training makes single-modality input viable but less reliable.\n"
    report += "- MC-Dropout uncertainty > 0.10 suggests the model is unsure — interpret with caution.\n"
    report += "\n⚕️ **Always consult healthcare professionals for proper medical evaluation.**\n"

    return report


# ════════════════════════════════════════════════════════════════
# ░░ 9. GRADIO INTERFACE ░░
# ════════════════════════════════════════════════════════════════

custom_css = """
.gradio-container { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
.main-header { text-align: center; margin-bottom: 20px; }
.tab-nav button { font-size: 16px !important; }
footer { display: none !important; }
"""

with gr.Blocks(css=custom_css, title="Parkinson's Disease Detection", theme=gr.themes.Soft()) as demo:

    gr.Markdown("""
# 🧠 Parkinson's Disease Detection System

### Multi-Modal Analysis — Handwriting · Speech · Cross-Modal Fusion

This system employs three deep-learning pipelines:
- **Handwriting** — EfficientNet-B4 + CBAM attention + Spatial Pyramid Pooling
- **Speech** — XLS-R 300M + CNN + MFCC + Acoustic features + SE attention (4-path fusion)
- **Fusion (CMAFN)** — Cross-Modal Transformer Attention + Gated Multimodal Unit (5-fold ensemble)

---
    """)

    with gr.Tabs():

        # ───────────── TAB 1: Handwriting ─────────────
        with gr.TabItem("✍️ Handwriting Analysis", id=1):
            gr.Markdown("### Upload a handwriting sample (spiral or wave drawings work best)")
            with gr.Row():
                with gr.Column(scale=1):
                    hw_input = gr.Image(label="Handwriting Sample", type="numpy", height=300)
                    hw_btn = gr.Button("🔍 Analyze Handwriting", variant="primary", size="lg")
                with gr.Column(scale=1):
                    hw_label = gr.Label(label="Classification", num_top_classes=2)
                    hw_text = gr.Markdown(label="Detailed Analysis")
            hw_btn.click(predict_handwriting, inputs=hw_input, outputs=[hw_label, hw_text])
            gr.Markdown("""
---
**Tips:** Use clear, well-lit images · Spiral drawings are most informative · Ensure writing fills most of the image
            """)

        # ───────────── TAB 2: Speech ──────────────────
        with gr.TabItem("🎤 Speech Analysis", id=2):
            gr.Markdown("### Record or upload speech (sustained vowels 'ahhh' or reading passages)")
            with gr.Row():
                with gr.Column(scale=1):
                    sp_input = gr.Audio(
                        label="Record or Upload Speech", type="numpy",
                        sources=["microphone", "upload"],
                    )
                    sp_btn = gr.Button("🔍 Analyze Speech", variant="primary", size="lg")
                with gr.Column(scale=1):
                    sp_label = gr.Label(label="Classification", num_top_classes=2)
                    sp_text = gr.Markdown(label="Detailed Analysis")
            sp_btn.click(predict_speech, inputs=sp_input, outputs=[sp_label, sp_text])
            gr.Markdown("""
---
**Tips:** Record in a quiet room · Speak for at least 3-5 seconds · Sustained vowels reveal voice tremor
            """)

        # ───────────── TAB 3: CMAFN Fusion ────────────
        with gr.TabItem("🧬 CMAFN Fusion", id=3):
            gr.Markdown("""### Cross-Modal Attention Fusion Network
Provide **both** handwriting and speech for maximum accuracy.
Single-modality input is supported but less reliable (the model uses learned default tokens for missing data).
            """)
            with gr.Row():
                with gr.Column(scale=1):
                    fusion_hw = gr.Image(label="Handwriting Sample", type="numpy", height=220)
                with gr.Column(scale=1):
                    fusion_sp = gr.Audio(
                        label="Speech Sample", type="numpy",
                        sources=["microphone", "upload"],
                    )
            fusion_btn = gr.Button("🧬 Run CMAFN Fusion Analysis", variant="primary", size="lg")
            fusion_out = gr.Markdown(label="CMAFN Report")
            fusion_btn.click(predict_fusion, inputs=[fusion_hw, fusion_sp], outputs=fusion_out)

        # ───────────── TAB 4: About ───────────────────
        with gr.TabItem("ℹ️ About", id=4):
            # Build dynamic dependency status
            dep_lines = []
            dep_lines.append(f"- **XLS-R (transformers):** {'✅ Available' if TRANSFORMERS_AVAILABLE else '⚠️ Not installed — audio path 1 uses zeros'}")
            dep_lines.append(f"- **Parselmouth:** {'✅ Available' if PARSELMOUTH_AVAILABLE else '⚠️ Not installed — voice quality features zeroed'}")
            dep_lines.append(f"- **Torchaudio:** {'✅ Available' if TORCHAUDIO_AVAILABLE else '⚠️ Fallback to librosa'}")
            dep_lines.append(f"- **CUDA:** {'✅ ' + torch.cuda.get_device_name(0) if torch.cuda.is_available() else '❌ CPU mode'}")
            dep_lines.append(f"- **Handwriting model:** {'✅ Loaded' if hw_loaded else '⚠️ Random weights'}")
            dep_lines.append(f"- **Audio model:** {'✅ ' + str(len(audio_models)) + ' fold(s) loaded' if audio_loaded else '⚠️ Random weights'}")
            dep_lines.append(f"- **Fusion models:** {f'✅ {len(fusion_models)} fold(s)' if fusion_models else '❌ Not loaded'}")
            dep_status = "\n".join(dep_lines)

            about_md = f"""
## Architecture

### 1. Handwriting Model (EfficientNet-B4 + CBAM + SPP)
| Component | Details |
|---|---|
| Backbone | EfficientNet-B4 (frozen, ImageNet pre-trained) |
| Attention | CBAM — Channel + Spatial Attention |
| Pooling | Spatial Pyramid Pooling (1x1 + 2x2 + 4x4) |
| Classifier | Linear(SPP -> 512 -> 256 -> 2) with BN + Dropout |
| Feature output | 512-dim (for fusion) |
| Standalone accuracy | 87.55% — AUC: 0.9354 |

### 2. Speech / Audio Model (4-Path Fusion + SE Attention)
| Component | Details |
|---|---|
| Path 1 | XLS-R 300M (frozen) -> 1024-d -> MLP -> 256-d |
| Path 2 | Mel-spectrogram (128 mels) -> 3-layer CNN -> 512-d |
| Path 3 | MFCC (40) + delta -> MLP -> 128-d |
| Path 4 | Acoustic features (39-d: voice quality + spectral) -> MLP -> 64-d |
| Fusion | SE-Attention on 960-d concat -> MLP -> 2 |
| Feature output | 512-dim (for fusion) |
| Standalone accuracy | 93.62% — AUC: 0.9661 |

### 3. CMAFN Fusion (Novel Architecture)
| Component | Details |
|---|---|
| Cross-Modal Transformer | 2 layers, 8 heads, bidirectional attention |
| GMU | Gated Multimodal Unit — adaptive per-sample weighting |
| Modality Dropout | 25% during training — handles missing modality |
| Contrastive Loss | Cross-modal alignment (InfoNCE, tau=0.07) |
| Ensemble | 5-fold CV models — averaged softmax |
| MC-Dropout | 10 samples x 5 models = 50 forward passes |
| Classifier input | 256 + 256 + 128 = 640-dim |
| **Test accuracy** | **96.94% — AUC: 0.9995** |

---

## System Status

{dep_status}

---

## How to Install Optional Dependencies

For full accuracy, install these optional packages:
```
pip install transformers     # XLS-R 300M embeddings
pip install praat-parselmouth  # Voice quality (jitter, shimmer, HNR)
```

---

## ⚠️ Disclaimer

This tool is for **screening purposes only** and must not replace professional medical diagnosis.
If you have concerns about Parkinson's disease, please consult a qualified neurologist.

---

**Version:** 2.0.0 | **Last Updated:** February 2026
            """
            gr.Markdown(about_md)

    gr.Markdown("""
---
<center>Made with ❤️ for healthcare · <a href="https://github.com/Tvenkatathanuj/SDP">GitHub</a></center>
    """)


# ════════════════════════════════════════════════════════════════
# ░░ LAUNCH ░░
# ════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    demo.launch(
        share=True,
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True,
    )
