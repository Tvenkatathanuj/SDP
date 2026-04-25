"""
Parkinson's Disease Speech Detection — Live Web App
====================================================
XLS-R 300M + Cross-Attention Fusion + 5-Fold Ensemble
Results: 91.7% accuracy | 0.97 AUC-ROC (patient-level 5-fold CV)

Deploy: gunicorn app:app --bind 0.0.0.0:$PORT --timeout 300
"""

import os
import sys
import warnings

# Prevent transformers from trying to import broken TensorFlow
os.environ["USE_TF"] = "0"
os.environ["TRANSFORMERS_NO_TF"] = "1"

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as T
import librosa
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from dataclasses import dataclass, field
from typing import List, Tuple
from pathlib import Path

warnings.filterwarnings("ignore")

DEVICE = torch.device("cpu")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ════════════════════════════════════════════════════════════════
# Audio Configuration (must match training)
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


config = AudioConfig()

# ════════════════════════════════════════════════════════════════
# Model Architecture (must match training notebook exactly)
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

        # Path 1: XLS-R + BiLSTM
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

        # Path 2: CNN for mel-spec
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

        # Path 3: MFCC MLP + skip
        self.mfcc_encoder = nn.Sequential(
            nn.Linear(n_mfcc_features, 256), nn.BatchNorm1d(256), nn.GELU(), nn.Dropout(cfg.dropout),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.GELU(), nn.Dropout(cfg.dropout * 0.6)
        )
        self.mfcc_skip = nn.Linear(n_mfcc_features, 128)

        # Path 4: Acoustic MLP
        self.acoustic_encoder = nn.Sequential(
            nn.Linear(n_acoustic_features, 128), nn.BatchNorm1d(128), nn.GELU(), nn.Dropout(cfg.dropout),
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.GELU()
        )

        # Fusion
        pathway_dims = [256, 512, 128, 64]
        self.cross_attn_fusion = MultiHeadCrossAttentionFusion(
            dims=pathway_dims, n_heads=cfg.n_cross_attn_heads,
            proj_dim=cfg.cross_attn_dim, dropout=cfg.dropout * 0.5
        )
        fusion_dim = self.cross_attn_fusion.output_dim

        # Classifier
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


# ════════════════════════════════════════════════════════════════
# XLS-R Feature Extractor (Attentive Statistical Pooling)
# ════════════════════════════════════════════════════════════════

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
# Hand-Crafted Feature Extraction
# ════════════════════════════════════════════════════════════════

def load_audio(filepath, sr=16000, max_length=8):
    """Load audio, resample, normalize, pad/truncate."""
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
    """MFCC + delta → 160 dims."""
    mfcc = librosa.feature.mfcc(y=waveform, sr=sr, n_mfcc=n_mfcc, n_fft=2048, hop_length=512)
    feats = np.concatenate([np.mean(mfcc, axis=1), np.std(mfcc, axis=1)])
    delta = librosa.feature.delta(mfcc, order=1)
    feats = np.concatenate([feats, np.mean(delta, axis=1), np.std(delta, axis=1)])
    return feats


def extract_voice_quality(waveform, sr=16000):
    """Praat-based voice quality features (with fallback)."""
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
    """Spectral contrast (7) + chroma (12) + tonnetz (6) = 25 dims."""
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
    """Mel spectrogram tensor (1, 128, T)."""
    mel_transform = T.MelSpectrogram(sample_rate=sr, n_fft=2048, hop_length=512, n_mels=128, power=2.0)
    waveform_t = torch.from_numpy(waveform).float().unsqueeze(0)
    mel = mel_transform(waveform_t)
    mel = T.AmplitudeToDB()(mel)
    return mel


# ════════════════════════════════════════════════════════════════
# Global State
# ════════════════════════════════════════════════════════════════

ensemble_models = []   # List of (model, val_f1) tuples
w2v_model = None
w2v_processor = None
asp_pooling = None
models_loaded = False
load_error = None


def load_all_models():
    """Load XLS-R + 5 fold PyTorch models at startup."""
    global ensemble_models, w2v_model, w2v_processor, asp_pooling
    global models_loaded, load_error

    print("[*] Loading models...")

    # ── Load XLS-R backbone ──
    try:
        from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor
        print("  Loading XLS-R 300M...")
        w2v_processor = Wav2Vec2FeatureExtractor.from_pretrained(config.wav2vec2_model_name)
        w2v_model = Wav2Vec2Model.from_pretrained(config.wav2vec2_model_name)
        w2v_model.eval()
        w2v_model.to(DEVICE)
        for param in w2v_model.parameters():
            param.requires_grad = False
        asp_pooling = AttentiveStatisticalPooling(config.wav2vec2_embed_dim, 128)
        asp_pooling.to(DEVICE)
        asp_pooling.eval()
        print(f"  [OK] XLS-R loaded ({sum(p.numel() for p in w2v_model.parameters())/1e6:.0f}M params)")
    except Exception as e:
        load_error = f"Failed to load XLS-R: {e}"
        print(f"  [!!] {load_error}")
        return

    # Patch AudioConfig into the module so torch.load can unpickle it
    sys.modules["__main__"].AudioConfig = AudioConfig

    # ── Load 5-fold ensemble models ──
    fold_files = sorted(Path(BASE_DIR).glob("fold_*_model*.pth"))
    if not fold_files:
        load_error = "No fold model .pth files found"
        print(f"  [!!] {load_error}")
        return

    for pth_path in fold_files:
        try:
            ckpt = torch.load(str(pth_path), map_location=DEVICE, weights_only=False)
            model = Wav2VecAudioModel(config)
            model.load_state_dict(ckpt["model_state_dict"])
            model.eval()
            model.to(DEVICE)
            val_f1 = ckpt.get("val_f1", 0)
            ensemble_models.append((model, val_f1))
            print(f"  [OK] {pth_path.name} (val F1={val_f1:.3f})")
        except Exception as e:
            print(f"  [!!] {pth_path.name}: {e}")

    if not ensemble_models:
        load_error = "All fold models failed to load"
        print(f"  [!!] {load_error}")
        return

    models_loaded = True
    print(f"[OK] Loaded {len(ensemble_models)} fold models + XLS-R backbone")


# ════════════════════════════════════════════════════════════════
# Feature Extraction Pipeline
# ════════════════════════════════════════════════════════════════

@torch.no_grad()
def extract_xlsr_features(waveform):
    """Extract XLS-R embedding (2048d) and sequence (100, 1024) from waveform."""
    inputs = w2v_processor(waveform, sampling_rate=config.sample_rate, return_tensors="pt", padding=True)
    input_values = inputs.input_values.to(DEVICE)
    outputs = w2v_model(input_values)
    hidden = outputs.last_hidden_state  # (1, T, 1024)

    # Attentive Statistical Pooling → 2048d
    pooled = asp_pooling(hidden).squeeze(0).cpu().numpy()

    # Sequence for BiLSTM (downsample to 100 frames)
    seq = hidden.squeeze(0).cpu().numpy()
    if seq.shape[0] > 100:
        indices = np.linspace(0, seq.shape[0] - 1, 100, dtype=int)
        seq = seq[indices]
    elif seq.shape[0] < 100:
        seq = np.pad(seq, ((0, 100 - seq.shape[0]), (0, 0)))

    return pooled, seq


# ════════════════════════════════════════════════════════════════
# Prediction Pipeline
# ════════════════════════════════════════════════════════════════

@torch.no_grad()
def predict_ensemble(audio_file):
    """Full pipeline: load → extract features → ensemble predict."""
    # Load audio
    waveform = load_audio(audio_file, sr=config.sample_rate, max_length=config.max_audio_length)

    # Extract all features
    mfcc_feat = extract_mfcc(waveform, config.sample_rate, config.n_mfcc)
    voice_feats = extract_voice_quality(waveform, config.sample_rate)
    extra_feats = extract_extra_features(waveform, config.sample_rate)
    acoustic_feat = np.concatenate([list(voice_feats.values()), extra_feats])
    mel_spec = extract_mel_spectrogram(waveform, config.sample_rate)
    w2v_emb, w2v_seq = extract_xlsr_features(waveform)

    # Convert to tensors (batch dim = 1)
    mel_t = mel_spec.unsqueeze(0).to(DEVICE)
    mfcc_t = torch.tensor(mfcc_feat, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    acoustic_t = torch.tensor(acoustic_feat, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    w2v_emb_t = torch.tensor(w2v_emb, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    w2v_seq_t = torch.tensor(w2v_seq, dtype=torch.float32).unsqueeze(0).to(DEVICE)

    # Fix NaN/Inf
    mfcc_t = torch.nan_to_num(mfcc_t)
    acoustic_t = torch.nan_to_num(acoustic_t)

    # Ensemble over fold models
    fold_probs = []
    for model, val_f1 in ensemble_models:
        model.eval()
        outputs = model(mel_t, mfcc_t, acoustic_t, w2v_emb_t, w2v_seq=w2v_seq_t)
        probs = F.softmax(outputs["logits"], dim=-1)
        pd_prob = probs[0, 1].item()
        fold_probs.append(pd_prob)

    # Weighted average (weight by validation F1)
    weights = np.array([f1 for _, f1 in ensemble_models])
    if weights.sum() > 0:
        weights = weights / weights.sum()
    else:
        weights = np.ones(len(fold_probs)) / len(fold_probs)
    combined_risk = float(np.average(fold_probs, weights=weights))

    # Status
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
# Flask App
# ════════════════════════════════════════════════════════════════

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB

ALLOWED_EXTENSIONS = {"wav", "mp3", "ogg", "flac", "m4a", "webm"}


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict_route():
    if not models_loaded:
        msg = load_error or "Models not loaded yet"
        return jsonify({"error": msg}), 503

    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400
    if not allowed_file(file.filename):
        return jsonify({"error": "Invalid file type. Allowed: WAV, MP3, OGG, FLAC, M4A"}), 400

    try:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)

        result = predict_ensemble(filepath)

        # Extract voice quality features for the UI breakdown
        waveform = load_audio(filepath, sr=config.sample_rate, max_length=config.max_audio_length)
        voice_feats = extract_voice_quality(waveform, config.sample_rate)
        duration = len(waveform) / config.sample_rate

        result["voice_features"] = {
            "duration": round(duration, 2),
            "mean_pitch": round(float(voice_feats.get("mean_pitch", 0)), 2),
            "std_pitch": round(float(voice_feats.get("std_pitch", 0)), 2),
            "jitter": round(float(voice_feats.get("jitter_local", 0)) * 100, 4),
            "shimmer": round(float(voice_feats.get("shimmer_local", 0)) * 100, 4),
            "hnr": round(float(voice_feats.get("hnr", 0)), 2),
            "spectral_centroid": round(float(voice_feats.get("spectral_centroid", 0)), 1),
            "f1_mean": round(float(voice_feats.get("f1_mean", 0)), 1),
            "f2_mean": round(float(voice_feats.get("f2_mean", 0)), 1),
        }

        try:
            os.remove(filepath)
        except OSError:
            pass

        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/health")
def health():
    return jsonify({
        "status": "ok" if models_loaded else "error",
        "models_loaded": models_loaded,
        "model_count": len(ensemble_models),
        "error": load_error,
    })


# ════════════════════════════════════════════════════════════════
# Startup
# ════════════════════════════════════════════════════════════════

load_all_models()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5003))
    app.run(host="0.0.0.0", port=port, debug=False)
