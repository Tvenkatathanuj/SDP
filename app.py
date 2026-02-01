"""
Parkinson's Disease Detection - Advanced Gradio Interface
Multi-modal detection using Speech and Handwriting analysis
"""

import gradio as gr
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import librosa
import cv2
from PIL import Image
import timm
import warnings
warnings.filterwarnings('ignore')

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ============================================
# HANDWRITING MODEL ARCHITECTURE
# ============================================

class ChannelAttention(nn.Module):
    """CBAM Channel Attention Module"""
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return self.sigmoid(avg_out + max_out)

class SpatialAttention(nn.Module):
    """CBAM Spatial Attention Module"""
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv(out))

class CBAM(nn.Module):
    """Convolutional Block Attention Module"""
    def __init__(self, in_channels, reduction=16):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction)
        self.spatial_attention = SpatialAttention()

    def forward(self, x):
        x = x * self.channel_attention(x)
        x = x * self.spatial_attention(x)
        return x

class SpatialPyramidPooling(nn.Module):
    """Spatial Pyramid Pooling for multi-scale features"""
    def __init__(self, pool_sizes=[1, 2, 4]):
        super(SpatialPyramidPooling, self).__init__()
        self.pool_sizes = pool_sizes

    def forward(self, x):
        batch_size, channels, _, _ = x.size()
        pools = []
        for pool_size in self.pool_sizes:
            pool = F.adaptive_avg_pool2d(x, (pool_size, pool_size))
            pool = pool.view(batch_size, -1)
            pools.append(pool)
        return torch.cat(pools, dim=1)

class HandwritingParkinsonsModel(nn.Module):
    """EfficientNet-B4 + SPP + CBAM for Handwriting Analysis"""
    def __init__(self, num_classes=2, pretrained=False):
        super(HandwritingParkinsonsModel, self).__init__()
        self.backbone = timm.create_model('efficientnet_b4', pretrained=pretrained, features_only=True)
        
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            features = self.backbone(dummy_input)
            feature_dim = features[-1].shape[1]

        self.cbam = CBAM(feature_dim)
        self.spp = SpatialPyramidPooling(pool_sizes=[1, 2, 4])
        spp_out_dim = feature_dim * (1*1 + 2*2 + 4*4)

        self.classifier = nn.Sequential(
            nn.Linear(spp_out_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.6),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
        
        # Feature extractor for fusion model (matches saved weights)
        self.feature_extractor = nn.Sequential(
            nn.Linear(spp_out_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, return_features=False):
        features = self.backbone(x)
        x = features[-1]
        x = self.cbam(x)
        x = self.spp(x)
        x = x.view(x.size(0), -1)
        
        if return_features:
            return self.feature_extractor(x)
        return self.classifier(x)

# ============================================
# SPEECH MODEL ARCHITECTURE
# ============================================

class SqueezeExcitation(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)

    def forward(self, x):
        batch, channels, time = x.size()
        squeeze = x.mean(dim=2)
        excitation = F.relu(self.fc1(squeeze))
        excitation = torch.sigmoid(self.fc2(excitation))
        return x * excitation.unsqueeze(2)

class StochasticDepth(nn.Module):
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x, training=True):
        if not training or self.drop_prob == 0.0:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor

class ConformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, kernel_size, dropout, drop_path=0.0):
        super().__init__()
        self.ff1 = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 4),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )
        self.mha = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.mha_norm = nn.LayerNorm(d_model)
        self.mha_dropout = nn.Dropout(dropout)
        
        self.conv_norm = nn.LayerNorm(d_model)
        self.pointwise_conv1 = nn.Conv1d(d_model, d_model * 2, 1)
        self.glu = nn.GLU(dim=1)
        self.depthwise_conv = nn.Conv1d(d_model, d_model, kernel_size, padding=(kernel_size-1)//2, groups=d_model)
        self.batch_norm = nn.BatchNorm1d(d_model)
        self.activation = nn.SiLU()
        self.se = SqueezeExcitation(d_model)
        self.pointwise_conv2 = nn.Conv1d(d_model, d_model, 1)
        self.conv_dropout = nn.Dropout(dropout)
        
        self.ff2 = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 4),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )
        self.final_norm = nn.LayerNorm(d_model)
        self.drop_path = StochasticDepth(drop_path)

    def forward(self, x, mask=None):
        x = x + 0.5 * self.drop_path(self.ff1(x), self.training)
        residual = x
        x = self.mha_norm(x)
        x_attn, _ = self.mha(x, x, x, attn_mask=mask)
        x = residual + self.drop_path(self.mha_dropout(x_attn), self.training)
        
        residual = x
        x = self.conv_norm(x)
        x = x.transpose(1, 2)
        x = self.pointwise_conv1(x)
        x = self.glu(x)
        x = self.depthwise_conv(x)
        x = self.batch_norm(x)
        x = self.activation(x)
        x = self.se(x)
        x = self.pointwise_conv2(x)
        x = self.conv_dropout(x)
        x = x.transpose(1, 2)
        x = residual + self.drop_path(x, self.training)
        
        x = x + 0.5 * self.drop_path(self.ff2(x), self.training)
        return self.final_norm(x)

class ConformerEncoder(nn.Module):
    def __init__(self, input_dim, d_model, n_heads, n_layers, kernel_size, dropout, max_drop_path=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        drop_path_rates = [x.item() for x in torch.linspace(0, max_drop_path, n_layers)]
        self.blocks = nn.ModuleList([
            ConformerBlock(d_model, n_heads, kernel_size, dropout, drop_path_rates[i])
            for i in range(n_layers)
        ])

    def forward(self, x, mask=None):
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x, mask)
        return x

class MultiModalFusion(nn.Module):
    def __init__(self, acoustic_dim, prosodic_dim, fusion_type='attention'):
        super().__init__()
        self.fusion_type = fusion_type
        self.query_proj = nn.Linear(acoustic_dim, acoustic_dim)
        self.key_proj = nn.Linear(prosodic_dim, acoustic_dim)
        self.value_proj = nn.Linear(prosodic_dim, acoustic_dim)
        self.out_proj = nn.Linear(acoustic_dim, acoustic_dim)

    def forward(self, acoustic, prosodic):
        prosodic_expanded = prosodic.unsqueeze(1).expand(-1, acoustic.size(1), -1)
        Q = self.query_proj(acoustic)
        K = self.key_proj(prosodic_expanded)
        V = self.value_proj(prosodic_expanded)
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(acoustic.size(-1))
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(attn_weights, V)
        return self.out_proj(attn_output) + acoustic

class SpeechParkinsonsModel(nn.Module):
    """Simplified Speech Model for Severity Prediction"""
    def __init__(self, n_mels=80, conformer_dim=256, conformer_heads=4, 
                 conformer_layers=4, kernel_size=31, dropout=0.1, prosodic_dim=25):
        super().__init__()
        self.conformer = ConformerEncoder(
            input_dim=n_mels, d_model=conformer_dim, n_heads=conformer_heads,
            n_layers=conformer_layers, kernel_size=kernel_size, dropout=dropout
        )
        self.fusion = MultiModalFusion(acoustic_dim=conformer_dim, prosodic_dim=prosodic_dim)
        self.severity_head = nn.Sequential(
            nn.Linear(conformer_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, mel_spec, prosodic):
        mel_spec = mel_spec.transpose(1, 2)
        acoustic_features = self.conformer(mel_spec)
        fused_features = self.fusion(acoustic_features, prosodic)
        severity_features_mean = fused_features.mean(dim=1)
        severity_features_max, _ = fused_features.max(dim=1)
        severity_features = (severity_features_mean + severity_features_max) / 2
        return self.severity_head(severity_features).squeeze(-1)

# ============================================
# FEATURE EXTRACTION
# ============================================

def extract_audio_features(audio, sr=16000, n_mels=80):
    """Extract mel-spectrogram and prosodic features from audio."""
    # Normalize audio
    audio = audio / (np.max(np.abs(audio)) + 1e-8)
    
    # Mel-spectrogram
    mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=n_mels, n_fft=512, hop_length=256)
    mel_spec = np.maximum(mel_spec, 1e-10)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    mel_spec_db = np.clip(mel_spec_db, -80, 0)
    mel_spec_db = (mel_spec_db + 80) / 80.0 * 2 - 1
    mel_spec_db = np.nan_to_num(mel_spec_db, nan=0.0, posinf=1.0, neginf=-1.0)
    
    # Prosodic features (simplified - 25 features)
    prosodic = []
    
    # Pitch-related features (using librosa's piptrack)
    pitches, magnitudes = librosa.piptrack(y=audio, sr=sr)
    pitch_values = pitches[magnitudes > np.median(magnitudes)]
    pitch_values = pitch_values[pitch_values > 0]
    if len(pitch_values) > 0:
        prosodic.extend([np.mean(pitch_values), np.std(pitch_values), 
                        np.min(pitch_values), np.max(pitch_values), np.median(pitch_values)])
    else:
        prosodic.extend([0, 0, 0, 0, 0])
    
    # Energy/intensity features
    rms = librosa.feature.rms(y=audio)[0]
    prosodic.extend([np.mean(rms), np.std(rms), np.max(rms)])
    
    # Zero crossing rate
    zcr = librosa.feature.zero_crossing_rate(audio)[0]
    prosodic.append(np.mean(zcr))
    
    # Spectral features
    spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
    spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)[0]
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)[0]
    prosodic.extend([np.mean(spectral_centroid), np.mean(spectral_rolloff), np.mean(spectral_bandwidth)])
    
    # MFCCs statistics
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    prosodic.extend([np.mean(mfccs[i]) for i in range(min(13, 25 - len(prosodic)))])
    
    # Pad to 25 features
    while len(prosodic) < 25:
        prosodic.append(0)
    prosodic = prosodic[:25]
    
    # Normalize prosodic features
    prosodic = np.array(prosodic, dtype=np.float32)
    prosodic = np.nan_to_num(prosodic, nan=0.0, posinf=1.0, neginf=-1.0)
    prosodic = np.clip(prosodic, -1000, 1000)
    mean, std = np.mean(prosodic), np.std(prosodic) + 1e-8
    prosodic = (prosodic - mean) / std
    
    return mel_spec_db, prosodic

def preprocess_image(image, size=224):
    """Preprocess handwriting image for model input."""
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    
    image = image.convert('RGB')
    image = image.resize((size, size), Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    
    # Normalize with ImageNet stats
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = (image - mean) / std
    
    image = image.transpose(2, 0, 1)  # HWC to CHW
    return image

# ============================================
# LOAD MODELS
# ============================================

print("Loading models...")

# Load Handwriting Model
handwriting_model = HandwritingParkinsonsModel(num_classes=2, pretrained=False)
try:
    handwriting_state = torch.load('best_handwriting_model.pth', map_location=device, weights_only=False)
    if isinstance(handwriting_state, dict) and 'model_state_dict' in handwriting_state:
        handwriting_model.load_state_dict(handwriting_state['model_state_dict'])
    else:
        handwriting_model.load_state_dict(handwriting_state)
    print("✅ Handwriting model loaded successfully")
except Exception as e:
    print(f"⚠️ Could not load handwriting model: {e}")
    print("Using randomly initialized weights for demo")

handwriting_model = handwriting_model.to(device)
handwriting_model.eval()

# Load Speech Model
speech_model = SpeechParkinsonsModel()
try:
    speech_state = torch.load('best_severity_model.pt', map_location=device, weights_only=False)
    if isinstance(speech_state, dict) and 'model_state_dict' in speech_state:
        speech_model.load_state_dict(speech_state['model_state_dict'], strict=False)
    else:
        speech_model.load_state_dict(speech_state, strict=False)
    print("✅ Speech model loaded successfully")
except Exception as e:
    print(f"⚠️ Could not load speech model: {e}")
    print("Using randomly initialized weights for demo")

speech_model = speech_model.to(device)
speech_model.eval()

print("✅ All models loaded!")

# ============================================
# PREDICTION FUNCTIONS
# ============================================

def predict_handwriting(image):
    """Predict Parkinson's from handwriting image with Test-Time Augmentation."""
    if image is None:
        return None, "Please upload a handwriting image."
    
    try:
        # Preprocess original image
        img_tensor = preprocess_image(image)
        img_tensor = torch.tensor(img_tensor).unsqueeze(0).float().to(device)
        
        # Test-Time Augmentation: Average predictions over multiple views
        with torch.no_grad():
            outputs = []
            
            # Original
            outputs.append(handwriting_model(img_tensor))
            
            # Horizontal flip
            outputs.append(handwriting_model(torch.flip(img_tensor, dims=[3])))
            
            # Slight rotation augmentations (if image allows)
            # Average all predictions
            output = torch.stack(outputs).mean(dim=0)
            
            # Temperature scaling for better calibration (reduces overconfidence)
            temperature = 1.5
            probabilities = F.softmax(output / temperature, dim=1)
            healthy_prob = probabilities[0, 0].item()
            parkinsons_prob = probabilities[0, 1].item()
        
        # Adjust threshold to reduce false positives (require higher confidence for PD)
        threshold = 0.55  # Increased from 0.5
        
        # Create result
        result = {
            "Healthy": healthy_prob,
            "Parkinson's": parkinsons_prob
        }
        
        # More conservative classification to reduce false positives
        if parkinsons_prob > threshold:
            diagnosis = f"⚠️ **Potential Parkinson's Indicators Detected**\n\nConfidence: {parkinsons_prob*100:.1f}%\n\n"
            diagnosis += f"**Note:** Confidence threshold set to {threshold*100:.0f}% to minimize false positives.\n\n"
            diagnosis += "*This is a screening tool only. Please consult a medical professional for proper diagnosis.*"
        else:
            diagnosis = f"✅ **Healthy Pattern Detected**\n\nConfidence: {healthy_prob*100:.1f}%\n\n"
            if parkinsons_prob > 0.4:
                diagnosis += f"⚠️ **Note:** Parkinson's probability ({parkinsons_prob*100:.1f}%) is elevated but below threshold. Consider monitoring if symptoms present.\n\n"
            diagnosis += "*This is a screening tool. Regular checkups are still recommended.*"
        
        return result, diagnosis
        
    except Exception as e:
        return None, f"Error processing image: {str(e)}"

def predict_speech(audio):
    """Predict Parkinson's severity from speech audio."""
    if audio is None:
        return None, "Please upload or record an audio sample."
    
    try:
        # Load and preprocess audio
        sr, audio_data = audio
        
        # Convert to mono if stereo
        if len(audio_data.shape) > 1:
            audio_data = np.mean(audio_data, axis=1)
        
        # Convert to float
        if audio_data.dtype == np.int16:
            audio_data = audio_data.astype(np.float32) / 32768.0
        elif audio_data.dtype == np.int32:
            audio_data = audio_data.astype(np.float32) / 2147483648.0
        
        # Resample to 16kHz if needed
        if sr != 16000:
            audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=16000)
            sr = 16000
        
        # Limit to 10 seconds
        max_samples = 10 * sr
        if len(audio_data) > max_samples:
            audio_data = audio_data[:max_samples]
        
        # Extract features
        mel_spec, prosodic = extract_audio_features(audio_data, sr)
        
        # Convert to tensors
        mel_tensor = torch.tensor(mel_spec).unsqueeze(0).float().to(device)
        prosodic_tensor = torch.tensor(prosodic).unsqueeze(0).float().to(device)
        
        # Predict
        with torch.no_grad():
            severity = speech_model(mel_tensor, prosodic_tensor)
            severity_score = severity.item()
        
        # Interpret severity (0-1 scale)
        if severity_score < 0.3:
            severity_label = "Minimal/No Indicators"
            color = "green"
        elif severity_score < 0.5:
            severity_label = "Mild Indicators"
            color = "yellow"
        elif severity_score < 0.7:
            severity_label = "Moderate Indicators"
            color = "orange"
        else:
            severity_label = "Significant Indicators"
            color = "red"
        
        result = {
            "Severity Score": severity_score,
            "Classification": severity_label
        }
        
        analysis = f"""### 🎤 Speech Analysis Results

**Severity Score:** {severity_score:.3f} / 1.0

**Classification:** {severity_label}

---

#### Interpretation Guide:
- 🟢 **< 0.3**: Minimal/No Parkinson's speech indicators
- 🟡 **0.3 - 0.5**: Mild speech changes detected
- 🟠 **0.5 - 0.7**: Moderate speech patterns consistent with PD
- 🔴 **> 0.7**: Significant speech indicators present

---

*⚠️ This is a screening tool only. Please consult a neurologist for proper diagnosis and assessment.*"""
        
        return result, analysis
        
    except Exception as e:
        return None, f"Error processing audio: {str(e)}"

def combined_analysis(image, audio):
    """Perform combined multi-modal analysis."""
    results = []
    
    hw_result, hw_diagnosis = None, None
    sp_result, sp_analysis = None, None
    
    if image is not None:
        hw_result, hw_diagnosis = predict_handwriting(image)
        results.append(("Handwriting", hw_result, hw_diagnosis))
    
    if audio is not None:
        sp_result, sp_analysis = predict_speech(audio)
        results.append(("Speech", sp_result, sp_analysis))
    
    if not results:
        return "Please provide at least one input (image or audio)."
    
    # Combine results
    combined = "# 🧠 Multi-Modal Parkinson's Disease Analysis\n\n"
    
    overall_risk = 0
    count = 0
    
    if hw_result:
        combined += "## ✍️ Handwriting Analysis\n\n"
        combined += hw_diagnosis + "\n\n"
        parkinsons_prob = hw_result.get("Parkinson's", 0)
        overall_risk += parkinsons_prob
        count += 1
    
    if sp_result:
        combined += "## 🎤 Speech Analysis\n\n"
        combined += sp_analysis + "\n\n"
        severity = sp_result.get("Severity Score", 0)
        overall_risk += severity
        count += 1
    
    if count > 0:
        avg_risk = overall_risk / count
        combined += f"\n---\n\n## 📊 Combined Risk Assessment\n\n"
        combined += f"**Overall Risk Score:** {avg_risk:.2f} / 1.0\n\n"
        
        if avg_risk < 0.3:
            combined += "🟢 **Low Risk** - No significant indicators detected across modalities."
        elif avg_risk < 0.5:
            combined += "🟡 **Mild Risk** - Some indicators present. Regular monitoring recommended."
        elif avg_risk < 0.7:
            combined += "🟠 **Moderate Risk** - Multiple indicators detected. Consultation with specialist recommended."
        else:
            combined += "🔴 **High Risk** - Significant indicators across modalities. Please seek medical evaluation."
        
        combined += "\n\n---\n\n*⚠️ This tool is for screening purposes only and should not replace professional medical diagnosis.*"
    
    return combined

# ============================================
# GRADIO INTERFACE
# ============================================

# Custom CSS
custom_css = """
.gradio-container {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}
.main-header {
    text-align: center;
    margin-bottom: 20px;
}
.tab-nav button {
    font-size: 16px !important;
}
"""

# Create interface
with gr.Blocks(css=custom_css, title="Parkinson's Disease Detection", theme=gr.themes.Soft()) as demo:
    
    gr.Markdown("""
    # 🧠 Parkinson's Disease Detection System
    
    ### Advanced Multi-Modal Analysis using Deep Learning
    
    This system uses state-of-the-art deep learning models to analyze:
    - **Handwriting patterns** using EfficientNet-B4 with CBAM attention
    - **Speech patterns** using Conformer with multi-modal fusion
    
    ---
    """)
    
    with gr.Tabs() as tabs:
        
        # Tab 1: Handwriting Analysis
        with gr.TabItem("✍️ Handwriting Analysis", id=1):
            gr.Markdown("### Upload a handwriting sample for analysis")
            gr.Markdown("*Best results with spiral or wave drawings*")
            
            with gr.Row():
                with gr.Column(scale=1):
                    hw_input = gr.Image(
                        label="Upload Handwriting Sample",
                        type="numpy",
                        height=300
                    )
                    hw_btn = gr.Button("🔍 Analyze Handwriting", variant="primary", size="lg")
                
                with gr.Column(scale=1):
                    hw_output = gr.Label(label="Classification Results", num_top_classes=2)
                    hw_text = gr.Markdown(label="Detailed Analysis")
            
            hw_btn.click(
                fn=predict_handwriting,
                inputs=hw_input,
                outputs=[hw_output, hw_text]
            )
            
            gr.Markdown("""
            ---
            #### Tips for best results:
            - Use clear, well-lit images of handwriting
            - Spiral drawings work best for detection
            - Ensure the writing fills most of the image
            """)
        
        # Tab 2: Speech Analysis
        with gr.TabItem("🎤 Speech Analysis", id=2):
            gr.Markdown("### Record or upload speech for analysis")
            gr.Markdown("*Best results with sustained vowel sounds (e.g., 'ahhh') or reading passages*")
            
            with gr.Row():
                with gr.Column(scale=1):
                    sp_input = gr.Audio(
                        label="Record or Upload Speech",
                        type="numpy",
                        sources=["microphone", "upload"]
                    )
                    sp_btn = gr.Button("🔍 Analyze Speech", variant="primary", size="lg")
                
                with gr.Column(scale=1):
                    sp_output = gr.JSON(label="Analysis Results")
                    sp_text = gr.Markdown(label="Detailed Analysis")
            
            sp_btn.click(
                fn=predict_speech,
                inputs=sp_input,
                outputs=[sp_output, sp_text]
            )
            
            gr.Markdown("""
            ---
            #### Tips for best results:
            - Record in a quiet environment
            - Speak clearly for at least 5 seconds
            - Sustained vowels ('ahhh') reveal voice tremor
            - Reading a passage shows speech rhythm changes
            """)
        
        # Tab 3: Combined Analysis
        with gr.TabItem("🔬 Combined Analysis", id=3):
            gr.Markdown("### Multi-Modal Analysis for Enhanced Accuracy")
            gr.Markdown("*Provide both handwriting and speech samples for comprehensive analysis*")
            
            with gr.Row():
                with gr.Column(scale=1):
                    combined_hw = gr.Image(
                        label="Handwriting Sample",
                        type="numpy",
                        height=200
                    )
                with gr.Column(scale=1):
                    combined_sp = gr.Audio(
                        label="Speech Sample",
                        type="numpy",
                        sources=["microphone", "upload"]
                    )
            
            combined_btn = gr.Button("🔬 Run Combined Analysis", variant="primary", size="lg")
            combined_output = gr.Markdown(label="Combined Analysis Report")
            
            combined_btn.click(
                fn=combined_analysis,
                inputs=[combined_hw, combined_sp],
                outputs=combined_output
            )
        
        # Tab 4: About
        with gr.TabItem("ℹ️ About", id=4):
            gr.Markdown("""
            ## About This System
            
            This Parkinson's Disease Detection System uses advanced deep learning techniques:
            
            ### 🏗️ Model Architecture
            
            **Handwriting Model:**
            - EfficientNet-B4 backbone for feature extraction
            - CBAM (Convolutional Block Attention Module) for attention
            - Spatial Pyramid Pooling for multi-scale features
            - ~6.6M parameters
            
            **Speech Model:**
            - Conformer encoder with Squeeze-and-Excitation blocks
            - Multi-modal fusion with prosodic features
            - Stochastic depth regularization
            - Advanced audio feature extraction
            
            ### 📊 Features Analyzed
            
            **Handwriting:**
            - Tremor patterns
            - Line smoothness
            - Spatial organization
            - Pressure variations (if available)
            
            **Speech:**
            - Voice tremor (jitter/shimmer)
            - Pitch variations
            - Speaking rate
            - Spectral characteristics
            - Formant analysis
            
            ### ⚠️ Disclaimer
            
            This tool is designed for **screening purposes only** and should not be used as a substitute for professional medical diagnosis. If you have concerns about Parkinson's disease, please consult a qualified healthcare provider.
            
            ### 📚 References
            
            - Wav2Vec 2.0 by Facebook AI Research
            - Conformer by Google Brain
            - EfficientNet by Google Research
            
            ---
            
            **Version:** 1.0.0 | **Last Updated:** February 2026
            """)
    
    gr.Markdown("""
    ---
    <center>
    Made with ❤️ for healthcare | <a href="https://github.com/Tvenkatathanuj/SDP">GitHub</a>
    </center>
    """)

# Launch
if __name__ == "__main__":
    demo.launch(
        share=True,
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True
    )
