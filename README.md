# 🧠 Parkinson's Disease Detection Using Multi-Modal Deep Learning

A comprehensive deep learning system for early Parkinson's Disease detection using both **speech analysis** and **handwriting pattern recognition**.

---

## 📋 Project Overview

This project implements state-of-the-art deep learning approaches for Parkinson's Disease detection through two primary modalities:

1. **Speech-Based Detection** - Analyzing speech patterns using Wav2Vec 2.0 + Conformer architecture
2. **Handwriting-Based Detection** - Analyzing handwriting samples using EfficientNet with attention mechanisms

---

## 🏗️ Project Structure

```
📁 SDP/
├── 📓 parkinsons_speech_recognition(2).ipynb   # Speech-based PD detection
├── 📓 handwriting_parkinsons_detection.ipynb   # Handwriting-based PD detection
├── 🔧 feature_config(1).json                   # Audio feature configuration
├── 📊 model_summary(1).json                    # Model architecture summary
├── 🧠 best_model(1).pt                         # Best speech model checkpoint
├── 🧠 best_severity_model.pt                   # Severity prediction model
├── 🧠 best_speech_model.pth                    # Speech model weights
├── 🧠 best_handwriting_model.pth               # Handwriting model weights
├── 🧠 handwriting_parkinsons_model_final.pth   # Final handwriting model
├── 📁 original-speech-dataset/                 # Original speech recordings
│   ├── DL/                                     # DL speaker recordings
│   ├── emma/                                   # Emma dataset
│   ├── Faces/                                  # Face recordings
│   ├── LW/                                     # LW speaker recordings
│   └── Tessi/                                  # Tessi dataset
├── 📁 denoised-speech-dataset/                 # Preprocessed/denoised audio
│   ├── DL/
│   ├── emma/
│   ├── Faces/
│   ├── LW/
│   └── Tessi/
├── 📁 handwritten dataset/                     # Handwriting samples
│   └── Dataset/
│       └── Dataset/
│           ├── Healthy/                        # Healthy control samples
│           └── Parkinson/                      # Parkinson's patient samples
└── 📁 speech/
    └── Parkinson-Patient-Speech-Dataset/
```

---

## 🚀 Key Features

### Speech Recognition Module
- **Wav2Vec 2.0** pre-trained feature extraction
- **Conformer** encoder with Squeeze-and-Excitation blocks
- **Multi-task learning** (CTC + Severity + Contrastive + Domain)
- **Advanced augmentation** (MixUp, CutMix, SpecAugment++, VTLP, RIR)
- **Prosodic-acoustic fusion** for enhanced detection
- **Contrastive learning** on paired datasets

### Handwriting Recognition Module
- **EfficientNet** backbone with attention mechanisms
- **Transfer learning** with fine-tuning
- **Albumentations** for robust data augmentation
- **Grad-CAM** visualization for interpretability

---

## 📊 Model Architecture

### Speech Model Configuration
```json
{
  "model_type": "Multi-Task Parkinson Speech Recognition",
  "architecture": "Wav2Vec2 + Conformer + Multi-Modal Fusion",
  "total_parameters": 6,614,051,
  "training_config": {
    "epochs": 100,
    "batch_size": 2,
    "learning_rate": 5e-05,
    "optimizer": "AdamW"
  }
}
```

### Audio Feature Configuration
```json
{
  "sample_rate": 16000,
  "n_mels": 80,
  "n_fft": 512,
  "hop_length": 256,
  "max_audio_length": 10.0,
  "prosodic_dim": 25
}
```

---

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended)
- PyTorch 1.9+

### Setup
```bash
# Clone the repository
git clone https://github.com/Tvenkatathanuj/SDP.git
cd SDP

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or
.\.venv\Scripts\Activate.ps1  # Windows PowerShell

# Install dependencies
pip install torch torchvision torchaudio
pip install transformers librosa soundfile
pip install efficientnet-pytorch albumentations grad-cam scikit-plot
pip install timm pandas numpy matplotlib seaborn tqdm
```

---

## 💻 Usage

### Speech-Based Detection
1. Open `parkinsons_speech_recognition(2).ipynb`
2. Place datasets in the appropriate folders
3. Run cells sequentially

### Handwriting-Based Detection
1. Open `handwriting_parkinsons_detection.ipynb`
2. Ensure handwriting dataset is in place
3. Run cells sequentially

### Google Colab
Both notebooks are **Google Colab ready**:
1. Upload notebook to Google Colab
2. Upload datasets or mount Google Drive
3. Run all cells sequentially

---

## 📈 Results

### Speech Model Performance
| Metric | Value |
|--------|-------|
| Severity MAE | 0.201 |
| Severity RMSE | 0.201 |

### Novel Contributions
- Wav2Vec 2.0 pre-training for Parkinson's speech
- Conformer encoder with SE blocks
- Stochastic depth regularization
- Multi-modal prosodic-acoustic fusion
- Mixed precision training (FP16)
- Exponential Moving Average (EMA)
- Cosine annealing with warmup

---

## 📁 Datasets

### Speech Datasets
- **DL Dataset**: 48 speaker recordings
- **LW Dataset**: 21 speaker recordings  
- **Tessi Dataset**: Spanish/Italian recordings
- **Emma Dataset**: IC and WP recordings
- **Faces Dataset**: Multi-speaker recordings

### Handwriting Dataset
- **Healthy Controls**: Handwriting samples from healthy individuals
- **Parkinson's Patients**: Handwriting samples from PD patients

---

## 🔬 Technical Details

### Training Configuration
- **Epochs**: 100
- **Batch Size**: 2
- **Learning Rate**: 5e-05
- **Optimizer**: AdamW
- **Mixed Precision**: Supported

### Augmentation Techniques
- MixUp
- CutMix
- SpecAugment++
- VTLP (Vocal Tract Length Perturbation)
- RIR (Room Impulse Response)

---

## 📝 Citation

If you use this work in your research, please cite:

```bibtex
@misc{parkinsons_multimodal_detection,
  title={Multi-Modal Deep Learning for Parkinson's Disease Detection},
  author={Venkata Thanuj T.},
  year={2026},
  url={https://github.com/Tvenkatathanuj/SDP}
}
```

---

## 📄 License

This project is for educational and research purposes.

---

## 👥 Contributors

- **Venkata Thanuj T.** - Project Lead

---

## 🙏 Acknowledgments

- Wav2Vec 2.0 by Facebook AI Research
- EfficientNet by Google Research
- Conformer architecture by Google Brain
