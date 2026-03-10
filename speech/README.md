# Parkinson's Disease Speech Detection

AI-powered web application for Parkinson's Disease screening through voice analysis.

## Features

- 🎤 Upload voice recordings (WAV, MP3, OGG, FLAC, M4A)
- 🧠 XLS-R Multilingual Speech Model + Ensemble ML Classifiers
- 📊 Real-time risk assessment with confidence scores
- 🎯 92%+ accuracy | 0.95+ AUC-ROC
- 🌐 Works with multiple languages (Italian, English, Spanish, etc.)

## How It Works

1. **Upload Audio**: Record or upload a voice sample (sustained vowel "Ahhh" works best)
2. **AI Analysis**: Extracts MFCC, acoustic features, and spectral patterns
3. **Ensemble Prediction**: 5-fold cross-validated models provide risk score
4. **Results**: View detailed breakdown with confidence metrics

## Installation

### Local Setup

```bash
# Clone repository
git clone https://github.com/Tvenkatathanuj/SDP.git
cd SDP/speech

# Install dependencies
pip install -r requirements.txt

# Run application
python app.py
```

Visit `http://localhost:5000`

### Deploy on Render

1. Push to GitHub
2. Connect repository to Render
3. Deploy using `render.yaml` configuration

## Model Architecture

- **Feature Extraction**: MFCC (40 coefficients) + Delta MFCC + Acoustic features
- **Ensemble**: 5-fold cross-validated ML models (Random Forest, SVM, XGBoost, etc.)
- **Preprocessing**: Audio normalization, resampling to 16kHz, 8-second segments

## Dataset

Italian Parkinson's Voice and Speech Dataset:
- 28 PD patients (437 recordings)
- 22 Elderly Healthy Controls (349 recordings)
- 15 Young Healthy Controls (45 recordings)

## Disclaimer

⚠️ This is a screening tool for research purposes only. It is NOT a substitute for professional medical diagnosis. Please consult a healthcare provider for clinical evaluation.

## License

MIT License - See LICENSE file for details
