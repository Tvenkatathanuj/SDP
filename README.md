<p align="center">
  <h1 align="center">Multimodal Parkinson's Disease Detection Using<br>Cross-Modal Attention Fusion of Handwriting and Speech Biomarkers</h1>
</p>

<p align="center">
  <em>A Deep Learning Framework for Early and Non-Invasive Parkinson's Disease Screening</em>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Accuracy-96.9%25-brightgreen" alt="accuracy"/>
  <img src="https://img.shields.io/badge/AUC--ROC-0.999-blue" alt="auc"/>
  <img src="https://img.shields.io/badge/Framework-PyTorch%202.2-red" alt="pytorch"/>
  <img src="https://img.shields.io/badge/Backbone-XLS--R%20300M-orange" alt="xlsr"/>
  <img src="https://img.shields.io/badge/Deploy-Flask%20%2B%20Render-purple" alt="deploy"/>
</p>

---

## Table of Contents

1. [Abstract](#1-abstract)
2. [Introduction](#2-introduction)
3. [Related Work](#3-related-work)
4. [System Architecture](#4-system-architecture)
5. [Methodology](#5-methodology)
   - 5.1 [Handwriting Analysis Module](#51-handwriting-analysis-module)
   - 5.2 [Speech Analysis Module](#52-speech-analysis-module)
   - 5.3 [Cross-Modal Attention Fusion Network (CMAFN)](#53-cross-modal-attention-fusion-network-cmafn)
   - 5.4 [Combined Web Application](#54-combined-web-application)
6. [Datasets](#6-datasets)
7. [Feature Engineering](#7-feature-engineering)
8. [Training Strategy](#8-training-strategy)
9. [Experimental Results](#9-experimental-results)
10. [Project Structure](#10-project-structure)
11. [Installation & Setup](#11-installation--setup)
12. [Deployment](#12-deployment)
13. [Future Work](#13-future-work)
14. [References](#14-references)

---

## 1. Abstract

Parkinson's Disease (PD) is a chronic, progressive neurodegenerative disorder affecting over 10 million people worldwide. Early detection remains a critical challenge as motor symptoms often manifest only after significant neuronal loss. This project presents a **multimodal deep learning framework** that leverages both **handwriting** and **speech** biomarkers for non-invasive PD screening. The system comprises three independent detection modules — a handwriting analysis engine using **EfficientNet-B0 + CBAM attention** with **16 spatial biomarkers**, a speech analysis engine using **XLS-R 300M + Cross-Attention Fusion**, and a novel **Cross-Modal Attention Fusion Network (CMAFN)** that integrates both modalities via **Cross-Modal Transformer Attention** and a **Gated Multimodal Unit (GMU)**. The CMAFN achieves **96.9% accuracy** and **0.999 AUC-ROC** on a held-out test set, surpassing single-modality baselines by significant margins. All modules are deployed as interactive web applications using Flask, enabling real-time clinical screening.

**Keywords:** *Parkinson's Disease, Deep Learning, Multimodal Fusion, Handwriting Analysis, Speech Analysis, EfficientNet, XLS-R, Cross-Modal Attention, Transfer Learning, Convolutional Block Attention Module (CBAM)*

---

## 2. Introduction

### 2.1 Background

Parkinson's Disease (PD) is the second most prevalent neurodegenerative condition globally, characterized by the progressive loss of dopaminergic neurons in the substantia nigra. Cardinal motor symptoms include tremor, rigidity, bradykinesia, and postural instability. However, by the time these motor symptoms are clinically observable, approximately 60–80% of dopaminergic neurons have already been lost, making early detection imperative for effective therapeutic intervention.

### 2.2 Motivation

Traditional diagnostic methods rely heavily on subjective clinical assessment (e.g., Unified Parkinson's Disease Rating Scale — UPDRS) and expensive neuroimaging (e.g., DaTSCAN). These approaches are:
- **Costly and inaccessible** in resource-limited settings
- **Subjective** and dependent on clinician expertise
- **Late-stage** — diagnosis typically occurs only after significant neuronal degeneration

Non-invasive biomarkers derived from **handwriting** (micrographia) and **speech** (dysarthria, hypophonia) present a compelling alternative for mass screening, as these modalities can be captured using simple consumer hardware (a camera/scanner and a microphone).

### 2.3 Contributions

This project makes the following key contributions:

1. **A 16-feature spatial biomarker framework** for handwriting analysis, extracting clinically relevant features including stroke width, fractal dimension, curvature statistics, and Hu moments from spiral/wave drawings.
2. **A multi-pathway speech model** integrating self-supervised representations (XLS-R 300M), spectral features (mel-spectrograms), cepstral features (MFCC with deltas), and acoustic voice quality measures (jitter, shimmer, HNR) via Cross-Attention Fusion.
3. **A Cross-Modal Attention Fusion Network (CMAFN)** that learns cross-modal interactions between handwriting and speech features through Transformer attention and fuses them with a Gated Multimodal Unit.
4. **End-to-end deployable web applications** with real-time inference, supporting drawing-canvas input, image/audio file upload, and comprehensive diagnostic dashboards.

---

## 3. Related Work

| Study | Modality | Method | Accuracy | AUC |
|-------|----------|--------|----------|-----|
| Pereira et al. (2018) | Handwriting | CNN (ImageNet) | 85.0% | — |
| Zham et al. (2017) | Handwriting | Kinematic features + SVM | 87.4% | — |
| Narendra & Alku (2021) | Speech | Glottal features + SVM | 82.5% | — |
| Vasquez-Correa et al. (2019) | Speech | CNN + Hand-crafted | 85.0% | — |
| Sakar et al. (2019) | Speech | SVM + MFCC | 86.0% | 0.91 |
| Orozco-Arroyave et al. (2016) | Speech | i-vectors + GMM | 87.5% | — |
| Quan et al. (2022) | Speech | Wav2Vec2 + SVM | 88.0% | 0.93 |
| Mughal et al. (2023) | Speech | CNN + BiLSTM | 89.5% | 0.94 |
| Diaz et al. (2021) | Handwriting | Transfer Learning | 90.1% | — |
| **This Work (Handwriting)** | **Handwriting** | **EfficientNet-B0 + CBAM + Meta-Learner** | **93.6%** | **0.985** |
| **This Work (Speech)** | **Speech** | **XLS-R + Cross-Attn + 6-Model Stack** | **91.7%** | **0.971** |
| **This Work (CMAFN)** | **Multimodal** | **Cross-Modal Transformer + GMU** | **96.9%** | **0.999** |

---

## 4. System Architecture

The system is organized into four independent but interoperable modules, each deployable as a standalone Flask web application:

```
┌─────────────────────────────────────────────────────────────────┐
│                     COMBINED WEB APP (Port 5000)                │
│   Unified interface exposing all three detection modalities     │
├─────────────┬───────────────────┬───────────────────────────────┤
│ HANDWRITING │    SPEECH         │  CROSS-MODAL FUSION (CMAFN)   │
│  Module     │    Module         │       Module                  │
│             │                   │                               │
│ EfficientNet│ XLS-R 300M        │ ┌──────────┐  ┌───────────┐  │
│ -B0 + CBAM  │ + Cross-Attn      │ │ HW Enc.  │  │Audio Enc. │  │
│ + Res-MLP   │   Fusion          │ │EffNet-B4 │  │XLS-R+SE   │  │
│ + Meta      │ + BiLSTM          │ │+SPP+CBAM │  │+CNN+MFCC  │  │
│  Learner    │ + 5-Fold          │ └────┬─────┘  └─────┬─────┘  │
│             │   Ensemble        │      └───────┬──────┘        │
│             │                   │     Cross-Modal Transformer   │
│             │                   │      + GMU + 5-Fold Ensemble  │
├─────────────┴───────────────────┴───────────────────────────────┤
│                   Flask + Gunicorn + Render                     │
└─────────────────────────────────────────────────────────────────┘
```

---

## 5. Methodology

### 5.1 Handwriting Analysis Module

**Directory:** `handwriting/`

#### 5.1.1 Feature Extraction Pipeline

The handwriting module extracts **16 spatial biomarkers** from grayscale handwriting images (spiral and wave drawings):

| # | Feature | Description | Clinical Relevance |
|---|---------|-------------|-------------------|
| 1 | `stroke_width_mean` | Mean stroke width via distance transform | Reduced motor control → thinner strokes |
| 2 | `stroke_width_std` | Stroke width variability | Tremor causes irregular stroke widths |
| 3 | `contour_roughness` | Isoperimetric quotient (P²/4πA) | Rougher contours indicate bradykinesia |
| 4 | `direction_changes` | Mean angular difference of stroke direction | Frequent changes suggest tremor |
| 5 | `n_components` | Number of connected components | Pen lifts due to motor impairment |
| 6 | `ink_density` | Ratio of ink pixels to total area | Micrographia reduces ink density |
| 7 | `solidity` | Contour area / convex hull area | Drawing irregularity measure |
| 8 | `intensity_variance` | Std. dev. of ink pixel intensities | Pressure variation indicator |
| 9 | `fractal_dimension` | Box-counting fractal dimension | Complexity of drawing patterns |
| 10 | `entropy` | Shannon entropy of ink histogram | Information content of strokes |
| 11 | `hu_moment_1` | First Hu moment (log-scale) | Rotation-invariant shape descriptor |
| 12 | `hu_moment_2` | Second Hu moment (log-scale) | Shape symmetry measure |
| 13 | `curvature_mean` | Mean curvature along contours | Smoothness of drawn curves |
| 14 | `curvature_std` | Curvature variability | Consistency of motor output |
| 15 | `aspect_ratio` | Bounding box width/height | Overall drawing proportions |
| 16 | `stroke_regularity` | FFT-based periodicity of stroke distances | Rhythmic regularity of drawing |

#### 5.1.2 Model Architecture

The handwriting module employs a **dual-pathway ensemble** with stacking meta-learner:

**Path A — Residual MLP (5-Fold):**
```
Input (16 features) → StandardScaler → Linear(16, 64) → BN → GELU → Dropout(0.3)
  → ResidualBlock(64, dropout=0.4) → Linear(64, 32) → BN → GELU → Dropout(0.35)
  → Linear(32, 1) → Sigmoid
```

**Path B — EfficientNet-B0 + CBAM (5-Fold):**
```
Input (224×224×3) → EfficientNet-B0 (ImageNet, 80% frozen)
  → CBAM(1280) [Channel Attention + Spatial Attention]
  → AdaptiveAvgPool2d(1) → Dropout(0.5)
  → Linear(1280, 128) → GELU → Dropout(0.4)
  → Linear(128, 1) → Sigmoid
```

**Meta-Learner:** Logistic Regression stacking over MLP, CNN, and CNN+TTA predictions.

**Test-Time Augmentation (TTA):** 7 views — original, horizontal flip, ±10° rotation, brightness jitter, vertical flip, center crop.

#### 5.1.3 Results

| Model | Accuracy | AUC-ROC |
|-------|----------|---------|
| Residual MLP (5-fold) | 83.98% | 0.916 |
| EfficientNet-B0 + CBAM (5-fold) | 89.64% | 0.964 |
| EfficientNet-B0 + CBAM + TTA | 92.13% | 0.977 |
| **Stacking Ensemble** | **93.57%** | **0.985** |

> **+3.47% improvement** over the prior state-of-the-art result of 90.1%.

---

### 5.2 Speech Analysis Module

**Directory:** `speech/` | **Notebook:** `audio_live_tracking.ipynb`

The speech module uses a **dual-engine** approach: a Deep Learning model (Cross-Attention Fusion) and a Classical ML Ensemble, combined via a **Stacking Meta-Learner**. Total DL model parameters: **3,284,742**.

#### 5.2.1 Deep Learning — Multi-Pathway Cross-Attention Fusion

The DL model processes audio through **four parallel feature pathways**:

**Path 1 — Self-Supervised Representations (XLS-R 300M, 315.4M params):**
- Pre-trained `facebook/wav2vec2-xls-r-300m` — **436K hours of speech in 128 languages**
- **Attentive Statistical Pooling** (Okabe et al., 2018) → weighted mean + weighted std → 2048-d
- **BiLSTM** (128 hidden, 1-layer, bidirectional) → 256-d temporal features
- Combined: 2304-d → MLP encoder (512 → 256) → 256-d

**Path 2 — Residual Mel-Spectrogram CNN:**
- 128-bin mel-spectrogram (n_fft=2048, hop_length=512)
- Three convolutional blocks: Conv2d → BN → GELU → Pool → Dropout2d
- Channels: 32 → 64 → 128, with residual skip (conv1→conv3 via 1×1 conv)
- Output: 512-d

**Path 3 — MFCC Features:**
- 40 MFCCs + delta coefficients (first-order), mean + std aggregation → 160-d
- MLP encoder with **skip connection** → 128-d

**Path 4 — Acoustic Voice Quality (39-d):**
- Praat-based: mean pitch, pitch std, jitter (local + RAP), shimmer (local), HNR, formants (F1–F3)
- Librosa-based: spectral centroid, spectral rolloff, ZCR, RMS energy (mean + std)
- Additional: spectral contrast (7-d), chroma (12-d), tonnetz (6-d)
- MLP encoder → 64-d

**Cross-Attention Fusion:**
```
[XLS-R(256) | CNN(512) | MFCC(128) | Acoustic(64)]
  → Project each to 256-d → Stack → Multi-Head Self-Attention (4 heads)
  → Residual Connection → Gated Fusion (Softmax) → LayerNorm → Dropout
  → Classifier: Linear(256, 256) → BN → GELU → Dropout(0.6) → Linear(256, 2)
```

#### 5.2.2 Classical ML Ensemble (with PCA Anti-Overfitting)

In parallel, a 5-model ML ensemble is trained on the same features (XLS-R embeddings + MFCC + acoustics = 2247-d), with **PCA reduction to 512-d** (retains 99.5–99.9% variance) to prevent speaker-ID memorization:

| Classifier | Key Anti-Overfit Settings |
|------------|---------------------------|
| **Random Forest** | 300 trees, max_depth=8, min_samples_leaf=5, balanced |
| **SVM (RBF)** | C=1.0 (reduced from 10), gamma=scale, balanced |
| **Gradient Boosting** | 200 trees, max_depth=4, subsample=0.7, lr=0.05 |
| **XGBoost** | max_depth=4, reg_alpha=0.1, reg_lambda=1.0, subsample=0.7 |
| **LightGBM** | max_depth=4, reg_alpha=0.1, reg_lambda=1.0, subsample=0.7 |

SMOTE oversampling is applied when the minority class has ≥2 samples.

#### 5.2.3 Stacking Meta-Learner

A **Ridge-regularized Logistic Regression** (C=0.1) stacks the 5 ML classifier probabilities + DL probability into a final prediction. F1-optimized threshold search is applied on validation data. A fallback **weighted average** (55% ML + 45% DL) is used if it outperforms stacking.

#### 5.2.4 Anti-Overfit v2 Strategies

| Technique | Value | Purpose |
|-----------|-------|---------|
| PCA before ML | 2247→512d | Prevents speaker-ID memorization |
| DL Dropout | 0.6 (↑ from 0.5) | Stronger feature dropout |
| Label Smoothing | 0.2 (↑ from 0.15) | Prevents overconfident outputs |
| Weight Decay | 0.08 (↑ from 0.05) | Stronger L2 regularization |
| Learning Rate | 3e-5 (↓ from 5e-5) | Slower convergence |
| Classifier | 256→2 (simplified) | Fewer params = less overfit |
| Meta-Learner C | 0.1 | Strong Ridge regularization on small val set |
| Gradient Accumulation | 2× (effective batch=32) | Stable gradients |
| Scheduler | OneCycleLR (pct_start=0.2) | Cosine annealing with warmup |
| Early Stopping | patience=10 | Stop before memorization |

#### 5.2.5 Results

**Overall Performance (Aggregated Across 5 Folds):**

| Metric | Value |
|--------|-------|
| Accuracy | **91.70%** |
| Balanced Accuracy | 91.46% |
| Precision | 91.96% |
| F1-Score (Weighted) | 91.66% |
| F1-Score (Macro) | 91.62% |
| AUC-ROC | **0.971** |
| PD Recall (Sensitivity) | 96.11% |
| PD Precision | 88.98% |
| HC Recall (Specificity) | 86.80% |
| HC Precision | 95.26% |

**Per-Fold Breakdown (ML vs DL vs Final):**

| Fold | Final F1 | ML F1 | DL F1 | PD Recall | HC Recall | Best Method |
|------|----------|-------|-------|-----------|-----------|-------------|
| 1 | 0.953 | 0.941 | 0.816 | 95.1% | 98.2% | Stacking |
| 2 | 0.945 | 0.955 | 0.845 | 100.0% | 89.5% | Stacking |
| 3 | 0.873 | 0.862 | 0.872 | 79.2% | 94.7% | Weighted Avg |
| 4 | 0.838 | 0.830 | 0.835 | 100.0% | 67.1% | Weighted Avg |
| 5 | 0.920 | 0.920 | 0.869 | 100.0% | 86.7% | Stacking |
| **Average ± Std** | **0.906 ± 0.044** | — | — | **94.9%** | **87.3%** | — |

**Individual ML Classifier Performance (Best Fold Scores):**

SVM consistently was the strongest individual ML classifier, achieving up to F1=0.988 (Fold 1) and F1=0.970 (Fold 2). The stacking approach successfully combined ML's strong feature engineering with DL's learned representations.

> **+2.2% accuracy improvement** over the prior best (Mughal et al., 2023 at 89.5%). Our system achieves this with **patient-level CV** (no data leakage), unlike many prior works.

---

### 5.3 Cross-Modal Attention Fusion Network (CMAFN)

**Directory:** `fusion_models/`

The CMAFN represents the core contribution — a multimodal fusion architecture that learns cross-modal interactions between handwriting and speech representations.

#### 5.3.1 Encoder Architecture

**Handwriting Encoder — EfficientNet-B4 + CBAM + SPP:**
```
Input (336×336×3) → EfficientNet-B4 (ImageNet pre-trained, frozen)
  → CBAM (Channel + Spatial Attention)
  → Spatial Pyramid Pooling [1×1, 2×2, 4×4]
  → Feature Extractor: Linear(SPP_out, 512) → BN → ReLU → 512-d
```
Pre-trained on the PD handwriting task (~107 MB model).

**Audio Encoder — XLS-R + CNN + MFCC + SE Attention:**
```
4-Path Fusion:
  XLS-R(1024) → MLP → 256-d
  Mel-Spec → 3×Conv2d → 512-d
  MFCC(160) → MLP → 128-d
  Acoustic(39) → MLP → 64-d
  ────────────────────────
  Concatenate → 960-d
  → Squeeze-and-Excitation (SE) Attention → 960-d
  → Audio Feature Extractor: Linear(960, 512) → BN → ReLU → 512-d
```
Pre-trained on the PD speech task (~147 MB model).

#### 5.3.2 Cross-Modal Transformer Fusion

```
HW Features (512-d)    Audio Features (512-d)
       │                        │
  ModalityProjection        ModalityProjection
  (512 → 256 → 256)       (512 → 256 → 256)
       │                        │
  + Type Embedding          + Type Embedding
       │                        │
       └────────┬───────────────┘
                │
     CrossModalTransformerLayer × 2
     ┌──────────────────────────────────┐
     │ HW-Cross-Attn(Q=HW, K=Aud, V=Aud)  │
     │   → LayerNorm → FFN → LayerNorm     │
     │ Aud-Cross-Attn(Q=Aud, K=HW, V=HW)  │
     │   → LayerNorm → FFN → LayerNorm     │
     └──────────────────────────────────┘
                │
     ┌──────────┴──────────┐
     │                     │
 HW_attended(256)   Audio_attended(256)
     │                     │
     └───┬─────────────┬───┘
         │             │
    GatedMultimodalUnit(256, 256 → 128)
         │
    [HW_attended ∥ Audio_attended ∥ GMU_out] = 640-d
         │
    Classifier: 640 → 256 → 128 → 2
```

**Key Design Choices:**
- **Modality Dropout (25%):** Randomly drops one modality during training for robustness to missing inputs
- **Default Tokens:** Learnable fallback tokens when a modality is unavailable
- **Auxiliary Heads:** Separate handwriting and audio classification heads for regularization
- **MC Dropout:** 10-sample Monte Carlo dropout for uncertainty estimation at inference

#### 5.3.3 Results

| Metric | Value |
|--------|-------|
| **Ensemble Accuracy** | **96.94%** |
| **Balanced Accuracy** | 96.94% |
| **F1 (Macro)** | 96.94% |
| **AUC-ROC** | **0.9995** |
| PD Precision | 100.0% |
| PD Recall | 93.88% |
| HC Precision | 94.23% |
| HC Recall | 100.0% |
| MC Dropout Accuracy | 97.14% |
| Mean Uncertainty | 0.061 |

**Cross-Validation (5-Fold):**

| Fold | Balanced Accuracy | F1 | AUC-ROC |
|------|-------------------|----|---------|
| 1 | 99.10% | 0.991 | 0.9998 |
| 2 | 96.40% | 0.964 | 0.9970 |
| 3 | 99.46% | 0.995 | 0.9996 |
| 4 | 99.82% | 0.998 | **1.0000** |
| 5 | 99.82% | 0.998 | **1.0000** |
| **Mean ± Std** | **98.92% ± 1.29%** | **0.989 ± 0.013** | **0.999 ± 0.001** |

**Comparison Against Single-Modality Baselines:**

| Method | Accuracy | AUC | F1 |
|--------|----------|-----|-----|
| Handwriting Only | 87.55% | 0.935 | 0.875 |
| Speech Only | 93.62% | 0.966 | 0.936 |
| **CMAFN (Ours)** | **96.94%** | **0.999** | **0.969** |

> The fusion model achieves **+9.4%** improvement over handwriting-only and **+3.3%** over speech-only baselines.

---

### 5.4 Combined Web Application

**Directory:** `combined/`

The combined module unifies all three modalities into a single deployable web application (1725 lines), providing:

- **Handwriting Detection** via canvas drawing or image upload
- **Speech Detection** via audio file upload (WAV, MP3, OGG, FLAC, M4A, WebM)
- **Multimodal Fusion** via simultaneous handwriting + audio input
- Real-time diagnostic dashboards with risk assessment (LOW / MODERATE / HIGH)
- Detailed feature breakdowns and per-fold prediction transparency

---

## 6. Datasets

### 6.1 Handwriting Dataset

| Property | Value |
|----------|-------|
| Total images | **3,264** |
| Classes | Healthy Control (1,632) / Parkinson's Disease (1,632) |
| Drawing types | Spiral and wave patterns |
| Preprocessing | Resize to 224×224 (handwriting) or 336×336 (fusion) |
| Augmentation | Horizontal/vertical flip, rotation (±10°), brightness jitter, center crop |

### 6.2 Speech Dataset — Italian Parkinson's Voice and Speech (IEEE DataPort)

| Property | Value |
|----------|-------|
| Source | IEEE DataPort |
| Total recordings | **831** |
| Unique patients | **61** (24 PD + 22 EHC + 15 YHC) |
| PD recordings | 437 (from 24 patients) |
| HC recordings | 394 (from 37 healthy controls) |
| Class ratio | HC:PD = 0.9:1 (well-balanced) |
| Sample rate | 16 kHz (resampled) |
| Max duration | 8 seconds (padded/truncated) |
| Mean duration | 24.04 seconds (min: 3.38s, max: 250.31s) |
| Languages supported | Italian (primary), English, Spanish, Turkish, Telugu, Hindi, German, French |

**Speech Task Distribution:**

| Task Type | Code | Samples | Description |
|-----------|------|---------|-------------|
| Sustained Vowel /a/ | VA | 99 | Sustained phonation |
| Sustained Vowel /e/ | VE | 99 | Sustained phonation |
| Sustained Vowel /i/ | VI | 99 | Sustained phonation |
| Sustained Vowel /o/ | VO | 99 | Sustained phonation |
| Sustained Vowel /u/ | VU | 99 | Sustained phonation |
| Diadochokinesis (DDK) | D | 100 | Rapid alternating movements |
| Balanced Sentence | B | 124 | Phonetically balanced reading |
| Prose Reading | PR | 65 | Continuous speech reading |
| Free Speech | FB | 47 | Spontaneous speech |

**Cross-Validation Strategy:** Patient-level Stratified Group 5-Fold CV using `StratifiedGroupKFold` — **no patient overlap** between train/val/test splits, preventing data leakage from recordings of the same speaker appearing in both training and test sets.

---

## 7. Feature Engineering

### 7.1 Handwriting Features (16 Biomarkers)

Extracted via OpenCV image processing:
- **Morphological:** Stroke width (distance transform), ink density, connected components, solidity
- **Geometric:** Contour roughness (isoperimetric quotient), aspect ratio, direction changes
- **Statistical:** Intensity variance, Shannon entropy, Hu moments (1st and 2nd)
- **Complexity:** Box-counting fractal dimension, curvature (mean + std), stroke regularity (FFT)

### 7.2 Speech Features (199-d → 4 Pathways)

| Pathway | Features | Dimensionality |
|---------|----------|----------------|
| XLS-R | Self-supervised speech embeddings (Attentive Pooling + BiLSTM) | 2048 + 256 |
| Mel-Spectrogram | 128-bin mel-spectrogram → 3-layer CNN | 128×T → 512 |
| MFCC | 40 MFCCs + delta (mean + std) | 160 |
| Acoustic Quality | Pitch, jitter, shimmer, HNR, formants, spectral features, chroma, tonnetz | 39 |

---

## 8. Training Strategy

### 8.1 Deep Learning Training

| Technique | Speech Module | Fusion Module |
|-----------|--------------|---------------|
| Cross-Validation | Patient-level Stratified Group 5-Fold | Stratified 5-Fold |
| Optimizer | AdamW (lr=3e-5, wd=0.08) | AdamW (lr=5e-5, wd=0.03) |
| Scheduler | OneCycleLR (pct_start=0.2, cosine) | CosineAnnealingWarmRestarts |
| Loss Function | Focal Loss (α=0.5, γ=3.0) + LS=0.2 | Focal Loss (α=0.75, γ=2.0) + LS=0.05 |
| Gradient Clipping | Max norm = 1.0 | Max norm = 1.0 |
| Gradient Accumulation | 2× (effective batch 32) | — |
| Class Balancing | WeightedRandomSampler | — |
| Early Stopping | Patience = 10 | Patience = 12 |
| Reproducibility | Fixed seed (42), deterministic cuDNN | Fixed seed (42) |

### 8.2 Classical ML Training (Speech Module)

| Step | Details |
|------|---------|
| Feature Extraction | XLS-R (2048-d) + MFCC (160-d) + Acoustic (39-d) = 2247-d |
| Preprocessing | StandardScaler → PCA (2247→512d, retains 99.5–99.9% variance) |
| Oversampling | SMOTE (k_neighbors=3) on training set |
| Classifiers | RF (300 trees) + SVM (RBF, C=1.0) + GB + XGBoost + LightGBM |
| Stacking | Ridge LogisticRegression (C=0.1) over 5 ML + 1 DL probabilities |
| Threshold | F1-macro optimized per fold (search 0.1–0.9, step 0.02) |

### 8.3 Data Augmentation

**Handwriting:**
- Horizontal/vertical flip, rotation (±10°), brightness jitter, center crop
- Mixup augmentation (α=0.3)

**Speech (DL):**
- **SpecAugment:** frequency masking (25 bins) + time masking (40 frames)
- **VTLP:** Vocal Tract Length Perturbation (warp factor: 0.9–1.1)
- **Mixup:** α=0.3, applied stochastically (50% of batches)
- **Light augmentation at data level:** Gaussian noise (scale=0.01), feature dropout (10% of MFCC dims)
- **Test-Time Augmentation (TTA):** 5× augmented inference with increasing noise (0.005×run)

**Fusion:**
- Modality dropout (25%) — randomly drops one modality per sample
- Contrastive learning loss (weight=0.3) for cross-modal alignment
- Albumentations-based image augmentation for handwriting encoder

---

## 9. Experimental Results

### 9.1 Summary of Results

| Module | Architecture | Accuracy | AUC-ROC | F1 |
|--------|-------------|----------|---------|-----|
| Handwriting | EfficientNet-B0 + CBAM + Stacking | 93.57% | 0.985 | 0.934 |
| Speech | XLS-R 300M + Cross-Attn + BiLSTM | 91.70% | 0.971 | 0.916 |
| **CMAFN Fusion** | **Cross-Modal Transformer + GMU** | **96.94%** | **0.999** | **0.969** |

### 9.2 Ablation Analysis

The fusion model demonstrates clear advantages over single-modality approaches:
- **Handwriting-only baseline:** 87.55% accuracy → CMAFN provides **+9.39%** improvement
- **Speech-only baseline:** 93.62% accuracy → CMAFN provides **+3.32%** improvement
- **MC Dropout uncertainty estimation** achieved 97.14% accuracy with mean uncertainty of 0.061, demonstrating well-calibrated confidence scores

### 9.3 Per-Fold Stability

The CMAFN shows remarkable stability across folds:
- Mean balanced accuracy: **98.92% ± 1.29%**
- Two folds achieved **perfect AUC-ROC = 1.0000**
- Best fold (Fold 4) converged in just **2 epochs**

---

## 10. Project Structure

```
SDP/
├── README.md                          # This file
│
├── handwriting/                       # Module 1: Handwriting Detection
│   ├── app.py                         # Flask app (547 lines)
│   ├── templates/index.html           # Web UI
│   ├── mlp_fold_{1-5}.pth            # 5× Residual MLP models
│   ├── cnn_fold_{1-5}.pth            # 5× EfficientNet-B0+CBAM models
│   ├── scaler.pkl                     # StandardScaler for 16 features
│   ├── meta_model.pkl                 # Stacking meta-learner (LogReg)
│   ├── training_results.json          # Evaluation metrics
│   └── requirements.txt
│
├── speech/                            # Module 2: Speech Detection
│   ├── app.py                         # Flask app (586 lines)
│   ├── audio_live_tracking.ipynb      # Training notebook (Kaggle/Colab)
│   ├── templates/index.html           # Web UI
│   ├── fold_{1-5}_model.pth          # 5× XLS-R+CrossAttn fold models
│   ├── best_audio_model.pth          # Best single fold model
│   ├── audio_results.json            # Evaluation metrics
│   └── requirements.txt
│
├── fusion_models/                     # Module 3: CMAFN Fusion
│   ├── app.py                         # Flask app (899 lines)
│   ├── templates/index.html           # Web UI
│   ├── checkpoint_fusion/             # CMAFN ensemble checkpoints
│   ├── checkpoints/                   # Pre-trained encoder weights
│   ├── best_handwriting_model.pth    # Pre-trained HW encoder (~107 MB)
│   ├── handwriting_parkinsons_model_final.pth  # Final HW encoder
│   ├── fusion_results.json           # Evaluation metrics
│   ├── *.ipynb                        # Training notebooks
│   └── requirements.txt
│
├── combined/                          # Module 4: Unified Web App
│   ├── app.py                         # Combined Flask app (1725 lines)
│   ├── templates/index.html           # Unified Web UI
│   ├── render.yaml                    # Render deployment config
│   └── requirements.txt
│
├── handwritten dataset/               # Raw handwriting dataset
│   └── Dataset/
│       ├── 28 People with Parkinson's disease/
│       ├── 22 Elderly Healthy Control/
│       └── 15 Young Healthy Control/
│
├── Italian Parkinson's Voice and speech/  # Raw speech dataset
│
├── review slides/                     # Presentation materials
│   └── *.html, *.pptx
│
└── SDP Project Proposal.pdf           # Original project proposal
```

---

## 11. Installation & Setup

### 11.1 Prerequisites

- **Python** ≥ 3.9
- **PyTorch** ≥ 2.0 (CPU or CUDA)
- **~2 GB** disk space for XLS-R model download (first run)

### 11.2 Install Dependencies

```bash
# For the combined application (includes all dependencies)
cd combined
pip install -r requirements.txt
```

Or install individual modules:

```bash
# Handwriting only
cd handwriting && pip install -r requirements.txt

# Speech only
cd speech && pip install -r requirements.txt

# Fusion only
cd fusion_models && pip install -r requirements.txt
```

### 11.3 Key Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| PyTorch | 2.2.0 | Deep learning framework |
| TorchVision | 0.17.0 | EfficientNet backbone, image transforms |
| TorchAudio | 2.2.0 | Mel-spectrogram extraction |
| Transformers | 4.40.0 | XLS-R 300M (Wav2Vec2) |
| librosa | 0.10.1 | Audio feature extraction (MFCC, chroma, etc.) |
| OpenCV | 4.9.0 | Image processing, feature extraction |
| praat-parselmouth | 0.4.5 | Voice quality analysis (jitter, shimmer, HNR) |
| timm | ≥0.9 | EfficientNet-B4 backbone (fusion module) |
| albumentations | ≥1.3 | Image augmentation (fusion module) |
| scikit-learn | 1.4.2 | StandardScaler, PCA, LogisticRegression meta-learner |
| XGBoost | latest | Gradient boosting classifier (speech ensemble) |
| LightGBM | latest | Gradient boosting classifier (speech ensemble) |
| imbalanced-learn | latest | SMOTE oversampling (speech training) |
| Flask | 3.0.3 | Web application framework |
| Gunicorn | 22.0.0 | Production WSGI server |

### 11.4 Run Locally

```bash
# Run individual modules
cd handwriting && python app.py     # Port 5000
cd speech && python app.py          # Port 5000
cd fusion_models && python app.py   # Port 5000

# Run the combined application
cd combined && python app.py        # Port 5000
```

Navigate to `http://localhost:5000` in your browser.

---

## 12. Deployment

Each module includes a `Procfile` and `render.yaml` for deployment on [Render](https://render.com/):

```yaml
# render.yaml example
services:
  - type: web
    name: pd-detection
    runtime: python
    buildCommand: pip install -r requirements.txt
    startCommand: gunicorn app:app --bind 0.0.0.0:$PORT --timeout 300
```

**Deployment Notes:**
- Set `--timeout 300` to allow time for XLS-R model loading (~60s on CPU)
- Maximum upload size: 16 MB
- Supported audio formats: WAV, MP3, OGG, FLAC, M4A, WebM
- Models run on CPU for deployment; GPU is not required for inference

---

## 13. Future Work

1. **Multimodal Data Fusion at Scale:** Incorporate additional biomarker modalities such as gait analysis (accelerometer data) and eye-tracking patterns.
2. **Longitudinal Tracking:** Develop a progression monitoring system that tracks biomarker changes over time to assess disease stage and treatment efficacy.
3. **Federated Learning:** Enable privacy-preserving model training across distributed hospital systems without sharing raw patient data.
4. **Explainability:** Integrate Grad-CAM and attention visualization for model interpretability, providing clinicians with visual explanations of diagnostic decisions.
5. **Mobile Deployment:** Port the lightweight handwriting model to TensorFlow Lite / ONNX for edge inference on smartphones and tablets.
6. **Multilingual Speech Models:** Extend the speech module to handle more languages and dialects, leveraging the multilingual capabilities of XLS-R.

---

## 14. References

1. Pereira, C. R., et al. "Handwriting dynamics assessment for early identification of Parkinson's disease." *Artificial Intelligence in Medicine*, 2018.
2. Diaz, M., et al. "A Transfer Learning Approach with MobileNetV2 for Parkinson's Disease Detection using Hand Drawings." 2021.
3. Orozco-Arroyave, J. R., et al. "Analysis of speech of people with Parkinson's disease." *INTERSPEECH*, 2016.
4. Sakar, B. E., et al. "A comparative analysis of speech signal processing algorithms for Parkinson's disease classification." *Computer Methods and Programs in Biomedicine*, 2019.
5. Vasquez-Correa, J. C., et al. "Multimodal assessment of Parkinson's disease: A deep learning approach." *IEEE Journal of Biomedical and Health Informatics*, 2019.
6. Narendra, N. P., & Alku, P. "Glottal source information for pathological voice detection." *IEEE Access*, 2021.
7. Quan, C., et al. "End-to-end Parkinson's disease detection using self-supervised speech pre-training." 2022.
8. Mughal, H. A., et al. "Parkinson's disease detection from speech using CNN-BiLSTM." 2023.
9. Baevski, A., et al. "wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations." *NeurIPS*, 2020.
10. Conneau, A., et al. "Unsupervised Cross-lingual Representation Learning for Speech Recognition." *INTERSPEECH*, 2021.
11. Tan, M., & Le, Q. "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks." *ICML*, 2019.
12. Woo, S., et al. "CBAM: Convolutional Block Attention Module." *ECCV*, 2018.
13. Arevalo, J., et al. "Gated Multimodal Units for Information Fusion." *ICLR Workshop*, 2017.
14. Park, D. S., et al. "SpecAugment: A Simple Data Augmentation Method for Automatic Speech Recognition." *INTERSPEECH*, 2019.
15. Okabe, K., et al. "Attentive Statistics Pooling for Deep Speaker Embedding." *INTERSPEECH*, 2018.
16. He, K., et al. "Deep Residual Learning for Image Recognition." *CVPR*, 2016.

---

<p align="center">
  <strong>Developed as a Senior Design Project (SDP)</strong><br>
  <em>Multimodal AI for Healthcare — Parkinson's Disease Early Detection</em>
</p>
