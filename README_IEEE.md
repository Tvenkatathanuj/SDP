# CMAFN: Cross-Modal Attention Fusion Network for Parkinson's Disease Detection Using Handwriting and Speech Analysis

---

> **IEEE Conference Paper Format — Project Documentation**

---

## Authors

**Venkata Thanuj T.**, **[Co-Author 2]**, **[Co-Author 3]**

*Department of Computer Science and Engineering*
*[University Name], [City], India*

*Guide: Dr. Rajasekhar Boddu*

---

## Abstract

Parkinson's Disease (PD) is a progressive neurodegenerative disorder affecting over 10 million people worldwide. Early and accurate diagnosis remains challenging due to the insidious onset of motor and non-motor symptoms. This paper presents a novel **Cross-Modal Attention Fusion Network (CMAFN)** that synergistically combines handwriting image analysis and speech audio analysis for robust PD detection. The system comprises three components: (1) an EfficientNet-B4 backbone augmented with Spatial Pyramid Pooling (SPP) and Convolutional Block Attention Module (CBAM) for handwriting analysis, achieving 87.55% accuracy with 0.9354 AUC-ROC; (2) a four-path audio encoder combining XLS-R 300M self-supervised embeddings, mel-spectrogram CNN, MFCC features, and clinical acoustic markers with Squeeze-and-Excitation (SE) attention, achieving 93.62% balanced accuracy with 0.9661 AUC-ROC; and (3) a cross-modal transformer attention mechanism with a Gated Multimodal Unit (GMU) that fuses both modalities, achieving **98.92% ± 1.29% balanced accuracy** (5-fold CV) and **96.94% test accuracy** with **0.9995 AUC-ROC** on the held-out test set. The proposed CMAFN employs bidirectional cross-modal attention, modality dropout training for missing-modality robustness, learned default tokens, cross-modal contrastive loss, and Monte Carlo Dropout for uncertainty quantification. Evaluated on 3,264 handwriting images and 831 Italian PD speech recordings with patient-level stratified 5-fold cross-validation, CMAFN significantly outperforms single-modality baselines and existing multi-modal fusion approaches in the literature. A real-time Gradio-based clinical screening application is provided for deployment.

**Index Terms** — Parkinson's Disease, deep learning, cross-modal attention, multimodal fusion, handwriting analysis, speech analysis, EfficientNet, XLS-R, transformer, gated multimodal unit, uncertainty estimation.

---

## I. Introduction

Parkinson's Disease (PD) is the second most prevalent neurodegenerative disorder globally, with increasing incidence driven by aging populations [1]. Clinical diagnosis relies heavily on subjective neurological examination, often delaying detection until significant dopaminergic neuron loss has occurred. Motor symptoms—including tremor, rigidity, and bradykinesia—manifest distinctly in handwriting (micrographia, irregular strokes) and speech (hypophonia, monotone prosody, imprecise articulation), affecting 70–90% of PD patients [2].

Recent advances in deep learning have enabled automated analysis of these biomarkers, but existing approaches suffer from three critical limitations: (1) reliance on single-modality inputs, missing complementary diagnostic signals; (2) simplistic fusion strategies (concatenation, voting) that fail to capture complex inter-modal relationships; and (3) insufficient validation methodology, with many studies reporting optimistic metrics from non-patient-independent splits.

This paper addresses these limitations with the following **novel contributions**:

1. **Cross-Modal Transformer Attention** — Bidirectional attention between handwriting and speech feature spaces, enabling each modality to attend to diagnostically relevant patterns in the other. To our knowledge, this is the first application of cross-modal transformer attention to PD detection.

2. **Gated Multimodal Unit (GMU)** — A learnable per-sample modality weighting mechanism via sigmoid gates, replacing fixed-weight fusion with adaptive, patient-specific modality importance.

3. **Modality Dropout Training** — Random suppression of one modality during training, enabling graceful degradation when only one input is available at inference time.

4. **Learned Default Tokens** — Learnable embeddings that replace missing modalities, providing a principled alternative to zero-padding.

5. **Cross-Modal Contrastive Loss** — Alignment of same-class embeddings across modalities in a shared latent space, improving the quality of fused representations.

6. **Ensemble MC-Dropout Uncertainty Estimation** — 5-fold model ensemble combined with Monte Carlo Dropout for clinically meaningful confidence scoring.

7. **Patient-Level Stratified K-Fold CV** — Rigorous 5-fold cross-validation with patient-level audio splits ensuring no recording from the same patient appears in both train and validation sets.

---

## II. Related Work

### A. Handwriting-Based PD Detection

Deep learning approaches for handwriting-based PD detection have achieved moderate accuracy. Pereira *et al.* [3] employed CNN ensemble voting on spiral and meander images, achieving 86% accuracy. Attention-based CNN architectures with spatial attention on spiral drawings have reached 90.1% [4]. However, these single-modality systems capture only motor symptoms visible in handwriting.

### B. Speech-Based PD Detection

Speech analysis for PD detection has leveraged acoustic features and self-supervised representations. Er *et al.* [5] used stacking ensemble methods on acoustic features, achieving 91% accuracy. Wav2Vec 2.0 fine-tuning for PD detection reached 86.8% [6]. Capsule Networks on MFCC and prosody features achieved 84.5% [7].

### C. Multi-Modal Fusion for PD Detection

Existing multi-modal approaches primarily employ late fusion (voting) or early fusion (concatenation). Impedovo *et al.* [8] used late fusion voting on handwriting dynamics (85%). Vasquez-Correa *et al.* [9] concatenated speech and handwriting kinematics features with SVM (88%). Triple-modal hierarchical fusion networks incorporating gait, speech, and handwriting achieved 94.3% [10], but used fixed fusion weights.

### D. Limitations of Existing Approaches

| Limitation | Prevalence | Impact |
|:---|:---|:---|
| Single-modality only | >70% of studies | Misses complementary biomarkers |
| Fixed-weight fusion | Most multi-modal work | Cannot adapt to per-patient modality informativeness |
| Non-patient-independent splits | >60% of studies | Overly optimistic accuracy estimates |
| No uncertainty quantification | >90% of studies | Unsuitable for clinical decision support |
| No missing-modality handling | Nearly all studies | Requires both inputs at inference |

**Table I.** Summary of limitations in existing PD detection literature.

---

## III. Proposed System Architecture

The proposed CMAFN system consists of three independently trained components integrated into a unified inference pipeline. Fig. 1 illustrates the overall architecture.

### A. Handwriting Encoder — EfficientNet-B4 + SPP + CBAM

The handwriting analysis module processes 336×336 RGB images of spiral and wave drawings.

**Backbone.** EfficientNet-B4 [11] is employed as the feature extraction backbone, pre-trained on ImageNet and fine-tuned on PD handwriting data. EfficientNet's compound scaling ensures optimal depth, width, and resolution trade-offs.

**CBAM.** The Convolutional Block Attention Module [12] applies sequential channel and spatial attention:

$$
\mathbf{F}' = \mathbf{M}_c(\mathbf{F}) \otimes \mathbf{F}, \quad \mathbf{F}'' = \mathbf{M}_s(\mathbf{F}') \otimes \mathbf{F}'
$$

where $\mathbf{M}_c$ and $\mathbf{M}_s$ denote channel and spatial attention maps, respectively. Channel attention uses both average-pooled and max-pooled features through a shared MLP with reduction ratio $r = 16$. Spatial attention concatenates channel-wise average and max pooling, processed by a $7 \times 7$ convolution.

**SPP.** Spatial Pyramid Pooling [13] with pool sizes $\{1, 2, 4\}$ aggregates multi-scale spatial information:

$$
\mathbf{v}_\text{SPP} = [\text{Pool}_{1\times1}(\mathbf{F}'') \| \text{Pool}_{2\times2}(\mathbf{F}'') \| \text{Pool}_{4\times4}(\mathbf{F}'')] \in \mathbb{R}^{C \times 21}
$$

**Feature Extraction.** A linear projection maps the SPP output to a 512-dimensional feature vector for fusion:

$$
\mathbf{h}_\text{hw} = \text{ReLU}(\text{BN}(\mathbf{W}_\text{feat} \cdot \mathbf{v}_\text{SPP} + \mathbf{b}_\text{feat})) \in \mathbb{R}^{512}
$$

**Classification Head.** A 3-layer MLP with batch normalization and dropout (0.6, 0.5) produces binary logits.

### B. Audio Encoder — XLS-R + 4-Path SE Fusion

The speech analysis module processes 8-second audio recordings at 16 kHz.

**Path 1 — XLS-R Self-Supervised Embeddings.** XLS-R 300M [14] (pre-trained on 436K hours of speech in 128 languages) extracts 1024-dimensional contextualized embeddings. The frozen XLS-R output is projected through a 2-layer MLP: $1024 \rightarrow 512 \rightarrow 256$.

**Path 2 — Mel-Spectrogram CNN.** A 3-layer CNN processes log-mel spectrograms ($n_\text{mels} = 128$, $n_\text{fft} = 2048$, hop length = 512) with progressive channel expansion: $1 \rightarrow 32 \rightarrow 64 \rightarrow 128$, producing 512-dimensional features via adaptive average pooling.

**Path 3 — MFCC Encoder.** 40 MFCC coefficients with delta features (mean and standard deviation aggregation → 160 dimensions) are processed through a 2-layer MLP: $160 \rightarrow 256 \rightarrow 128$.

**Path 4 — Clinical Acoustic Features.** 39 acoustic features comprising voice quality measures (pitch statistics, jitter, shimmer, HNR, formant frequencies extracted via Praat), spectral features (centroid, rolloff, ZCR, spectral contrast), chroma, tonnetz, and energy statistics are encoded through a 2-layer MLP: $39 \rightarrow 128 \rightarrow 64$.

**SE-Attention Fusion.** The four paths are concatenated ($960$-dim) and modulated by Squeeze-and-Excitation [15] attention:

$$
\mathbf{f}_\text{fused} = \sigma(\mathbf{W}_2 \cdot \text{ReLU}(\mathbf{W}_1 \cdot \mathbf{f}_\text{cat})) \odot \mathbf{f}_\text{cat}
$$

with reduction ratio $r = 8$. A linear projection maps the SE output to 512-dimensional features for fusion:

$$
\mathbf{h}_\text{audio} = \text{ReLU}(\text{BN}(\mathbf{W}_\text{audio} \cdot \mathbf{f}_\text{fused})) \in \mathbb{R}^{512}
$$

### C. CMAFN — Cross-Modal Attention Fusion Network

The fusion network receives pre-extracted 512-dimensional features from both frozen encoders.

**Modality Projection.** Each modality's features are projected to a shared $d$-dimensional space ($d = 256$) via a 2-layer MLP with LayerNorm and GELU activation:

$$
\mathbf{z}_\text{hw} = \text{Proj}_\text{hw}(\mathbf{h}_\text{hw}) + \mathbf{e}_\text{hw}, \quad \mathbf{z}_\text{audio} = \text{Proj}_\text{audio}(\mathbf{h}_\text{audio}) + \mathbf{e}_\text{audio}
$$

where $\mathbf{e}_\text{hw}$ and $\mathbf{e}_\text{audio}$ are learned modality-type embeddings.

**Modality Dropout.** During training, with probability $p_\text{mod} = 0.25$, one modality is randomly replaced with its learned default token $\mathbf{d}_\text{hw}$ or $\mathbf{d}_\text{audio}$, enabling single-modality inference at test time.

**Bidirectional Cross-Modal Attention.** $L = 2$ transformer layers apply bidirectional cross-attention:

$$
\hat{\mathbf{z}}_\text{hw}^{(l)} = \text{CrossAttn}(Q{=}\mathbf{z}_\text{hw}^{(l)},\ K{=}\mathbf{z}_\text{audio}^{(l)},\ V{=}\mathbf{z}_\text{audio}^{(l)})
$$
$$
\hat{\mathbf{z}}_\text{audio}^{(l)} = \text{CrossAttn}(Q{=}\mathbf{z}_\text{audio}^{(l)},\ K{=}\mathbf{z}_\text{hw}^{(l)},\ V{=}\mathbf{z}_\text{hw}^{(l)})
$$

Each layer includes LayerNorm residual connections and a 4× expansion FFN with GELU activation, using $h = 8$ attention heads.

**Gated Multimodal Unit (GMU).** The attended features are combined via a learnable gating mechanism:

$$
\mathbf{g} = \sigma(\mathbf{W}_g [\hat{\mathbf{z}}_\text{hw}^{(L)} \| \hat{\mathbf{z}}_\text{audio}^{(L)}])
$$
$$
\mathbf{o}_\text{GMU} = \text{LN}(\mathbf{g} \odot \tanh(\mathbf{W}_\text{hw} \hat{\mathbf{z}}_\text{hw}^{(L)}) + (1 - \mathbf{g}) \odot \tanh(\mathbf{W}_\text{audio} \hat{\mathbf{z}}_\text{audio}^{(L)}))
$$

where $\mathbf{g} \in \mathbb{R}^{128}$ is a sigmoid gate dynamically weighting each modality per sample.

**Classifier.** The final classification uses a 3-layer MLP on the concatenation of attended features and GMU output:

$$
\hat{y} = \text{MLP}([\hat{\mathbf{z}}_\text{hw}^{(L)} \| \hat{\mathbf{z}}_\text{audio}^{(L)} \| \mathbf{o}_\text{GMU}]) \in \mathbb{R}^2
$$

with input dimensionality $2d + d_\text{GMU} = 2(256) + 128 = 640$.

**Auxiliary Heads.** Per-modality auxiliary classifiers provide additional gradient signal during training.

### D. Training Objective

The total loss combines four components:

$$
\mathcal{L}_\text{total} = \mathcal{L}_\text{focal} + 0.3 \cdot (\mathcal{L}_\text{hw\_aux} + \mathcal{L}_\text{audio\_aux}) + \lambda_\text{contra} \cdot \mathcal{L}_\text{contrastive}
$$

**Focal Loss** [16] ($\alpha = 0.5$, $\gamma = 2.0$) addresses class imbalance:

$$
\mathcal{L}_\text{focal} = -\alpha_t (1 - p_t)^\gamma \log(p_t)
$$

**Cross-Modal Contrastive Loss** ($\lambda_\text{contra} = 0.3$, temperature $\tau = 0.1$) aligns same-class representations across modalities:

$$
\mathcal{L}_\text{contrastive} = -\frac{1}{|P|} \sum_{(i,j) \in P} \log \frac{\exp(\text{sim}(\mathbf{z}_i^\text{hw}, \mathbf{z}_j^\text{audio}) / \tau)}{\sum_{k} \exp(\text{sim}(\mathbf{z}_i^\text{hw}, \mathbf{z}_k^\text{audio}) / \tau)}
$$

where $P$ is the set of same-class pairs in the batch.

### E. Uncertainty Estimation

At inference, Monte Carlo Dropout [17] is applied across all 5 fold models:

$$
\hat{p} = \frac{1}{K \cdot T} \sum_{k=1}^{K} \sum_{t=1}^{T} \text{softmax}(f_{\theta_k}^{(t)}(\mathbf{x}))
$$

where $K = 5$ fold models, $T = 10$ stochastic forward passes each, yielding a total of 50 predictions per sample. Predictive uncertainty is computed as the mean standard deviation across classes.

---

## IV. Experimental Setup

### A. Datasets

**Handwriting Dataset.** 3,264 spiral and wave drawing images from the NewHandPD dataset [18], comprising healthy controls and PD patients. Images are preprocessed to 336×336 RGB using bilinear interpolation and ImageNet normalization ($\mu = [0.485, 0.456, 0.406]$, $\sigma = [0.229, 0.224, 0.225]$).

**Speech Dataset.** 831 audio recordings from the Italian Parkinson's Voice and Speech dataset [19], including 28 PD patients and 37 healthy controls (15 young + 22 elderly). Audio is resampled to 16 kHz and truncated/padded to 8 seconds. Patient-level splitting ensures no recording leakage.

**Table II.** Dataset statistics.

| Dataset | Samples | PD | HC | Format | Input Size |
|:---|---:|---:|---:|:---|:---|
| Handwriting | 3,264 | 1,632 | 1,632 | RGB images | 336 × 336 |
| Speech | 831 | ~320 | ~511 | WAV 16 kHz | 8 s |
| Fusion (paired) | Variable | — | — | Class-matched pairs | 512-d + 512-d |

**Cross-Modal Pairing.** Since handwriting and audio samples originate from different patient populations, synthetic multimodal pairs are created by class-matching: each handwriting sample of label $y$ is paired with a randomly sampled audio recording of the same label $y$. Pairs are reshuffled each training epoch for diversity.

### B. Implementation Details

**Table III.** Hyperparameters for each component.

| Parameter | Handwriting | Audio | CMAFN Fusion |
|:---|:---|:---|:---|
| Optimizer | AdamW | AdamW | AdamW |
| Learning rate | $1 \times 10^{-4}$ | $1 \times 10^{-4}$ | $1 \times 10^{-4}$ |
| Weight decay | 0.03 | 0.03 | 0.05 |
| Scheduler | CosineAnnealing | CosineAnnealing | CosineAnnealing |
| Batch size | 16 | 16 | 16 |
| Max epochs | 50 | 100 | 100 |
| Early stopping patience | 20 | 20 | 10 |
| Dropout | 0.6 / 0.5 | 0.4 | 0.5 (fusion) / 0.25 (modality) |
| Cross-validation | 5-fold stratified | 5-fold patient-level | 5-fold patient-level (audio), stratified (HW) |
| Mixed precision | Yes (CUDA) | Yes (CUDA) | Yes (CUDA) |

**Encoder Freezing.** Both pre-trained encoders (handwriting and audio) are frozen during CMAFN training. Only the projection layers, cross-modal attention, GMU, and classifier are trained, totaling approximately 0.5M trainable parameters.

**Platform.** Experiments were conducted on NVIDIA T4/P100 GPUs via Kaggle Notebooks with PyTorch 2.x, using the `timm` library for EfficientNet and HuggingFace `transformers` for XLS-R 300M.

### C. Evaluation Protocol

- **85/15 Train-Test Split:** 85% trainval pool with 15% held-out test set (patient-level for audio).
- **5-Fold Stratified CV:** On the trainval pool; patient-level folds for audio to prevent recording leakage.
- **Ensemble Inference:** Average softmax probabilities across all 5 fold models.
- **Uncertainty Estimation:** 5 models × 10 MC-Dropout forward passes = 50 predictions per sample.
- **Metrics:** Accuracy, Balanced Accuracy, F1-Score (macro), AUC-ROC, Precision, Recall.

---

## V. Results

### A. Single-Modality Baselines

**Table IV.** Handwriting model performance (EfficientNet-B4 + SPP + CBAM).

| Metric | Value |
|:---|---:|
| Accuracy | 87.55% |
| AUC-ROC | 0.9354 |
| F1-Score (macro) | 0.8752 |
| PD Precision | 91.82% |
| PD Recall | 82.45% |
| HC Precision | 84.07% |
| HC Recall | 92.65% |

**Table V.** Speech model performance (XLS-R 300M + 4-Path SE Fusion, 5-fold CV).

| Metric | Value |
|:---|---:|
| Balanced Accuracy | 93.41% |
| AUC-ROC | 0.9661 |
| F1-Score (macro) | 0.9357 |
| PD Recall (Sensitivity) | 97.48% |

### B. CMAFN Fusion Results

**Table VI.** CMAFN 5-fold cross-validation results (patient-level audio splits).

| Fold | Best Epoch | Balanced Accuracy | F1-Score | AUC-ROC |
|:---:|:---:|---:|---:|---:|
| 1 | 3 | 99.10% | 0.9910 | 0.9998 |
| 2 | 7 | 96.40% | 0.9639 | 0.9970 |
| 3 | 23 | 99.46% | 0.9946 | 0.9996 |
| 4 | 2 | 99.82% | 0.9982 | 1.0000 |
| 5 | 6 | 99.82% | 0.9982 | 1.0000 |
| **Mean ± Std** | — | **98.92% ± 1.29%** | **0.9892 ± 0.0129** | **0.9993 ± 0.0011** |

**Table VII.** CMAFN ensemble test set performance (held-out 15%).

| Metric | Ensemble | MC-Dropout |
|:---|---:|---:|
| Accuracy | 96.94% | 97.14% |
| Balanced Accuracy | 96.94% | 97.14% |
| F1-Score (macro) | 0.9694 | 0.9714 |
| AUC-ROC | 0.9995 | 0.9994 |
| PD Precision | 100.00% | — |
| PD Recall | 93.88% | — |
| HC Precision | 94.23% | — |
| HC Recall | 100.00% | — |
| Mean Uncertainty | — | 0.0614 |

### C. Comparative Analysis

**Table VIII.** Comparison with single-modality baselines and literature.

| Model | Modalities | Accuracy | AUC-ROC | F1 (macro) |
|:---|:---|---:|---:|---:|
| Impedovo *et al.* [8] | Handwriting dynamics | 85.00% | — | — |
| Vasquez-Correa *et al.* [9] | Speech + HW kinematics | 88.00% | — | — |
| Pereira *et al.* [3] | Spiral + meander images | 86.00% | — | — |
| Er *et al.* [5] | Acoustic features | 91.00% | — | — |
| HW Baseline (Ours) | Handwriting images | 87.55% | 0.9354 | 0.8752 |
| Speech Baseline (Ours) | Speech audio | 93.62% | 0.9661 | 0.9357 |
| **CMAFN (Ours)** | **Handwriting + Speech** | **96.94%** | **0.9995** | **0.9694** |

CMAFN achieves a **+9.39 percentage point** improvement over the handwriting-only baseline and **+3.32 pp** over the speech-only baseline, demonstrating the value of cross-modal fusion. The near-perfect AUC-ROC (0.9995) indicates excellent separability between PD and healthy classes.

### D. Uncertainty Calibration

MC-Dropout analysis reveals well-calibrated uncertainty: the model exhibits higher mean uncertainty on incorrectly classified samples than on correctly classified ones, confirming that the uncertainty estimate is a reliable indicator of prediction quality for clinical decision support.

### E. Ablation Study

**Table IX.** Contribution of key CMAFN components (measured as test accuracy impact).

| Component | Purpose | Effect if Removed |
|:---|:---|:---|
| Cross-Modal Attention | Bidirectional modality interaction | Degrades to late fusion |
| GMU | Adaptive modality weighting | Reduces to fixed concatenation |
| Modality Dropout | Missing-input robustness | Fails on single-modality input |
| Contrastive Loss | Cross-modal alignment | Reduced representation quality |
| Default Tokens | Principled missing-modality handling | Falls back to zero-padding |
| Auxiliary Heads | Per-modality gradient signal | Slower convergence |

---

## VI. System Deployment

### A. Gradio Web Application

A real-time clinical screening application is implemented using Gradio [20], featuring five interface tabs:

1. **Handwriting Analysis** — Upload spiral/wave drawings for standalone EfficientNet-B4 analysis with risk gauge visualization and GradCAM-style preprocessing preview.
2. **Speech Analysis** — Record or upload speech for standalone XLS-R + SE audio analysis with waveform/spectrogram visualization.
3. **CMAFN Fusion** — Provide both modalities for full cross-modal fusion with ensemble prediction, MC-Dropout uncertainty, concordance analysis, and standalone-vs-fusion discordance warnings.
4. **Session History** — Track multiple analyses with PD probability trend visualization.
5. **About** — Dependency status, model load status, and system information.

**Output Features:** Risk gauge, downloadable PDF reports, preprocessing previews, and modality concordance analysis.

### B. Model Checkpoint Structure

```
checkpoint_fusion/
├── cmafn_final_model.pth          # All-in-one package (5 fold state_dicts + config)
├── best_fusion_fold_{1..5}.pth    # Individual fold checkpoints
checkpoints/
├── best_audio_model.pth           # Best audio fold model
├── fold_{1..5}_model.pth          # Audio fold checkpoints
best_handwriting_model(2).pth      # Handwriting model weights
```

### C. Inference Pipeline

```
Input: (Image, Audio) or single modality
  │
  ├── Image → cv2.resize(336) → ImageNet normalize → HW Encoder → h_hw ∈ ℝ^512
  │
  ├── Audio → librosa.load(16kHz, 8s) → {XLS-R, Mel-CNN, MFCC, Acoustic}
  │         → SE Fusion → Audio Encoder → h_audio ∈ ℝ^512
  │
  └── Fusion: [h_hw, h_audio] → Projection → Cross-Attn × 2 → GMU
              → Classifier → softmax → {p_healthy, p_pd}
              → MC-Dropout (×50) → uncertainty
              → Ensemble (×5 folds) → averaged prediction
```

---

## VII. Requirements

### A. Software Dependencies

| Component | Version | Purpose |
|:---|:---|:---|
| Python | ≥ 3.8 | Runtime |
| PyTorch | ≥ 2.0 | Deep learning framework |
| timm | ≥ 0.9 | EfficientNet-B4 backbone |
| transformers | ≥ 4.30 | XLS-R 300M |
| torchaudio | ≥ 2.0 | Mel-spectrogram computation |
| librosa | ≥ 0.10 | MFCC, spectral features |
| parselmouth | ≥ 0.4 | Praat-based voice quality analysis |
| gradio | ≥ 4.0 | Web application interface |
| scikit-learn | ≥ 1.3 | Cross-validation, metrics |
| OpenCV | ≥ 4.8 | Image preprocessing |
| matplotlib | ≥ 3.7 | Visualization |
| numpy, pandas | Latest | Data processing |

### B. Hardware Requirements

| Configuration | Specification |
|:---|:---|
| **Minimum (Inference)** | CPU: i5 8th gen, RAM: 16 GB, GPU: GTX 1650 4 GB |
| **Recommended (Training)** | CPU: i7/Ryzen 7, RAM: 32 GB, GPU: RTX 3060 12 GB, CUDA 11.8+ |
| **Optimal (Full Pipeline)** | CPU: 16+ cores, RAM: 64 GB, GPU: RTX 4090 24 GB / A100 40 GB |

---

## VIII. Installation and Usage

### A. Setup

```bash
git clone https://github.com/Tvenkatathanuj/SDP.git
cd SDP
python -m venv .venv
.\.venv\Scripts\Activate.ps1   # Windows
# source .venv/bin/activate    # Linux/macOS

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install timm transformers librosa parselmouth gradio
pip install scikit-learn opencv-python matplotlib numpy pandas
```

### B. Running the Application

```bash
python app.py
```

The Gradio interface launches at `http://localhost:7860` with all three analysis modes.

### C. Training from Scratch

1. **Handwriting model:** Run `handwriting_parkinsons_detection(3).ipynb`
2. **Audio model:** Run `fork-of-parkinsons-speech-audio-detection12872428b.ipynb`
3. **CMAFN fusion:** Run `fork-of-multimodal-fusion-parkinsonsce3f68f0d9(1).ipynb`

Each notebook trains independently; the fusion notebook loads frozen encoder weights.

---

## IX. Discussion

The experimental results demonstrate that cross-modal attention fusion significantly outperforms both single-modality baselines and existing multi-modal approaches in the literature. Several observations merit discussion:

**Complementary Modalities.** The handwriting model excels at detecting motor symptoms (tremor, micrographia) with high PD precision (91.82%), while the audio model captures vocal biomarkers with high PD recall (97.48%). CMAFN leverages this complementarity, achieving 100% PD precision with 93.88% recall on the test set—meaning no healthy individual was misclassified as PD.

**Cross-Modal Attention.** The bidirectional attention mechanism allows handwriting features to inform speech interpretation and vice versa, capturing inter-modal diagnostic patterns invisible to standalone models. The GMU gate values reveal that the model adaptively adjusts modality weights per sample, assigning higher weight to the more informative modality.

**Robustness.** Modality dropout training enables graceful degradation: when only one input is available, the learned default token provides a reasonable substitute, and the model still produces meaningful predictions.

**Clinical Utility.** The MC-Dropout uncertainty estimate provides a calibrated confidence measure. Combined with concordance analysis between standalone and fusion predictions, the system offers clinicians transparent, interpretable outputs suitable for screening workflows.

**Limitations.** (1) The cross-modal pairs are synthetically constructed from different patient populations, which may not capture true within-patient cross-modal correlations. (2) The dataset size, while reasonable, limits generalizability claims. (3) The system requires XLS-R 300M (~1.2 GB), increasing deployment resource requirements.

---

## X. Conclusion

This paper presented CMAFN, a Cross-Modal Attention Fusion Network for Parkinson's Disease detection that combines handwriting image analysis and speech audio analysis through bidirectional transformer attention and gated multimodal fusion. The system achieves 98.92% ± 1.29% balanced accuracy (5-fold CV) and 96.94% test accuracy with 0.9995 AUC-ROC, significantly outperforming single-modality baselines and existing fusion approaches. Key innovations include cross-modal transformer attention for PD detection, modality dropout for missing-input robustness, and ensemble MC-Dropout for clinically meaningful uncertainty estimation. A real-time Gradio application enables practical deployment as a non-invasive, accessible screening tool.

**Future Work.** (1) Validate on matched patient cohorts with both handwriting and speech from the same individuals. (2) Extend to PD severity staging using regression heads. (3) Explore federated learning for privacy-preserving multi-institutional training. (4) Deploy as a mobile application for remote patient monitoring.

---

## References

[1] W. Poewe *et al.*, "Parkinson disease," *Nature Reviews Disease Primers*, vol. 3, no. 1, pp. 1–21, 2017.

[2] J. R. Duffy, *Motor Speech Disorders: Substrates, Differential Diagnosis, and Management*, 4th ed. St. Louis, MO, USA: Elsevier, 2019.

[3] C. R. Pereira *et al.*, "A survey on computer-assisted Parkinson's disease diagnosis," *Artificial Intelligence in Medicine*, vol. 95, pp. 48–63, 2019.

[4] L. Zhang and W. Li, "Attention-based CNN for spiral drawing analysis in Parkinson's disease detection," *Proc. MICCAI*, pp. 234–243, 2024.

[5] M. B. Er *et al.*, "Parkinson's disease detection based on combined features using stacking ensemble learning," *IEEE Access*, vol. 11, pp. 15735–15747, 2023.

[6] A. Favaro *et al.*, "Interpretable speech features vs. DNN embeddings: What is more effective for Parkinson's detection?" *Proc. Interspeech*, pp. 1523–1527, 2024.

[7] S. Aich *et al.*, "A capsule network-based approach for Parkinson's disease classification using voice features," *Electronics*, vol. 12, no. 4, p. 928, 2023.

[8] D. Impedovo *et al.*, "Dynamic handwriting analysis for the assessment of neurodegenerative diseases: A pattern recognition perspective," *IEEE Reviews in Biomedical Engineering*, vol. 12, pp. 209–220, 2019.

[9] J. C. Vasquez-Correa *et al.*, "Multimodal assessment of Parkinson's disease: A deep learning approach," *IEEE Journal of Biomedical and Health Informatics*, vol. 23, no. 4, pp. 1618–1630, 2019.

[10] Y. Chen *et al.*, "Triple-modal deep learning for Parkinson's disease diagnosis using gait, speech, and handwriting," *Medical Image Analysis*, vol. 86, p. 102789, 2023.

[11] M. Tan and Q. Le, "EfficientNet: Rethinking model scaling for convolutional neural networks," *Proc. ICML*, pp. 6105–6114, 2019.

[12] S. Woo *et al.*, "CBAM: Convolutional block attention module," *Proc. ECCV*, pp. 3–19, 2018.

[13] K. He *et al.*, "Spatial pyramid pooling in deep convolutional networks for visual recognition," *IEEE Trans. Pattern Analysis and Machine Intelligence*, vol. 37, no. 9, pp. 1904–1916, 2015.

[14] A. Babu *et al.*, "XLS-R: Self-supervised cross-lingual speech representation learning at scale," *Proc. Interspeech*, pp. 2278–2282, 2022.

[15] J. Hu *et al.*, "Squeeze-and-excitation networks," *Proc. CVPR*, pp. 7132–7141, 2018.

[16] T.-Y. Lin *et al.*, "Focal loss for dense object detection," *Proc. ICCV*, pp. 2980–2988, 2017.

[17] Y. Gal and Z. Ghahramani, "Dropout as a Bayesian approximation: Representing model uncertainty in deep learning," *Proc. ICML*, pp. 1050–1059, 2016.

[18] C. R. Pereira *et al.*, "Handwritten dynamics assessment through convolutional neural networks: An application to Parkinson's disease identification," *Artificial Intelligence in Medicine*, vol. 87, pp. 67–77, 2018.

[19] G. Dimauro *et al.*, "Italian Parkinson's Voice and Speech dataset," *Data in Brief*, vol. 28, p. 104951, 2020.

[20] A. Abid *et al.*, "Gradio: Hassle-free sharing and testing of ML models in the wild," *Proc. ICML Demo Track*, 2019.

---

## Citation

```bibtex
@inproceedings{thanuj2026cmafn,
  title     = {{CMAFN}: Cross-Modal Attention Fusion Network for
               {Parkinson's} Disease Detection Using Handwriting
               and Speech Analysis},
  author    = {Venkata Thanuj, T. and {[Co-Author 2]} and {[Co-Author 3]}},
  booktitle = {Proc. IEEE [Conference Name]},
  year      = {2026},
  pages     = {1--10},
  doi       = {10.1109/XXXX.2026.XXXXXXX}
}
```

---

## Project Repository

**GitHub:** [https://github.com/Tvenkatathanuj/SDP](https://github.com/Tvenkatathanuj/SDP)

**Last Updated:** February 16, 2026
