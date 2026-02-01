# Healthy Speech Samples from Dataset

## 📁 Location: `original-speech-dataset/emma/IC/`

The **IC (Healthy Controls)** folder contains speech samples from healthy individuals. These can be used for testing the speech model.

## 🎤 Sample Files

Here are some healthy speech samples you can use:

### Sample 1: IC1.wav
- **Location**: `original-speech-dataset/emma/IC/IC1.wav`
- **Text**: "Firstly, I would say you've got to have confidence to wear trousers like this."
- **Use for**: Testing normal speech patterns

### Sample 2: IC2.wav
- **Location**: `original-speech-dataset/emma/IC/IC2.wav`
- **Use for**: Baseline comparison

### Sample 3: IC10.wav
- **Location**: `original-speech-dataset/emma/IC/IC10.wav`
- **Use for**: Testing model accuracy

### Sample 4: IC15.wav
- **Location**: `original-speech-dataset/emma/IC/IC15.wav`
- **Use for**: Healthy voice characteristics

### Sample 5: IC20.wav
- **Location**: `original-speech-dataset/emma/IC/IC20.wav`
- **Use for**: Normal prosody patterns

## 📊 Full Dataset Structure

### Healthy Controls (IC)
- **Path**: `original-speech-dataset/emma/IC/`
- **Files**: IC1.wav through IC123.wav
- **Total**: 123 healthy speech samples
- **Type**: Conversational speech from healthy individuals

### Also Available (WP - Healthy Controls)
- **Path**: `original-speech-dataset/emma/WP/`
- **Files**: WP1.wav through WP240.wav
- **Total**: 240 healthy speech samples
- **Type**: Word pairs from healthy speakers

## 🧪 How to Test with Healthy Samples

### Using the Gradio App:

1. **Run the app**:
   ```bash
   python app.py
   ```

2. **Upload a healthy sample**:
   - Go to the "Speech Analysis" tab
   - Click "Upload Audio"
   - Select a file from `original-speech-dataset/emma/IC/`
   - Example: `IC1.wav`, `IC10.wav`, etc.

3. **Expected Result for Healthy Speech**:
   - ✅ Severity Score: < 0.3 (Minimal/No Indicators)
   - ✅ Classification: "Minimal/No Indicators"
   - 🟢 Green status

### Using Python Script:

```python
import librosa
import torch
import numpy as np

# Load healthy speech sample
audio, sr = librosa.load('original-speech-dataset/emma/IC/IC1.wav', sr=16000)

# Use in your model...
```

## 📋 Quick Access List

### Top 10 Recommended Healthy Samples:
1. `emma/IC/IC1.wav` - Clear speech
2. `emma/IC/IC5.wav` - Good volume
3. `emma/IC/IC10.wav` - Natural prosody
4. `emma/IC/IC15.wav` - Steady pitch
5. `emma/IC/IC20.wav` - Clear articulation
6. `emma/IC/IC25.wav` - Normal rhythm
7. `emma/IC/IC30.wav` - Good quality
8. `emma/IC/IC35.wav` - Varied intonation
9. `emma/IC/IC40.wav` - Conversational
10. `emma/IC/IC50.wav` - Natural speech

## 🔍 Parkinson's Speech Samples (For Comparison)

### DL Dataset (Parkinson's Patients):
- **Path**: `original-speech-dataset/DL/`
- **Files**: DL1.wav through DL65.wav
- **Type**: Speech from diagnosed Parkinson's patients

### Expected Results for Parkinson's Samples:
- ⚠️ Severity Score: 0.5-0.8 (Moderate to High)
- ⚠️ Classification: "Moderate/Significant Indicators"
- 🟠/🔴 Orange/Red status

## 💡 Testing Tips

1. **Test Multiple Samples**: Try 5-10 healthy samples to see consistency
2. **Compare Results**: Test both healthy (IC) and PD (DL) samples
3. **Check Severity Scores**:
   - Healthy should be < 0.3
   - Parkinson's should be > 0.5
4. **Look for Patterns**: Healthy speech should show consistent low scores

## 📂 Full Directory Structure

```
original-speech-dataset/
├── emma/
│   ├── IC/          ← Healthy Controls (123 files)
│   │   ├── IC1.wav
│   │   ├── IC2.wav
│   │   └── ...
│   └── WP/          ← Healthy Word Pairs (240 files)
│       ├── WP1.wav
│       ├── WP2.wav
│       └── ...
├── DL/              ← Parkinson's Patients (65 files)
├── LW/              ← Parkinson's Patients (21 files)
└── Tessi/           ← Mixed samples
```

## 🎯 Quick Test Commands

### Copy a sample to test folder:
```bash
# Windows PowerShell
Copy-Item "original-speech-dataset/emma/IC/IC1.wav" -Destination "test_healthy.wav"

# Test in Gradio app
python app.py
# Then upload test_healthy.wav
```

### Batch test multiple samples:
```bash
# Copy first 5 healthy samples
1..5 | ForEach-Object { 
    Copy-Item "original-speech-dataset/emma/IC/IC$_.wav" -Destination "test_samples/"
}
```

---

**Summary**: The `emma/IC/` folder contains 123 healthy speech samples perfect for testing. Expected severity scores should be **< 0.3** for all these samples.
