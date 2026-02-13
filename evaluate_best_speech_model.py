"""
Evaluate and Compare All K-Fold Models to Find the Best One
Run this in Google Colab after training all folds
"""

# ============================================
# CONFIGURATION
# ============================================

# Path to your saved models
MODEL_SAVE_DIR = "/content/drive/MyDrive/speech_models"  # Adjust this

# Path to test/validation dataset
AUDIO_DATASET_PATH = "/content/drive/MyDrive/denoised-speech-dataset"

# Total number of folds you trained
TOTAL_FOLDS = 5

# Device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ============================================
# IMPORTS
# ============================================

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchaudio
import torchaudio.transforms as T

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)

print(f"✅ Using device: {DEVICE}")

# ============================================
# DATASET CLASS (Same as training)
# ============================================

class ParkinsonsAudioDataset(Dataset):
    """Dataset for Parkinson's speech audio files"""
    
    def __init__(self, file_paths, labels, target_length=48000, sr=16000):
        self.file_paths = file_paths
        self.labels = labels
        self.target_length = target_length
        self.sr = sr
        
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        audio_path = self.file_paths[idx]
        label = self.labels[idx]
        
        try:
            waveform, sr = torchaudio.load(audio_path)
            
            if sr != self.sr:
                resampler = T.Resample(sr, self.sr)
                waveform = resampler(waveform)
            
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            if waveform.shape[1] < self.target_length:
                padding = self.target_length - waveform.shape[1]
                waveform = F.pad(waveform, (0, padding))
            else:
                waveform = waveform[:, :self.target_length]
            
            mel_spec = T.MelSpectrogram(
                sample_rate=self.sr,
                n_fft=1024,
                hop_length=512,
                n_mels=128
            )(waveform)
            
            mel_spec_db = T.AmplitudeToDB()(mel_spec)
            mel_spec_db = (mel_spec_db - mel_spec_db.mean()) / (mel_spec_db.std() + 1e-6)
            
            return mel_spec_db, label
            
        except Exception as e:
            print(f"Error loading {audio_path}: {e}")
            return torch.zeros(1, 128, 94), label

# ============================================
# MODEL ARCHITECTURE (Same as training)
# ============================================

class SpeechCNN(nn.Module):
    """CNN for Mel-Spectrogram analysis"""
    
    def __init__(self, num_classes=2):
        super(SpeechCNN, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# ============================================
# LOAD MODEL FUNCTION
# ============================================

def load_model(model_path, device):
    """Load a trained model from checkpoint"""
    model = SpeechCNN(num_classes=2).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, checkpoint

# ============================================
# EVALUATE MODEL FUNCTION
# ============================================

def evaluate_model(model, dataloader, device):
    """Evaluate model and return detailed metrics"""
    model.eval()
    
    all_preds = []
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for spectrograms, labels in tqdm(dataloader, desc="Evaluating"):
            spectrograms, labels = spectrograms.to(device), labels.to(device)
            
            outputs = model(spectrograms)
            probs = F.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())  # Probability of class 1 (Parkinson's)
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='binary', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='binary', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='binary', zero_division=0)
    
    # ROC AUC if we have both classes
    try:
        roc_auc = roc_auc_score(all_labels, all_probs)
    except:
        roc_auc = 0.0
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'confusion_matrix': cm,
        'predictions': all_preds,
        'probabilities': all_probs,
        'labels': all_labels
    }

# ============================================
# LOAD DATA FUNCTION
# ============================================

def load_audio_files(audio_dir):
    """Load all audio files and labels from the dataset"""
    
    file_paths = []
    labels = []
    
    patient_labels = {
        'DL': 1, 'LW': 1, 'ES': 1, 'SI': 1,
        'IC': 0, 'WP': 0, 'BG': 0, 'JC': 0, 'MJ': 0, 'SK': 0, 'TP': 0, 'TS': 0,
    }
    
    main_folders = ['DL', 'LW', 'Tessi', 'emma', 'Faces']
    
    for main_folder in main_folders:
        main_path = os.path.join(audio_dir, main_folder)
        
        if not os.path.exists(main_path):
            continue
        
        items = os.listdir(main_path)
        
        for item in items:
            item_path = os.path.join(main_path, item)
            
            if os.path.isdir(item_path):
                for audio_file in os.listdir(item_path):
                    if audio_file.endswith('.wav'):
                        patient_id = audio_file.split('.')[0][:2]
                        if patient_id in patient_labels:
                            file_paths.append(os.path.join(item_path, audio_file))
                            labels.append(patient_labels[patient_id])
            else:
                if item.endswith('.wav'):
                    patient_id = item.split('.')[0][:2]
                    if patient_id in patient_labels:
                        file_paths.append(item_path)
                        labels.append(patient_labels[patient_id])
    
    return np.array(file_paths), np.array(labels)

# ============================================
# COMPARE ALL MODELS
# ============================================

def compare_all_models():
    """Compare all fold models and identify the best one"""
    
    print("="*60)
    print("🔍 COMPARING ALL FOLD MODELS")
    print("="*60)
    
    # Find all model files
    model_files = []
    for fold in range(1, TOTAL_FOLDS + 1):
        model_path = os.path.join(MODEL_SAVE_DIR, f'speech_model_fold{fold}_best.pth')
        if os.path.exists(model_path):
            model_files.append((fold, model_path))
        else:
            print(f"⚠️  Warning: Fold {fold} model not found at {model_path}")
    
    if len(model_files) == 0:
        print("❌ No model files found! Check MODEL_SAVE_DIR path.")
        return
    
    print(f"\n✅ Found {len(model_files)} model files\n")
    
    # Load full dataset for evaluation
    print("📂 Loading audio dataset...")
    file_paths, labels = load_audio_files(AUDIO_DATASET_PATH)
    
    dataset = ParkinsonsAudioDataset(file_paths, labels)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=2)
    
    # Evaluate each model
    results = []
    
    for fold, model_path in model_files:
        print(f"\n{'='*60}")
        print(f"📊 Evaluating Fold {fold} Model")
        print(f"{'='*60}")
        
        # Load model and checkpoint info
        model, checkpoint = load_model(model_path, DEVICE)
        
        print(f"   Model trained for {checkpoint.get('epoch', 'N/A')} epochs")
        print(f"   Saved val accuracy: {checkpoint.get('val_accuracy', 0)*100:.2f}%")
        print(f"   Saved val F1: {checkpoint.get('val_f1', 0):.4f}")
        
        # Evaluate on full dataset
        metrics = evaluate_model(model, dataloader, DEVICE)
        
        print(f"\n   📈 Full Dataset Performance:")
        print(f"      Accuracy:  {metrics['accuracy']*100:.2f}%")
        print(f"      Precision: {metrics['precision']:.4f}")
        print(f"      Recall:    {metrics['recall']:.4f}")
        print(f"      F1-Score:  {metrics['f1']:.4f}")
        print(f"      ROC-AUC:   {metrics['roc_auc']:.4f}")
        
        # Store results
        results.append({
            'fold': fold,
            'model_path': model_path,
            'saved_val_acc': checkpoint.get('val_accuracy', 0),
            'saved_val_f1': checkpoint.get('val_f1', 0),
            'test_accuracy': metrics['accuracy'],
            'test_precision': metrics['precision'],
            'test_recall': metrics['recall'],
            'test_f1': metrics['f1'],
            'test_roc_auc': metrics['roc_auc'],
            'confusion_matrix': metrics['confusion_matrix']
        })
    
    # Create results DataFrame
    df_results = pd.DataFrame(results)
    
    # Display comparison table
    print("\n" + "="*80)
    print("📊 MODEL COMPARISON TABLE")
    print("="*80 + "\n")
    
    display_df = df_results[['fold', 'test_accuracy', 'test_precision', 'test_recall', 'test_f1', 'test_roc_auc']].copy()
    display_df['test_accuracy'] = display_df['test_accuracy'] * 100
    display_df.columns = ['Fold', 'Accuracy (%)', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
    
    print(display_df.to_string(index=False))
    
    # Identify best models
    print("\n" + "="*80)
    print("🏆 BEST MODELS")
    print("="*80)
    
    best_acc_idx = df_results['test_accuracy'].idxmax()
    best_f1_idx = df_results['test_f1'].idxmax()
    best_auc_idx = df_results['test_roc_auc'].idxmax()
    
    print(f"\n🥇 Best Accuracy:  Fold {df_results.loc[best_acc_idx, 'fold']} - {df_results.loc[best_acc_idx, 'test_accuracy']*100:.2f}%")
    print(f"🥇 Best F1-Score:  Fold {df_results.loc[best_f1_idx, 'fold']} - {df_results.loc[best_f1_idx, 'test_f1']:.4f}")
    print(f"🥇 Best ROC-AUC:   Fold {df_results.loc[best_auc_idx, 'fold']} - {df_results.loc[best_auc_idx, 'test_roc_auc']:.4f}")
    
    # Overall recommendation
    print("\n" + "="*80)
    print("💡 RECOMMENDATION")
    print("="*80)
    
    # Calculate composite score (weighted average)
    df_results['composite_score'] = (
        0.4 * df_results['test_accuracy'] + 
        0.3 * df_results['test_f1'] + 
        0.3 * df_results['test_roc_auc']
    )
    
    best_overall_idx = df_results['composite_score'].idxmax()
    best_fold = df_results.loc[best_overall_idx, 'fold']
    
    print(f"\n🎯 RECOMMENDED MODEL: Fold {best_fold}")
    print(f"   Path: {df_results.loc[best_overall_idx, 'model_path']}")
    print(f"\n   Performance:")
    print(f"      Accuracy:  {df_results.loc[best_overall_idx, 'test_accuracy']*100:.2f}%")
    print(f"      Precision: {df_results.loc[best_overall_idx, 'test_precision']:.4f}")
    print(f"      Recall:    {df_results.loc[best_overall_idx, 'test_recall']:.4f}")
    print(f"      F1-Score:  {df_results.loc[best_overall_idx, 'test_f1']:.4f}")
    print(f"      ROC-AUC:   {df_results.loc[best_overall_idx, 'test_roc_auc']:.4f}")
    
    # Display confusion matrix of best model
    print(f"\n   Confusion Matrix:")
    cm = df_results.loc[best_overall_idx, 'confusion_matrix']
    print(f"      [[TN={cm[0,0]:3d}  FP={cm[0,1]:3d}]")
    print(f"       [FN={cm[1,0]:3d}  TP={cm[1,1]:3d}]]")
    
    # Save comparison results
    save_path = os.path.join(MODEL_SAVE_DIR, 'model_comparison_results.csv')
    df_results[['fold', 'test_accuracy', 'test_precision', 'test_recall', 'test_f1', 'test_roc_auc']].to_csv(save_path, index=False)
    print(f"\n💾 Comparison results saved to: {save_path}")
    
    # Visualization
    plot_comparison(df_results)
    
    return df_results, best_fold

# ============================================
# VISUALIZATION
# ============================================

def plot_comparison(df_results):
    """Create comparison plots"""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Model Performance Comparison Across Folds', fontsize=16, fontweight='bold')
    
    # Plot 1: Accuracy comparison
    axes[0, 0].bar(df_results['fold'], df_results['test_accuracy'] * 100, color='skyblue', edgecolor='navy')
    axes[0, 0].set_xlabel('Fold')
    axes[0, 0].set_ylabel('Accuracy (%)')
    axes[0, 0].set_title('Test Accuracy by Fold')
    axes[0, 0].grid(axis='y', alpha=0.3)
    
    # Plot 2: F1-Score comparison
    axes[0, 1].bar(df_results['fold'], df_results['test_f1'], color='lightgreen', edgecolor='darkgreen')
    axes[0, 1].set_xlabel('Fold')
    axes[0, 1].set_ylabel('F1-Score')
    axes[0, 1].set_title('F1-Score by Fold')
    axes[0, 1].grid(axis='y', alpha=0.3)
    
    # Plot 3: All metrics comparison
    metrics_df = df_results[['fold', 'test_precision', 'test_recall', 'test_f1']].set_index('fold')
    metrics_df.plot(kind='bar', ax=axes[1, 0], width=0.8)
    axes[1, 0].set_xlabel('Fold')
    axes[1, 0].set_ylabel('Score')
    axes[1, 0].set_title('Precision, Recall, F1-Score Comparison')
    axes[1, 0].legend(['Precision', 'Recall', 'F1-Score'])
    axes[1, 0].grid(axis='y', alpha=0.3)
    axes[1, 0].set_xticklabels(axes[1, 0].get_xticklabels(), rotation=0)
    
    # Plot 4: ROC-AUC comparison
    axes[1, 1].bar(df_results['fold'], df_results['test_roc_auc'], color='coral', edgecolor='darkred')
    axes[1, 1].set_xlabel('Fold')
    axes[1, 1].set_ylabel('ROC-AUC')
    axes[1, 1].set_title('ROC-AUC by Fold')
    axes[1, 1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(MODEL_SAVE_DIR, 'model_comparison_plots.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"📊 Comparison plots saved to: {plot_path}")
    
    plt.show()

# ============================================
# MAIN EXECUTION
# ============================================

if __name__ == "__main__":
    df_results, best_fold = compare_all_models()
    
    print("\n" + "="*80)
    print(f"✅ EVALUATION COMPLETE!")
    print(f"🏆 Use Fold {best_fold} model for deployment/inference")
    print("="*80)
