"""
Live Handwriting Tracking - Backend API
Phase 3: Flask API for feature extraction, inference, and report generation
"""

import os
import json
import pickle
import numpy as np
import torch
import torch.nn as nn
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from datetime import datetime
from pathlib import Path

from feature_extractor import FeatureExtractor

# ============================================================================
# CONFIGURATION
# ============================================================================

app = Flask(__name__)
CORS(app)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_DIR = 'models'
UPLOAD_DIR = 'uploads'

os.makedirs(UPLOAD_DIR, exist_ok=True)

# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================

class PDDetectionModel(nn.Module):
    """4-layer MLP for PD detection"""
    
    def __init__(self, input_size=8, hidden_sizes=[64, 32, 16]):
        super(PDDetectionModel, self).__init__()
        
        self.fc1 = nn.Linear(input_size, hidden_sizes[0])
        self.bn1 = nn.BatchNorm1d(hidden_sizes[0])
        self.dropout1 = nn.Dropout(0.3)
        
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.bn2 = nn.BatchNorm1d(hidden_sizes[1])
        self.dropout2 = nn.Dropout(0.3)
        
        self.fc3 = nn.Linear(hidden_sizes[1], hidden_sizes[2])
        self.dropout3 = nn.Dropout(0.2)
        
        self.fc4 = nn.Linear(hidden_sizes[2], 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.dropout1(x)
        
        x = self.fc2(x)
        x = self.bn2(x)
        x = torch.relu(x)
        x = self.dropout2(x)
        
        x = self.fc3(x)
        x = torch.relu(x)
        x = self.dropout3(x)
        
        x = self.fc4(x)
        x = self.sigmoid(x)
        
        return x


# ============================================================================
# GLOBAL STATE
# ============================================================================

extractor = FeatureExtractor(sampling_rate=100)
models = []
scaler = None

def load_models():
    """Load trained models and scaler"""
    global models, scaler
    
    print("[*] Loading models...")
    
    # Load scaler
    scaler_path = os.path.join(MODEL_DIR, 'scaler.pkl')
    if os.path.exists(scaler_path):
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        print(f"[✓] Loaded scaler")
    
    # Load models
    for fold in range(1, 6):
        model_path = os.path.join(MODEL_DIR, f'fold_{fold}_model.pth')
        if os.path.exists(model_path):
            model = PDDetectionModel().to(DEVICE)
            model.load_state_dict(torch.load(model_path, map_location=DEVICE))
            model.eval()
            models.append(model)
            print(f"[✓] Loaded fold {fold} model")
    
    if len(models) == 0:
        print("[!] No models found. Please train models first.")
    else:
        print(f"[✓] Loaded {len(models)} models")

# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.route('/', methods=['GET'])
def index():
    """Serve frontend"""
    return render_template('index.html')


@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'ok',
        'device': str(DEVICE),
        'models_loaded': len(models),
        'timestamp': datetime.now().isoformat()
    })


@app.route('/api/extract_features', methods=['POST'])
def extract_features():
    """
    Extract biomarkers from drawing points
    
    Request:
    {
        "points": [
            {"x": 100, "y": 150, "t": 0},
            {"x": 102, "y": 152, "t": 10},
            ...
        ]
    }
    
    Response:
    {
        "tremor_frequency": 5.2,
        "tremor_amplitude": 0.68,
        "jerkiness": 0.72,
        "consistency": 0.75,
        "fatigue": 0.18,
        "pause_frequency": 0.08,
        "avg_velocity": 2.1,
        "velocity_variance": 0.45,
        "normalized": {...}
    }
    """
    try:
        data = request.get_json()
        points = data.get('points', [])
        
        if not points or len(points) < 3:
            return jsonify({'error': 'Insufficient points'}), 400
        
        # Extract features
        features = extractor.extract_all_features(points)
        normalized = extractor.normalize_features(features)
        
        return jsonify({
            'features': features,
            'normalized': normalized,
            'num_points': len(points),
            'timestamp': datetime.now().isoformat()
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/predict', methods=['POST'])
def predict():
    """
    Predict PD risk from biomarkers
    
    Request:
    {
        "biomarkers": [5.2, 0.68, 0.72, 0.75, 0.18, 0.08, 2.1, 0.45]
    }
    
    Response:
    {
        "pd_risk": 0.71,
        "confidence": 0.89,
        "status": "HIGH RISK",
        "ensemble_predictions": [0.70, 0.72, 0.71, 0.70, 0.71]
    }
    """
    try:
        if len(models) == 0:
            return jsonify({'error': 'No models loaded'}), 500
        
        data = request.get_json()
        biomarkers = np.array(data.get('biomarkers', []))
        
        if len(biomarkers) != 8:
            return jsonify({'error': 'Expected 8 biomarkers'}), 400
        
        # Normalize biomarkers
        if scaler is not None:
            biomarkers_normalized = scaler.transform([biomarkers])[0]
        else:
            biomarkers_normalized = biomarkers
        
        # Ensemble prediction
        predictions = []
        with torch.no_grad():
            for model in models:
                input_tensor = torch.FloatTensor(biomarkers_normalized).unsqueeze(0).to(DEVICE)
                output = model(input_tensor).cpu().numpy()[0][0]
                predictions.append(float(output))
        
        # Average prediction
        pd_risk = np.mean(predictions)
        confidence = 1.0 - np.std(predictions)  # Higher agreement = higher confidence
        
        # Determine status
        if pd_risk < 0.33:
            status = "LOW RISK"
            color = "green"
        elif pd_risk < 0.66:
            status = "MODERATE RISK"
            color = "orange"
        else:
            status = "HIGH RISK"
            color = "red"
        
        return jsonify({
            'pd_risk': float(pd_risk),
            'confidence': float(confidence),
            'status': status,
            'color': color,
            'ensemble_predictions': predictions,
            'timestamp': datetime.now().isoformat()
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/analyze', methods=['POST'])
def analyze():
    """
    Complete analysis: extract features + predict
    
    Request:
    {
        "points": [...]
    }
    
    Response:
    {
        "features": {...},
        "normalized": {...},
        "pd_risk": 0.71,
        "confidence": 0.89,
        "status": "HIGH RISK"
    }
    """
    try:
        data = request.get_json()
        points = data.get('points', [])
        
        if not points or len(points) < 3:
            return jsonify({'error': 'Insufficient points'}), 400
        
        # Extract features
        features = extractor.extract_all_features(points)
        normalized = extractor.normalize_features(features)
        
        # Prepare biomarkers for prediction
        biomarkers = np.array([
            normalized['tremor_frequency'],
            normalized['tremor_amplitude'],
            normalized['jerkiness'],
            normalized['consistency'],
            normalized['fatigue'],
            normalized['pause_frequency'],
            normalized['avg_velocity'],
            normalized['velocity_variance'],
        ])
        
        # Predict
        if len(models) > 0:
            predictions = []
            with torch.no_grad():
                for model in models:
                    if scaler is not None:
                        biomarkers_scaled = scaler.transform([biomarkers])[0]
                    else:
                        biomarkers_scaled = biomarkers
                    
                    input_tensor = torch.FloatTensor(biomarkers_scaled).unsqueeze(0).to(DEVICE)
                    output = model(input_tensor).cpu().numpy()[0][0]
                    predictions.append(float(output))
            
            pd_risk = np.mean(predictions)
            confidence = 1.0 - np.std(predictions)
        else:
            pd_risk = 0.0
            confidence = 0.0
            predictions = []
        
        # Determine status
        if pd_risk < 0.33:
            status = "LOW RISK"
            color = "green"
        elif pd_risk < 0.66:
            status = "MODERATE RISK"
            color = "orange"
        else:
            status = "HIGH RISK"
            color = "red"
        
        return jsonify({
            'features': features,
            'normalized': normalized,
            'pd_risk': float(pd_risk),
            'confidence': float(confidence),
            'status': status,
            'color': color,
            'ensemble_predictions': predictions,
            'num_points': len(points),
            'timestamp': datetime.now().isoformat()
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/report', methods=['POST'])
def generate_report():
    """
    Generate detailed clinical report
    
    Request:
    {
        "features": {...},
        "pd_risk": 0.71,
        "confidence": 0.89
    }
    
    Response:
    {
        "report": {...},
        "interpretation": "..."
    }
    """
    try:
        data = request.get_json()
        features = data.get('features', {})
        pd_risk = data.get('pd_risk', 0.0)
        confidence = data.get('confidence', 0.0)
        
        # Generate interpretation
        interpretation = generate_interpretation(features, pd_risk)
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'pd_risk_score': float(pd_risk),
            'confidence': float(confidence),
            'status': 'HIGH RISK' if pd_risk > 0.66 else ('MODERATE RISK' if pd_risk > 0.33 else 'LOW RISK'),
            'biomarkers': {
                'tremor_frequency_hz': float(features.get('tremor_frequency', 0)),
                'tremor_amplitude': float(features.get('tremor_amplitude', 0)),
                'jerkiness': float(features.get('jerkiness', 0)),
                'consistency': float(features.get('consistency', 1)),
                'fatigue': float(features.get('fatigue', 0)),
                'pause_frequency': float(features.get('pause_frequency', 0)),
                'avg_velocity': float(features.get('avg_velocity', 0)),
                'velocity_variance': float(features.get('velocity_variance', 0)),
            },
            'interpretation': interpretation,
            'recommendations': generate_recommendations(pd_risk),
            'disclaimer': 'This is a screening tool and not a diagnostic tool. Please consult a neurologist for proper diagnosis.'
        }
        
        return jsonify(report)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def generate_interpretation(features, pd_risk):
    """Generate clinical interpretation"""
    
    tremor_freq = features.get('tremor_frequency', 0)
    jerkiness = features.get('jerkiness', 0)
    fatigue = features.get('fatigue', 0)
    consistency = features.get('consistency', 1)
    
    findings = []
    
    if 4 <= tremor_freq <= 8:
        findings.append(f"Tremor detected at {tremor_freq:.1f} Hz (PD range: 4-8 Hz)")
    
    if jerkiness > 0.6:
        findings.append("High jerkiness detected (indicates tremor/dyskinesia)")
    
    if fatigue > 0.5:
        findings.append("Fatigue pattern detected (velocity decline during drawing)")
    
    if consistency < 0.7:
        findings.append("Low stroke consistency (variable drawing patterns)")
    
    if not findings:
        findings.append("No significant PD indicators detected")
    
    interpretation = "Clinical Findings:\n" + "\n".join(f"• {f}" for f in findings)
    
    if pd_risk > 0.66:
        interpretation += "\n\nRisk Assessment: HIGH - Recommend neurological evaluation"
    elif pd_risk > 0.33:
        interpretation += "\n\nRisk Assessment: MODERATE - Consider follow-up assessment"
    else:
        interpretation += "\n\nRisk Assessment: LOW - Continue routine monitoring"
    
    return interpretation


def generate_recommendations(pd_risk):
    """Generate clinical recommendations"""
    
    if pd_risk > 0.66:
        return [
            "Refer to neurologist for comprehensive evaluation",
            "Consider additional diagnostic tests (MRI, DaTscan)",
            "Repeat screening in 3-6 months",
            "Discuss medication options if PD confirmed"
        ]
    elif pd_risk > 0.33:
        return [
            "Schedule follow-up assessment in 3 months",
            "Monitor for additional PD symptoms",
            "Maintain healthy lifestyle (exercise, sleep)",
            "Consider repeat screening"
        ]
    else:
        return [
            "Continue routine health monitoring",
            "Maintain active lifestyle",
            "Annual screening recommended",
            "Report any new symptoms to healthcare provider"
        ]


# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Not found'}), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    print("=" * 80)
    print("LIVE HANDWRITING TRACKING - BACKEND API")
    print("=" * 80)
    print(f"Device: {DEVICE}")
    
    # Load models
    load_models()
    
    print("\n[*] Starting Flask server...")
    print("[*] Access at http://localhost:5000")
    print("=" * 80)
    
    app.run(debug=True, host='0.0.0.0', port=5000)
