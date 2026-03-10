"""
Feature Extraction Pipeline for Live Handwriting Tracking
Extracts 8 biomarkers from drawing coordinates for PD detection
"""

import numpy as np
from scipy import signal
from scipy.fft import fft, fftfreq
import warnings
warnings.filterwarnings('ignore')


class FeatureExtractor:
    """Extract temporal biomarkers from handwriting coordinates"""
    
    def __init__(self, sampling_rate=100):
        """
        Args:
            sampling_rate: Points per second (Hz)
        """
        self.sampling_rate = sampling_rate
        self.dt = 1.0 / sampling_rate
    
    def extract_all_features(self, points):
        """
        Extract all 8 biomarkers from drawing points
        
        Args:
            points: List of dicts with keys 'x', 'y', 't' (timestamp in ms)
        
        Returns:
            dict with 8 biomarkers
        """
        if len(points) < 3:
            return self._get_zero_features()
        
        # Calculate velocity and acceleration
        velocity = self.calculate_velocity(points)
        acceleration = self.calculate_acceleration(velocity)
        
        if len(velocity) < 2:
            return self._get_zero_features()
        
        features = {
            'tremor_frequency': self.detect_tremor_frequency(velocity),
            'tremor_amplitude': self.detect_tremor_amplitude(velocity),
            'jerkiness': self.calculate_jerkiness(acceleration),
            'consistency': self.track_stroke_consistency(points),
            'fatigue': self.detect_fatigue(velocity),
            'pause_frequency': self.count_pauses(points),
            'avg_velocity': np.mean(velocity) if len(velocity) > 0 else 0.0,
            'velocity_variance': np.var(velocity) if len(velocity) > 1 else 0.0,
        }
        
        return features
    
    def calculate_velocity(self, points):
        """
        Calculate velocity from coordinates
        
        Args:
            points: List of dicts with 'x', 'y', 't'
        
        Returns:
            List of velocities in pixels/ms
        """
        velocities = []
        
        for i in range(1, len(points)):
            dx = points[i]['x'] - points[i-1]['x']
            dy = points[i]['y'] - points[i-1]['y']
            dt = points[i]['t'] - points[i-1]['t']
            
            if dt > 0:
                distance = np.sqrt(dx**2 + dy**2)
                velocity = distance / dt
                velocities.append(velocity)
        
        return np.array(velocities)
    
    def calculate_acceleration(self, velocities):
        """
        Calculate acceleration from velocity
        
        Args:
            velocities: Array of velocities
        
        Returns:
            Array of accelerations
        """
        if len(velocities) < 2:
            return np.array([])
        
        accelerations = np.diff(velocities)
        return np.abs(accelerations)
    
    def detect_tremor_frequency(self, velocity_signal):
        """
        Detect tremor frequency using FFT
        PD tremor: 4-8 Hz
        
        Args:
            velocity_signal: Array of velocities
        
        Returns:
            Tremor frequency in Hz (0 if not detected)
        """
        if len(velocity_signal) < 10:
            return 0.0
        
        try:
            # Apply Hann window to reduce spectral leakage
            windowed = velocity_signal * signal.hann(len(velocity_signal))
            
            # Compute FFT
            fft_result = fft(windowed)
            frequencies = fftfreq(len(windowed), self.dt / 1000.0)  # Convert ms to s
            
            # Look for tremor in 4-8 Hz range
            pd_range_mask = (frequencies >= 4) & (frequencies <= 8)
            
            if not np.any(pd_range_mask):
                return 0.0
            
            # Find peak frequency in PD range
            pd_magnitudes = np.abs(fft_result[pd_range_mask])
            pd_frequencies = frequencies[pd_range_mask]
            
            if len(pd_magnitudes) == 0:
                return 0.0
            
            peak_idx = np.argmax(pd_magnitudes)
            tremor_freq = float(pd_frequencies[peak_idx])
            
            return max(0.0, tremor_freq)
        
        except Exception as e:
            print(f"Error in tremor detection: {e}")
            return 0.0
    
    def detect_tremor_amplitude(self, velocity_signal):
        """
        Detect tremor amplitude (magnitude of tremor component)
        
        Args:
            velocity_signal: Array of velocities
        
        Returns:
            Tremor amplitude (normalized 0-1)
        """
        if len(velocity_signal) < 10:
            return 0.0
        
        try:
            # Apply Hann window
            windowed = velocity_signal * signal.hann(len(velocity_signal))
            
            # Compute FFT
            fft_result = fft(windowed)
            frequencies = fftfreq(len(windowed), self.dt / 1000.0)
            
            # Look for tremor in 4-8 Hz range
            pd_range_mask = (frequencies >= 4) & (frequencies <= 8)
            
            if not np.any(pd_range_mask):
                return 0.0
            
            pd_magnitudes = np.abs(fft_result[pd_range_mask])
            
            if len(pd_magnitudes) == 0:
                return 0.0
            
            # Normalize by total power
            total_power = np.sum(np.abs(fft_result))
            tremor_power = np.max(pd_magnitudes)
            
            if total_power == 0:
                return 0.0
            
            amplitude = tremor_power / total_power
            return float(np.clip(amplitude, 0.0, 1.0))
        
        except Exception as e:
            print(f"Error in tremor amplitude: {e}")
            return 0.0
    
    def calculate_jerkiness(self, acceleration_signal):
        """
        Jerkiness = variance of acceleration
        High jerkiness indicates tremor/PD
        
        Args:
            acceleration_signal: Array of accelerations
        
        Returns:
            Jerkiness score (normalized 0-1)
        """
        if len(acceleration_signal) < 2:
            return 0.0
        
        jerkiness = np.var(acceleration_signal)
        # Normalize to 0-1 range (empirically determined)
        jerkiness_normalized = float(np.clip(jerkiness / 10.0, 0.0, 1.0))
        
        return jerkiness_normalized
    
    def track_stroke_consistency(self, points):
        """
        Track consistency across strokes
        Detects if strokes are similar in size/shape
        
        Args:
            points: List of dicts with 'x', 'y', 't'
        
        Returns:
            Consistency score (0-1, higher is more consistent)
        """
        if len(points) < 10:
            return 1.0
        
        try:
            # Detect strokes (gaps in drawing)
            strokes = self._detect_strokes(points)
            
            if len(strokes) < 2:
                return 1.0
            
            # Calculate size of each stroke
            sizes = []
            for stroke in strokes:
                if len(stroke) > 1:
                    xs = [p['x'] for p in stroke]
                    ys = [p['y'] for p in stroke]
                    size = (max(xs) - min(xs)) * (max(ys) - min(ys))
                    sizes.append(size)
            
            if len(sizes) < 2:
                return 1.0
            
            # Consistency = 1 / (1 + variance)
            size_variance = np.var(sizes)
            consistency = 1.0 / (1.0 + size_variance / (np.mean(sizes) + 1e-6))
            
            return float(np.clip(consistency, 0.0, 1.0))
        
        except Exception as e:
            print(f"Error in consistency: {e}")
            return 1.0
    
    def detect_fatigue(self, velocity_signal):
        """
        Detect fatigue (velocity decline over time)
        
        Args:
            velocity_signal: Array of velocities
        
        Returns:
            Fatigue score (0-1, higher indicates more fatigue)
        """
        if len(velocity_signal) < 5:
            return 0.0
        
        try:
            # Fit line to velocity over time
            x = np.arange(len(velocity_signal))
            coeffs = np.polyfit(x, velocity_signal, 1)
            slope = coeffs[0]
            
            # Negative slope = fatigue
            mean_velocity = np.mean(velocity_signal)
            if mean_velocity == 0:
                return 0.0
            
            fatigue_score = max(0.0, -slope / mean_velocity)
            return float(np.clip(fatigue_score, 0.0, 1.0))
        
        except Exception as e:
            print(f"Error in fatigue detection: {e}")
            return 0.0
    
    def count_pauses(self, points, pause_threshold_ms=200):
        """
        Count pauses in drawing (gaps > threshold)
        
        Args:
            points: List of dicts with 'x', 'y', 't'
            pause_threshold_ms: Threshold for pause detection
        
        Returns:
            Pause frequency (pauses per second)
        """
        if len(points) < 2:
            return 0.0
        
        pauses = 0
        total_time = points[-1]['t'] - points[0]['t']
        
        if total_time == 0:
            return 0.0
        
        for i in range(1, len(points)):
            time_gap = points[i]['t'] - points[i-1]['t']
            if time_gap > pause_threshold_ms:
                pauses += 1
        
        pause_frequency = pauses / (total_time / 1000.0)  # Convert to per second
        return float(np.clip(pause_frequency, 0.0, 10.0))
    
    def _detect_strokes(self, points, pause_threshold_ms=200):
        """
        Detect individual strokes (separated by pauses)
        
        Args:
            points: List of dicts with 'x', 'y', 't'
            pause_threshold_ms: Threshold for stroke separation
        
        Returns:
            List of strokes (each stroke is a list of points)
        """
        strokes = []
        current_stroke = [points[0]]
        
        for i in range(1, len(points)):
            time_gap = points[i]['t'] - points[i-1]['t']
            
            if time_gap > pause_threshold_ms:
                if len(current_stroke) > 1:
                    strokes.append(current_stroke)
                current_stroke = [points[i]]
            else:
                current_stroke.append(points[i])
        
        if len(current_stroke) > 1:
            strokes.append(current_stroke)
        
        return strokes
    
    def _get_zero_features(self):
        """Return zero features when insufficient data"""
        return {
            'tremor_frequency': 0.0,
            'tremor_amplitude': 0.0,
            'jerkiness': 0.0,
            'consistency': 1.0,
            'fatigue': 0.0,
            'pause_frequency': 0.0,
            'avg_velocity': 0.0,
            'velocity_variance': 0.0,
        }
    
    def normalize_features(self, features):
        """
        Normalize features to 0-1 range
        
        Args:
            features: Dict of biomarkers
        
        Returns:
            Normalized features dict
        """
        normalized = {}
        
        # Define normalization ranges (empirically determined)
        ranges = {
            'tremor_frequency': (0, 10),      # Hz
            'tremor_amplitude': (0, 1),       # Already normalized
            'jerkiness': (0, 1),              # Already normalized
            'consistency': (0, 1),            # Already normalized
            'fatigue': (0, 1),                # Already normalized
            'pause_frequency': (0, 10),       # Per second
            'avg_velocity': (0, 10),          # pixels/ms
            'velocity_variance': (0, 10),     # pixels²/ms²
        }
        
        for key, value in features.items():
            if key in ranges:
                min_val, max_val = ranges[key]
                normalized[key] = np.clip((value - min_val) / (max_val - min_val + 1e-6), 0.0, 1.0)
            else:
                normalized[key] = float(np.clip(value, 0.0, 1.0))
        
        return normalized


def extract_features_from_image(image_array, num_points=100):
    """
    Extract features from a static handwriting image
    by simulating temporal drawing sequence
    
    Args:
        image_array: Binary image (0 or 1)
        num_points: Number of points to extract
    
    Returns:
        List of points with x, y, t
    """
    # Find all non-zero pixels
    coords = np.argwhere(image_array > 0)
    
    if len(coords) == 0:
        return []
    
    # Sample points along the drawing
    if len(coords) > num_points:
        indices = np.linspace(0, len(coords) - 1, num_points, dtype=int)
        sampled_coords = coords[indices]
    else:
        sampled_coords = coords
    
    # Create temporal sequence
    points = []
    for i, (y, x) in enumerate(sampled_coords):
        points.append({
            'x': float(x),
            'y': float(y),
            't': float(i * 10)  # 10ms between points
        })
    
    return points
