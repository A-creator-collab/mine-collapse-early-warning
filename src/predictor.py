"""Real-time predictor with rolling feature engineering."""
import numpy as np
import joblib
from collections import deque
from typing import Dict
import yaml


class Predictor:
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        
        self.window = self.config['data']['rolling_window']
        self.buffer = deque(maxlen=self.window)
        
        self.scaler = joblib.load(self.config['paths']['scaler'])
        self.feature_cols = joblib.load(self.config['paths']['feature_cols'])
        
        clf_path = self.config['paths']['classifier_model'].replace('.keras', '.pkl')
        self.classifier = joblib.load(clf_path)
        
        try:
            anomaly_path = self.config['paths']['anomaly_model'].replace('.keras', '.pkl')
            self.anomaly_model = joblib.load(anomaly_path)
        except Exception:
            self.anomaly_model = None
    
    def _engineer_features(self, reading: Dict) -> Dict:
        self.buffer.append(reading)
        buf = list(self.buffer)
        
        vib = [r['vibration_magnitude'] for r in buf]
        load = [r['load_cell'] for r in buf]
        disp = [r['displacement'] for r in buf]
        
        features = dict(reading)
        features['vib_rolling_mean'] = float(np.mean(vib))
        features['vib_rolling_std'] = float(np.std(vib)) if len(vib) > 1 else 0.0
        features['load_rolling_mean'] = float(np.mean(load))
        features['load_rolling_std'] = float(np.std(load)) if len(load) > 1 else 0.0
        features['disp_rolling_mean'] = float(np.mean(disp))
        features['disp_rolling_std'] = float(np.std(disp)) if len(disp) > 1 else 0.0
        features['vib_change_rate'] = float(vib[-1] - vib[-2]) if len(vib) > 1 else 0.0
        features['load_change_rate'] = float(load[-1] - load[-2]) if len(load) > 1 else 0.0
        features['disp_change_rate'] = float(disp[-1] - disp[-2]) if len(disp) > 1 else 0.0
        
        return features
    
    def predict(self, reading: Dict) -> Dict:
        features = self._engineer_features(reading)
        x = np.array([features[c] for c in self.feature_cols], dtype=np.float32)
        x_scaled = self.scaler.transform(x.reshape(1, -1))
        
        probs = self.classifier.predict_proba(x_scaled)[0]
        level = int(np.argmax(probs))
        confidence = float(probs[level])
        
        anomaly_score = None
        if self.anomaly_model is not None:
            anomaly_score = float(self.anomaly_model.score_samples(x_scaled)[0])
        
        return {
            'level': level,
            'confidence': confidence,
            'probabilities': probs.tolist(),
            'anomaly_score': anomaly_score,
            'features': features
        }