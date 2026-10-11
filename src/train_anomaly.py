"""Train IsolationForest for anomaly detection."""
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_score, recall_score, f1_score
import joblib
from pathlib import Path
import yaml


class AnomalyTrainer:
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        self.cfg = self.config['model']['anomaly']
    
    def train(self):
        print("Loading preprocessed data...")
        X_train = np.load('data/processed/X_train.npy')
        y_train = np.load('data/processed/y_train.npy')
        X_test = np.load('data/processed/X_test.npy')
        y_test = np.load('data/processed/y_test.npy')
        
        X_normal = X_train[y_train == 0]
        print(f"Training IsolationForest on {len(X_normal):,} normal samples")
        
        model = IsolationForest(
            contamination=self.cfg['contamination'],
            random_state=42,
            n_estimators=100,
            verbose=0
        )
        model.fit(X_normal)
        
        print("\n=== Anomaly Detection ===")
        preds = model.predict(X_test)
        preds_binary = (preds == -1).astype(int)
        truth = (y_test != 0).astype(int)
        
        print(f"Precision: {precision_score(truth, preds_binary, zero_division=0):.4f}")
        print(f"Recall:    {recall_score(truth, preds_binary, zero_division=0):.4f}")
        print(f"F1:        {f1_score(truth, preds_binary, zero_division=0):.4f}")
        
        Path("models").mkdir(exist_ok=True)
        joblib.dump(model, self.config['paths']['anomaly_model'].replace('.keras', '.pkl'))
        print("Anomaly model saved.")


if __name__ == "__main__":
    Path("models").mkdir(exist_ok=True)
    AnomalyTrainer().train()