"""Train autoencoder for anomaly detection."""
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers, callbacks
from sklearn.metrics import precision_score, recall_score, f1_score
import joblib
from pathlib import Path
import yaml


class AnomalyTrainer:
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        self.cfg = self.config['model']['anomaly']
    
    def build_autoencoder(self, input_dim: int) -> keras.Model:
        inp = layers.Input(shape=(input_dim,))
        x = inp
        for dim in self.cfg['encoding_dims']:
            x = layers.Dense(dim, activation='relu')(x)
        for dim in reversed(self.cfg['encoding_dims'][:-1]):
            x = layers.Dense(dim, activation='relu')(x)
        out = layers.Dense(input_dim, activation='linear')(x)
        ae = keras.Model(inp, out)
        ae.compile(optimizer=keras.optimizers.Adam(self.cfg['learning_rate']), loss='mse')
        return ae
    
    def train(self):
        X_train = np.load('data/processed/X_train.npy')
        y_train = np.load('data/processed/y_train.npy')
        X_test = np.load('data/processed/X_test.npy')
        y_test = np.load('data/processed/y_test.npy')
        X_normal = X_train[y_train == 0]
        print(f"Training autoencoder on {len(X_normal):,} normal samples")
        ae = self.build_autoencoder(X_train.shape[1])
        ae.summary()
        ae.fit(X_normal, X_normal, validation_split=0.1,
            epochs=self.cfg['epochs'], batch_size=self.cfg['batch_size'],
            callbacks=[callbacks.EarlyStopping(monitor='val_loss', patience=5,
                restore_best_weights=True)], verbose=1)
        recon = ae.predict(X_normal, verbose=0)
        mse = np.mean((X_normal - recon) ** 2, axis=1)
        threshold = np.percentile(mse, 100 * (1 - self.cfg['contamination']))
        print(f"Anomaly threshold: {threshold:.6f}")
        recon_test = ae.predict(X_test, verbose=0)
        mse_test = np.mean((X_test - recon_test) ** 2, axis=1)
        preds = (mse_test > threshold).astype(int)
        truth = (y_test != 0).astype(int)
        print(f"Precision: {precision_score(truth, preds, zero_division=0):.4f}")
        print(f"Recall:    {recall_score(truth, preds, zero_division=0):.4f}")
        print(f"F1:        {f1_score(truth, preds, zero_division=0):.4f}")
        ae.save(self.config['paths']['anomaly_model'])
        joblib.dump(threshold, self.config['paths']['anomaly_threshold'])


if __name__ == "__main__":
    Path("models").mkdir(exist_ok=True)
    AnomalyTrainer().train()