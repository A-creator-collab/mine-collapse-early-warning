"""Feature engineering and preprocessing."""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
from pathlib import Path
import yaml


class Preprocessor:
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        self.raw_features = self.config['features']['raw']
        self.engineered_features = self.config['features']['engineered']
        self.all_features = self.raw_features + self.engineered_features
        self.window = self.config['data']['rolling_window']
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        w = self.window
        for col in ['vibration_magnitude', 'load_cell', 'displacement']:
            prefix = {'vibration_magnitude': 'vib',
                     'load_cell': 'load',
                     'displacement': 'disp'}[col]
            df[f'{prefix}_rolling_mean'] = df[col].rolling(w, min_periods=1).mean()
            df[f'{prefix}_rolling_std'] = df[col].rolling(w, min_periods=1).std().fillna(0)
        df['vib_change_rate'] = df['vibration_magnitude'].diff().fillna(0)
        df['load_change_rate'] = df['load_cell'].diff().fillna(0)
        df['disp_change_rate'] = df['displacement'].diff().fillna(0)
        return df
    
    def prepare(self, input_path: str = "data/raw/mine_sensor_data.csv"):
        print("Loading data...")
        df = pd.read_csv(input_path)
        print("Engineering features...")
        df = self.engineer_features(df)
        X = df[self.all_features].values.astype(np.float32)
        y = df['label'].values.astype(np.int64)
        train_split = self.config['data']['train_split']
        val_split = self.config['data']['val_split']
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=(1 - train_split), random_state=42, stratify=y)
        val_ratio = val_split / (1 - train_split)
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=(1 - val_ratio), random_state=42, stratify=y_temp)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)
        out_dir = Path("data/processed")
        out_dir.mkdir(parents=True, exist_ok=True)
        np.save(out_dir / "X_train.npy", X_train)
        np.save(out_dir / "X_val.npy", X_val)
        np.save(out_dir / "X_test.npy", X_test)
        np.save(out_dir / "y_train.npy", y_train)
        np.save(out_dir / "y_val.npy", y_val)
        np.save(out_dir / "y_test.npy", y_test)
        Path("models").mkdir(exist_ok=True)
        joblib.dump(scaler, self.config['paths']['scaler'])
        joblib.dump(self.all_features, self.config['paths']['feature_cols'])
        print(f"Train: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}")
        print(f"Features: {len(self.all_features)}")
        return X_train, X_val, X_test, y_train, y_val, y_test


if __name__ == "__main__":
    Preprocessor().prepare()