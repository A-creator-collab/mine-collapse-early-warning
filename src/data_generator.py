"""Synthetic mine sensor data generator."""
import numpy as np
import pandas as pd
from pathlib import Path
import yaml


class MineDataGenerator:
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        np.random.seed(self.config['project']['random_seed'])
        self.n_samples = self.config['data']['n_samples']
        self.sampling_rate = self.config['data']['sampling_rate_hz']
    
    def generate_normal(self, n: int) -> pd.DataFrame:
        t = np.arange(n)
        accel_x = 0.005 + 0.02 * np.sin(2 * np.pi * 0.5 * t / self.sampling_rate) + np.random.normal(0, 0.015, n)
        accel_y = 0.005 + 0.02 * np.sin(2 * np.pi * 0.7 * t / self.sampling_rate) + np.random.normal(0, 0.015, n)
        accel_z = 1.0 + np.random.normal(0, 0.02, n)
        gyro_x = np.random.normal(0, 0.5, n)
        gyro_y = np.random.normal(0, 0.5, n)
        gyro_z = np.random.normal(0, 0.5, n)
        total = np.sqrt(accel_x**2 + accel_y**2 + accel_z**2)
        vibration_magnitude = np.abs(total - 1.0)
        load_cell = 2.5 + np.random.normal(0, 0.02, n)
        displacement = 512 + np.random.normal(0, 3, n)
        water_level = np.random.poisson(2, n)
        return pd.DataFrame({
            'timestamp': t / self.sampling_rate,
            'accel_x': accel_x, 'accel_y': accel_y, 'accel_z': accel_z,
            'gyro_x': gyro_x, 'gyro_y': gyro_y, 'gyro_z': gyro_z,
            'vibration_magnitude': vibration_magnitude,
            'load_cell': load_cell, 'displacement': displacement,
            'water_level': water_level, 'label': 0
        })
    
    def generate_warning(self, n: int) -> pd.DataFrame:
        t = np.arange(n)
        envelope = 0.3 + 0.7 * (np.sin(2 * np.pi * 0.05 * t / self.sampling_rate) ** 2)
        accel_x = 0.02 + 0.15 * envelope * np.sin(2 * np.pi * 3 * t / self.sampling_rate) + np.random.normal(0, 0.05, n)
        accel_y = 0.02 + 0.15 * envelope * np.sin(2 * np.pi * 4 * t / self.sampling_rate) + np.random.normal(0, 0.05, n)
        accel_z = 1.0 + 0.1 * envelope * np.sin(2 * np.pi * 2 * t / self.sampling_rate) + np.random.normal(0, 0.05, n)
        gyro_x = np.random.normal(0, 3, n)
        gyro_y = np.random.normal(0, 3, n)
        gyro_z = np.random.normal(0, 3, n)
        total = np.sqrt(accel_x**2 + accel_y**2 + accel_z**2)
        vibration_magnitude = np.abs(total - 1.0)
        drift = 0.4 * np.sin(2 * np.pi * 0.02 * t / self.sampling_rate)
        load_cell = 2.5 + drift + np.random.normal(0, 0.08, n)
        displacement = 512 + 100 * np.sin(2 * np.pi * 0.01 * t / self.sampling_rate) + np.random.normal(0, 10, n)
        water_level = np.random.poisson(3, n)
        return pd.DataFrame({
            'timestamp': t / self.sampling_rate,
            'accel_x': accel_x, 'accel_y': accel_y, 'accel_z': accel_z,
            'gyro_x': gyro_x, 'gyro_y': gyro_y, 'gyro_z': gyro_z,
            'vibration_magnitude': vibration_magnitude,
            'load_cell': load_cell, 'displacement': displacement,
            'water_level': water_level, 'label': 1
        })
    
    def generate_critical(self, n: int) -> pd.DataFrame:
        t = np.arange(n)
        accel_x = 0.1 + 0.5 * np.sin(2 * np.pi * 8 * t / self.sampling_rate) + np.random.normal(0, 0.15, n)
        accel_y = 0.1 + 0.5 * np.sin(2 * np.pi * 10 * t / self.sampling_rate) + np.random.normal(0, 0.15, n)
        accel_z = 1.0 + 0.3 * np.sin(2 * np.pi * 6 * t / self.sampling_rate) + np.random.normal(0, 0.2, n)
        spike_idx = np.random.choice(n, size=max(1, n // 100), replace=False)
        accel_x[spike_idx] += np.random.uniform(-1.5, 1.5, len(spike_idx))
        accel_y[spike_idx] += np.random.uniform(-1.5, 1.5, len(spike_idx))
        gyro_x = np.random.normal(0, 15, n)
        gyro_y = np.random.normal(0, 15, n)
        gyro_z = np.random.normal(0, 15, n)
        total = np.sqrt(accel_x**2 + accel_y**2 + accel_z**2)
        vibration_magnitude = np.abs(total - 1.0)
        drop = -1.5 * (t / n)
        load_cell = 2.5 + drop + np.random.normal(0, 0.3, n)
        displacement = 512 + 300 * (t / n) + np.random.normal(0, 30, n)
        water_level = np.random.poisson(8, n)
        return pd.DataFrame({
            'timestamp': t / self.sampling_rate,
            'accel_x': accel_x, 'accel_y': accel_y, 'accel_z': accel_z,
            'gyro_x': gyro_x, 'gyro_y': gyro_y, 'gyro_z': gyro_z,
            'vibration_magnitude': vibration_magnitude,
            'load_cell': load_cell, 'displacement': displacement,
            'water_level': water_level, 'label': 2
        })
    
    def generate_dataset(self) -> pd.DataFrame:
        print(f"Generating {self.n_samples:,} samples...")
        segments = []
        remaining = self.n_samples
        while remaining > 0:
            seg_len = min(np.random.randint(200, 2000), remaining)
            r = np.random.random()
            if r < 0.70:
                seg = self.generate_normal(seg_len)
            elif r < 0.90:
                seg = self.generate_warning(seg_len)
            else:
                seg = self.generate_critical(seg_len)
            segments.append(seg)
            remaining -= seg_len
        df = pd.concat(segments, ignore_index=True)
        df['timestamp'] = np.arange(len(df)) / self.sampling_rate
        print(f"Generated: {len(df):,} samples")
        print(df['label'].value_counts().sort_index().to_string())
        return df
    
    def save(self, path: str = "data/raw/mine_sensor_data.csv"):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        df = self.generate_dataset()
        df.to_csv(path, index=False)
        print(f"Saved to {path}")
        return df


if __name__ == "__main__":
    MineDataGenerator().save()