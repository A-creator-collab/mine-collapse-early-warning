"""Live sensor stream simulator (no hardware needed)."""
import time
import numpy as np
from typing import Generator, Dict
import yaml


class StreamSimulator:
    def __init__(self, config_path: str = "config.yaml", scenario: str = "mixed"):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        np.random.seed()
        self.scenario = scenario
        self.t = 0
    
    def _baseline(self) -> Dict:
        self.t += 1
        n = self.t
        accel_x = 0.005 + np.random.normal(0, 0.015)
        accel_y = 0.005 + np.random.normal(0, 0.015)
        accel_z = 1.0 + np.random.normal(0, 0.02)
        gyro_x = np.random.normal(0, 0.5)
        gyro_y = np.random.normal(0, 0.5)
        gyro_z = np.random.normal(0, 0.5)
        total = (accel_x**2 + accel_y**2 + accel_z**2) ** 0.5
        vib = abs(total - 1.0)
        return {
            'timestamp': n / self.config['data']['sampling_rate_hz'],
            'accel_x': accel_x, 'accel_y': accel_y, 'accel_z': accel_z,
            'gyro_x': gyro_x, 'gyro_y': gyro_y, 'gyro_z': gyro_z,
            'vibration_magnitude': vib,
            'load_cell': 2.5 + np.random.normal(0, 0.02),
            'displacement': int(512 + np.random.normal(0, 3)),
            'water_level': int(np.random.poisson(2)),
        }
    
    def _warning(self) -> Dict:
        base = self._baseline()
        base['accel_x'] += np.random.normal(0, 0.15)
        base['accel_y'] += np.random.normal(0, 0.15)
        total = (base['accel_x']**2 + base['accel_y']**2 + base['accel_z']**2) ** 0.5
        base['vibration_magnitude'] = abs(total - 1.0)
        base['load_cell'] += np.random.normal(0, 0.2)
        base['displacement'] += int(np.random.normal(0, 40))
        return base
    
    def _critical(self) -> Dict:
        base = self._baseline()
        base['accel_x'] += np.random.normal(0, 0.5)
        base['accel_y'] += np.random.normal(0, 0.5)
        base['accel_z'] += np.random.normal(0, 0.3)
        total = (base['accel_x']**2 + base['accel_y']**2 + base['accel_z']**2) ** 0.5
        base['vibration_magnitude'] = abs(total - 1.0)
        base['load_cell'] += np.random.normal(-0.8, 0.4)
        base['displacement'] += int(np.random.normal(150, 50))
        base['water_level'] = int(np.random.poisson(8))
        return base
    
    def get_reading(self) -> Dict:
        if self.scenario == 'normal':
            return self._baseline()
        elif self.scenario == 'warning':
            return self._warning()
        elif self.scenario == 'critical':
            return self._critical()
        else:
            r = np.random.random()
            if r < 0.85:
                return self._baseline()
            elif r < 0.96:
                return self._warning()
            else:
                return self._critical()
    
    def stream(self, rate_hz: float = 4.0) -> Generator[Dict, None, None]:
        interval = 1.0 / rate_hz
        while True:
            yield self.get_reading()
            time.sleep(interval)