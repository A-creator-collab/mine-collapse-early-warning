"""
Adapter: converts mine_monitoring_synthetic_1000000.csv (your dataset)
into the format the pipeline expects (mine_sensor_data.csv).

Uses risk_level_simulated to derive 3-class labels:
    LOW     -> 0 (Normal)
    MEDIUM  -> 1 (Warning)
    HIGH    -> 1 (Warning)
    VERY_HIGH -> 2 (Critical)
    UNKNOWN_SENSOR_FAULT -> dropped

Usage:
    python scripts/prepare_real_data.py
"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys

SRC = Path(r"C:\Users\athar\Downloads\mine_monitoring_synthetic_1000000.csv")
DST = Path("data/raw/mine_sensor_data.csv")
CHUNK_SIZE = 200_000
TARGET_ROWS = 100_000


def main():
    if not SRC.exists():
        print(f"Source not found: {SRC}")
        sys.exit(1)

    print(f"Reading from {SRC} in chunks...")
    total_rows = 0
    sampled_chunks = []

    for i, chunk in enumerate(pd.read_csv(SRC, chunksize=CHUNK_SIZE)):
        total_rows += len(chunk)
        frac = TARGET_ROWS / 1_000_000
        n_sample = max(1, int(len(chunk) * frac))
        sampled_chunks.append(
            chunk.sample(n=min(n_sample, len(chunk)), random_state=42)
        )
        print(f"  Chunk {i+1}: read {len(chunk):,}, kept {n_sample}")

    df = pd.concat(sampled_chunks, ignore_index=True)
    print(f"\nTotal read: {total_rows:,}")
    print(f"Sampled:    {len(df):,}")

    # --- Column mapping ---
    df = df.rename(columns={
        'vibration_rms_g': 'vibration_magnitude',
        'support_load_kN': 'load_cell',
        'roof_displacement_mm': 'displacement',
        'water_level_cm': 'water_level',
    })

    # Fabricate accel/gyro from tilt and trends
    df['accel_x'] = df['tilt_x_deg']
    df['accel_y'] = df['tilt_y_deg']
    df['accel_z'] = 1.0 + df['vibration_magnitude'] * 0.1
    df['gyro_x'] = df['vibration_trend_simulated']
    df['gyro_y'] = df['displacement_trend_simulated']
    df['gyro_z'] = df['load_trend_simulated']

    # --- Label from risk_level_simulated ---
    label_map = {
        'LOW': 0,
        'MEDIUM': 1,
        'HIGH': 1,
        'VERY_HIGH': 2,
    }
    df['label'] = df['risk_level_simulated'].map(label_map)

    print("\nOriginal risk_level_simulated counts (in sample):")
    print(df['risk_level_simulated'].value_counts().to_string())

    before = len(df)
    df = df.dropna(subset=['label'])
    df['label'] = df['label'].astype(int)
    dropped = before - len(df)
    if dropped:
        print(f"\nDropped {dropped} rows (UNKNOWN_SENSOR_FAULT or unmapped)")

    # --- Keep only what models need ---
    keep = [
        'accel_x', 'accel_y', 'accel_z',
        'gyro_x', 'gyro_y', 'gyro_z',
        'vibration_magnitude', 'load_cell', 'displacement', 'water_level',
        'label'
    ]
    df = df[keep].dropna()

    # --- Save ---
    DST.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(DST, index=False)
    print(f"\nSaved {len(df):,} rows to {DST}")
    print(f"File size: {DST.stat().st_size / 1e6:.1f} MB")
    print("\nFinal label distribution:")
    counts = df['label'].value_counts().sort_index()
    for lbl, cnt in counts.items():
        name = {0: 'Normal', 1: 'Warning', 2: 'Critical'}.get(lbl, str(lbl))
        print(f"  {lbl} ({name:8}): {cnt:>7,}  ({100*cnt/len(df):5.2f}%)")


if __name__ == "__main__":
    main()