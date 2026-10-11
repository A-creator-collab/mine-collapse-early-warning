"""One-shot pipeline."""
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_generator import MineDataGenerator
from src.preprocessor import Preprocessor
from src.train_classifier import ClassifierTrainer
from src.train_anomaly import AnomalyTrainer
from src.stream_simulator import StreamSimulator


def quick_pipeline():
    print("=== QUICK PIPELINE (small dataset) ===")
    gen = MineDataGenerator()
    gen.n_samples = 5000
    gen.save("data/raw/mine_sensor_data.csv")
    Preprocessor().prepare("data/raw/mine_sensor_data.csv")
    ClassifierTrainer().train()
    AnomalyTrainer().train()
    print("\n✅ Done. Run: streamlit run dashboard/app.py")


def full_pipeline():
    print("=== FULL PIPELINE ===")
    MineDataGenerator().save("data/raw/mine_sensor_data.csv")
    Preprocessor().prepare("data/raw/mine_sensor_data.csv")
    ClassifierTrainer().train()
    AnomalyTrainer().train()
    print("\n✅ Done. Run: streamlit run dashboard/app.py")


def stream_demo():
    print("=== STREAM DEMO ===")
    sim = StreamSimulator(scenario='mixed')
    for i, r in enumerate(sim.stream(rate_hz=5)):
        if i >= 20:
            break
        print(f"vib={r['vibration_magnitude']:.3f}  load={r['load_cell']:.3f}  disp={r['displacement']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--quick', action='store_true')
    parser.add_argument('--stream', action='store_true')
    args = parser.parse_args()
    if args.quick:
        quick_pipeline()
    elif args.train:
        full_pipeline()
    elif args.stream:
        stream_demo()
    else:
        parser.print_help()