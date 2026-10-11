import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.data_generator import MineDataGenerator


def test_generates_correct_shape():
    gen = MineDataGenerator()
    gen.n_samples = 500
    df = gen.generate_dataset()
    assert len(df) == 500
    assert 'label' in df.columns
    assert df['label'].isin([0, 1, 2]).all()


def test_label_distribution():
    gen = MineDataGenerator()
    gen.n_samples = 2000
    df = gen.generate_dataset()
    counts = df['label'].value_counts(normalize=True)
    assert counts[0] > 0.4