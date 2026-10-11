import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.action_engine import ActionEngine


def test_normal_recommendation():
    engine = ActionEngine()
    rec = engine.recommend(0, 0.95, {})
    assert rec.level == "NORMAL"
    assert not rec.evacuation_required


def test_critical_recommendation():
    engine = ActionEngine()
    rec = engine.recommend(2, 0.99, {})
    assert rec.level == "CRITICAL"
    assert rec.evacuation_required