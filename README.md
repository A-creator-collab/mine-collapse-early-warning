# ⛏️ AI-Powered Mine Collapse Early-Warning System

Software-first prototype that detects mine tunnel collapse risk from
multi-sensor data using machine learning.

**Trained on 1,000,000 synthetic mine monitoring samples.**

![Classifier Accuracy](https://img.shields.io/badge/accuracy-99%25-brightgreen)
![Critical Precision](https://img.shields.io/badge/critical_precision-100%25-brightgreen)

## 🎯 Results

Trained on 98,978 real-format samples (sampled from 1M-row source):

| Class | Precision | Recall | F1 |
|-------|-----------|--------|----|
| Normal | 0.99 | 1.00 | 0.99 |
| Warning | 0.98 | 0.93 | 0.96 |
| **Critical** | **1.00** | **0.86** | **0.93** |
| **Overall accuracy** | | | **0.99** |

**100% precision on Critical** — zero false alarms on the highest-urgency class.

## 🏗️ Architecture

Raw mine sensor stream (1M rows)
│
▼
Data adapter (column mapping, sampling, labeling)
│
▼
Preprocessor (rolling stats, rate-of-change, scaling)
│
├──► MLPClassifier ──► 3-class prediction
└──► IsolationForest ──► anomaly score
│
▼
Action Engine (rule-based safety recommendations)
│
▼
Streamlit dashboard (real-time visualization)


## 🚀 Quick Start

```bash
git clone https://github.com/A-creator-collab/mine-collapse-early-warning.git
cd mine-collapse-early-warning
pip install -r requirements.txt
streamlit run dashboard/app.py

🔬 Retrain

If you have your own dataset:

# Place your CSV somewhere
python scripts/prepare_real_data.py     # adapts columns
python -c "from src.preprocessor import Preprocessor; Preprocessor().prepare('data/raw/mine_sensor_data.csv')"
python -c "from src.train_classifier import ClassifierTrainer; ClassifierTrainer().train()"
python -c "from src.train_anomaly import AnomalyTrainer; AnomalyTrainer().train()"

📁 Layout

src/          Core library (data, models, prediction)
dashboard/    Streamlit UI
scripts/      One-click pipeline + real-data adapter
models/       Trained artifacts (included)
data/         Datasets (gitignored, regenerate)
tests/        pytest suite

 Safety Disclaimer
Competition prototype. Never use as the sole warning system in a real mine.
Real deployments require certified instrumentation, fail-safe design,
underground radio validation, and regulatory approval.
📜 License

MIT
