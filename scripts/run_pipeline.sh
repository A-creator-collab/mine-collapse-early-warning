#!/bin/bash
set -e
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/demo.py --quick
echo "Ready! Run: streamlit run dashboard/app.py"