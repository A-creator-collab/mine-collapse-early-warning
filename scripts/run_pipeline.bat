@echo off
python -m venv .venv
call .venv\Scripts\activate
pip install -r requirements.txt
python scripts\demo.py --quick
echo Ready! Run: streamlit run dashboard/app.py