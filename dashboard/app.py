"""Streamlit dashboard — no hardware needed."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from collections import deque
from datetime import datetime
import time

from src.stream_simulator import StreamSimulator
from src.action_engine import ActionEngine
from src.predictor import Predictor


st.set_page_config(page_title="Mine Collapse Early-Warning", page_icon="⛏️", layout="wide")

st.markdown("""
<style>
    .status-card { padding: 1.2rem; border-radius: 12px; text-align: center;
        font-weight: 600; font-size: 1.3rem; margin-bottom: 1rem; }
    .status-normal   { background: linear-gradient(135deg,#1abc9c,#16a085); color:white; }
    .status-warning  { background: linear-gradient(135deg,#f39c12,#e67e22); color:white; }
    .status-critical { background: linear-gradient(135deg,#e74c3c,#c0392b); color:white; }
</style>
""", unsafe_allow_html=True)

st.title("⛏️ AI-Powered Mine Collapse Early-Warning System")
st.caption("Software-only simulation — no hardware required")


if 'history' not in st.session_state:
    st.session_state.history = deque(maxlen=300)
if 'alerts' not in st.session_state:
    st.session_state.alerts = []
if 'running' not in st.session_state:
    st.session_state.running = False
if 'sim' not in st.session_state:
    st.session_state.sim = StreamSimulator(scenario='mixed')


@st.cache_resource
def load_models():
    return Predictor(), ActionEngine()


try:
    predictor, action_engine = load_models()
    models_ok = True
except Exception as e:
    st.error(f"⚠️ Models not loaded: {e}")
    st.info("Run first: `python scripts/demo.py --quick`")
    models_ok = False


with st.sidebar:
    st.header("⚙️ Control Panel")
    scenario = st.selectbox("Simulation scenario", ['mixed', 'normal', 'warning', 'critical'], index=0)
    st.session_state.sim.scenario = scenario
    rate = st.slider("Update rate (Hz)", 1.0, 10.0, 4.0, 0.5)
    st.divider()
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("▶️ Start", use_container_width=True):
            st.session_state.running = True
            st.rerun()
    with col_b:
        if st.button("⏸️ Stop", use_container_width=True):
            st.session_state.running = False
            st.rerun()
    if st.button("🔄 Reset", use_container_width=True):
        st.session_state.history.clear()
        st.session_state.alerts.clear()
        st.rerun()
    if st.session_state.history:
        df = pd.DataFrame(list(st.session_state.history))
        st.download_button("📥 Export CSV", df.to_csv(index=False),
            f"mine_log_{datetime.now():%Y%m%d_%H%M%S}.csv", "text/csv",
            use_container_width=True)


if not models_ok:
    st.stop()

latest = st.session_state.history[-1] if st.session_state.history else None

if latest:
    lvl = latest['level']
    labels = {0: "NORMAL", 1: "WARNING", 2: "CRITICAL"}
    css_class = {0: "status-normal", 1: "status-warning", 2: "status-critical"}[lvl]
    st.markdown(f'<div class="status-card {css_class}">{labels[lvl]} — Confidence: {latest["confidence"]:.1%}</div>',
                unsafe_allow_html=True)
else:
    st.info("Press ▶️ Start to begin simulation.")

if latest:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Vibration (g)", f"{latest['vibration_magnitude']:.4f}")
    c2.metric("Load (kg)", f"{latest['load_cell']:.3f}")
    c3.metric("Displacement", f"{latest['displacement']}")
    c4.metric("Water Level", f"{latest['water_level']}")

if len(st.session_state.history) > 2:
    df = pd.DataFrame(list(st.session_state.history))
    st.subheader("📈 Live Sensor Trends")
    tab1, tab2, tab3 = st.tabs(["Vibration", "Load", "Displacement"])
    with tab1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=df['vibration_magnitude'], mode='lines',
            line=dict(color='#00d4ff', width=2)))
        fig.update_layout(template='plotly_dark', height=320, yaxis_title="Vibration (g)")
        st.plotly_chart(fig, use_container_width=True)
    with tab2:
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=df['load_cell'], mode='lines',
            line=dict(color='#ff6b6b', width=2)))
        fig.update_layout(template='plotly_dark', height=320, yaxis_title="Load (kg)")
        st.plotly_chart(fig, use_container_width=True)
    with tab3:
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=df['displacement'], mode='lines',
            line=dict(color='#f39c12', width=2)))
        fig.update_layout(template='plotly_dark', height=320, yaxis_title="Displacement")
        st.plotly_chart(fig, use_container_width=True)

st.divider()
st.subheader("📋 Recommended Actions")
if latest:
    rec = action_engine.recommend(latest['level'], latest['confidence'], latest['features'])
    st.markdown(f"### {rec.title}")
    st.write(f"**Description:** {rec.description}")
    st.write(f"**Urgency:** {rec.urgency}")
    st.write(f"**Trend:** {rec.details.get('trend')}")
    if rec.evacuation_required:
        st.error("🚨 **EVACUATION REQUIRED**")
    for a in rec.actions:
        st.write(f"- {a}")

st.divider()
st.subheader("🚨 Alerts History")
if st.session_state.alerts:
    for a in st.session_state.alerts[-10:][::-1]:
        st.write(f"`{a['time']}` — **{a['level']}** — {a['message']}")
else:
    st.caption("No alerts yet.")

if st.session_state.running:
    reading = st.session_state.sim.get_reading()
    prediction = predictor.predict(reading)
    record = {**reading, 'level': prediction['level'],
              'confidence': prediction['confidence'], 'features': prediction['features']}
    st.session_state.history.append(record)
    if prediction['level'] > 0:
        prev_lvl = st.session_state.history[-2]['level'] if len(st.session_state.history) > 1 else 0
        if prediction['level'] != prev_lvl:
            labels = {1: "WARNING", 2: "CRITICAL"}
            st.session_state.alerts.append({
                'time': datetime.now().strftime("%H:%M:%S"),
                'level': labels[prediction['level']],
                'message': f"Level → {labels[prediction['level']]} (conf {prediction['confidence']:.1%})"
            })
    time.sleep(1.0 / rate)
    st.rerun()