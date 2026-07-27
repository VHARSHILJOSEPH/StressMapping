import sys
import time
import socket
import os
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np

import config
from desktop_app.receiver import UDPDataReceiver
from desktop_app.data_logger import DataLogger
from desktop_app.preprocessing import BioSignalPreprocessor
from desktop_app.model_inference import StressClassifier
from desktop_app.report_generator import SessionReportGenerator

# Page Configuration
st.set_page_config(
    page_title="ESP32 Bio-Signal Stress Monitor",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# PREMIUM CLINICAL-TECH CSS DESIGN SYSTEM
# ============================================================================
st.markdown("""
<style>
    /* ── Google Font Import ── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* ── Root Variables ── */
    :root {
        --bg-primary: #06090F;
        --bg-secondary: #0C1118;
        --bg-card: rgba(15, 23, 42, 0.65);
        --bg-card-hover: rgba(20, 30, 52, 0.8);
        --bg-elevated: rgba(22, 33, 55, 0.7);
        --border-subtle: rgba(56, 82, 130, 0.25);
        --border-accent: rgba(37, 99, 235, 0.35);
        --text-primary: #F1F5F9;
        --text-secondary: #94A3B8;
        --text-muted: #64748B;
        --accent-blue: #2563EB;
        --accent-cyan: #06B6D4;
        --accent-green: #22C55E;
        --accent-amber: #F59E0B;
        --accent-red: #EF4444;
        --accent-purple: #A855F7;
        --glow-blue: rgba(37, 99, 235, 0.15);
        --glow-cyan: rgba(6, 182, 212, 0.12);
        --radius-sm: 8px;
        --radius-md: 12px;
        --radius-lg: 16px;
        --radius-xl: 20px;
        --shadow-card: 0 1px 3px rgba(0,0,0,0.3), 0 8px 24px rgba(0,0,0,0.15);
        --shadow-elevated: 0 4px 16px rgba(0,0,0,0.4), 0 16px 48px rgba(0,0,0,0.2);
        --transition-fast: 150ms cubic-bezier(0.4, 0, 0.2, 1);
        --transition-smooth: 300ms cubic-bezier(0.4, 0, 0.2, 1);
    }

    /* ── Global ── */
    .main {
        background: linear-gradient(170deg, var(--bg-primary) 0%, #070B12 40%, #0A0F1A 100%) !important;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif !important;
    }
    .block-container { padding-top: 1.5rem !important; }
    h1, h2, h3, h4, h5, h6, p, span, label, div {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif !important;
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0B1120 0%, #0D1424 50%, #091018 100%) !important;
        border-right: 1px solid var(--border-subtle) !important;
    }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
        color: var(--text-secondary) !important;
        font-size: 13px !important;
    }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
        color: var(--text-primary) !important;
    }
    [data-testid="stSidebar"] hr {
        border-color: var(--border-subtle) !important;
        margin: 16px 0 !important;
    }

    /* ── Buttons ── */
    .stButton > button {
        background: linear-gradient(135deg, var(--accent-blue) 0%, #1D4ED8 100%) !important;
        color: #FFFFFF !important;
        border: 1px solid rgba(59, 130, 246, 0.5) !important;
        border-radius: var(--radius-sm) !important;
        font-weight: 600 !important;
        font-size: 13px !important;
        letter-spacing: 0.01em !important;
        padding: 8px 16px !important;
        transition: all var(--transition-fast) !important;
        box-shadow: 0 1px 2px rgba(0,0,0,0.2), inset 0 1px 0 rgba(255,255,255,0.08) !important;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #3B82F6 0%, var(--accent-blue) 100%) !important;
        border-color: rgba(96, 165, 250, 0.6) !important;
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.35), inset 0 1px 0 rgba(255,255,255,0.1) !important;
        transform: translateY(-1px) !important;
    }
    .stButton > button:active {
        transform: translateY(0px) !important;
        box-shadow: 0 1px 2px rgba(0,0,0,0.3) !important;
    }
    .stButton > button:disabled {
        background: rgba(30, 41, 59, 0.8) !important;
        color: var(--text-muted) !important;
        border-color: var(--border-subtle) !important;
        box-shadow: none !important;
        cursor: not-allowed !important;
    }

    /* ── Download Buttons ── */
    .stDownloadButton > button {
        background: linear-gradient(135deg, rgba(22, 163, 74, 0.15) 0%, rgba(22, 163, 74, 0.08) 100%) !important;
        color: var(--accent-green) !important;
        border: 1px solid rgba(34, 197, 94, 0.3) !important;
        border-radius: var(--radius-sm) !important;
        font-weight: 600 !important;
        transition: all var(--transition-fast) !important;
    }
    .stDownloadButton > button:hover {
        background: linear-gradient(135deg, rgba(22, 163, 74, 0.25) 0%, rgba(22, 163, 74, 0.15) 100%) !important;
        border-color: rgba(34, 197, 94, 0.5) !important;
        box-shadow: 0 4px 12px rgba(34, 197, 94, 0.2) !important;
    }

    /* ── Toggle Switch ── */
    [data-testid="stToggle"] label span {
        color: var(--text-secondary) !important;
        font-weight: 500 !important;
    }

    /* ── Metric Cards ── */
    [data-testid="stMetric"] {
        background: var(--bg-card) !important;
        backdrop-filter: blur(12px) !important;
        -webkit-backdrop-filter: blur(12px) !important;
        border: 1px solid var(--border-subtle) !important;
        padding: 20px 22px !important;
        border-radius: var(--radius-lg) !important;
        box-shadow: var(--shadow-card) !important;
        transition: all var(--transition-smooth) !important;
        position: relative !important;
        overflow: hidden !important;
    }
    [data-testid="stMetric"]:hover {
        border-color: var(--border-accent) !important;
        box-shadow: var(--shadow-elevated), 0 0 20px var(--glow-blue) !important;
        transform: translateY(-2px) !important;
    }
    [data-testid="stMetric"]::before {
        content: '' !important;
        position: absolute !important;
        top: 0 !important;
        left: 0 !important;
        right: 0 !important;
        height: 2px !important;
        background: linear-gradient(90deg, var(--accent-blue), var(--accent-cyan)) !important;
        opacity: 0 !important;
        transition: opacity var(--transition-smooth) !important;
    }
    [data-testid="stMetric"]:hover::before { opacity: 1 !important; }
    [data-testid="stMetric"] label {
        color: var(--text-muted) !important;
        font-size: 12px !important;
        font-weight: 600 !important;
        letter-spacing: 0.06em !important;
        text-transform: uppercase !important;
    }
    [data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: var(--text-primary) !important;
        font-size: 28px !important;
        font-weight: 700 !important;
        letter-spacing: -0.02em !important;
    }
    [data-testid="stMetric"] [data-testid="stMetricDelta"] {
        font-size: 12px !important;
        font-weight: 500 !important;
    }

    /* ── Tabs ── */
    .stTabs [data-baseweb="tab-list"] {
        background: rgba(11, 17, 32, 0.6) !important;
        border-radius: var(--radius-md) !important;
        padding: 4px !important;
        gap: 4px !important;
        border: 1px solid var(--border-subtle) !important;
    }
    .stTabs [data-baseweb="tab"] {
        color: var(--text-muted) !important;
        font-weight: 500 !important;
        font-size: 13px !important;
        border-radius: var(--radius-sm) !important;
        padding: 8px 16px !important;
        transition: all var(--transition-fast) !important;
        border: none !important;
        background: transparent !important;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: var(--text-secondary) !important;
        background: rgba(37, 99, 235, 0.06) !important;
    }
    .stTabs [aria-selected="true"] {
        color: var(--text-primary) !important;
        background: rgba(37, 99, 235, 0.12) !important;
        border: 1px solid var(--border-accent) !important;
        box-shadow: 0 0 12px var(--glow-blue) !important;
    }
    .stTabs [data-baseweb="tab-highlight"] { display: none !important; }
    .stTabs [data-baseweb="tab-border"] { display: none !important; }

    /* ── Info / Warning / Success Alerts ── */
    [data-testid="stAlert"] {
        border-radius: var(--radius-md) !important;
        backdrop-filter: blur(8px) !important;
        border-left-width: 3px !important;
    }

    /* ── Dataframe / Tables ── */
    [data-testid="stDataFrame"] {
        border-radius: var(--radius-md) !important;
        overflow: hidden !important;
        border: 1px solid var(--border-subtle) !important;
    }

    /* ── Selectbox ── */
    [data-testid="stSelectbox"] > div > div {
        background-color: var(--bg-card) !important;
        border-color: var(--border-subtle) !important;
        border-radius: var(--radius-sm) !important;
    }

    /* ── Progress Bar ── */
    .stProgress > div > div {
        border-radius: 6px !important;
        overflow: hidden !important;
    }
    .stProgress > div > div > div {
        background: linear-gradient(90deg, var(--accent-blue) 0%, var(--accent-cyan) 100%) !important;
        border-radius: 6px !important;
    }

    /* ═══ Custom HTML Elements ═══ */

    /* ── Header Bar ── */
    .dashboard-header {
        display: flex;
        align-items: center;
        gap: 16px;
        margin-bottom: 8px;
    }
    .dashboard-header .logo-mark {
        width: 40px;
        height: 40px;
        border-radius: var(--radius-md);
        background: linear-gradient(135deg, var(--accent-blue) 0%, var(--accent-cyan) 100%);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 20px;
        flex-shrink: 0;
        box-shadow: 0 2px 10px rgba(37, 99, 235, 0.3);
    }
    .dashboard-header .title-group h1 {
        margin: 0 !important;
        font-size: 22px !important;
        font-weight: 700 !important;
        color: var(--text-primary) !important;
        letter-spacing: -0.03em !important;
        line-height: 1.2 !important;
    }
    .dashboard-header .title-group .subtitle {
        font-size: 13px;
        color: var(--text-muted);
        font-weight: 400;
        margin-top: 2px;
    }

    /* ── Status Strip ── */
    .status-strip {
        background: var(--bg-card);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid var(--border-subtle);
        border-radius: var(--radius-lg);
        padding: 14px 24px;
        margin-bottom: 20px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        flex-wrap: wrap;
        gap: 12px;
        box-shadow: var(--shadow-card);
    }
    .status-strip .segment {
        display: flex;
        align-items: center;
        gap: 8px;
        font-size: 13px;
        color: var(--text-secondary);
    }
    .status-strip .segment .label {
        color: var(--text-muted);
        font-weight: 500;
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }
    .status-strip .segment code {
        background: rgba(37, 99, 235, 0.08);
        border: 1px solid rgba(37, 99, 235, 0.15);
        padding: 2px 8px;
        border-radius: 6px;
        font-size: 12px;
        color: var(--text-secondary);
        font-family: 'SF Mono', 'Fira Code', 'Cascadia Code', monospace;
    }

    /* ── Status Badges ── */
    .badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 5px 14px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 12px;
        letter-spacing: 0.02em;
        text-transform: uppercase;
    }
    .badge-live {
        background: linear-gradient(135deg, rgba(34, 197, 94, 0.15) 0%, rgba(34, 197, 94, 0.08) 100%);
        color: var(--accent-green);
        border: 1px solid rgba(34, 197, 94, 0.3);
        box-shadow: 0 0 12px rgba(34, 197, 94, 0.12);
    }
    .badge-sim {
        background: linear-gradient(135deg, rgba(37, 99, 235, 0.15) 0%, rgba(6, 182, 212, 0.08) 100%);
        color: var(--accent-cyan);
        border: 1px solid rgba(6, 182, 212, 0.3);
        box-shadow: 0 0 12px rgba(6, 182, 212, 0.1);
    }
    .badge-off {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.12) 0%, rgba(239, 68, 68, 0.05) 100%);
        color: var(--accent-red);
        border: 1px solid rgba(239, 68, 68, 0.25);
    }
    .badge-rec {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.15) 0%, rgba(239, 68, 68, 0.06) 100%);
        color: var(--accent-red);
        border: 1px solid rgba(239, 68, 68, 0.25);
    }
    .badge-idle {
        background: rgba(30, 41, 59, 0.5);
        color: var(--text-muted);
        border: 1px solid var(--border-subtle);
    }

    /* ── Pulse Dot Animation ── */
    .pulse-dot {
        width: 7px;
        height: 7px;
        border-radius: 50%;
        display: inline-block;
        position: relative;
    }
    .pulse-dot.green { background-color: var(--accent-green); }
    .pulse-dot.cyan { background-color: var(--accent-cyan); }
    .pulse-dot.red { background-color: var(--accent-red); }
    .pulse-dot.gray { background-color: var(--text-muted); }
    .pulse-dot.active::after {
        content: '';
        position: absolute;
        top: -3px;
        left: -3px;
        width: 13px;
        height: 13px;
        border-radius: 50%;
        border: 1.5px solid currentColor;
        opacity: 0;
        animation: pulse-ring 1.8s ease-out infinite;
    }
    @keyframes pulse-ring {
        0% { transform: scale(0.6); opacity: 0.8; }
        100% { transform: scale(1.8); opacity: 0; }
    }

    /* ── Stress Gauge Card ── */
    .stress-card {
        background: var(--bg-card);
        backdrop-filter: blur(12px);
        border: 1px solid var(--border-subtle);
        border-radius: var(--radius-xl);
        padding: 28px;
        box-shadow: var(--shadow-card);
        transition: all var(--transition-smooth);
    }
    .stress-card:hover {
        border-color: var(--border-accent);
        box-shadow: var(--shadow-elevated), 0 0 24px var(--glow-blue);
    }
    .stress-card .stress-label {
        font-size: 32px;
        font-weight: 800;
        letter-spacing: -0.03em;
        margin: 12px 0 4px;
    }
    .stress-label-relaxed { color: var(--accent-green); }
    .stress-label-low { color: var(--accent-cyan); }
    .stress-label-moderate { color: var(--accent-amber); }
    .stress-label-high { color: var(--accent-red); }
    .stress-card .score-bar {
        height: 6px;
        border-radius: 3px;
        background: rgba(30, 41, 59, 0.6);
        margin: 16px 0 8px;
        overflow: hidden;
    }
    .stress-card .score-fill {
        height: 100%;
        border-radius: 3px;
        transition: width 600ms cubic-bezier(0.4, 0, 0.2, 1);
    }
    .stress-card .meta-row {
        display: flex;
        justify-content: space-between;
        font-size: 12px;
        color: var(--text-muted);
        font-weight: 500;
    }

    /* ── Feature Table Card ── */
    .feature-card {
        background: var(--bg-card);
        backdrop-filter: blur(12px);
        border: 1px solid var(--border-subtle);
        border-radius: var(--radius-xl);
        padding: 24px;
        box-shadow: var(--shadow-card);
    }
    .feature-card h3 {
        color: var(--text-primary);
        font-size: 16px;
        font-weight: 600;
        margin: 0 0 16px;
        letter-spacing: -0.01em;
    }
    .feature-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 10px 0;
        border-bottom: 1px solid rgba(56, 82, 130, 0.12);
    }
    .feature-row:last-child { border-bottom: none; }
    .feature-row .feat-name {
        font-size: 13px;
        color: var(--text-secondary);
        font-weight: 500;
    }
    .feature-row .feat-value {
        font-size: 14px;
        color: var(--text-primary);
        font-weight: 600;
        font-family: 'SF Mono', 'Fira Code', 'Cascadia Code', monospace;
    }
    .feature-row .feat-ref {
        font-size: 11px;
        color: var(--text-muted);
        font-weight: 400;
    }

    /* ── Pipeline Config Cards ── */
    .pipeline-card {
        background: var(--bg-card);
        backdrop-filter: blur(12px);
        border: 1px solid var(--border-subtle);
        border-radius: var(--radius-lg);
        padding: 24px;
        box-shadow: var(--shadow-card);
    }
    .pipeline-card h4 {
        color: var(--text-primary);
        font-size: 15px;
        font-weight: 600;
        margin: 0 0 14px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .pipeline-card h4 .icon {
        width: 28px;
        height: 28px;
        border-radius: var(--radius-sm);
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font-size: 14px;
        flex-shrink: 0;
    }
    .pipeline-card .param {
        display: flex;
        justify-content: space-between;
        padding: 7px 0;
        font-size: 13px;
        border-bottom: 1px solid rgba(56, 82, 130, 0.1);
    }
    .pipeline-card .param:last-child { border-bottom: none; }
    .pipeline-card .param .key { color: var(--text-muted); font-weight: 500; }
    .pipeline-card .param .val {
        color: var(--text-primary);
        font-weight: 600;
        font-family: 'SF Mono', 'Fira Code', 'Cascadia Code', monospace;
        font-size: 12px;
    }

    /* ── Empty State ── */
    .empty-state {
        text-align: center;
        padding: 60px 40px;
        background: var(--bg-card);
        border: 1px dashed var(--border-subtle);
        border-radius: var(--radius-xl);
    }
    .empty-state .icon { font-size: 48px; margin-bottom: 16px; opacity: 0.5; }
    .empty-state .heading {
        font-size: 18px;
        font-weight: 600;
        color: var(--text-secondary);
        margin-bottom: 6px;
    }
    .empty-state .desc {
        font-size: 13px;
        color: var(--text-muted);
        max-width: 380px;
        margin: 0 auto;
        line-height: 1.5;
    }

    /* ── Sidebar Section Headers ── */
    .sidebar-section {
        font-size: 11px;
        font-weight: 700;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 10px;
    }
    .sidebar-info-card {
        background: rgba(37, 99, 235, 0.06);
        border: 1px solid rgba(37, 99, 235, 0.12);
        border-radius: var(--radius-sm);
        padding: 12px 14px;
        margin-bottom: 8px;
    }
    .sidebar-info-card .info-label {
        font-size: 10px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: var(--text-muted);
        margin-bottom: 4px;
    }
    .sidebar-info-card .info-value {
        font-size: 14px;
        font-weight: 600;
        color: var(--accent-cyan);
        font-family: 'SF Mono', 'Fira Code', 'Cascadia Code', monospace;
    }
</style>
""", unsafe_allow_html=True)

# Helper function to get local IP address
def get_local_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"

# Initialize Session State Objects
if "receiver" not in st.session_state:
    st.session_state.receiver = UDPDataReceiver()
    st.session_state.data_logger = DataLogger()
    st.session_state.preprocessor = BioSignalPreprocessor()
    st.session_state.classifier = StressClassifier()
    st.session_state.report_gen = SessionReportGenerator()
    st.session_state.active_report = None
    st.session_state.last_logged_packet_counter = -1

receiver: UDPDataReceiver = st.session_state.receiver
data_logger: DataLogger = st.session_state.data_logger
preprocessor: BioSignalPreprocessor = st.session_state.preprocessor
classifier: StressClassifier = st.session_state.classifier
report_gen: SessionReportGenerator = st.session_state.report_gen

# ============================================================================
# SIDEBAR CONTROLS & NETWORKING
# ============================================================================
st.sidebar.markdown("""
<div style="display:flex; align-items:center; gap:10px; margin-bottom:4px;">
    <div style="width:32px;height:32px;border-radius:10px;background:linear-gradient(135deg,#2563EB,#06B6D4);display:flex;align-items:center;justify-content:center;font-size:16px;box-shadow:0 2px 8px rgba(37,99,235,0.3);">&#9889;</div>
    <div>
        <div style="font-size:16px;font-weight:700;color:#F1F5F9;letter-spacing:-0.02em;">Telemetry Control</div>
        <div style="font-size:11px;color:#64748B;font-weight:400;">ESP32 Receiver Management</div>
    </div>
</div>
""", unsafe_allow_html=True)

local_ip = get_local_ip()

st.sidebar.markdown(f"""
<div class="sidebar-info-card">
    <div class="info-label">Computer Host IP</div>
    <div class="info-value">{local_ip}</div>
</div>
<div class="sidebar-info-card">
    <div class="info-label">UDP Target Port</div>
    <div class="info-value">{config.UDP_PORT}</div>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")
st.sidebar.markdown('<div class="sidebar-section">Connection Setup</div>', unsafe_allow_html=True)

sim_mode_toggle = st.sidebar.toggle("Simulated Hardware Mode", value=receiver.simulation_mode)
if sim_mode_toggle != receiver.simulation_mode:
    receiver.set_simulation_mode(sim_mode_toggle)

col_conn1, col_conn2 = st.sidebar.columns(2)
with col_conn1:
    if st.button("Start Receiver", use_container_width=True):
        receiver.start(simulation_mode=sim_mode_toggle)
        st.session_state.last_logged_packet_counter = -1
        st.toast("Telemetry Receiver Started!", icon="🚀")

with col_conn2:
    if st.button("Stop Receiver", use_container_width=True):
        receiver.stop()
        st.toast("Receiver Stopped.", icon="🛑")

st.sidebar.markdown("---")
st.sidebar.markdown('<div class="sidebar-section">Recording Session</div>', unsafe_allow_html=True)

col_rec1, col_rec2 = st.sidebar.columns(2)
with col_rec1:
    if st.button("Start Session", disabled=data_logger.is_recording, use_container_width=True):
        sess_id = data_logger.start_session()
        st.session_state.last_logged_packet_counter = -1
        st.toast(f"Recording Session Started: {sess_id}", icon="🔴")

with col_rec2:
    if st.button("Stop Session", disabled=not data_logger.is_recording, use_container_width=True):
        summary = data_logger.stop_session()
        st.toast(f"Session Saved! Duration: {summary.get('duration_sec', 0)}s", icon="✅")

if st.sidebar.button("Export Last Session Report", use_container_width=True):
    sessions = data_logger.list_recorded_sessions()
    if sessions:
        report_files = report_gen.generate_report_from_csv(str(sessions[0]))
        st.session_state.active_report = report_files
        st.toast("Session Report Exported!", icon="📄")
    else:
        st.sidebar.warning("No recorded session CSV found yet.")

# ============================================================================
# MAIN DASHBOARD HEADER & STATUS BAR
# ============================================================================
st.markdown("""
<div class="dashboard-header">
    <div class="logo-mark">&#9889;</div>
    <div class="title-group">
        <h1>Wearable Stress-Monitoring Research Dashboard</h1>
        <div class="subtitle">ESP32 Multi-Modal Bio-Signal Telemetry &middot; Real-Time Analytics Platform</div>
    </div>
</div>
""", unsafe_allow_html=True)

status_summary = receiver.get_status_summary()
conn_status = status_summary["status"]

if conn_status == "CONNECTED":
    badge_html = '<span class="badge badge-live"><span class="pulse-dot green active"></span>LIVE UDP</span>'
elif conn_status == "SIMULATED":
    badge_html = '<span class="badge badge-sim"><span class="pulse-dot cyan active"></span>SIMULATION</span>'
else:
    badge_html = '<span class="badge badge-off"><span class="pulse-dot gray"></span>DISCONNECTED</span>'

if data_logger.is_recording:
    rec_badge_html = f'<span class="badge badge-rec"><span class="pulse-dot red active"></span>REC &middot; {data_logger.samples_logged}</span>'
else:
    rec_badge_html = '<span class="badge badge-idle">IDLE</span>'

st.markdown(f"""
<div class="status-strip">
    <div style="display:flex; align-items:center; gap:16px; flex-wrap:wrap;">
        <div class="segment">
            <span class="label">Status</span>
            {badge_html}
        </div>
        <div class="segment">
            <span class="label">Source</span>
            <code>{status_summary['last_remote_ip']}</code>
        </div>
        <div class="segment">
            <span class="label">Rate</span>
            <code>{status_summary['rate_hz']} Hz</code>
        </div>
    </div>
    <div class="segment">
        <span class="label">Session</span>
        {rec_badge_html}
    </div>
</div>
""", unsafe_allow_html=True)

# Fetch latest raw telemetry packets from receiver
latest_packets = receiver.get_latest_data(count=config.BUFFER_SIZE)

# Run preprocessing & classification pipeline
processed_batch = preprocessor.process_batch(latest_packets)
stress_result = classifier.predict({
    "rmssd": processed_batch["rmssd"],
    "scl_mean": processed_batch["scl_mean"],
    "scr_count": processed_batch["scr_count"],
    "activity_index": processed_batch["activity_index"],
    "bpm": processed_batch["bpm"]
})

# Log all new packets to CSV when session recording is active
if data_logger.is_recording and latest_packets:
    last_counter = st.session_state.last_logged_packet_counter
    for pkt in latest_packets:
        counter = pkt.get("packet_counter", 0)
        if counter > last_counter:
            pkt["stress_state"] = stress_result["label"]
            pkt["confidence"] = stress_result["confidence"]
            data_logger.log_packet(pkt)
            last_counter = counter
    st.session_state.last_logged_packet_counter = last_counter

# ============================================================================
# REAL-TIME SENSOR METRICS CARDS
# ============================================================================
col_m1, col_m2, col_m3, col_m4 = st.columns(4)

with col_m1:
    st.metric(
        label="Heart Rate (PPG)",
        value=f"{processed_batch['bpm']} BPM",
        delta=f"HRV RMSSD: {processed_batch['rmssd']} ms"
    )

with col_m2:
    st.metric(
        label="Skin Conductance (GSR)",
        value=f"{processed_batch['scl_mean']} \u03bcS",
        delta=f"SCR Spikes: {processed_batch['scr_count']}"
    )

with col_m3:
    st.metric(
        label="Motion Activity (IMU)",
        value=f"{processed_batch['activity_index']} g",
        delta=f"State: {processed_batch['motion_state']}"
    )

with col_m4:
    stress_label = stress_result["label"]
    st.metric(
        label="Stress Classification",
        value=stress_label,
        delta=f"Score: {stress_result['stress_score']} | {stress_result['trend']}"
    )

# ============================================================================
# PLOTLY CHART THEME HELPERS
# ============================================================================
def get_chart_layout(title: str, height: int = 260) -> dict:
    """Shared premium Plotly dark layout config."""
    return dict(
        title=dict(
            text=title,
            font=dict(size=14, color="#94A3B8", family="Inter, sans-serif"),
            x=0.01, y=0.97
        ),
        height=height,
        margin=dict(l=12, r=12, t=40, b=16),
        template="plotly_dark",
        paper_bgcolor="rgba(15, 23, 42, 0.5)",
        plot_bgcolor="rgba(11, 17, 32, 0.4)",
        font=dict(family="Inter, sans-serif", color="#94A3B8", size=11),
        xaxis=dict(
            gridcolor="rgba(56, 82, 130, 0.12)",
            zerolinecolor="rgba(56, 82, 130, 0.15)",
            showgrid=True, gridwidth=1,
        ),
        yaxis=dict(
            gridcolor="rgba(56, 82, 130, 0.12)",
            zerolinecolor="rgba(56, 82, 130, 0.15)",
            showgrid=True, gridwidth=1,
        ),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            font=dict(size=11, color="#94A3B8"),
            orientation="h", y=1.08,
        ),
        hoverlabel=dict(
            bgcolor="#1E293B",
            bordercolor="#334155",
            font=dict(color="#F1F5F9", size=12, family="Inter, sans-serif"),
        ),
    )

# ============================================================================
# REAL-TIME CHARTS & SIGNAL VISUALIZATION
# ============================================================================
tab_live, tab_stress, tab_pipeline, tab_report = st.tabs([
    "Live Bio-Signals",
    "Stress Telemetry & Model",
    "Preprocessing Pipeline",
    "Session Reports"
])

with tab_live:
    if not latest_packets:
        st.markdown("""
        <div class="empty-state">
            <div class="icon">&#128225;</div>
            <div class="heading">Waiting for data stream</div>
            <div class="desc">Click <strong>Start Receiver</strong> in the sidebar, or toggle <strong>Simulated Hardware Mode</strong> to begin streaming synthetic bio-signals.</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        df_packets = pd.DataFrame(latest_packets)
        time_idx = np.arange(len(df_packets)) / config.SAMPLING_RATE_HZ

        # PPG Chart
        fig_ppg = go.Figure()
        fig_ppg.add_trace(go.Scatter(
            y=df_packets["ppg_raw"], name="Raw PPG ADC",
            line=dict(color="rgba(148, 163, 184, 0.45)", width=1)
        ))
        if "ppg_filtered" in processed_batch and len(processed_batch["ppg_filtered"]) == len(df_packets):
            fig_ppg.add_trace(go.Scatter(
                y=processed_batch["ppg_filtered"], name="Filtered Pulse",
                line=dict(color="#3B82F6", width=2.5),
                fill='tozeroy', fillcolor='rgba(59, 130, 246, 0.06)'
            ))
        fig_ppg.update_layout(**get_chart_layout("PPG Cardiac Waveform (Photoplethysmography)", 270))
        st.plotly_chart(fig_ppg, use_container_width=True)

        # GSR Chart
        fig_gsr = go.Figure()
        fig_gsr.add_trace(go.Scatter(
            y=df_packets["gsr_raw"], name="Raw Conductance (\u03bcS)",
            line=dict(color="rgba(245, 158, 11, 0.6)", width=1.5)
        ))
        if "gsr_tonic" in processed_batch and len(processed_batch["gsr_tonic"]) == len(df_packets):
            fig_gsr.add_trace(go.Scatter(
                y=processed_batch["gsr_tonic"], name="Tonic SCL Level",
                line=dict(color="#22C55E", width=2.5),
                fill='tozeroy', fillcolor='rgba(34, 197, 94, 0.05)'
            ))
        fig_gsr.update_layout(**get_chart_layout("Galvanic Skin Response / Electrodermal Activity", 250))
        st.plotly_chart(fig_gsr, use_container_width=True)

        # IMU Chart
        fig_imu = go.Figure()
        fig_imu.add_trace(go.Scatter(
            y=df_packets["imu_ax"], name="Accel X",
            line=dict(color="#EF4444", width=1.5)
        ))
        fig_imu.add_trace(go.Scatter(
            y=df_packets["imu_ay"], name="Accel Y",
            line=dict(color="#A855F7", width=1.5)
        ))
        fig_imu.add_trace(go.Scatter(
            y=df_packets["imu_az"], name="Accel Z",
            line=dict(color="#3B82F6", width=1.5)
        ))
        fig_imu.update_layout(**get_chart_layout("IMU 3-Axis Accelerometer Dynamics (g)", 250))
        st.plotly_chart(fig_imu, use_container_width=True)

with tab_stress:
    col_s1, col_s2 = st.columns([1, 1])

    with col_s1:
        # Determine stress color class
        sl = stress_result['label']
        if sl == "RELAXED":
            stress_color_cls = "stress-label-relaxed"
            bar_gradient = "linear-gradient(90deg, #22C55E, #4ADE80)"
        elif sl == "LOW_STRESS":
            stress_color_cls = "stress-label-low"
            bar_gradient = "linear-gradient(90deg, #06B6D4, #22D3EE)"
        elif sl == "MODERATE_STRESS":
            stress_color_cls = "stress-label-moderate"
            bar_gradient = "linear-gradient(90deg, #F59E0B, #FBBF24)"
        else:
            stress_color_cls = "stress-label-high"
            bar_gradient = "linear-gradient(90deg, #EF4444, #F87171)"

        score_pct = min(100, max(0, stress_result['stress_score']))
        confidence_pct = int(stress_result['confidence'] * 100)

        st.markdown(f"""
        <div class="stress-card">
            <div style="font-size:12px; font-weight:600; text-transform:uppercase; letter-spacing:0.06em; color:#64748B;">Current Stress Assessment</div>
            <div class="stress-label {stress_color_cls}">{stress_result['label'].replace('_', ' ')}</div>
            <div class="score-bar"><div class="score-fill" style="width:{score_pct}%; background:{bar_gradient};"></div></div>
            <div class="meta-row">
                <span>Autonomic Stress Index: <strong style="color:#F1F5F9;">{stress_result['stress_score']} / 100</strong></span>
                <span>Confidence: <strong style="color:#F1F5F9;">{confidence_pct}%</strong></span>
            </div>
            <div style="margin-top:16px; padding-top:14px; border-top:1px solid rgba(56,82,130,0.15);">
                <div style="font-size:12px; color:#64748B; font-weight:500;">Short-Term Trend</div>
                <div style="font-size:18px; font-weight:600; color:#94A3B8; margin-top:4px;">{stress_result['trend']}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_s2:
        features_data = [
            ("HRV RMSSD", f"{processed_batch['rmssd']} ms", "20 - 70 ms"),
            ("GSR Conductance", f"{processed_batch['scl_mean']} \u03bcS", "1.0 - 5.0 \u03bcS"),
            ("SCR Phasic Spikes", f"{processed_batch['scr_count']}", "0 - 3 / 10s"),
            ("Motion Index", f"{processed_batch['activity_index']} g", "< 0.2 g (Rest)"),
            ("Heart Rate", f"{processed_batch['bpm']} BPM", "60 - 100 BPM"),
        ]
        rows_html = ""
        for name, val, ref in features_data:
            rows_html += f"""
            <div class="feature-row">
                <span class="feat-name">{name}</span>
                <span>
                    <span class="feat-value">{val}</span>
                    <span class="feat-ref" style="margin-left:10px;">{ref}</span>
                </span>
            </div>
            """
        st.markdown(f"""
        <div class="feature-card">
            <h3>Feature Matrix Contribution</h3>
            {rows_html}
        </div>
        """, unsafe_allow_html=True)

with tab_pipeline:
    col_p1, col_p2 = st.columns(2)
    with col_p1:
        st.markdown(f"""
        <div class="pipeline-card">
            <h4><span class="icon" style="background:rgba(59,130,246,0.12); color:#3B82F6;">&#9835;</span> PPG Bandpass Butterworth Filter</h4>
            <div class="param"><span class="key">Filter Order</span><span class="val">{config.PPG_FILTER_ORDER}</span></div>
            <div class="param"><span class="key">Low Cutoff</span><span class="val">{config.PPG_LOWCUT} Hz (30 BPM)</span></div>
            <div class="param"><span class="key">High Cutoff</span><span class="val">{config.PPG_HIGHCUT} Hz (240 BPM)</span></div>
            <div class="param"><span class="key">Method</span><span class="val">Zero-Phase Filtfilt</span></div>
        </div>
        """, unsafe_allow_html=True)

    with col_p2:
        st.markdown(f"""
        <div class="pipeline-card">
            <h4><span class="icon" style="background:rgba(34,197,94,0.12); color:#22C55E;">&#9889;</span> GSR / EDA Decomposition Filter</h4>
            <div class="param"><span class="key">Filter Order</span><span class="val">{config.GSR_FILTER_ORDER}</span></div>
            <div class="param"><span class="key">Tonic Lowpass Cutoff</span><span class="val">{config.GSR_LOWCUT} Hz</span></div>
            <div class="param"><span class="key">SCR Spike Threshold</span><span class="val">&gt; 0.05 &mu;S</span></div>
            <div class="param"><span class="key">Method</span><span class="val">Tonic/Phasic Separation</span></div>
        </div>
        """, unsafe_allow_html=True)

with tab_report:
    sessions_available = data_logger.list_recorded_sessions()
    if sessions_available:
        selected_session = st.selectbox("Select Recorded Session CSV", [s.name for s in sessions_available])
        if st.button("Generate & Preview Report"):
            selected_path = config.DATA_DIR / selected_session
            report_files = report_gen.generate_report_from_csv(str(selected_path))
            st.session_state.active_report = report_files

    if st.session_state.active_report:
        rep = st.session_state.active_report
        st.success(f"Report Generated for: {rep['session_id']}")

        col_r1, col_r2 = st.columns(2)
        with col_r1:
            if os.path.exists(rep["html_path"]):
                with open(rep["html_path"], "r", encoding="utf-8") as f:
                    html_bytes = f.read().encode("utf-8")
                st.download_button(
                    label="Download HTML Report",
                    data=html_bytes,
                    file_name=os.path.basename(rep["html_path"]),
                    mime="text/html",
                    use_container_width=True
                )
        with col_r2:
            if os.path.exists(rep["md_path"]):
                with open(rep["md_path"], "r", encoding="utf-8") as f:
                    md_bytes = f.read().encode("utf-8")
                st.download_button(
                    label="Download Markdown Summary",
                    data=md_bytes,
                    file_name=os.path.basename(rep["md_path"]),
                    mime="text/markdown",
                    use_container_width=True
                )
    else:
        st.markdown("""
        <div class="empty-state">
            <div class="icon">&#128203;</div>
            <div class="heading">No report generated yet</div>
            <div class="desc">Record a session using the sidebar controls, then click <strong>Export Last Session Report</strong> or select a recorded CSV above.</div>
        </div>
        """, unsafe_allow_html=True)

# ============================================================================
# AUTO-REFRESH UI LOOP
# ============================================================================
time.sleep(0.2)
st.rerun()
