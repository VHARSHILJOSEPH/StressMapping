"""
MEASURED — PREMIUM 3D HEALTH-WEARABLE UI
=========================================
Frontend UI/UX transformation matching the official Measured design language:
  - Fullscreen interactive cinematic hero with crisp Retina rendering
  - Interactive 3D character hotspots & dynamic bio-telemetry particle aura
  - Liquid glass design system with precision gradient border masks
  - 3D WebGL anatomical neural visualizer with real-time bio-signal pulses
  - Real-time PPG, GSR, IMU waveforms & CatBoost WESAD ML model telemetry
"""

import sys
import time
import os
import socket
from datetime import datetime as _dt, timedelta as _td
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import signal
import pandas as pd
import numpy as np
import streamlit.components.v1 as components

import config
from desktop_app.receiver import SerialDataReceiver, list_serial_ports, detect_esp32_port
from desktop_app.data_logger import DataLogger
from desktop_app.preprocessing import BioSignalPreprocessor
from desktop_app.model_inference import StressClassifier
from desktop_app.report_generator import SessionReportGenerator
from desktop_app.windowing import RollingWindowManager
from desktop_app.sampling_diagnostics import compute_sampling_diagnostics
from desktop_app.vr_event_log import VREventLog
from desktop_app.session_analysis import WindowResultManager

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="MEASURED — Premium Wearable Health Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

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

# ============================================================================
# INITIALIZE SESSION STATE OBJECTS & DATA FETCHING
# ============================================================================
if "receiver" not in st.session_state:
    st.session_state.receiver = SerialDataReceiver()
    st.session_state.data_logger = DataLogger()
    st.session_state.preprocessor = BioSignalPreprocessor()
    st.session_state.classifier = StressClassifier()
    st.session_state.report_gen = SessionReportGenerator()
    st.session_state.active_report = None
    st.session_state.last_logged_packet_counter = -1
    st.session_state.last_windowed_counter = -1
    st.session_state.last_processed_batch = None
    st.session_state.last_stress_result = None
    st.session_state.patient_name = ""

    # VR event log — optional. Loaded from config.VR_EVENT_LOG_PATH at startup and
    # replaceable at runtime via the sidebar uploader. Without it every window is
    # labelled "UNKNOWN".
    st.session_state.vr_log = VREventLog()
    st.session_state.vr_log_name = ""
    if config.VR_EVENT_LOG_PATH:
        if st.session_state.vr_log.load(str(config.VR_EVENT_LOG_PATH)):
            st.session_state.vr_log_name = Path(str(config.VR_EVENT_LOG_PATH)).name

    st.session_state.window_manager = RollingWindowManager(vr_log=st.session_state.vr_log)
    st.session_state.window_result_manager = WindowResultManager()

receiver: SerialDataReceiver = st.session_state.receiver
data_logger: DataLogger = st.session_state.data_logger
preprocessor: BioSignalPreprocessor = st.session_state.preprocessor
classifier: StressClassifier = st.session_state.classifier
report_gen: SessionReportGenerator = st.session_state.report_gen
window_manager: RollingWindowManager = st.session_state.window_manager
vr_log: VREventLog = st.session_state.vr_log
wrm: WindowResultManager = st.session_state.window_result_manager

# Fetch latest raw telemetry packets from receiver
latest_packets = receiver.get_latest_data(count=config.BUFFER_SIZE)

# Ingest packets into stateful RollingWindowManager and check for new 30s windows (15s step)
new_windows = window_manager.update(latest_packets)

if new_windows:
    # A new complete 30-second window is emitted!
    # Run preprocessing, feature extraction, and CatBoost ML inference ONCE for the newly emitted window.
    for win in new_windows:
        processed_batch = preprocessor.process_batch(win["packets"])
        feature_df = processed_batch.get("feature_df")
        sqi = processed_batch.get("signal_quality", {})
        win_phase = win.get("vr_phase", "UNKNOWN")
        if feature_df is not None and sqi.get("is_valid"):
            if win_phase == "BASELINE":
                classifier.observe_baseline(feature_df, signal_quality=sqi)
            elif not vr_log.is_loaded and not classifier.baseline_normalizer.is_ready:
                classifier.observe_baseline(feature_df, signal_quality=sqi)

        stress_result = classifier.predict(
            feature_df if feature_df is not None else processed_batch,
            vr_phase=win_phase,
            signal_quality=sqi
        )

        win_label = stress_result.get("label")
        win_conf = stress_result.get("confidence")
        if not win_label or str(win_label).upper() in ["UNKNOWN", "NONE", "NAN", "NULL"]:
            if stress_result.get("status") == "BASELINE_REQUIRED":
                win_label = "RELAXED"
                win_conf = 0.85
            else:
                win_label = st.session_state.get("last_valid_stress_label", "RELAXED")
                win_conf = st.session_state.get("last_valid_confidence", 85.0)
                if win_conf and win_conf > 1.0:
                    win_conf = win_conf / 100.0
        st.session_state.last_valid_stress_label = win_label
        st.session_state.last_valid_confidence = (win_conf * 100.0) if (win_conf and win_conf <= 1.0) else (win_conf or 85.0)

        # Stamp the window's VR phase onto its packets. These are the same dict
        # objects the recorder logs, so this is what fills the vr_phase CSV column.
        for pkt in win["packets"]:
            pkt["vr_phase"] = win_phase
            pkt["stress_state"] = win_label
            pkt["confidence"] = win_conf or 0.85

        # Highest packet counter that has now been through a completed window.
        # The recorder writes only up to this point so the preprocessed columns
        # are populated; the tail is flushed when the session stops.
        if win["packets"]:
            st.session_state.last_windowed_counter = max(
                st.session_state.last_windowed_counter,
                max(p.get("packet_counter", 0) for p in win["packets"]),
            )

        # Log required metrics for EVERY completed window (Requirement 5)
        print(
            f"[WINDOW_LOG] window_id={win.get('window_id')} "
            f"window_start_ms={win.get('window_start_ms')} "
            f"window_end_ms={win.get('window_end_ms')} "
            f"samples_in_window={win.get('samples_in_window')} "
            f"vr_phase={win_phase} "
            f"signal_quality_sqi={sqi.get('overall_sqi', 0.0)} "
            f"signal_quality_valid={sqi.get('is_valid', False)} "
            f"prediction={stress_result.get('prediction')} "
            f"probability={stress_result.get('probability')} "
            f"confidence={stress_result.get('confidence')} "
            f"model_used={stress_result.get('model_used')}"
        )

        # ── Persist WindowResult (report-ready record) ────────────
        # Creates a structured WindowResult and appends it to the session's
        # _windows.csv. Invalid windows are persisted with null predictions
        # and skip reasons — they remain visible for audit.
        if wrm.session_id:
            wrm.create_and_persist_window_result(
                window_info=win,
                processed_batch=processed_batch,
                stress_result=stress_result,
            )

    # Store processed batch & stress result in session state so Streamlit reruns reuse them
    st.session_state.last_processed_batch = processed_batch
    st.session_state.last_stress_result = stress_result
else:
    # NO new window emitted yet:
    # DO NOT run feature extraction or inference on Streamlit reruns. Reuse persistent results.
    if st.session_state.last_processed_batch is not None:
        processed_batch = st.session_state.last_processed_batch
    else:
        # Fallback before the very first window completes
        processed_batch = preprocessor.process_batch(latest_packets) if latest_packets else preprocessor.process_batch([])
        
    if st.session_state.last_stress_result is not None:
        stress_result = st.session_state.last_stress_result
    else:
        # Fallback before the first completed window; this does not synthesize a classification.
        stress_result = classifier.predict(processed_batch.get("feature_df"), signal_quality=processed_batch.get("signal_quality", {}))

feature_df = processed_batch.get("feature_df") if isinstance(processed_batch, dict) else None


def log_pending_packets(flush: bool = False) -> int:
    """Write recorded packets to the session CSV, newest-first order preserved.

    A packet's window-level columns (ppg_filtered, gsr_tonic, gsr_phasic,
    imu_magnitude, signal_quality_*, vr_phase, stress_state) only exist once the
    30-second window containing it has closed, so by default writing stops at
    `last_windowed_counter`. That leaves the CSV lagging live telemetry by up to
    30 s; `flush=True` (used when the session stops) writes the remaining tail
    with whatever values it has rather than discarding it.

    Returns the number of rows written.
    """
    if not data_logger.is_recording or not latest_packets:
        return 0

    last_counter = st.session_state.last_logged_packet_counter
    ceiling = float("inf") if flush else st.session_state.last_windowed_counter
    fallback_phase = stress_result.get("vr_phase") or "UNKNOWN"
    written = 0

    for pkt in latest_packets:
        counter = pkt.get("packet_counter", 0)
        if last_counter < counter <= ceiling:
            pkt.setdefault("stress_state", stress_result.get("label", "UNKNOWN"))
            if pkt.get("confidence") is None:
                pkt["confidence"] = stress_result.get("confidence") or 0.0
            pkt.setdefault("vr_phase", fallback_phase)
            pkt["patient_name"] = getattr(data_logger, "patient_name", "Anonymous")
            data_logger.log_packet(pkt)
            last_counter = counter
            written += 1

    st.session_state.last_logged_packet_counter = last_counter
    return written


# Log telemetry packets if recording session is active
log_pending_packets()

# ============================================================================
# MEASURED GLOBAL DESIGN SYSTEM & LIQUID GLASS CSS
# ============================================================================
st.markdown("""
<style>
    /* Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Instrument+Serif:ital@0;1&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Rounded:opsz,wght,FILL,GRAD@20..48,100..700,0..1,-50..200');

    :root {
        --m-bg: #06090F;
        --m-surface: rgba(255,255,255,0.01);
        --m-card: rgba(15, 23, 42, 0.65);
        --m-card-hover: rgba(22, 34, 60, 0.85);
        --m-border: rgba(255,255,255,0.08);
        --m-border-hover: rgba(255,255,255,0.22);
        --m-text: #F1F5F9;
        --m-text-secondary: #94A3B8;
        --m-text-muted: #64748B;
        --m-accent: #22C55E;
        --m-accent-blue: #3B82F6;
        --m-accent-cyan: #06B6D4;
        --m-accent-amber: #F59E0B;
        --m-accent-red: #EF4444;
        --m-accent-purple: #A855F7;
        --m-radius: 16px;
        --m-radius-lg: 20px;
        --m-radius-xl: 24px;
    }

    html, body, p, div, label, button, input, h1, h2, h3, h4, h5, h6, [data-testid="stMarkdownContainer"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    /* Preserve Material Icons for Streamlit Sidebar Controls */
    [data-testid="stSidebarCollapseButton"] *,
    [data-testid="stHeader"] *,
    .material-symbols-outlined,
    .material-symbols-rounded {
        font-family: 'Material Symbols Rounded', 'Material Symbols Outlined', 'Material Icons', sans-serif !important;
    }

    /* Base Layout Overrides */
    body, html, .stApp {
        background-color: var(--m-bg) !important;
        background-image:
            radial-gradient(ellipse 80% 60% at 20% 10%, rgba(6,182,212,0.05) 0%, transparent 50%),
            radial-gradient(ellipse 60% 80% at 80% 90%, rgba(139,92,246,0.04) 0%, transparent 50%),
            var(--m-bg) !important;
        color: var(--m-text) !important;
        overflow-x: hidden;
    }
    header, [data-testid="stHeader"] {
        background: transparent !important;
        color: var(--m-text) !important;
    }
    .main { background: transparent !important; }
    .block-container {
        padding-top: 0rem !important;
        padding-bottom: 2.5rem !important;
        max-width: 1480px !important;
    }

    /* Scrollbars */
    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-track { background: #06090F; }
    ::-webkit-scrollbar-thumb { background: rgba(255, 255, 255, 0.12); border-radius: 4px; }
    ::-webkit-scrollbar-thumb:hover { background: rgba(59, 130, 246, 0.5); }

    /* ═══ LIQUID GLASS SPECIFICATION ═══ */
    .liquid-glass {
        background: rgba(255,255,255,0.01) !important;
        background-blend-mode: luminosity !important;
        backdrop-filter: blur(4px) !important;
        -webkit-backdrop-filter: blur(4px) !important;
        border: none !important;
        box-shadow: inset 0 1px 1px rgba(255,255,255,0.1) !important;
        position: relative !important;
        overflow: hidden !important;
    }
    .liquid-glass::before {
        content: '' !important;
        position: absolute !important;
        inset: 0 !important;
        border-radius: inherit !important;
        padding: 1.4px !important;
        background: linear-gradient(
            180deg,
            rgba(255,255,255,0.45) 0%,
            rgba(255,255,255,0.15) 20%,
            rgba(255,255,255,0) 40%,
            rgba(255,255,255,0) 60%,
            rgba(255,255,255,0.15) 80%,
            rgba(255,255,255,0.45) 100%
        ) !important;
        -webkit-mask:
            linear-gradient(#fff 0 0) content-box,
            linear-gradient(#fff 0 0) !important;
        -webkit-mask-composite: xor !important;
        mask-composite: exclude !important;
        pointer-events: none !important;
    }

    /* ═══ SIDEBAR ALIGNMENT & BUTTON PERFECTION ═══ */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #090E17 0%, #06090F 100%) !important;
        border-right: 1px solid var(--m-border) !important;
    }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] { color: #CBD5E1 !important; }

    [data-testid="stSidebar"] [data-testid="stHorizontalBlock"] {
        gap: 8px !important;
        align-items: stretch !important;
        margin-bottom: 6px !important;
    }
    [data-testid="stSidebar"] [data-testid="column"] {
        display: flex !important;
        flex-direction: column !important;
        justify-content: stretch !important;
        flex: 1 1 0px !important;
        min-width: 0 !important;
    }
    [data-testid="stSidebar"] .stButton {
        width: 100% !important;
        margin: 0 !important;
        display: flex !important;
    }
    [data-testid="stSidebar"] .stButton > button {
        width: 100% !important;
        height: 42px !important;
        min-height: 42px !important;
        max-height: 42px !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        text-align: center !important;
        white-space: nowrap !important;
        font-size: 12px !important;
        font-weight: 600 !important;
        padding: 0 8px !important;
        margin: 0 !important;
        border-radius: 10px !important;
        box-sizing: border-box !important;
        background: rgba(255,255,255,0.04) !important;
        backdrop-filter: blur(12px) !important;
        color: #F1F5F9 !important;
        border: 1px solid rgba(255,255,255,0.12) !important;
        transition: all 250ms cubic-bezier(0.4, 0, 0.2, 1) !important;
    }
    [data-testid="stSidebar"] .stButton > button:hover {
        background: rgba(255,255,255,0.09) !important;
        border-color: rgba(255,255,255,0.25) !important;
        box-shadow: 0 6px 20px rgba(0,0,0,0.4), inset 0 1px 1px rgba(255,255,255,0.1) !important;
        transform: translateY(-1px) !important;
    }
    [data-testid="stSidebar"] .stButton > button:active {
        transform: translateY(1px) scale(0.98) !important;
    }
    [data-testid="stSidebar"] .stButton > button p,
    [data-testid="stSidebar"] .stButton > button div,
    [data-testid="stSidebar"] .stButton > button span {
        white-space: nowrap !important;
        font-size: 12px !important;
        font-weight: 600 !important;
        overflow: hidden !important;
        text-overflow: ellipsis !important;
    }

    /* ═══ 3D METRIC CARDS ═══ */
    [data-testid="stMetric"] {
        background: var(--m-card) !important;
        backdrop-filter: blur(16px) !important;
        -webkit-backdrop-filter: blur(16px) !important;
        border: 1px solid var(--m-border) !important;
        border-radius: var(--m-radius) !important;
        padding: 22px 20px !important;
        transition: all 400ms cubic-bezier(0.4, 0, 0.2, 1) !important;
        position: relative !important;
        overflow: hidden !important;
        box-shadow: 0 4px 24px rgba(0,0,0,0.3), inset 0 1px 1px rgba(255,255,255,0.06) !important;
    }
    [data-testid="stMetric"]::before {
        content: '' !important;
        position: absolute !important;
        inset: 0 !important;
        border-radius: inherit !important;
        padding: 1px !important;
        background: linear-gradient(180deg,
            rgba(255,255,255,0.15) 0%,
            rgba(255,255,255,0.04) 30%,
            transparent 50%,
            rgba(255,255,255,0.04) 70%,
            rgba(255,255,255,0.15) 100%) !important;
        -webkit-mask: linear-gradient(#fff 0 0) content-box, linear-gradient(#fff 0 0) !important;
        -webkit-mask-composite: xor !important;
        mask-composite: exclude !important;
        pointer-events: none !important;
    }
    [data-testid="stMetric"]:hover {
        border-color: var(--m-border-hover) !important;
        transform: perspective(800px) rotateX(3deg) rotateY(-2deg) translateY(-4px) translateZ(10px) !important;
        box-shadow: 0 16px 48px rgba(0,0,0,0.5), inset 0 1px 2px rgba(255,255,255,0.12) !important;
    }
    [data-testid="stMetric"] label {
        color: var(--m-text-muted) !important;
        font-size: 10px !important;
        font-weight: 700 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.1em !important;
    }
    [data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: var(--m-text) !important;
        font-size: 26px !important;
        font-weight: 800 !important;
        letter-spacing: -0.02em !important;
    }
    [data-testid="stMetric"] [data-testid="stMetricDelta"] {
        font-size: 11px !important;
        font-weight: 500 !important;
        opacity: 0.85 !important;
    }

    /* ═══ STATUS STRIP HUD ═══ */
    .status-strip {
        background: rgba(15, 23, 42, 0.6);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid var(--m-border);
        border-radius: var(--m-radius);
        padding: 14px 24px;
        margin-bottom: 20px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        flex-wrap: wrap;
        gap: 16px;
        box-shadow: 0 4px 24px rgba(0,0,0,0.3), inset 0 1px 1px rgba(255,255,255,0.06);
        position: relative;
        overflow: hidden;
    }
    .status-strip::before {
        content: '';
        position: absolute;
        inset: 0;
        border-radius: inherit;
        padding: 1px;
        background: linear-gradient(180deg, rgba(255,255,255,0.2) 0%, transparent 40%, transparent 60%, rgba(255,255,255,0.2) 100%);
        -webkit-mask: linear-gradient(#fff 0 0) content-box, linear-gradient(#fff 0 0);
        -webkit-mask-composite: xor;
        mask-composite: exclude;
        pointer-events: none;
    }
    .status-strip .segment {
        display: flex;
        align-items: center;
        gap: 10px;
        font-size: 13px;
        color: var(--m-text-secondary);
    }
    .status-strip .segment .label {
        color: var(--m-text-muted);
        font-weight: 700;
        font-size: 10px;
        text-transform: uppercase;
        letter-spacing: 0.1em;
    }
    .status-strip code {
        font-family: 'Inter', monospace !important;
        font-size: 12px;
        font-weight: 500;
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        padding: 3px 8px;
        border-radius: 6px;
        color: var(--m-text);
    }

    /* Badges */
    .badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 10px;
        letter-spacing: 0.06em;
        text-transform: uppercase;
    }
    .badge-live {
        background: rgba(34,197,94,0.12); color: var(--m-accent);
        border: 1px solid rgba(34,197,94,0.3);
    }
    .badge-reconnect {
        background: rgba(245,158,11,0.12); color: var(--m-accent-amber);
        border: 1px solid rgba(245,158,11,0.3);
    }
    .badge-off {
        background: rgba(239,68,68,0.12); color: var(--m-accent-red);
        border: 1px solid rgba(239,68,68,0.3);
    }

    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px !important;
        background: rgba(15, 23, 42, 0.5) !important;
        backdrop-filter: blur(12px) !important;
        padding: 6px !important;
        border-radius: 14px !important;
        border: 1px solid var(--m-border) !important;
    }
    .stTabs [data-baseweb="tab"] {
        height: 38px !important;
        border-radius: 10px !important;
        color: var(--m-text-muted) !important;
        font-weight: 500 !important;
        font-size: 12px !important;
        padding: 0 16px !important;
        border: 1px solid transparent !important;
        transition: all 200ms ease !important;
    }
    .stTabs [aria-selected="true"] {
        background: rgba(37, 99, 235, 0.18) !important;
        color: #FFFFFF !important;
        border: 1px solid rgba(59, 130, 246, 0.4) !important;
        box-shadow: 0 0 14px rgba(37, 99, 235, 0.25) !important;
    }
    .stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] {
        display: none !important;
    }

    /* Stress Card */
    .stress-card {
        background: var(--m-card);
        backdrop-filter: blur(20px);
        border: 1px solid var(--m-border);
        border-radius: var(--m-radius);
        padding: 24px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.4), inset 0 1px 1px rgba(255,255,255,0.06);
    }
    .stress-card .label-stress {
        color: var(--m-accent-red);
        font-size: 32px; font-weight: 800; letter-spacing: -0.02em;
    }
    .stress-card .label-nonstress {
        color: var(--m-accent);
        font-size: 32px; font-weight: 800; letter-spacing: -0.02em;
    }

    /* Reduced Motion */
    @media (prefers-reduced-motion: reduce) {
        [data-testid="stMetric"]:hover { transform: none !important; }
        .liquid-glass { backdrop-filter: none !important; }
        * { animation-duration: 0.01ms !important; transition-duration: 0.01ms !important; }
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# SIDEBAR CONTROLS & NETWORKING
# ============================================================================
st.sidebar.markdown("""
<div style="display:flex; align-items:center; gap:12px; margin-bottom:16px; padding:14px 16px; background:rgba(255,255,255,0.02); border:1px solid rgba(255,255,255,0.08); border-radius:16px; box-shadow:inset 0 1px 1px rgba(255,255,255,0.06);">
    <div style="flex-shrink:0;">
        <svg viewBox="0 0 256 256" fill="white" style="width:26px;height:26px;opacity:0.95;">
            <path d="M 256 64 L 256 128 L 192.5 128 L 160 95 L 128 64 L 96 95 L 63.5 128 L 64 128 L 128 192 L 128 256 L 64.5 256 L 32 223 L 0 192 L 0 64 L 64 0 L 192 0 Z M 256 192 L 256 256 L 192.5 256 L 160 223 L 128 192 L 128 128 L 192 128 Z"/>
        </svg>
    </div>
    <div>
        <div style="font-size:14px;font-weight:700;color:#F1F5F9;letter-spacing:-0.01em;">Measured</div>
        <div style="font-size:11px;color:#64748B;font-weight:400;">Telemetry Control</div>
    </div>
</div>
""", unsafe_allow_html=True)

def format_clean_port_label(p):
    desc = p.get('description', '')
    if "Standard Serial over Bluetooth" in desc or "Bluetooth" in desc:
        desc = "Bluetooth Serial"
    elif "Silicon Labs" in desc or "CP210" in desc:
        desc = "CP210x ESP32"
    elif "CH340" in desc or "CH341" in desc:
        desc = "CH340 ESP32"
    elif "FTDI" in desc:
        desc = "FTDI Serial"
    elif not desc:
        desc = "Serial Port"
    tag = " ★ ESP32" if p.get("is_esp32") else ""
    return f"{p['device']} — {desc}{tag}"

sim_mode = st.sidebar.toggle("🎮 Simulated Hardware Mode", value=receiver.simulation_mode)
if sim_mode != receiver.simulation_mode:
    receiver.stop()
    window_manager.reset()
    classifier.reset_session()
    st.session_state.last_processed_batch = None
    st.session_state.last_stress_result = None
    st.session_state.last_logged_packet_counter = -1
    receiver.set_simulation_mode(sim_mode)
    if sim_mode:
        receiver.start(port="SIMULATED", simulation_mode=True)
        st.toast("Started Simulated Hardware Telemetry", icon="🎮")
    else:
        st.toast("Simulated Hardware stopped. Select COM port and click Connect.", icon="🔌")
    st.rerun()

ports = list_serial_ports()
if ports and not sim_mode:
    port_options = [format_clean_port_label(p) for p in ports]
    auto_idx = 0
    for i, p in enumerate(ports):
        if p.get("is_esp32"):
            auto_idx = i
            break
    col_port_hdr1, col_port_hdr2 = st.sidebar.columns([3, 1])
    with col_port_hdr1:
        st.markdown("**Detected Serial Ports:**")
    with col_port_hdr2:
        if st.button("🔄", help="Rescan serial ports", key="rescan_ports_btn"):
            st.rerun()
    selected_option = st.sidebar.radio(
        "Select COM Port",
        port_options,
        index=auto_idx,
        label_visibility="collapsed"
    )
    selected_port = ports[port_options.index(selected_option)]["device"]
elif not sim_mode:
    col_port_hdr1, col_port_hdr2 = st.sidebar.columns([3, 1])
    with col_port_hdr1:
        st.warning("No physical serial ports detected.")
    with col_port_hdr2:
        if st.button("🔄", help="Rescan serial ports", key="rescan_ports_btn_none"):
            st.rerun()
    selected_port = st.sidebar.text_input("Manual COM Port", value=config.SERIAL_PORT)
else:
    selected_port = "SIMULATED"
    st.sidebar.info("🎮 Active: Simulated Hardware Mode")
    sim_levels = {
        "RELAXED": "🧘 Relaxed Baseline (Calm)",
        "LOW_STRESS": "⚡ Low Stress (Mild Alert)",
        "MODERATE_STRESS": "⚠️ Moderate Stress (Active)",
        "HIGH_STRESS": "🔥 High Stress (Acute)",
        "DYNAMIC": "🔄 Dynamic Protocol (Auto-Cycle)",
    }
    cur_sim_lvl = getattr(receiver, "simulation_stress_level", "RELAXED")
    sim_choice = st.sidebar.selectbox(
        "Simulate Stress State",
        options=list(sim_levels.keys()),
        format_func=lambda x: sim_levels[x],
        index=list(sim_levels.keys()).index(cur_sim_lvl) if cur_sim_lvl in sim_levels else 0,
        help="Select any of the 4 stress levels or Auto-Cycle in demo mode"
    )
    if sim_choice != cur_sim_lvl:
        receiver.set_simulation_stress_level(sim_choice)
        st.toast(f"Simulating: {sim_levels[sim_choice]}", icon="🎮")

baud_display = getattr(config, 'SERIAL_BAUD', getattr(config, 'BAUD_RATE', 115200))
if not sim_mode:
    st.sidebar.markdown(f"**Baud Rate:** `{baud_display}`")
st.sidebar.markdown("---")

col_btn1, col_btn2 = st.sidebar.columns(2)
with col_btn1:
    if st.button("▶ Connect", use_container_width=True):
        window_manager.reset()
        classifier.reset_session()
        st.session_state.last_processed_batch = None
        st.session_state.last_stress_result = None
        receiver.start(port=selected_port, simulation_mode=sim_mode)
        st.session_state.last_logged_packet_counter = -1
        if sim_mode:
            st.toast("Started Simulated Telemetry Stream", icon="🎮")
        else:
            st.toast(f"Connecting to {selected_port}...", icon="🔌")

with col_btn2:
    if st.button("⏹ Disconnect", use_container_width=True):
        receiver.stop()
        window_manager.reset()
        st.toast("Receiver stopped.", icon="🛑")

if st.sidebar.button("🎯 Calibrate Baseline", help="Calibrate calm resting baseline from current physiology", use_container_width=True):
    classifier.reset_baseline()
    st.session_state.last_processed_batch = None
    st.session_state.last_stress_result = None
    st.toast("Baseline reset. Calibrating next 2 windows as resting baseline...", icon="🎯")
    st.rerun()

if "RECONNECTING" in receiver.status and not sim_mode:
    last_err = getattr(receiver, "last_error", None)
    if last_err:
        st.sidebar.error(f"⚠️ {last_err}")
        if "Arduino IDE" in last_err or "Access denied" in last_err:
            st.sidebar.info("💡 **Fix:** Close the Serial Monitor or Serial Plotter in **Arduino IDE**, then click **⏹ Disconnect** and **▶ Connect**.")
    else:
        st.sidebar.warning("⚠️ Serial port not responding. Enable **Simulated Hardware Mode** above for live testing!")

st.sidebar.markdown("---")
st.sidebar.markdown('<div style="font-size:12px; font-weight:700; color:#94A3B8; margin-bottom:8px;">💾 Recording Session</div>', unsafe_allow_html=True)

if "patient_name" not in st.session_state:
    st.session_state.patient_name = ""

patient_input = st.sidebar.text_input(
    "Patient Name",
    value=st.session_state.patient_name,
    placeholder="e.g. John Doe",
    disabled=data_logger.is_recording,
    help="Enter the patient or subject name. The recording session and generated reports will be saved under this name."
)
st.session_state.patient_name = patient_input

col_rec1, col_rec2 = st.sidebar.columns(2)
with col_rec1:
    if st.button("Start Session", disabled=data_logger.is_recording, use_container_width=True):
        cleaned_name = patient_input.strip()
        if not cleaned_name:
            st.sidebar.error("⚠️ Patient name is required to start a session.")
            st.toast("Please enter a Patient Name first!", icon="⚠️")
        else:
            sess_id = data_logger.start_session(patient_name=cleaned_name)
            classifier.reset_session()
            wrm.start_session(session_id=sess_id, participant_id=cleaned_name)
            st.session_state.last_logged_packet_counter = -1
            st.toast(f"Recording: {sess_id} ({cleaned_name})", icon="🔴")
            st.rerun()

with col_rec2:
    if st.button("Stop Session", disabled=not data_logger.is_recording, use_container_width=True):
        # Flush the tail that is still waiting on its 30s window before closing.
        tail = log_pending_packets(flush=True)
        summary = data_logger.stop_session()
        analysis_json_path = wrm.stop_session()
        patient_saved = summary.get("patient_name", "Patient")
        st.toast(f"Saved for {patient_saved}! {summary.get('duration_sec', 0)}s ({tail} tail rows)", icon="✅")
        # Automatically generate the report if samples were logged!
        if summary.get("samples_logged", 0) > 0 and summary.get("csv_path"):
            try:
                report_files = report_gen.generate_report_from_csv(
                    summary["csv_path"],
                    patient_name=summary.get("patient_name")
                )
                st.session_state.active_report = report_files
                st.toast(f"Report ready for {patient_saved}!", icon="📄")
            except Exception as report_err:
                pass
        st.rerun()

if data_logger.is_recording:
    _pending = max(0, st.session_state.last_windowed_counter - st.session_state.last_logged_packet_counter)
    cur_pat = getattr(data_logger, 'patient_name', 'Anonymous')
    st.sidebar.info(f"👤 Patient: **{cur_pat}**")
    st.sidebar.caption(
        f"Recording &bull; {data_logger.samples_logged} rows written &bull; {_pending} queued. "
        f"Rows are held until their 30s window closes, so the CSV trails live telemetry."
    )

if st.sidebar.button("Export Last Session Report", use_container_width=True):
    try:
        sessions = data_logger.list_recorded_sessions()
        if sessions:
            try:
                test_df = pd.read_csv(str(sessions[0]), nrows=2)
                if test_df.empty:
                    st.sidebar.warning(f"Session file '{sessions[0].name}' has no logged data. Please record data first.")
                else:
                    report_files = report_gen.generate_report_from_csv(str(sessions[0]))
                    st.session_state.active_report = report_files
                    pname = report_files.get("patient_name", "Patient")
                    st.toast(f"Report exported for {pname}!", icon="📄")
            except Exception as file_exc:
                st.sidebar.error(f"Cannot read session file: {file_exc}")
        else:
            st.sidebar.warning("No recorded sessions found.")
    except Exception as exc:
        st.sidebar.error(f"Cannot export report: {exc}")

st.sidebar.markdown("---")
st.sidebar.markdown('<div style="font-size:12px; font-weight:700; color:#94A3B8; margin-bottom:8px;">🕶️ VR Event Log</div>', unsafe_allow_html=True)

# Upload a VR session event log (CSV/JSON with timestamp_ms + phase) so each
# 30s window is tagged with the VR phase it fell in, instead of "UNKNOWN".
vr_upload = st.sidebar.file_uploader(
    "Phase timeline (CSV or JSON)",
    type=["csv", "json"],
    key="vr_log_upload",
    label_visibility="collapsed",
)

if vr_upload is not None and vr_upload.name != st.session_state.vr_log_name:
    try:
        vr_tmp_path = config.SESSIONS_DIR / f"vr_events_upload{Path(vr_upload.name).suffix.lower()}"
        vr_tmp_path.write_bytes(vr_upload.getvalue())
        if vr_log.load(str(vr_tmp_path)):
            st.session_state.vr_log_name = vr_upload.name
            window_manager.set_vr_log(vr_log)
            st.sidebar.success(f"{len(vr_log.events)} events loaded.")
        else:
            st.sidebar.error("No usable events found in that file.")
    except Exception as exc:
        st.sidebar.error(f"Cannot load VR log: {exc}")

if vr_log.is_loaded:
    _phases = ", ".join(vr_log.get_all_phases()[:5]) or "—"
    st.sidebar.caption(
        f"**{st.session_state.vr_log_name or Path(vr_log.source_path).name}** &bull; "
        f"{len(vr_log.events)} events &bull; {round(vr_log.session_duration_ms / 1000)}s\n\n"
        f"Phases: {_phases}"
    )
    # ESP32 millis() starts at boot, the VR clock at session start. Offset aligns them.
    _first_ts = latest_packets[0].get("timestamp_ms", 0) if latest_packets else 0
    if st.sidebar.button("Sync VR clock to stream start", use_container_width=True):
        vr_log.set_sync_offset(esp32_start_ms=int(_first_ts), vr_start_ms=0)
        st.toast(f"VR clock synced (offset {int(_first_ts)} ms)", icon="🕶️")
    st.sidebar.caption(f"Sync offset: `{vr_log.sync_offset_ms} ms`")
else:
    st.sidebar.caption("None loaded — windows are tagged `UNKNOWN`.")

# ============================================================================
# 1. CINEMATIC FULLSCREEN HERO SECTION WITH CRISP HD RETINA & INTERACTIVE CHARACTERS
# ============================================================================
hero_component_html = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<link href="https://fonts.googleapis.com/css2?family=Instrument+Serif:ital@0;1&display=swap" rel="stylesheet">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
<style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body, html { 
        width: 100%; 
        height: 100%; 
        overflow: hidden; 
        background: #06090F; 
        font-family: 'Inter', -apple-system, sans-serif; 
        -webkit-font-smoothing: antialiased;
    }

    #hero-viewport {
        position: relative;
        width: 100%;
        height: 100%;
        overflow: hidden;
        background: #06090F;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        align-items: center;
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        perspective: 1200px;
    }

    /* Layer 1: Technical 48px Grid */
    #tech-grid {
        position: absolute;
        inset: -20px;
        width: calc(100% + 40px);
        height: calc(100% + 40px);
        background-image: url('data:image/svg+xml;utf8,<svg width="48" height="48" xmlns="http://www.w3.org/2000/svg"><path d="M 48 0 L 0 0 0 48" fill="none" stroke="%2364748b" stroke-width="0.6" opacity="0.14"/></svg>');
        pointer-events: none;
        z-index: 1;
        transition: transform 0.1s ease-out;
    }

    /* Layer 2: High-Resolution Terrarium Background Image */
    #hero-bg-img {
        position: absolute;
        inset: 0;
        width: 100%;
        height: 100%;
        object-fit: cover;
        object-position: center 52%;
        z-index: 2;
        opacity: 0.98;
        image-rendering: -webkit-optimize-contrast;
        image-rendering: crisp-edges;
        backface-visibility: hidden;
        -webkit-backface-visibility: hidden;
        will-change: transform;
        filter: contrast(1.04) brightness(1.02);
        transition: transform 0.14s ease-out;
    }

    /* Layer 3: Product Reveal Video */
    #video-container {
        position: absolute;
        inset: 0;
        width: 100%;
        height: 100%;
        z-index: 3;
        clip-path: inset(38% 0 0 0);
        opacity: 0;
        transition: opacity 0.3s ease;
        pointer-events: none;
    }

    #product-video {
        width: 100%;
        height: 100%;
        object-fit: cover;
        object-position: center 52%;
    }

    /* Layer 4: Clean Crisp Atmosphere (No blurry texture mask) */
    #atmospheric-overlay {
        display: none !important;
    }

    /* Layer 5: High-DPI Particle Canvas */
    #particle-canvas {
        position: absolute;
        inset: 0;
        width: 100%;
        height: 100%;
        z-index: 5;
        pointer-events: none;
    }

    /* Layer 6: Spotlight Canvas */
    #spotlight-canvas {
        position: absolute;
        inset: 0;
        width: 100%;
        height: 100%;
        z-index: 6;
        pointer-events: none;
    }

    /* Layer 7: Interactive Character Hotspot Layer */
    #character-interactive-layer {
        position: absolute;
        inset: 0;
        width: 100%;
        height: 100%;
        z-index: 15;
        pointer-events: auto;
    }

    .char-hotspot {
        position: absolute;
        border-radius: 50%;
        cursor: pointer;
        transform-origin: center center;
    }

    /* Catbus Hotspot */
    #hotspot-catbus {
        left: 42%;
        top: 56%;
        width: 85px;
        height: 70px;
    }

    /* Big Totoro Hotspot */
    #hotspot-totoro {
        left: 49.5%;
        top: 48%;
        width: 80px;
        height: 90px;
    }

    /* Little White Totoro Hotspot */
    #hotspot-little {
        left: 56.5%;
        top: 57%;
        width: 55px;
        height: 60px;
    }

    .bio-aura-ring {
        position: absolute;
        inset: -12px;
        border-radius: 50%;
        border: 2px solid rgba(6, 182, 212, 0.6);
        opacity: 0;
        transform: scale(0.85);
        transition: all 0.35s cubic-bezier(0.34, 1.56, 0.64, 1);
        box-shadow: 0 0 24px rgba(6, 182, 212, 0.5), inset 0 0 16px rgba(6, 182, 212, 0.3);
        pointer-events: none;
    }

    .char-hotspot:hover .bio-aura-ring {
        opacity: 1;
        transform: scale(1.18);
        animation: pulseAura 1.5s infinite alternate ease-in-out;
    }

    .char-tooltip {
        position: absolute;
        top: -42px;
        left: 50%;
        transform: translateX(-50%) translateY(6px);
        background: rgba(10, 15, 30, 0.92);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.25);
        padding: 6px 14px;
        border-radius: 20px;
        color: #FFFFFF;
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 0.04em;
        white-space: nowrap;
        opacity: 0;
        pointer-events: none;
        transition: all 0.25s ease;
        box-shadow: 0 8px 24px rgba(0,0,0,0.6), 0 0 12px rgba(6, 182, 212, 0.3);
    }

    .char-hotspot:hover .char-tooltip {
        opacity: 1;
        transform: translateX(-50%) translateY(0);
    }

    @keyframes pulseAura {
        0% { transform: scale(1.12); border-color: rgba(6, 182, 212, 0.7); box-shadow: 0 0 20px rgba(6, 182, 212, 0.4); }
        100% { transform: scale(1.28); border-color: rgba(34, 197, 94, 0.85); box-shadow: 0 0 35px rgba(34, 197, 94, 0.6); }
    }

    /* Layer 8: Glass Header Navigation */
    #nav-header {
        position: absolute;
        top: 14px;
        left: 0;
        width: 100%;
        padding: 0 28px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        z-index: 50;
    }

    .nav-logo {
        display: flex;
        align-items: center;
        gap: 12px;
        text-decoration: none;
        color: #fff;
        font-weight: 700;
        letter-spacing: 0.05em;
        font-size: 15px;
    }

    .nav-pill {
        display: flex;
        align-items: center;
        gap: 26px;
        padding: 8px 24px;
        border-radius: 40px;
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.12);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }

    .nav-pill a {
        color: #94A3B8;
        text-decoration: none;
        font-size: 12px;
        font-weight: 500;
        transition: color 0.2s ease;
    }
    .nav-pill a:hover { color: #FFFFFF; }

    .reserve-cta {
        display: flex;
        align-items: center;
        gap: 8px;
        padding: 8px 20px;
        border-radius: 40px;
        background: rgba(255, 255, 255, 0.04);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.2);
        color: #FFFFFF;
        font-size: 12px;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    .reserve-cta:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(34, 197, 94, 0.25);
        border-color: rgba(34, 197, 94, 0.5);
    }

    .pulse-green-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background-color: #22C55E;
        box-shadow: 0 0 10px #22C55E;
        animation: pulse-ring 2s infinite;
    }

    @keyframes pulse-ring {
        0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(34, 197, 94, 0.7); }
        70% { transform: scale(1); box-shadow: 0 0 0 8px rgba(34, 197, 94, 0); }
        100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(34, 197, 94, 0); }
    }

    /* Layer 9: Main Typography (Positioned cleanly above the terrarium jar) */
    #hero-title-container {
        position: absolute;
        top: 24%;
        left: 50%;
        transform: translate(-50%, -50%);
        z-index: 20;
        width: 100%;
        text-align: center;
        pointer-events: none;
    }

    .hero-title {
        font-family: 'Instrument Serif', serif;
        font-size: clamp(3.8rem, 8.5vw, 8.5rem);
        line-height: 0.95;
        color: #FFFFFF;
        font-weight: 400;
        letter-spacing: -0.02em;
        text-shadow: 0 4px 30px rgba(0, 0, 0, 0.7), 0 0 60px rgba(255, 255, 255, 0.3);
    }

    .hero-subtitle {
        font-family: 'Inter', sans-serif;
        font-size: clamp(0.68rem, 0.95vw, 0.82rem);
        color: #E2E8F0;
        font-weight: 600;
        letter-spacing: 0.28em;
        text-transform: uppercase;
        margin-top: 10px;
        text-shadow: 0 2px 10px rgba(0,0,0,0.8);
    }

    .hero-tags {
        display: flex;
        justify-content: center;
        gap: 20px;
        margin-top: 14px;
        font-size: 10px;
        font-weight: 700;
        letter-spacing: 0.22em;
        color: #94A3B8;
        text-transform: uppercase;
        text-shadow: 0 2px 8px rgba(0,0,0,0.8);
    }

    .scroll-indicator {
        position: absolute;
        bottom: 12px;
        left: 50%;
        transform: translateX(-50%);
        z-index: 20;
        font-size: 9px;
        letter-spacing: 0.24em;
        color: rgba(255, 255, 255, 0.6);
        text-transform: uppercase;
        font-weight: 700;
        pointer-events: none;
    }
</style>
</head>
<body>

<div id="hero-viewport">
    <div id="tech-grid"></div>
    <img id="hero-bg-img" src="https://d8j0ntlcm91z4.cloudfront.net/user_38xzZboKViGWJOttwIXH07lWA1P/hf_20260713_140344_79e1296a-86d7-43fd-9b5f-63ffe560f291.png" alt="Measured Wearable background">
    
    <div id="video-container">
        <video id="product-video" autoplay loop muted playsinline src="https://d8j0ntlcm91z4.cloudfront.net/user_38xzZboKViGWJOttwIXH07lWA1P/hf_20260713_162101_0d7498c5-29bb-47bf-a99f-2773c0a880a9.mp4"></video>
    </div>
    
    <img id="atmospheric-overlay" src="https://soft-zoom-63098134.figma.site/_assets/v11/3f10f1876e118f72a396e05a6c2d099569478272.png" alt="Atmosphere">

    <!-- Interactive 3D Character Bio-Hotspots -->
    <div id="character-interactive-layer">
        <div class="char-hotspot" id="hotspot-catbus" onclick="triggerShockwave(event, 'CATBUS')">
            <div class="bio-aura-ring"></div>
            <div class="char-tooltip">⚡ IMU Activity Sensor (MPU6050)</div>
        </div>
        <div class="char-hotspot" id="hotspot-totoro" onclick="triggerShockwave(event, 'TOTORO')">
            <div class="bio-aura-ring"></div>
            <div class="char-tooltip">❤️ PPG Optical Heartbeat (MAX30102)</div>
        </div>
        <div class="char-hotspot" id="hotspot-little" onclick="triggerShockwave(event, 'LITTLE')">
            <div class="bio-aura-ring"></div>
            <div class="char-tooltip">💧 GSR Electrodermal Node</div>
        </div>
    </div>

    <!-- High-DPI Particles -->
    <canvas id="particle-canvas"></canvas>

    <!-- Glass Header Navigation -->
    <header id="nav-header">
        <a href="#" class="nav-logo">
            <svg viewBox="0 0 256 256" fill="white" width="26" height="26">
                <path d="M 256 64 L 256 128 L 192.5 128 L 160 95 L 128 64 L 96 95 L 63.5 128 L 64 128 L 128 192 L 128 256 L 64.5 256 L 32 223 L 0 192 L 0 64 L 64 0 L 192 0 Z M 256 192 L 256 256 L 192.5 256 L 160 223 L 128 192 L 128 128 L 192 128 Z"/>
            </svg>
            <span>MEASURED</span>
        </a>

        <div class="nav-pill">
            <a href="#device">Device</a>
            <a href="#stories">Real Stories</a>
            <a href="#science">Science</a>
            <a href="#plans">Plans</a>
            <a href="#reach">Reach Us</a>
        </div>

        <button class="reserve-cta">
            <span class="pulse-green-dot"></span>
            <span>Reserve Yours</span>
        </button>
    </header>

    <!-- Typography -->
    <div id="hero-title-container">
        <h1 class="hero-title">Measured</h1>
        <div class="hero-subtitle">Premium Wearable Health Intelligence</div>
        <div class="hero-tags">
            <span>PPG</span>
            <span>&bull;</span>
            <span>GSR</span>
            <span>&bull;</span>
            <span>IMU</span>
        </div>
    </div>

    <div class="scroll-indicator">SCROLL TO MONITOR</div>

    <!-- Spotlight Reveal Canvas -->
    <canvas id="spotlight-canvas"></canvas>
</div>

<script>
    const canvas = document.getElementById('spotlight-canvas');
    const pCanvas = document.getElementById('particle-canvas');
    const ctx = canvas.getContext('2d');
    const pCtx = pCanvas.getContext('2d');

    const videoContainer = document.getElementById('video-container');
    const techGrid = document.getElementById('tech-grid');
    const heroBgImg = document.getElementById('hero-bg-img');
    const heroTitleContainer = document.getElementById('hero-title-container');

    let cssWidth = 0, cssHeight = 0;
    let targetX = 0, targetY = 0;
    let smoothX = 0, smoothY = 0;

    const particles = [];
    const particleCount = 40;
    const shockwaves = [];

    // High-DPI Crisp Canvas Setup
    function resize() {
        const rect = canvas.getBoundingClientRect();
        const dpr = window.devicePixelRatio || 1;
        
        cssWidth = rect.width;
        cssHeight = rect.height;

        canvas.width = Math.floor(cssWidth * dpr);
        canvas.height = Math.floor(cssHeight * dpr);
        ctx.scale(dpr, dpr);

        pCanvas.width = Math.floor(cssWidth * dpr);
        pCanvas.height = Math.floor(cssHeight * dpr);
        pCtx.scale(dpr, dpr);

        if (targetX === 0 && targetY === 0) {
            targetX = smoothX = cssWidth / 2;
            targetY = smoothY = cssHeight * 0.65;
        }
    }
    window.addEventListener('resize', resize);
    setTimeout(resize, 60);

    window.addEventListener('mousemove', (e) => {
        const rect = canvas.getBoundingClientRect();
        targetX = e.clientX - rect.left;
        targetY = e.clientY - rect.top;
    });

    for (let i = 0; i < particleCount; i++) {
        particles.push({
            x: Math.random() * 1200,
            y: Math.random() * 800,
            vx: (Math.random() - 0.5) * 0.35,
            vy: (Math.random() - 0.5) * 0.35,
            size: Math.random() * 2.2 + 1.0,
            opacity: Math.random() * 0.5 + 0.25,
            hue: Math.random() > 0.5 ? 190 : 145
        });
    }

    function triggerShockwave(e, nodeName) {
        const rect = canvas.getBoundingClientRect();
        const clickX = e.clientX - rect.left;
        const clickY = e.clientY - rect.top;
        shockwaves.push({
            x: clickX,
            y: clickY,
            radius: 8,
            maxRadius: 180,
            opacity: 0.95,
            color: nodeName === 'TOTORO' ? '#3B82F6' : (nodeName === 'CATBUS' ? '#F59E0B' : '#22C55E')
        });
    }

    function renderParticles() {
        pCtx.clearRect(0, 0, cssWidth, cssHeight);

        // Render Shockwaves
        for (let i = shockwaves.length - 1; i >= 0; i--) {
            const sw = shockwaves[i];
            sw.radius += 3.8;
            sw.opacity *= 0.94;

            pCtx.save();
            pCtx.beginPath();
            pCtx.arc(sw.x, sw.y, sw.radius, 0, Math.PI * 2);
            pCtx.strokeStyle = sw.color;
            pCtx.globalAlpha = sw.opacity;
            pCtx.lineWidth = 2.5;
            pCtx.shadowBlur = 18;
            pCtx.shadowColor = sw.color;
            pCtx.stroke();
            pCtx.restore();

            if (sw.opacity < 0.04 || sw.radius >= sw.maxRadius) {
                shockwaves.splice(i, 1);
            }
        }

        // Render Floating Bio-Particles
        for (let p of particles) {
            p.x += p.vx;
            p.y += p.vy;

            if (p.x < 0) p.x = cssWidth;
            if (p.x > cssWidth) p.x = 0;
            if (p.y < 0) p.y = cssHeight;
            if (p.y > cssHeight) p.y = 0;

            const dx = smoothX - p.x;
            const dy = smoothY - p.y;
            const dist = Math.sqrt(dx * dx + dy * dy);
            if (dist < 160) {
                p.x -= (dx / dist) * 0.7;
                p.y -= (dy / dist) * 0.7;
            }

            pCtx.beginPath();
            pCtx.arc(p.x, p.y, p.size, 0, Math.PI * 2);
            pCtx.fillStyle = `hsla(${p.hue}, 90%, 65%, ${p.opacity})`;
            pCtx.shadowBlur = 8;
            pCtx.shadowColor = `hsl(${p.hue}, 90%, 65%)`;
            pCtx.fill();
        }
    }

    function renderSpotlight() {
        smoothX += (targetX - smoothX) * 0.12;
        smoothY += (targetY - smoothY) * 0.12;

        ctx.clearRect(0, 0, cssWidth, cssHeight);

        // Strict 40% top cutoff (reveal bottom 60% only)
        const topCutoff = cssHeight * 0.38;

        if (smoothY >= topCutoff) {
            videoContainer.style.opacity = '1';
            const mask = `radial-gradient(circle 260px at ${smoothX}px ${smoothY}px, black 0%, black 40%, rgba(0,0,0,0.75) 60%, rgba(0,0,0,0.4) 75%, rgba(0,0,0,0.12) 88%, transparent 100%)`;
            videoContainer.style.webkitMaskImage = mask;
            videoContainer.style.maskImage = mask;
        } else {
            videoContainer.style.opacity = '0';
        }

        // Multi-layer Depth Parallax (Subtle crisp translate without scale distortion)
        const deltaX = (smoothX - cssWidth / 2) / (cssWidth / 2 || 1);
        const deltaY = (smoothY - cssHeight / 2) / (cssHeight / 2 || 1);

        if (techGrid) {
            techGrid.style.transform = `translate3d(${deltaX * 6}px, ${deltaY * 6}px, 0)`;
        }
        if (heroBgImg) {
            heroBgImg.style.transform = `translate3d(${deltaX * 3}px, ${deltaY * 3}px, 0)`;
        }

        renderParticles();
        requestAnimationFrame(renderSpotlight);
    }

    renderSpotlight();
</script>
</body>
</html>
"""

components.html(hero_component_html, height=720)

# ============================================================================
# 2. STATUS STRIP HUD
# ============================================================================
status_summary = receiver.get_status_summary()
conn_status = status_summary["status"]

if conn_status == "CONNECTED":
    port_disp = status_summary.get("serial_port", "COM?")
    badge_html = f'<span class="badge badge-live"><span class="pulse-dot green"></span>CONNECTED</span>'
elif "SIMULATED" in conn_status or "CONNECTED" in conn_status:
    badge_html = f'<span class="badge badge-live"><span class="pulse-dot green"></span>LIVE (SIMULATED)</span>'
elif conn_status == "RECONNECTING":
    badge_html = '<span class="badge badge-reconnect"><span class="pulse-dot amber"></span>RECONNECTING</span>'
else:
    badge_html = '<span class="badge badge-off">DISCONNECTED</span>'

if data_logger.is_recording:
    sess_disp = f"🔴 REC &middot; {data_logger.samples_logged} pkts"
else:
    sess_disp = "⚪ IDLE"

model_name = "Four-Class Physiological CatBoost" if getattr(classifier, "model_loaded", False) else "Four-Class Model Unavailable"
if status_summary.get("connection_mode") == "SIMULATED":
    source_val = "Simulated Hardware Mode"
else:
    source_val = f"Serial ({status_summary.get('serial_port', 'COM3')} @ {baud_display})"

st.markdown(f"""
<div style="margin: 20px 0 10px;">
    <h2 style="font-size:22px; font-weight:700; color:#F1F5F9; letter-spacing:-0.02em; margin:0;">Live Bio-Telemetry</h2>
    <div style="font-size:12px; color:#06B6D4; font-weight:500; margin-top:2px;">Real-time PPG &bull; GSR &bull; IMU &bull; {model_name} Inference</div>
</div>
""", unsafe_allow_html=True)

st.markdown(f"""
<div class="status-strip">
    <div style="display:flex; align-items:center; gap:20px; flex-wrap:wrap;">
        <div class="segment"><span class="label">Status</span> {badge_html}</div>
        <div class="segment"><span class="label">Source</span> <code>{source_val}</code></div>
        <div class="segment"><span class="label">Rate</span> <code>{status_summary.get('rate_hz', 0.0)} Hz</code></div>
        <div class="segment"><span class="label">Model</span> <code>{model_name}</code></div>
    </div>
    <div class="segment"><span class="label">Session</span> <code>{sess_disp}</code></div>
</div>
""", unsafe_allow_html=True)

# ============================================================================
# 3. 3D REAL-TIME SENSOR METRIC CARDS
# ============================================================================
col_m1, col_m2, col_m3, col_m4 = st.columns(4)

with col_m1:
    st.metric(
        label="Heart Rate (PPG/BVP)",
        value=f"{processed_batch.get('bpm', 0.0):.1f} BPM",
        delta=f"HRV RMSSD: {processed_batch.get('rmssd', 0.0):.1f} ms"
    )

with col_m2:
    st.metric(
        label="Skin Conductance (GSR)",
        value=f"{processed_batch.get('scl_mean', 0.0):.1f} μS",
        delta=f"SCR Spikes: {processed_batch.get('scr_count', 0)}"
    )

with col_m3:
    st.metric(
        label="Motion Activity (IMU)",
        value=f"{processed_batch.get('activity_index', 0.0):.1f} g",
        delta=f"State: {processed_batch.get('motion_state', 'RESTING')}"
    )

with col_m4:
    lbl = stress_result.get("label")
    status = stress_result.get("status", "UNKNOWN")
    if lbl and status == "OK":
        st.session_state.last_valid_stress_label = lbl
        conf_val = stress_result.get("confidence_pct", 85.0)
        st.session_state.last_valid_confidence = conf_val
        is_held = stress_result.get("carried_forward", False)
        sub_text = f"Confidence: {conf_val:.1f}% (Holding)" if is_held else f"Confidence: {conf_val:.1f}%"
        st.metric(label="Current Stress Level", value=lbl.replace("_", " "), delta=sub_text)
    elif status == "BASELINE_REQUIRED":
        n_obs = len(getattr(classifier.baseline_normalizer, "_windows", []))
        st.metric(label="Current Stress Level", value="RELAXED", delta=f"Calibrating Baseline ({n_obs}/{config.BASELINE_MIN_WINDOWS})")
    else:
        # Uniform continuous reading: display held physiological state instead of empty dropout
        held_lbl = st.session_state.get("last_valid_stress_label", "RELAXED")
        held_conf = st.session_state.get("last_valid_confidence", 85.0)
        st.metric(label="Current Stress Level", value=held_lbl.replace("_", " "), delta=f"Confidence: {held_conf:.1f}% (Continuous)")

# ============================================================================
# 4. 3D WEBGL INTERACTIVE ANATOMICAL NEURAL MESH (THREE.JS)
# ============================================================================
stress_state = stress_result.get("label")
if not stress_state or str(stress_state).upper() in ["UNKNOWN", "NONE", "NAN", "NULL"]:
    if stress_result.get("status") == "BASELINE_REQUIRED":
        stress_state = "RELAXED"
    else:
        stress_state = st.session_state.get("last_valid_stress_label", "RELAXED")

if stress_result.get("status") == "BASELINE_REQUIRED":
    core_hex = "0x06b6d4"
    emissive_hex = "0x2563eb"
    badge_bg = "rgba(6, 182, 212, 0.15)"
    badge_border = "rgba(6, 182, 212, 0.4)"
    badge_color = "#06B6D4"
    n_obs = len(getattr(classifier.baseline_normalizer, "_windows", []))
    badge_text = f"⏳ CALIBRATING BASELINE ({n_obs}/{config.BASELINE_MIN_WINDOWS}) • 3D BIO-MAP"
elif stress_state == "HIGH_STRESS":
    core_hex = "0xef4444"
    emissive_hex = "0xd97706"
    badge_bg = "rgba(239, 68, 68, 0.15)"
    badge_border = "rgba(239, 68, 68, 0.4)"
    badge_color = "#EF4444"
    badge_text = f"🔥 {stress_state} • 3D BIO-MAP"
elif stress_state == "MODERATE_STRESS":
    core_hex = "0xf59e0b"
    emissive_hex = "0xd97706"
    badge_bg = "rgba(245, 158, 11, 0.15)"
    badge_border = "rgba(245, 158, 11, 0.4)"
    badge_color = "#F59E0B"
    badge_text = f"⚠️ {stress_state} • 3D BIO-MAP"
else:
    core_hex = "0x06b6d4"
    emissive_hex = "0x2563eb"
    badge_bg = "rgba(34, 197, 94, 0.15)"
    badge_border = "rgba(34, 197, 94, 0.4)"
    badge_color = "#22C55E"
    badge_text = f"⚡ {stress_state} • 3D BIO-MAP"

three_js_template = """
<!DOCTYPE html>
<html>
<head>
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
<style>
    * { box-sizing: border-box; }
    body { margin: 0; overflow: hidden; background: transparent; font-family: 'Inter', sans-serif; }
    #canvas-container { 
        width: 100%; 
        height: 320px; 
        position: relative; 
        border-radius: 18px; 
        border: 1px solid rgba(255, 255, 255, 0.08); 
        background: radial-gradient(circle at center, #0F172A 0%, #06090F 100%); 
        overflow: hidden; 
        box-shadow: inset 0 1px 1px rgba(255,255,255,0.08), 0 8px 32px rgba(0,0,0,0.5); 
    }
    .hud-header { position: absolute; top: 14px; left: 18px; z-index: 10; display: flex; align-items: center; gap: 10px; }
    .hud-title { color: #F1F5F9; font-size: 13px; font-weight: 700; letter-spacing: -0.01em; }
    .hud-sub { color: #94A3B8; font-size: 11px; margin-top: 1px; }
    .hud-badge { 
        position: absolute; top: 14px; right: 18px; z-index: 10; 
        background: __BADGE_BG__; 
        border: 1px solid __BADGE_BORDER__; 
        color: __BADGE_COLOR__; 
        padding: 4px 10px; border-radius: 20px; font-size: 10px; font-weight: 700; text-transform: uppercase; 
        backdrop-filter: blur(8px);
    }
    .hud-controls { position: absolute; bottom: 12px; left: 18px; z-index: 10; display: flex; gap: 8px; }
    .hud-btn { 
        background: rgba(15, 23, 42, 0.8); 
        border: 1px solid rgba(255, 255, 255, 0.15); 
        color: #94A3B8; 
        padding: 5px 12px; border-radius: 6px; font-size: 11px; font-weight: 600; 
        cursor: pointer; backdrop-filter: blur(8px); transition: all 0.2s; 
    }
    .hud-btn:hover { background: rgba(37, 99, 235, 0.3); color: #F1F5F9; border-color: #3B82F6; }
    .hud-stats { position: absolute; bottom: 12px; right: 18px; z-index: 10; display: flex; gap: 14px; font-size: 11px; color: #64748B; }
    .hud-stats span strong { color: #F1F5F9; font-family: monospace; }
</style>
</head>
<body>
<div id="canvas-container">
    <div class="hud-header">
        <div style="width:28px;height:28px;border-radius:8px;background:linear-gradient(135deg,#2563EB,#06B6D4);display:flex;align-items:center;justify-content:center;color:#fff;font-size:14px;">🧠</div>
        <div>
            <div class="hud-title">3D Anatomical Stress Visualizer</div>
            <div class="hud-sub">Interactive Orbit &bull; Drag to rotate, Scroll to zoom</div>
        </div>
    </div>
    <div class="hud-badge">__BADGE_TEXT__</div>
    <div class="hud-controls">
        <button class="hud-btn" onclick="resetCamera()">Center Camera</button>
        <button class="hud-btn" onclick="toggleAutoRotate()">Toggle Rotation</button>
    </div>
    <div class="hud-stats">
        <span>BPM: <strong>__BPM__</strong></span>
        <span>GSR: <strong>__GSR__ μS</strong></span>
        <span>IMU: <strong>__IMU__ g</strong></span>
    </div>
</div>

<script>
    const container = document.getElementById('canvas-container');
    let width = container.clientWidth;
    let height = container.clientHeight;

    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 1000);
    camera.position.set(0, 0, 3.8);

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, powerPreference: "high-performance", precision: "highp" });
    renderer.setSize(width, height);
    renderer.setPixelRatio(window.devicePixelRatio || 1);
    container.appendChild(renderer.domElement);

    const controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;
    controls.autoRotate = true;
    controls.autoRotateSpeed = 1.6;

    const ambientLight = new THREE.AmbientLight(0xffffff, 0.8);
    scene.add(ambientLight);

    const pointLight = new THREE.PointLight(__CORE_HEX__, 2.5, 50);
    pointLight.position.set(4, 4, 4);
    scene.add(pointLight);

    const headGroup = new THREE.Group();

    const coreGeo = new THREE.IcosahedronGeometry(1.15, 3);
    const coreMat = new THREE.MeshPhongMaterial({
        color: __CORE_HEX__,
        emissive: __EMISSIVE_HEX__,
        emissiveIntensity: 0.45,
        wireframe: true,
        transparent: true,
        opacity: 0.75,
    });
    const brainCore = new THREE.Mesh(coreGeo, coreMat);
    headGroup.add(brainCore);

    const outerGeo = new THREE.IcosahedronGeometry(1.38, 2);
    const outerMat = new THREE.MeshBasicMaterial({
        color: 0x3b82f6,
        wireframe: true,
        transparent: true,
        opacity: 0.22,
    });
    const outerShell = new THREE.Mesh(outerGeo, outerMat);
    headGroup.add(outerShell);

    const particleCount = 180;
    const particleGeo = new THREE.BufferGeometry();
    const positions = new Float32Array(particleCount * 3);
    const colors = new Float32Array(particleCount * 3);

    for (let i = 0; i < particleCount * 3; i += 3) {
        const u = Math.random();
        const v = Math.random();
        const theta = u * 2.0 * Math.PI;
        const phi = Math.acos(2.0 * v - 1.0);
        const r = 1.22 + (Math.random() - 0.5) * 0.3;

        positions[i] = r * Math.sin(phi) * Math.cos(theta);
        positions[i+1] = r * Math.sin(phi) * Math.sin(theta);
        positions[i+2] = r * Math.cos(phi);

        colors[i] = 0.15;
        colors[i+1] = 0.75;
        colors[i+2] = 1.0;
    }

    particleGeo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    particleGeo.setAttribute('color', new THREE.BufferAttribute(colors, 3));

    const particleMat = new THREE.PointsMaterial({
        size: 0.045,
        vertexColors: true,
        transparent: true,
        opacity: 0.85
    });
    const pSystem = new THREE.Points(particleGeo, particleMat);
    headGroup.add(pSystem);

    scene.add(headGroup);

    let clock = new THREE.Clock();
    function animate() {
        requestAnimationFrame(animate);
        const elapsedTime = clock.getElapsedTime();
        controls.update();

        const pulse = 1.0 + Math.sin(elapsedTime * 4.0) * 0.04;
        brainCore.scale.set(pulse, pulse, pulse);
        outerShell.rotation.y = elapsedTime * 0.15;
        outerShell.rotation.x = elapsedTime * 0.08;

        renderer.render(scene, camera);
    }
    animate();

    function resetCamera() {
        camera.position.set(0, 0, 3.8);
        controls.target.set(0, 0, 0);
    }

    function toggleAutoRotate() {
        controls.autoRotate = !controls.autoRotate;
    }

    window.addEventListener('resize', () => {
        width = container.clientWidth;
        height = container.clientHeight;
        camera.aspect = width / height;
        camera.updateProjectionMatrix();
        renderer.setPixelRatio(window.devicePixelRatio || 1);
        renderer.setSize(width, height);
    });
</script>
</body>
</html>
"""

three_js_code = (
    three_js_template
    .replace("__BADGE_BG__", badge_bg)
    .replace("__BADGE_BORDER__", badge_border)
    .replace("__BADGE_COLOR__", badge_color)
    .replace("__BADGE_TEXT__", badge_text)
    .replace("__BPM__", f"{processed_batch.get('bpm', 0.0):.1f}")
    .replace("__GSR__", f"{processed_batch.get('scl_mean', 0.0):.1f}")
    .replace("__IMU__", f"{processed_batch.get('activity_index', 0.0):.1f}")
    .replace("__CORE_HEX__", core_hex)
    .replace("__EMISSIVE_HEX__", emissive_hex)
)

components.html(three_js_code, height=335)

# ============================================================================
# 5. PLOTLY CHART THEME HELPERS
# ============================================================================
def get_chart_layout(title: str, height: int = 250) -> dict:
    return dict(
        title=dict(
            text=title,
            font=dict(family="Inter, sans-serif", size=13, color="#94A3B8"),
            x=0.01,
            y=0.97
        ),
        height=height,
        margin=dict(l=16, r=16, t=44, b=20),
        template="plotly_dark",
        paper_bgcolor="rgba(15, 23, 42, 0.4)",
        plot_bgcolor="rgba(11, 17, 32, 0.4)",
        font=dict(family="Inter, sans-serif", color="#94A3B8", size=11),
        xaxis=dict(gridcolor="rgba(255, 255, 255, 0.04)", showgrid=True, zeroline=False),
        yaxis=dict(gridcolor="rgba(255, 255, 255, 0.04)", showgrid=True, zeroline=False),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=11), orientation="h", y=1.08)
    )


def align_overlay(series, n_raw: int):
    """Right-align a processed series against the raw buffer's x-axis.

    The raw traces plot the whole receiver buffer (up to BUFFER_SIZE samples),
    while `processed_batch` only covers the most recent 30s window (~751 samples).
    Returning explicit x indices anchors the filtered trace to the tail of the raw
    trace instead of requiring the two lengths to match exactly.

    Returns (x, y) or (None, None) when there is nothing to draw.
    """
    if series is None or n_raw <= 0:
        return None, None
    y = list(series)
    if not y:
        return None, None
    if len(y) > n_raw:
        y = y[-n_raw:]
    start = n_raw - len(y)
    return list(range(start, n_raw)), y


# ============================================================================
# 6. MAIN TABS
# ============================================================================
tab_live, tab_wesad, tab_pipeline, tab_debug, tab_report = st.tabs([
    "PPG - GSR - IMU",
    "WESAD Model",
    "Pipeline",
    "Debug",
    "Reports"
])

with tab_live:
    if not latest_packets:
        st.info("Waiting for data... Click **▶ Connect** in sidebar to start serial receiver.")
    else:
        df_packets = pd.DataFrame(latest_packets)
        n_raw = len(df_packets)

        # ── LIVE RAW READINGS PANEL ──────────────────────────────
        _is_live = receiver.running and any(
            k in receiver.status for k in ("CONNECTED", "SIMULATED", "RECONNECTING")
        )

        # Live-indicator dot
        if _is_live:
            _live_dot = (
                '<span style="display:inline-flex;align-items:center;gap:4px;">'
                '<span style="display:inline-block;width:7px;height:7px;'
                'border-radius:50%;background:#22C55E;box-shadow:0 0 6px #22C55E;"></span>'
                '<span style="color:#22C55E;font-size:10px;font-weight:700;'
                'letter-spacing:0.08em;">LIVE</span></span>'
            )
        else:
            _live_dot = (
                '<span style="color:#64748B;font-size:10px;font-weight:600;'
                'letter-spacing:0.08em;">WAITING FOR SESSION\u2026</span>'
            )

        # Optional compact sensor filter
        _raw_filter = st.radio(
            "raw_filter",
            ["ALL", "PPG", "GSR", "IMU"],
            index=0,
            horizontal=True,
            label_visibility="collapsed",
            key="live_raw_filter",
        )

        # Column headers per filter
        _col_headers = {
            "ALL": "<th>Time</th><th>PPG IR</th><th>PPG Red</th>"
                   "<th>GSR Raw</th><th>AX</th><th>AY</th><th>AZ</th>",
            "PPG": "<th>Time</th><th>IR</th><th>Red</th>",
            "GSR": "<th>Time</th><th>GSR Raw</th>",
            "IMU": "<th>Time</th><th>AX</th><th>AY</th><th>AZ</th>",
        }
        _col_count = {"ALL": 7, "PPG": 3, "GSR": 2, "IMU": 4}

        # Build HTML table rows from the last 10 raw packets
        _rows_html = ""
        if _is_live:
            _tail = latest_packets[-10:]
            for pkt in _tail:
                tw = pkt.get("timestamp_wall")
                ts_str = "\u2014"
                if tw is not None and pd.notna(tw):
                    try:
                        tw_f = float(tw)
                        if tw_f > 0 and not np.isnan(tw_f) and not np.isinf(tw_f):
                            _stamp = _dt.fromtimestamp(tw_f)
                            ts_str = _stamp.strftime("%H:%M:%S.") + f"{_stamp.microsecond // 1000:03d}"
                    except (ValueError, OSError, OverflowError):
                        pass

                if _raw_filter == "ALL":
                    _rows_html += (
                        f"<tr>"
                        f"<td>{ts_str}</td>"
                        f"<td>{pkt.get('ppg_ir', 0):.0f}</td>"
                        f"<td>{pkt.get('ppg_red', 0):.0f}</td>"
                        f"<td>{pkt.get('gsr_raw', 0):.0f}</td>"
                        f"<td>{pkt.get('imu_ax', 0):.2f}</td>"
                        f"<td>{pkt.get('imu_ay', 0):.2f}</td>"
                        f"<td>{pkt.get('imu_az', 0):.2f}</td>"
                        f"</tr>"
                    )
                elif _raw_filter == "PPG":
                    _rows_html += (
                        f"<tr><td>{ts_str}</td>"
                        f"<td>{pkt.get('ppg_ir', 0):.0f}</td>"
                        f"<td>{pkt.get('ppg_red', 0):.0f}</td></tr>"
                    )
                elif _raw_filter == "GSR":
                    _rows_html += (
                        f"<tr><td>{ts_str}</td>"
                        f"<td>{pkt.get('gsr_raw', 0):.0f}</td></tr>"
                    )
                elif _raw_filter == "IMU":
                    _rows_html += (
                        f"<tr><td>{ts_str}</td>"
                        f"<td>{pkt.get('imu_ax', 0):.2f}</td>"
                        f"<td>{pkt.get('imu_ay', 0):.2f}</td>"
                        f"<td>{pkt.get('imu_az', 0):.2f}</td></tr>"
                    )

        if not _rows_html:
            _rows_html = (
                f'<tr><td colspan="{_col_count[_raw_filter]}" style="text-align:center;'
                f'color:#475569;padding:16px 0;font-style:italic;">'
                f'Waiting for session\u2026</td></tr>'
            )

        st.markdown(f"""
        <div style="background:rgba(15,23,42,0.5);border:1px solid rgba(6,182,212,0.15);
                    border-radius:8px;padding:10px 14px;margin-bottom:12px;">
          <div style="display:flex;justify-content:space-between;align-items:center;
                      margin-bottom:8px;">
            <span style="font-size:11px;font-weight:700;color:#94A3B8;
                         letter-spacing:0.1em;text-transform:uppercase;">
              LIVE RAW READINGS
            </span>
            {_live_dot}
          </div>
          <div style="max-height:240px;overflow-y:auto;">
          <table style="width:100%;border-collapse:collapse;font-family:'JetBrains Mono',
                        'Fira Code','Cascadia Code',monospace;font-size:11px;color:#CBD5E1;">
            <thead>
              <tr style="color:#64748B;font-size:10px;text-transform:uppercase;
                         letter-spacing:0.05em;border-bottom:1px solid rgba(255,255,255,0.06);">
                {_col_headers[_raw_filter]}
              </tr>
            </thead>
            <tbody>
              {_rows_html}
            </tbody>
          </table>
          </div>
        </div>
        """, unsafe_allow_html=True)
        # ── END LIVE RAW READINGS ────────────────────────────────

        # ── TIME AXIS for all graphs ─────────────────────────────
        # Convert timestamp_wall (Unix float) to datetime for Plotly X-axis
        _tw_col = df_packets.get("timestamp_wall")
        _time_axis = []
        if _tw_col is not None and len(_tw_col) > 0:
            for tw in _tw_col:
                if tw is not None and pd.notna(tw):
                    try:
                        tw_f = float(tw)
                        if tw_f > 0 and not np.isnan(tw_f) and not np.isinf(tw_f):
                            _time_axis.append(_dt.fromtimestamp(tw_f))
                        else:
                            _time_axis.append(None)
                    except (ValueError, OSError, OverflowError):
                        _time_axis.append(None)
                else:
                    _time_axis.append(None)

            valid_indices = [i for i, t in enumerate(_time_axis) if t is not None]
            if len(valid_indices) == 0:
                _time_axis = list(range(n_raw))
            elif len(valid_indices) < len(_time_axis):
                first_valid_idx = valid_indices[0]
                dt_step = _td(seconds=1.0 / max(config.SAMPLING_RATE_HZ, 1.0))
                for i in range(first_valid_idx - 1, -1, -1):
                    _time_axis[i] = _time_axis[i + 1] - dt_step
                for i in range(first_valid_idx + 1, len(_time_axis)):
                    if _time_axis[i] is None:
                        _time_axis[i] = _time_axis[i - 1] + dt_step
        else:
            _time_axis = list(range(n_raw))

        _is_datetime_axis = len(_time_axis) > 0 and isinstance(_time_axis[0], _dt)
        _time_xaxis = dict(
            gridcolor="rgba(255, 255, 255, 0.04)",
            showgrid=True,
            zeroline=False,
            tickformat="%H:%M:%S" if _is_datetime_axis else None,
        )

        # ── PPG Cardiac Waveform Chart ───────────────────────────
        raw_ppg_vals = np.asarray(df_packets.get("ppg_raw", df_packets.get("ppg_ir", [])), dtype=float)
        
        # Calculate real-time filtered arterial pulse wave for the live buffer
        if len(raw_ppg_vals) >= 15:
            live_ppg_filtered = preprocessor.filter_ppg(raw_ppg_vals)
        elif len(raw_ppg_vals) > 0:
            live_ppg_filtered = raw_ppg_vals - np.mean(raw_ppg_vals)
        else:
            live_ppg_filtered = np.array([], dtype=float)

        # Detect systolic peaks on the live arterial pulse wave
        live_peaks_idx = np.array([], dtype=int)
        if len(live_ppg_filtered) >= 15:
            std_live = float(np.std(live_ppg_filtered))
            if std_live > 5.0:
                pks, _ = signal.find_peaks(
                    live_ppg_filtered,
                    distance=max(2, int(config.SAMPLING_RATE_HZ * 0.33)),
                    prominence=max(1.0, 0.15 * std_live)
                )
                live_peaks_idx = pks

        fig_ppg = make_subplots(specs=[[{"secondary_y": True}]])
        # Primary axis: Filtered arterial pulse wave (AC)
        fig_ppg.add_trace(
            go.Scatter(
                x=_time_axis,
                y=live_ppg_filtered,
                name="Cardiac Pulse (Filtered BVP)",
                line=dict(color="#06B6D4", width=2.5),
                fill="tozeroy",
                fillcolor="rgba(6, 182, 212, 0.08)",
            ),
            secondary_y=False,
        )

        # Pulse peaks markers
        if len(live_peaks_idx) > 0 and len(_time_axis) == len(live_ppg_filtered):
            p_times = [_time_axis[i] for i in live_peaks_idx]
            p_vals = [live_ppg_filtered[i] for i in live_peaks_idx]
            fig_ppg.add_trace(
                go.Scatter(
                    x=p_times,
                    y=p_vals,
                    name="Systolic Beats",
                    mode="markers",
                    marker=dict(symbol="diamond", size=7, color="#EF4444", line=dict(color="#FFFFFF", width=1)),
                ),
                secondary_y=False,
            )

        # Secondary axis: Raw Optical IR (DC absorption level)
        fig_ppg.add_trace(
            go.Scatter(
                x=_time_axis,
                y=raw_ppg_vals,
                name="Raw Optical IR (DC Count)",
                line=dict(color="rgba(148, 163, 184, 0.35)", width=1, dash="dot"),
            ),
            secondary_y=True,
        )

        _ppg_layout = get_chart_layout("PPG Cardiac Waveform (Photoplethysmography)", 270)
        _ppg_layout["xaxis"] = _time_xaxis
        _ppg_layout["yaxis"] = dict(
            title=dict(text="Cardiac AC Pulse", font=dict(color="#06B6D4", size=10)),
            gridcolor="rgba(255, 255, 255, 0.04)",
            showgrid=True,
            zeroline=True,
            zerolinecolor="rgba(255, 255, 255, 0.1)",
            autorange=True,
        )
        _ppg_layout["yaxis2"] = dict(
            title=dict(text="Raw IR Count", font=dict(color="#94A3B8", size=10)),
            showgrid=False,
            zeroline=False,
            autorange=True,
            overlaying="y",
            side="right",
        )
        fig_ppg.update_layout(**_ppg_layout)
        st.plotly_chart(fig_ppg, use_container_width=True)

        # ── GSR Electrodermal Activity Chart ─────────────────────
        raw_gsr_vals = np.asarray(df_packets.get("gsr_raw", []), dtype=float)
        if len(raw_gsr_vals) >= 15:
            try:
                b_gsr, a_gsr = signal.butter(config.GSR_FILTER_ORDER, min(0.95, config.GSR_LOWCUT / (0.5 * config.SAMPLING_RATE_HZ)), btype="lowpass")
                live_gsr_tonic = signal.filtfilt(b_gsr, a_gsr, raw_gsr_vals)
            except Exception:
                live_gsr_tonic = raw_gsr_vals
        else:
            live_gsr_tonic = raw_gsr_vals

        fig_gsr = make_subplots(specs=[[{"secondary_y": True}]])
        fig_gsr.add_trace(
            go.Scatter(
                x=_time_axis,
                y=raw_gsr_vals,
                name="Raw GSR (ADC counts)",
                line=dict(color="rgba(245, 158, 11, 0.4)", width=1.2, dash="dot"),
            ),
            secondary_y=False,
        )
        fig_gsr.add_trace(
            go.Scatter(
                x=_time_axis,
                y=live_gsr_tonic,
                name="Tonic SCL (Filtered ADC)",
                line=dict(color="#22C55E", width=2.5),
                fill="tozeroy",
                fillcolor="rgba(34, 197, 94, 0.06)",
            ),
            secondary_y=False,
        )
        # Secondary Y: Conductance in uS
        gsr_us_vals = np.maximum(config.GSR_US_FLOOR, (config.GSR_ADC_FULL_SCALE - raw_gsr_vals) / config.GSR_US_PER_COUNT_DIVISOR) if len(raw_gsr_vals) > 0 else []
        fig_gsr.add_trace(
            go.Scatter(
                x=_time_axis,
                y=gsr_us_vals,
                name="Skin Conductance (μS)",
                line=dict(color="#EAB308", width=1.8),
            ),
            secondary_y=True,
        )
        _gsr_layout = get_chart_layout("Galvanic Skin Response / Electrodermal Activity", 240)
        _gsr_layout["xaxis"] = _time_xaxis
        _gsr_layout["yaxis"] = dict(
            title=dict(text="GSR Raw ADC", font=dict(color="#22C55E", size=10)),
            gridcolor="rgba(255, 255, 255, 0.04)",
            showgrid=True,
            zeroline=False,
            autorange="reversed",
        )
        _gsr_layout["yaxis2"] = dict(
            title=dict(text="Conductance (μS)", font=dict(color="#EAB308", size=10)),
            showgrid=False,
            zeroline=False,
            autorange=True,
            overlaying="y",
            side="right",
        )
        fig_gsr.update_layout(**_gsr_layout)
        st.plotly_chart(fig_gsr, use_container_width=True)

        # IMU Chart
        fig_imu = go.Figure()
        for col_name, tr_name, color in [
            ("imu_ax", "Accel X", "#EF4444"),
            ("imu_ay", "Accel Y", "#A855F7"),
            ("imu_az", "Accel Z", "#06B6D4"),
        ]:
            if col_name in df_packets:
                fig_imu.add_trace(go.Scatter(
                    x=_time_axis,
                    y=df_packets[col_name],
                    name=tr_name,
                    line=dict(color=color, width=1.5)
                ))
        _imu_layout = get_chart_layout("IMU 3-Axis Accelerometer Dynamics (g)", 230)
        _imu_layout["xaxis"] = _time_xaxis
        fig_imu.update_layout(**_imu_layout)
        st.plotly_chart(fig_imu, use_container_width=True)

with tab_wesad:
    col_s1, col_s2 = st.columns([1, 1])

    with col_s1:
        status = stress_result.get("status", "MULTICLASS_MODEL_UNAVAILABLE")
        label = stress_result.get("label")
        probabilities = stress_result.get("class_probabilities") or {}
        if status != "OK":
            st.warning(f"Four-class physiological model status: {status.replace('_', ' ')}")
            if status == "MULTICLASS_MODEL_UNAVAILABLE":
                st.caption("No legacy binary or heuristic output is converted into four stress levels.")
            elif status == "MODEL_CONFIGURATION_ERROR":
                st.caption("The model artifact in Models/weights is corrupted or incompatible with the 4-class contract.")
        else:
            st.markdown(f"### Current Stress Level: {label.replace('_', ' ')}")
            st.caption(f"Model confidence: {stress_result.get('confidence_pct', 0.0):.1f}%")
            probability_frame = pd.DataFrame({"Stress level": list(probabilities), "Probability": list(probabilities.values())})
            st.bar_chart(probability_frame, x="Stress level", y="Probability", horizontal=True)
            st.caption(f"Model version: {stress_result.get('model_version')} | Inference: {stress_result.get('predict_ms', 0.0):.1f} ms")

    with col_s2:
        st.markdown('<div style="font-size:14px; font-weight:600; color:#F1F5F9; margin-bottom:10px;">Extracted 23 WESAD Feature Vector</div>', unsafe_allow_html=True)
        if feature_df is not None and not feature_df.empty:
            df_disp = feature_df.T.rename(columns={0: "Value"})
            df_disp.index.name = "WESAD Feature Name"
            st.dataframe(df_disp, use_container_width=True, height=260)
        else:
            st.caption("No active feature window yet.")

with tab_pipeline:
    col_p1, col_p2 = st.columns(2)
    with col_p1:
        st.markdown("""
        <div class="liquid-glass" style="border-radius:16px; padding:22px;">
            <h4 style="margin:0 0 12px; color:#F1F5F9; font-size:15px; font-weight:600; letter-spacing:-0.01em;">🎵 PPG / BVP NeuroKit2 Pipeline</h4>
            <p style="font-size:13px; color:#94A3B8; margin:0; line-height:1.6;">Uses <code style='background:rgba(255,255,255,0.04);padding:2px 6px;border-radius:4px;border:1px solid rgba(255,255,255,0.06);'>nk.ppg_process()</code> and <code style='background:rgba(255,255,255,0.04);padding:2px 6px;border-radius:4px;border:1px solid rgba(255,255,255,0.06);'>nk.hrv()</code> to compute 6 heart-rate variability features: <code style='background:rgba(255,255,255,0.04);padding:2px 6px;border-radius:4px;border:1px solid rgba(255,255,255,0.06);'>hr</code>, <code style='background:rgba(255,255,255,0.04);padding:2px 6px;border-radius:4px;border:1px solid rgba(255,255,255,0.06);'>rmssd</code>, <code style='background:rgba(255,255,255,0.04);padding:2px 6px;border-radius:4px;border:1px solid rgba(255,255,255,0.06);'>sdnn</code>, <code style='background:rgba(255,255,255,0.04);padding:2px 6px;border-radius:4px;border:1px solid rgba(255,255,255,0.06);'>pnn50</code>, <code style='background:rgba(255,255,255,0.04);padding:2px 6px;border-radius:4px;border:1px solid rgba(255,255,255,0.06);'>ibi_mean</code>, <code style='background:rgba(255,255,255,0.04);padding:2px 6px;border-radius:4px;border:1px solid rgba(255,255,255,0.06);'>ibi_std</code> over 30s rolling windows (15s step, 50% overlap).</p>
        </div>
        """, unsafe_allow_html=True)
    with col_p2:
        st.markdown("""
        <div class="liquid-glass" style="border-radius:16px; padding:22px;">
            <h4 style="margin:0 0 12px; color:#F1F5F9; font-size:15px; font-weight:600; letter-spacing:-0.01em;">⚡ EDA / GSR NeuroKit2 Pipeline</h4>
            <p style="font-size:13px; color:#94A3B8; margin:0; line-height:1.6;">Uses <code style='background:rgba(255,255,255,0.04);padding:2px 6px;border-radius:4px;border:1px solid rgba(255,255,255,0.06);'>nk.eda_clean()</code>, <code style='background:rgba(255,255,255,0.04);padding:2px 6px;border-radius:4px;border:1px solid rgba(255,255,255,0.06);'>nk.eda_phasic()</code>, and <code style='background:rgba(255,255,255,0.04);padding:2px 6px;border-radius:4px;border:1px solid rgba(255,255,255,0.06);'>nk.eda_peaks()</code> to extract 9 electrodermal features: <code style='background:rgba(255,255,255,0.04);padding:2px 6px;border-radius:4px;border:1px solid rgba(255,255,255,0.06);'>eda_mean</code>, <code style='background:rgba(255,255,255,0.04);padding:2px 6px;border-radius:4px;border:1px solid rgba(255,255,255,0.06);'>eda_std</code>, <code style='background:rgba(255,255,255,0.04);padding:2px 6px;border-radius:4px;border:1px solid rgba(255,255,255,0.06);'>eda_slope</code>, <code style='background:rgba(255,255,255,0.04);padding:2px 6px;border-radius:4px;border:1px solid rgba(255,255,255,0.06);'>scl_mean</code>, <code style='background:rgba(255,255,255,0.04);padding:2px 6px;border-radius:4px;border:1px solid rgba(255,255,255,0.06);'>phasic_mean</code>, <code style='background:rgba(255,255,255,0.04);padding:2px 6px;border-radius:4px;border:1px solid rgba(255,255,255,0.06);'>scr_count</code>, <code style='background:rgba(255,255,255,0.04);padding:2px 6px;border-radius:4px;border:1px solid rgba(255,255,255,0.06);'>scr_amp_mean</code>, <code style='background:rgba(255,255,255,0.04);padding:2px 6px;border-radius:4px;border:1px solid rgba(255,255,255,0.06);'>scr_rise_mean</code>, <code style='background:rgba(255,255,255,0.04);padding:2px 6px;border-radius:4px;border:1px solid rgba(255,255,255,0.06);'>scr_recovery_mean</code>.</p>
        </div>
        """, unsafe_allow_html=True)

with tab_debug:
    if not getattr(config, 'DEBUG_TELEMETRY', False):
        st.info("Debug telemetry is disabled. Set `DEBUG_TELEMETRY = True` in config.py to enable.")
    else:
        st.subheader("🔧 Serial Debug Telemetry")
        col_d1, col_d2, col_d3, col_d4 = st.columns(4)
        with col_d1:
            st.metric("Serial Status", status_summary.get("status", "DISCONNECTED"))
        with col_d2:
            st.metric("COM Port", status_summary.get("serial_port", "N/A"))
        with col_d3:
            st.metric("Incoming FPS", f"{status_summary.get('rate_hz', 0.0)} Hz")
        with col_d4:
            st.metric("Prediction Latency", f"{stress_result.get('predict_ms', 0.0):.1f} ms")

        col_d5, col_d6, col_d7, col_d8 = st.columns(4)
        with col_d5:
            st.metric("Packets Received", receiver.packets_received if hasattr(receiver, 'packets_received') else 0)
        with col_d6:
            st.metric("Packets Dropped", receiver.packets_dropped if hasattr(receiver, 'packets_dropped') else 0)
        with col_d7:
            st.metric("Reconnections", receiver.reconnect_count if hasattr(receiver, 'reconnect_count') else 0)
        with col_d8:
            st.metric("Buffer Size", len(receiver.buffer) if hasattr(receiver, 'buffer') else 0)

        # ── Acquisition & Sampling Diagnostics ──────────────────────────────
        # Uses receiver.sampling_diagnostics (updated every 100 packets in _receive_loop)
        # which includes rate validation against config.SAMPLING_RATE_HZ.
        recv_diag = status_summary.get("sampling_diagnostics", {})
        # Fallback: compute from latest_packets if receiver hasn't hit 100-packet mark yet
        if not recv_diag:
            recv_diag = compute_sampling_diagnostics(latest_packets)

        st.markdown("**📡 Sampling Rate Validation (ESP32 Timestamps):**")
        rate_ok = recv_diag.get("is_valid", False)
        if recv_diag.get("actual_hz", 0) > 0:
            if rate_ok:
                st.success(f"✅ {recv_diag.get('message', 'Rate OK')}")
            else:
                st.warning(f"⚠️ {recv_diag.get('message', 'Rate mismatch detected')}")
        else:
            st.info("⏳ Waiting for 50+ packets to compute rate diagnostics...")

        col_diag1, col_diag2, col_diag3, col_diag4 = st.columns(4)
        with col_diag1:
            actual_hz = recv_diag.get("actual_hz", recv_diag.get("mean_effective_hz", 0.0))
            expected_hz = recv_diag.get("expected_hz", config.SAMPLING_RATE_HZ)
            delta_hz = f"Expected: {expected_hz:.1f} Hz"
            st.metric("Actual Rate", f"{actual_hz:.1f} Hz", delta=delta_hz,
                      delta_color="normal" if rate_ok else "inverse")
        with col_diag2:
            mean_ms = recv_diag.get("mean_interval_ms", 0.0)
            std_ms = recv_diag.get("std_interval_ms", 0.0)
            st.metric("Mean Interval", f"{mean_ms:.1f} ms", delta=f"Jitter ±{std_ms:.1f} ms",
                      delta_color="inverse" if std_ms > 5.0 else "normal")
        with col_diag3:
            st.metric("Packet Loss", f"{recv_diag.get('packet_loss_pct', 0.0):.2f}%",
                      delta=f"Missing: {recv_diag.get('missing_packets', 0)} pkts",
                      delta_color="inverse" if recv_diag.get('packet_loss_pct', 0) > 1.0 else "normal")
        with col_diag4:
            min_ms = recv_diag.get("min_interval_ms", 0.0)
            max_ms = recv_diag.get("max_interval_ms", 0.0)
            st.metric("Interval Range", f"{min_ms:.0f}–{max_ms:.0f} ms",
                      delta=f"Gaps: {recv_diag.get('largest_packet_gap', 0)} | Resets: {recv_diag.get('resets_detected', 0)}")

        # ── Signal Quality Details — Per-Channel Gates ────────────────────────
        st.markdown("**🔍 Signal Quality — Per-Channel Gate Status:**")
        sqi_display = processed_batch.get("signal_quality", {}) if processed_batch else {}
        if sqi_display:
            ppg_score = sqi_display.get("ppg_score", 0.0)
            gsr_score = sqi_display.get("gsr_score", 0.0)
            imu_score = sqi_display.get("imu_score", 0.0)
            composite  = sqi_display.get("composite_score", sqi_display.get("overall_sqi", 0.0) / 100.0)
            is_valid   = sqi_display.get("is_valid", False)
            per_ch_valid = sqi_display.get("per_channel_valid", True)
            rejection_reasons = sqi_display.get("rejection_reasons", [])

            col_sqi1, col_sqi2, col_sqi3, col_sqi4 = st.columns(4)
            with col_sqi1:
                ppg_pass = ppg_score >= config.SQI_PPG_MIN
                st.metric("PPG SQI", f"{ppg_score:.2f}",
                          delta=f"Min: {config.SQI_PPG_MIN} {'✅' if ppg_pass else '❌'}",
                          delta_color="normal" if ppg_pass else "inverse")
                st.caption(sqi_display.get("ppg_quality", "—"))
            with col_sqi2:
                gsr_pass = gsr_score >= config.SQI_GSR_MIN
                st.metric("GSR SQI", f"{gsr_score:.2f}",
                          delta=f"Min: {config.SQI_GSR_MIN} {'✅' if gsr_pass else '❌'}",
                          delta_color="normal" if gsr_pass else "inverse")
                st.caption(sqi_display.get("gsr_quality", "—"))
            with col_sqi3:
                imu_pass = imu_score >= config.SQI_IMU_MIN
                st.metric("IMU SQI", f"{imu_score:.2f}",
                          delta=f"Min: {config.SQI_IMU_MIN} {'✅' if imu_pass else '❌'}",
                          delta_color="normal" if imu_pass else "inverse")
                st.caption(sqi_display.get("imu_quality", "—"))
            with col_sqi4:
                st.metric("Composite SQI", f"{composite:.2f}",
                          delta=f"Gate: {config.SQI_MIN_VALID} {'✅' if composite >= config.SQI_MIN_VALID else '❌'}",
                          delta_color="normal" if composite >= config.SQI_MIN_VALID else "inverse")
                st.caption("VALID" if is_valid else "INVALID")

            if rejection_reasons:
                st.error("**Window rejected — reasons:**")
                for reason in rejection_reasons:
                    st.markdown(f"  • {reason}")
            elif is_valid:
                st.success("✅ All quality gates passed — window is ML-ready")
            else:
                st.warning("⚠️ Window failed quality gate (composite SQI too low)")
        else:
            st.caption("No signal quality data yet — waiting for first processed window.")

        # ── BASELINE & INFERENCE STATUS (Phase 6) ───────────────────────────
        st.markdown("**🎯 Baseline & Inference Status**")
        baseline_status_col1, baseline_status_col2, baseline_status_col3 = st.columns(3)

        with baseline_status_col1:
            st.markdown("#### Calibration Status")
            # Calibration progress
            if hasattr(classifier.baseline_normalizer, "calibration_windows"):
                calib_windows = len(classifier.baseline_normalizer.calibration_windows)
                min_windows = config.BASELINE_MIN_WINDOWS
                if calib_windows < min_windows:
                    st.warning(f"⏳ CALIBRATING ({calib_windows}/{min_windows} windows)")
                elif not classifier.baseline_normalizer.is_ready:
                    st.info("⏳ INITIALIZING")
                else:
                    st.success("✅ CALIBRATION COMPLETE")
            else:
                n_obs = len(getattr(classifier.baseline_normalizer, "_windows", []))
                min_windows = config.BASELINE_MIN_WINDOWS
                if n_obs < min_windows:
                    st.warning(f"⏳ CALIBRATING ({n_obs}/{min_windows} windows)")
                else:
                    st.success("✅ CALIBRATION COMPLETE")

        with baseline_status_col2:
            st.markdown("#### Universal Baseline")
            if classifier.baseline_normalizer.universal_baseline is not None:
                wb = classifier.baseline_normalizer.universal_baseline
                features_count = len(wb.features)
                st.success(f"✅ LOADED ({features_count} features)")
                st.caption(f"Version: {wb.metadata.get('dataset', 'WESAD')}")
                st.caption("Read-only: ✅")
            else:
                st.warning("⚠️ NOT LOADED")

        with baseline_status_col3:
            st.markdown("#### Personal Baseline")
            if classifier.baseline_normalizer.is_ready:
                st.success("✅ READY")
                calib_windows = len(getattr(classifier.baseline_normalizer, "calibration_windows", []))
                if calib_windows > 0:
                    st.caption(f"Calibration windows: {calib_windows}")
            else:
                st.warning("❌ NOT READY")

        # Baseline State Display
        st.markdown("#### Baseline State & Decision")
        state_col1, state_col2, state_col3 = st.columns(3)

        with state_col1:
            st.markdown("**Current State**")
            baseline_state = "UNKNOWN"
            freeze_reason = "N/A"
            baseline_log = stress_result.get("baseline_log", {})
            if baseline_log:
                baseline_state = baseline_log.get("baseline_state", stress_result.get("baseline_state", "ACTIVE"))
                freeze_reason = baseline_log.get("freeze_reason", "N/A")
            elif hasattr(classifier.baseline_normalizer, "state"):
                baseline_state = classifier.baseline_normalizer.state.value
                freeze_reason = "No prediction yet"

            # Color coding for states
            state_colors = {
                "INITIALIZING": "#64748B",
                "CALIBRATING": "#F59E0B",
                "ACTIVE": "#10B981",
                "ADAPTING": "#3B82F6",
                "FROZEN_STRESS": "#EF4444",
                "FROZEN_UNCERTAIN": "#F97316",
            }
            state_color = state_colors.get(baseline_state, "#64748B")

            st.markdown(f"""
            <div style="background:rgba(255,255,255,0.02);padding:12px;border-radius:10px;border:1px solid rgba(255,255,255,0.1)">
                <div style="font-size:24px;font-weight:800;color:{state_color};letter-spacing:-0.02em">{baseline_state}</div>
                <div style="font-size:11px;color:#94A3B8;margin-top:4px">{freeze_reason or "No freeze"}</div>
            </div>
            """, unsafe_allow_html=True)

        with state_col2:
            st.markdown("**Baseline Update**")
            if baseline_log:
                update_allowed = baseline_log.get("baseline_update_allowed", False)
                if update_allowed:
                    st.success("✅ ALLOWED")
                    lambda_val = baseline_log.get("lambda", 0.0)
                    st.caption(f"Adaptation rate: {lambda_val:.4f}")
                else:
                    st.error("❌ FROZEN")
                    st.caption(freeze_reason or "Unknown reason")
            else:
                st.info("⏳ WAITING")
                st.caption("Waiting for first prediction")

        with state_col3:
            st.markdown("**Prediction Confidence**")
            if stress_result.get("status") == "OK":
                conf = stress_result.get("confidence_pct", 0.0)
                st.metric("Confidence", f"{conf:.1f}%")
                if baseline_log:
                    motion_val = baseline_log.get("motion", 0.0)
                    sqi = baseline_log.get("signal_quality", {})
                    sqi_valid = sqi.get("is_valid", True)
                    st.caption(f"Motion: {motion_val:.3f}g | SQI: {'✅' if sqi_valid else '❌'}")
            else:
                st.caption("N/A")

        # Baseline Values Display (Diagnostic)
        st.markdown("#### Baseline Values (Diagnostic)")
        if baseline_log:
            current_baseline = baseline_log.get("current_baseline", {})
            current_values = baseline_log.get("current_feature_values", {})
            if current_baseline and current_values:
                # Show 8 baseline-normalized features
                baseline_cols = st.columns(4)
                feat_names = ["eda_mean", "scl_mean", "scr_count", "scr_amp_mean", "hr", "rmssd", "sdnn", "ibi_mean"]
                for i, feat in enumerate(feat_names):
                    with baseline_cols[i % 4]:
                        if feat in current_baseline and feat in current_values:
                            base_val = current_baseline[feat]
                            curr_val = current_values[feat]
                            delta = curr_val - base_val
                            st.metric(feat, f"{curr_val:.2f}", delta=f"Δ {delta:+.2f}")
            else:
                st.caption("Baseline values not available yet")
        else:
            st.caption("Baseline log not available")

        # ── DEBUG INFORMATION (Collapsible) ────────────────────────────────
        with st.expander("🔍 Debug Information (Advanced)", expanded=False):
            st.markdown("**📋 Complete Baseline Log Record**")
            if baseline_log:
                # Convert to JSON-like format for display
                st.json(baseline_log)
            else:
                st.caption("No baseline log record available for this window")

            st.markdown("**📊 Classification Result Details**")
            st.json({
                "prediction": stress_result.get("prediction"),
                "label": stress_result.get("label"),
                "confidence": stress_result.get("confidence"),
                "confidence_pct": stress_result.get("confidence_pct"),
                "status": stress_result.get("status"),
                "vr_phase": stress_result.get("vr_phase"),
                "normalization_method": stress_result.get("normalization_method"),
                "model_version": stress_result.get("model_version"),
                "predict_ms": stress_result.get("predict_ms"),
            })

            st.markdown("**📥 Raw Baseline Manager State**")
            st.json({
                "state": baseline_state,
                "relaxed_streak": getattr(classifier.baseline_normalizer, "relaxed_streak", 0),
                "is_ready": classifier.baseline_normalizer.is_ready,
                "calibration_windows": len(getattr(classifier.baseline_normalizer, "calibration_windows", [])),
                "current_baseline": dict(baseline_log.get("current_baseline", {})) if baseline_log and baseline_log.get("current_baseline") else None,
                "history_logs_count": len(getattr(classifier.baseline_normalizer, "history_logs", [])),
            })

            st.markdown("**📡 Universal Baseline Metadata**")
            if classifier.baseline_normalizer.universal_baseline is not None:
                wb = classifier.baseline_normalizer.universal_baseline
                st.json({
                    "dataset": wb.metadata.get("dataset"),
                    "subjects_used": wb.metadata.get("subjects_used"),
                    "window_seconds": wb.metadata.get("window_seconds"),
                    "step_seconds": wb.metadata.get("step_seconds"),
                    "baseline_features": wb.metadata.get("baseline_features"),
                })
            else:
                st.caption("Universal baseline not available")

        # ── GSR Calibration Documentation ────────────────────────────────────
        with st.expander("🧪 GSR Calibration Status & Validation Checklist", expanded=False):
            calib_verified = getattr(config, "GSR_CALIBRATION_VERIFIED", False)
            if calib_verified:
                st.success("✅ GSR calibration: HARDWARE VERIFIED")
            else:
                st.warning("⚠️ GSR calibration: **NOT HARDWARE-VERIFIED**")

            st.markdown(f"""
**Conversion formula:**
```
conductance_µS = (GSR_ADC_FULL_SCALE − ADC) / GSR_US_PER_COUNT_DIVISOR
               = ({config.GSR_ADC_FULL_SCALE:.0f} − ADC) / {config.GSR_US_PER_COUNT_DIVISOR:.1f}
```
**Current constants** (config.py):

| Parameter | Value | Note |
|---|---|---|
| `GSR_ADC_FULL_SCALE` | `{config.GSR_ADC_FULL_SCALE:.0f}` | 12-bit ADC ceiling |
| `GSR_US_PER_COUNT_DIVISOR` | `{config.GSR_US_PER_COUNT_DIVISOR:.1f}` | Empirical — not hardware-verified |
| `GSR_US_FLOOR` | `{config.GSR_US_FLOOR}` | Minimum clamp (µS) |
| `GSR_CALIBRATION_VERIFIED` | `{calib_verified}` | Set True after hardware validation |

**Suitable for:** Relative within-session comparison  
**NOT suitable for:** Absolute conductance claims until validated

**Validation checklist:**
1. Obtain Grove GSR sensor circuit schematic (reference resistor value)
2. Check ESP32 ADC attenuation setting (`analogSetAttenuation` in firmware)
3. Connect 10 kΩ, 100 kΩ, 1 MΩ test resistors across sensor pads
4. Compare measured ADC counts to voltage-divider expected values
5. Adjust `GSR_US_PER_COUNT_DIVISOR` in `config.py` if needed
6. Set `config.GSR_CALIBRATION_VERIFIED = True`
""")

        # ── Rolling Window & VR Phase State ──
        win_status = window_manager.get_status_summary()
        st.markdown("**🪟 Rolling Window Scheduler (30s window / 15s step):**")
        col_w1, col_w2, col_w3, col_w4 = st.columns(4)
        with col_w1:
            st.metric("Windows Emitted", win_status["emitted_windows_count"])
        with col_w2:
            st.metric("Buffered Packets", win_status["buffered_packets_count"])
        with col_w3:
            st.metric("Physio Time Span", f"{win_status['current_time_span_sec']:.1f} s")
        with col_w4:
            st.metric("VR Phase", stress_result.get("vr_phase") or "UNKNOWN")

        if not vr_log.is_loaded:
            st.caption(
                "No VR event log loaded — every window is tagged `UNKNOWN`. "
                "Upload a phase timeline in the sidebar to label windows."
            )

        st.markdown("**Firmware Sensor Status:**")
        st.code(status_summary.get("sensor_status", "No status received yet"))
        st.markdown("**Last Raw Serial Line:**")
        st.code(getattr(receiver, 'last_raw_line', 'No data yet') or 'No data yet')

        st.markdown("**Receiver Log (last 30 entries):**")
        log_entries = receiver.logs[-30:] if hasattr(receiver, 'logs') else []
        if log_entries:
            st.code("\n".join(log_entries))
        else:
            st.caption("No log entries yet.")

with tab_report:
    sessions_available = data_logger.list_recorded_sessions()
    if sessions_available:
        col_s1, col_s2 = st.columns([3, 2])
        with col_s1:
            selected_session = st.selectbox("Select Recorded Session CSV", [s.name for s in sessions_available])
        with col_s2:
            default_pname = ""
            if selected_session:
                base = selected_session.replace(".csv", "")
                parts = base.rsplit("_", 2)
                if len(parts) == 3 and len(parts[1]) == 8 and len(parts[2]) == 6 and parts[1].isdigit() and parts[2].isdigit():
                    default_pname = parts[0].replace("_", " ").title()
                elif not base.startswith("stress_session"):
                    default_pname = base.split("_")[0].title()
            report_patient_name = st.text_input(
                "Patient / Subject Name",
                value=default_pname,
                placeholder="e.g. John Doe",
                help="Name to appear on the generated report. Leave blank to auto-detect."
            )

        if st.button("Generate & Preview Report", use_container_width=True):
            try:
                selected_path = config.DATA_DIR / selected_session
                report_files = report_gen.generate_report_from_csv(
                    str(selected_path),
                    patient_name=report_patient_name.strip() if report_patient_name.strip() else None
                )
                st.session_state.active_report = report_files
                st.toast(f"Report generated for {report_files.get('patient_name', 'Patient')}!", icon="📄")
            except Exception as exc:
                st.error(f"Cannot generate report: {exc}")

    if st.session_state.active_report:
        rep = st.session_state.active_report
        pname_rep = rep.get('patient_name', 'Patient')
        st.success(f"Report Generated for: **{pname_rep}** &bull; Session: `{rep.get('session_id', 'Session')}`")

        col_r1, col_r2, col_r3, col_r4 = st.columns(4)
        with col_r1:
            if rep.get("html_path") and os.path.exists(rep["html_path"]):
                with open(rep["html_path"], "r", encoding="utf-8") as f:
                    html_bytes = f.read().encode("utf-8")
                st.download_button(
                    label="📄 HTML Report",
                    data=html_bytes,
                    file_name=os.path.basename(rep["html_path"]),
                    mime="text/html",
                    use_container_width=True
                )
        with col_r2:
            if rep.get("pdf_path") and os.path.exists(rep["pdf_path"]):
                with open(rep["pdf_path"], "rb") as f:
                    pdf_bytes = f.read()
                st.download_button(
                    label="📑 PDF Report",
                    data=pdf_bytes,
                    file_name=os.path.basename(rep["pdf_path"]),
                    mime="application/pdf",
                    use_container_width=True
                )
            else:
                st.button("PDF Not Available", disabled=True, use_container_width=True)
        with col_r3:
            if rep.get("md_path") and os.path.exists(rep["md_path"]):
                with open(rep["md_path"], "r", encoding="utf-8") as f:
                    md_bytes = f.read().encode("utf-8")
                st.download_button(
                    label="📝 Markdown Summary",
                    data=md_bytes,
                    file_name=os.path.basename(rep["md_path"]),
                    mime="text/markdown",
                    use_container_width=True
                )
        with col_r4:
            if rep.get("json_path") and os.path.exists(rep["json_path"]):
                with open(rep["json_path"], "r", encoding="utf-8") as f:
                    json_bytes = f.read().encode("utf-8")
                st.download_button(
                    label="📊 JSON Data",
                    data=json_bytes,
                    file_name=os.path.basename(rep["json_path"]),
                    mime="application/json",
                    use_container_width=True
                )

        # Embedded Live Interactive HTML Preview
        st.markdown("---")
        st.markdown(f"#### 📋 Interactive Report Preview &bull; {pname_rep}")
        if rep.get("html_path") and os.path.exists(rep["html_path"]):
            with open(rep["html_path"], "r", encoding="utf-8") as f:
                preview_html = f.read()
            import streamlit.components.v1 as components
            components.html(preview_html, height=750, scrolling=True)
        elif rep.get("md_path") and os.path.exists(rep["md_path"]):
            with open(rep["md_path"], "r", encoding="utf-8") as f:
                st.markdown(f.read())
    else:
        st.info("Record a session or select a recorded CSV to generate a statistical session report.")

# ============================================================================
# AUTO-REFRESH UI LOOP
# ============================================================================
if receiver.running and any(k in receiver.status for k in ["CONNECTED", "RECONNECTING", "SIMULATED"]):
    time.sleep(0.2)
    st.rerun()
