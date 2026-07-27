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

# Custom Premium Design System CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    :root {
        --bg-primary: #06090F;
        --bg-secondary: #0C1118;
        --bg-card: rgba(15, 23, 42, 0.65);
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
        --radius-md: 12px;
        --radius-lg: 16px;
        --radius-xl: 20px;
    }

    .main {
        background: linear-gradient(170deg, var(--bg-primary) 0%, #070B12 40%, #0A0F1A 100%) !important;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }
    .block-container { padding-top: 1.5rem !important; }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0B1120 0%, #0D1424 50%, #091018 100%) !important;
        border-right: 1px solid var(--border-subtle) !important;
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, var(--accent-blue) 0%, #1D4ED8 100%) !important;
        color: #FFFFFF !important;
        border: 1px solid rgba(59, 130, 246, 0.5) !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        font-size: 13px !important;
        padding: 8px 16px !important;
        transition: all 150ms ease !important;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #3B82F6 0%, var(--accent-blue) 100%) !important;
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.35) !important;
    }

    /* Metric Cards */
    [data-testid="stMetric"] {
        background: var(--bg-card) !important;
        backdrop-filter: blur(12px) !important;
        border: 1px solid var(--border-subtle) !important;
        padding: 18px 20px !important;
        border-radius: var(--radius-lg) !important;
        transition: all 300ms ease !important;
    }
    [data-testid="stMetric"]:hover {
        border-color: var(--border-accent) !important;
        transform: translateY(-2px) !important;
    }
    [data-testid="stMetric"] label {
        color: var(--text-muted) !important;
        font-size: 11px !important;
        font-weight: 600 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.06em !important;
    }
    [data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: var(--text-primary) !important;
        font-size: 26px !important;
        font-weight: 700 !important;
    }

    /* Status Bar */
    .status-strip {
        background: var(--bg-card);
        backdrop-filter: blur(16px);
        border: 1px solid var(--border-subtle);
        border-radius: var(--radius-lg);
        padding: 14px 22px;
        margin-bottom: 20px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        flex-wrap: wrap;
        gap: 12px;
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

    /* Badges */
    .badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 5px 14px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 12px;
        text-transform: uppercase;
    }
    .badge-live { background: rgba(34, 197, 94, 0.15); color: var(--accent-green); border: 1px solid rgba(34, 197, 94, 0.3); }
    .badge-sim { background: rgba(6, 182, 212, 0.15); color: var(--accent-cyan); border: 1px solid rgba(6, 182, 212, 0.3); }
    .badge-serial { background: rgba(168, 85, 247, 0.15); color: #A855F7; border: 1px solid rgba(168, 85, 247, 0.3); }
    .badge-off { background: rgba(239, 68, 68, 0.12); color: var(--accent-red); border: 1px solid rgba(239, 68, 68, 0.25); }

    .pulse-dot { width: 7px; height: 7px; border-radius: 50%; display: inline-block; }
    .pulse-dot.green { background-color: var(--accent-green); }
    .pulse-dot.cyan { background-color: var(--accent-cyan); }
    .pulse-dot.purple { background-color: #A855F7; }

    /* Stress Card */
    .stress-card {
        background: var(--bg-card);
        backdrop-filter: blur(12px);
        border: 1px solid var(--border-subtle);
        border-radius: var(--radius-xl);
        padding: 26px;
    }
    .stress-card .label-stress { color: var(--accent-red); font-size: 34px; font-weight: 800; }
    .stress-card .label-nonstress { color: var(--accent-green); font-size: 34px; font-weight: 800; }
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
<div style="display:flex; align-items:center; gap:10px; margin-bottom:12px;">
    <div style="width:34px;height:34px;border-radius:10px;background:linear-gradient(135deg,#2563EB,#06B6D4);display:flex;align-items:center;justify-content:center;font-size:18px;">⚡</div>
    <div>
        <div style="font-size:16px;font-weight:700;color:#F1F5F9;">Telemetry Control</div>
        <div style="font-size:11px;color:#64748B;">ESP32 Wi-Fi & Serial Receiver</div>
    </div>
</div>
""", unsafe_allow_html=True)

local_ip = get_local_ip()

st.sidebar.markdown(f"**Computer Host IP**: `{local_ip}`")
st.sidebar.markdown(f"**UDP Port**: `{config.UDP_PORT}`")

st.sidebar.markdown("---")
st.sidebar.subheader("📡 Receiver Source")

input_mode = st.sidebar.radio("Select Input Mode", ["Wi-Fi UDP", "Serial COM Port", "Simulation Mode"])
selected_serial_port = st.sidebar.text_input("Serial Port", value=config.SERIAL_PORT)

col_conn1, col_conn2 = st.sidebar.columns(2)
with col_conn1:
    if st.button("Start Receiver", use_container_width=True):
        mode_key = "UDP" if input_mode == "Wi-Fi UDP" else ("SERIAL" if input_mode == "Serial COM Port" else "SIMULATION")
        receiver.start(mode=mode_key, serial_port=selected_serial_port)
        st.session_state.last_logged_packet_counter = -1
        st.toast(f"Receiver Started ({input_mode})!", icon="🚀")

with col_conn2:
    if st.button("Stop Receiver", use_container_width=True):
        receiver.stop()
        st.toast("Receiver Stopped.", icon="🛑")

st.sidebar.markdown("---")
st.sidebar.subheader("💾 Recording Session")

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
<div style="margin-bottom: 12px;">
    <h1 style="margin:0; font-size:24px; font-weight:700; color:#F1F5F9;">Wearable Stress-Monitoring Research Dashboard</h1>
    <div style="font-size:13px; color:#64748B;">CatBoost WESAD Machine Learning Model &amp; NeuroKit2 Bio-Signal Engine</div>
</div>
""", unsafe_allow_html=True)

status_summary = receiver.get_status_summary()
conn_mode = status_summary.get("connection_mode", "SIMULATION")
conn_status = status_summary["status"]

if conn_mode == "UDP" and conn_status == "CONNECTED":
    badge_html = '<span class="badge badge-live"><span class="pulse-dot green"></span>LIVE UDP</span>'
elif conn_mode == "SERIAL" and conn_status == "CONNECTED":
    badge_html = '<span class="badge badge-serial"><span class="pulse-dot purple"></span>SERIAL (COM3)</span>'
elif conn_mode == "SIMULATION" or conn_status == "SIMULATED":
    badge_html = '<span class="badge badge-sim"><span class="pulse-dot cyan"></span>SIMULATION</span>'
else:
    badge_html = '<span class="badge badge-off">DISCONNECTED</span>'

rec_status_str = f"🔴 RECORDING ({data_logger.samples_logged} samples)" if data_logger.is_recording else "⚪ IDLE"
model_status_str = "CatBoost WESAD (.cbm)" if classifier.model_loaded else "Heuristic Standby"

st.markdown(f"""
<div class="status-strip">
    <div style="display:flex; align-items:center; gap:16px; flex-wrap:wrap;">
        <div class="segment"><span class="label">Status</span> {badge_html}</div>
        <div class="segment"><span class="label">Source</span> <code>{status_summary['last_remote_ip']}</code></div>
        <div class="segment"><span class="label">Rate</span> <code>{status_summary['rate_hz']} Hz</code></div>
        <div class="segment"><span class="label">Model Engine</span> <code>{model_status_str}</code></div>
    </div>
    <div class="segment"><span class="label">Session</span> <code>{rec_status_str}</code></div>
</div>
""", unsafe_allow_html=True)

# Fetch latest raw telemetry packets from receiver
latest_packets = receiver.get_latest_data(count=config.BUFFER_SIZE)

# Run preprocessing (NeuroKit2 23 feature extraction) & CatBoost model inference
processed_batch = preprocessor.process_batch(latest_packets)
feature_df = processed_batch.get("feature_df")
stress_result = classifier.predict(feature_df if feature_df is not None else processed_batch)

# Log packets to CSV when session recording is active
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
        label="Heart Rate (PPG/BVP)",
        value=f"{processed_batch['bpm']} BPM",
        delta=f"HRV RMSSD: {processed_batch['rmssd']} ms"
    )

with col_m2:
    st.metric(
        label="Skin Conductance (GSR)",
        value=f"{processed_batch['scl_mean']} μS",
        delta=f"SCR Spikes: {processed_batch['scr_count']}"
    )

with col_m3:
    st.metric(
        label="Motion Activity (IMU)",
        value=f"{processed_batch['activity_index']} g",
        delta=f"State: {processed_batch['motion_state']}"
    )

with col_m4:
    st.metric(
        label="CatBoost Model Prediction",
        value=stress_result['label'],
        delta=f"Conf: {stress_result['confidence_pct']}% | {stress_result['trend']}"
    )

# ============================================================================
# REAL-TIME CHARTS & SIGNAL VISUALIZATION
# ============================================================================
tab_live, tab_wesad, tab_pipeline, tab_report = st.tabs([
    "📈 Live Bio-Signals",
    "🧠 WESAD CatBoost Model (23 Features)",
    "⚙️ Preprocessing Pipeline",
    "📑 Session Reports"
])

def get_chart_layout(title: str, height: int = 250) -> dict:
    return dict(
        title=dict(text=title, font=dict(size=14, color="#94A3B8"), x=0.01, y=0.97),
        height=height, margin=dict(l=12, r=12, t=40, b=16),
        template="plotly_dark",
        paper_bgcolor="rgba(15, 23, 42, 0.5)",
        plot_bgcolor="rgba(11, 17, 32, 0.4)",
        font=dict(color="#94A3B8", size=11),
        xaxis=dict(gridcolor="rgba(56, 82, 130, 0.12)", showgrid=True),
        yaxis=dict(gridcolor="rgba(56, 82, 130, 0.12)", showgrid=True),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=11), orientation="h", y=1.08)
    )

with tab_live:
    if not latest_packets:
        st.info("Waiting for data stream... Click 'Start Receiver' in sidebar or select 'Simulation Mode'.")
    else:
        df_packets = pd.DataFrame(latest_packets)

        # PPG Chart
        fig_ppg = go.Figure()
        fig_ppg.add_trace(go.Scatter(y=df_packets["ppg_raw"], name="Raw PPG ADC", line=dict(color="rgba(148, 163, 184, 0.45)", width=1)))
        if "ppg_filtered" in processed_batch and len(processed_batch["ppg_filtered"]) == len(df_packets):
            fig_ppg.add_trace(go.Scatter(y=processed_batch["ppg_filtered"], name="Filtered Cardiac Pulse", line=dict(color="#3B82F6", width=2.5), fill='tozeroy', fillcolor='rgba(59, 130, 246, 0.06)'))
        fig_ppg.update_layout(**get_chart_layout("PPG Cardiac Waveform (Photoplethysmography)", 260))
        st.plotly_chart(fig_ppg, use_container_width=True)

        # GSR Chart
        fig_gsr = go.Figure()
        fig_gsr.add_trace(go.Scatter(y=df_packets["gsr_raw"], name="Raw Conductance (μS)", line=dict(color="rgba(245, 158, 11, 0.6)", width=1.5)))
        if "gsr_tonic" in processed_batch and len(processed_batch["gsr_tonic"]) == len(df_packets):
            fig_gsr.add_trace(go.Scatter(y=processed_batch["gsr_tonic"], name="Tonic SCL Level", line=dict(color="#22C55E", width=2.5), fill='tozeroy', fillcolor='rgba(34, 197, 94, 0.05)'))
        fig_gsr.update_layout(**get_chart_layout("Galvanic Skin Response / Electrodermal Activity", 240))
        st.plotly_chart(fig_gsr, use_container_width=True)

        # IMU Chart
        fig_imu = go.Figure()
        fig_imu.add_trace(go.Scatter(y=df_packets["imu_ax"], name="Accel X", line=dict(color="#EF4444", width=1.5)))
        fig_imu.add_trace(go.Scatter(y=df_packets["imu_ay"], name="Accel Y", line=dict(color="#A855F7", width=1.5)))
        fig_imu.add_trace(go.Scatter(y=df_packets["imu_az"], name="Accel Z", line=dict(color="#3B82F6", width=1.5)))
        fig_imu.update_layout(**get_chart_layout("IMU 3-Axis Accelerometer Dynamics (g)", 240))
        st.plotly_chart(fig_imu, use_container_width=True)

with tab_wesad:
    col_s1, col_s2 = st.columns([1, 1])

    with col_s1:
        lbl_cls = "label-stress" if stress_result["label"] == "STRESS" else "label-nonstress"
        color_fill = "linear-gradient(90deg, #EF4444, #F87171)" if stress_result["label"] == "STRESS" else "linear-gradient(90deg, #22C55E, #4ADE80)"

        st.markdown(f"""
        <div class="stress-card">
            <div style="font-size:12px; font-weight:600; text-transform:uppercase; color:#64748B;">CatBoost WESAD Model Prediction</div>
            <div class="{lbl_cls}">{stress_result['label']}</div>
            <div style="height:6px; border-radius:3px; background:rgba(30,41,59,0.6); margin:14px 0;">
                <div style="width:{stress_result['confidence_pct']}%; height:100%; border-radius:3px; background:{color_fill};"></div>
            </div>
            <div style="display:flex; justify-content:space-between; font-size:13px; color:#94A3B8;">
                <span>Model Confidence: <strong style="color:#F1F5F9;">{stress_result['confidence_pct']}%</strong></span>
                <span>Stress Probability: <strong style="color:#F1F5F9;">{stress_result['stress_score']}%</strong></span>
            </div>
            <div style="margin-top:14px; padding-top:12px; border-top:1px solid rgba(56,82,130,0.15); font-size:12px; color:#64748B;">
                Source: <code>{stress_result['model_source']}</code>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_s2:
        st.subheader("Extracted 23 WESAD Feature Vector")
        if feature_df is not None:
            # Display transpose table of exact 23 features
            feat_table = feature_df.T.reset_index()
            feat_table.columns = ["WESAD Feature Name", "Value"]
            st.dataframe(feat_table, use_container_width=True, hide_index=True, height=280)
        else:
            st.caption("No active feature window extracted yet.")

with tab_pipeline:
    col_p1, col_p2 = st.columns(2)
    with col_p1:
        st.markdown(f"""
        <div style="background:rgba(15,23,42,0.65); border:1px solid rgba(56,82,130,0.25); border-radius:12px; padding:20px;">
            <h4 style="margin:0 0 12px; color:#F1F5F9;">🎵 PPG / BVP NeuroKit2 Pipeline</h4>
            <p style="font-size:13px; color:#94A3B8; margin:0;">Uses <code>nk.ppg_process()</code> and <code>nk.hrv()</code> to compute 6 heart-rate variability features: <code>hr</code>, <code>rmssd</code>, <code>sdnn</code>, <code>pnn50</code>, <code>ibi_mean</code>, <code>ibi_std</code> over 30s windows.</p>
        </div>
        """, unsafe_allow_html=True)

    with col_p2:
        st.markdown(f"""
        <div style="background:rgba(15,23,42,0.65); border:1px solid rgba(56,82,130,0.25); border-radius:12px; padding:20px;">
            <h4 style="margin:0 0 12px; color:#F1F5F9;">⚡ EDA / GSR NeuroKit2 Pipeline</h4>
            <p style="font-size:13px; color:#94A3B8; margin:0;">Uses <code>nk.eda_clean()</code>, <code>nk.eda_phasic()</code>, and <code>nk.eda_peaks()</code> to extract 9 electrodermal features: <code>eda_mean</code>, <code>eda_std</code>, <code>eda_slope</code>, <code>scl_mean</code>, <code>phasic_mean</code>, <code>scr_count</code>, <code>scr_amp_mean</code>, <code>scr_rise_mean</code>, <code>scr_recovery_mean</code>.</p>
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
                st.download_button(label="Download HTML Report", data=html_bytes, file_name=os.path.basename(rep["html_path"]), mime="text/html", use_container_width=True)
        with col_r2:
            if os.path.exists(rep["md_path"]):
                with open(rep["md_path"], "r", encoding="utf-8") as f:
                    md_bytes = f.read().encode("utf-8")
                st.download_button(label="Download Markdown Summary", data=md_bytes, file_name=os.path.basename(rep["md_path"]), mime="text/markdown", use_container_width=True)
    else:
        st.info("Record a session or select a recorded CSV to generate a statistical session report.")

# ============================================================================
# AUTO-REFRESH UI LOOP
# ============================================================================
time.sleep(0.2)
st.rerun()
