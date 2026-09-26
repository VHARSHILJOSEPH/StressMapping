import os
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import config

class SessionReportGenerator:
    """Generates Mental Health Summary (MHS) session report documents in HTML and Markdown."""

    def __init__(self, reports_dir: Path = config.REPORTS_DIR):
        self.reports_dir = Path(reports_dir)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    def generate_report_from_csv(self, csv_file_path: str, patient_name: Optional[str] = None) -> Dict[str, str]:
        """Read session CSV file and compile MHS statistical report files."""
        if not os.path.exists(csv_file_path):
            raise FileNotFoundError(f"CSV file not found: {csv_file_path}")

        try:
            df = pd.read_csv(csv_file_path)
        except Exception as exc:
            raise ValueError(f"Could not read CSV file: {exc}")

        if df.empty:
            raise ValueError(f"Session CSV file '{os.path.basename(csv_file_path)}' is empty (0 samples). Please ensure data is logged before generating a report.")

        # Compute Session Metrics
        session_id = str(df["session_id"].iloc[0]) if "session_id" in df.columns and pd.notna(df["session_id"].iloc[0]) else Path(csv_file_path).stem
        total_samples = len(df)

        # Extract Patient Name
        if not patient_name:
            if "patient_name" in df.columns and pd.notna(df["patient_name"].iloc[0]):
                p_val = str(df["patient_name"].iloc[0]).strip()
                if p_val and p_val.lower() != "nan":
                    patient_name = p_val

        if not patient_name:
            parts = session_id.rsplit("_", 2)
            if len(parts) == 3 and len(parts[1]) == 8 and len(parts[2]) == 6 and parts[1].isdigit() and parts[2].isdigit():
                patient_name = parts[0].replace("_", " ").title()
            elif session_id.startswith("stress_session_"):
                patient_name = "Anonymous"
            else:
                patient_name = session_id.replace("_", " ").title()
        
        # Estimate duration from ISO timestamps if available
        duration_sec = total_samples / config.SAMPLING_RATE_HZ
        if "timestamp_iso" in df.columns and len(df) > 1:
            try:
                t_start = datetime.fromisoformat(str(df["timestamp_iso"].iloc[0]))
                t_end = datetime.fromisoformat(str(df["timestamp_iso"].iloc[-1]))
                duration_sec = (t_end - t_start).total_seconds()
            except Exception:
                pass

        # Compute actual sampling rate from timestamps
        actual_rate_hz = round(total_samples / max(duration_sec, 0.1), 1)

        # Determine connection mode from data
        mode_col = df["mode"] if "mode" in df.columns else pd.Series(["UNKNOWN"])
        primary_mode = mode_col.mode().iloc[0] if not mode_col.empty else "UNKNOWN"
        connection_label = "USB Serial (C-to-C)" if primary_mode == "SERIAL" else (
            "Live UDP (Wi-Fi)" if primary_mode == "LIVE_UDP" else (
            "Simulation" if primary_mode == "SIMULATION" else primary_mode
        ))

        # Signal Statistics
        ppg_raw = df["ppg_raw"] if "ppg_raw" in df.columns else pd.Series([0])
        gsr_raw = df["gsr_raw"] if "gsr_raw" in df.columns else pd.Series([0])
        imu_mag = df["imu_magnitude"] if "imu_magnitude" in df.columns else pd.Series([1.0])

        ppg_avg = float(ppg_raw.mean())
        ppg_std = float(ppg_raw.std())
        gsr_avg = float(gsr_raw.mean())
        gsr_max = float(gsr_raw.max())
        motion_avg = float(imu_mag.mean())

        # Stress State Breakdown
        stress_col = df["stress_state"] if "stress_state" in df.columns else pd.Series(["UNKNOWN"])
        state_counts = stress_col.value_counts(normalize=True) * 100.0
        
        pct_relaxed = round(float(state_counts.get("RELAXED", 0.0)), 1)
        pct_low = round(float(state_counts.get("LOW_STRESS", 0.0)), 1)
        pct_mod = round(float(state_counts.get("MODERATE_STRESS", 0.0)), 1)
        pct_high = round(float(state_counts.get("HIGH_STRESS", 0.0)), 1)

        # Primary Stress Classification for Session
        primary_state = stress_col.mode().iloc[0] if not stress_col.empty else "UNKNOWN"

        stats = {
            "session_id": session_id,
            "patient_name": patient_name,
            "date_str": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "duration_sec": round(duration_sec, 1),
            "total_samples": total_samples,
            "actual_rate_hz": actual_rate_hz,
            "connection_label": connection_label,
            "primary_state": primary_state,
            "ppg_avg": round(ppg_avg, 1),
            "ppg_std": round(ppg_std, 1),
            "gsr_avg_uS": round(gsr_avg, 2),
            "gsr_max_uS": round(gsr_max, 2),
            "motion_avg_g": round(motion_avg, 3),
            "pct_relaxed": pct_relaxed,
            "pct_low": pct_low,
            "pct_mod": pct_mod,
            "pct_high": pct_high,
        }

        # Render Output Files
        html_path = self.reports_dir / f"{session_id}_mhs_report.html"
        md_path = self.reports_dir / f"{session_id}_mhs_report.md"
        json_path = self.reports_dir / f"{session_id}_mhs_report.json"
        pdf_path = self.reports_dir / f"{session_id}_mhs_report.pdf"

        # Generate Non-Clinical Decision Support Guidance
        recommendations = []
        if pct_high > 30.0:
            recommendations.append("HIGH STRESS ALERT: Subject experienced sustained high stress >30% of session duration. Recommended: Guided breathing / VR decompression protocol.")
        elif pct_mod + pct_high > 50.0:
            recommendations.append("MODERATE STRESS ELEVATION: Subject displayed elevated stress >50% of session. Recommended: Adjust task difficulty or introduce rest breaks.")
        else:
            recommendations.append("STABLE AUTONOMIC STATE: Physiological markers remained largely within baseline/low stress range.")

        if motion_avg > 1.2:
            recommendations.append("MOTION ARTIFACT NOTICE: High average physical movement detected (>1.2g). Ensure electrodes remained securely attached.")

        stats["recommendations"] = recommendations

        self._render_html_report(stats, html_path)
        self._render_markdown_report(stats, md_path)
        self._render_json_report(stats, json_path)

        try:
            from report_generator import generate_pdf_report
            generate_pdf_report(csv_file_path, output_pdf_path=str(pdf_path), patient_name=patient_name)
        except Exception:
            pass

        result = {
            "html_path": str(html_path),
            "md_path": str(md_path),
            "json_path": str(json_path),
            "session_id": session_id,
            "patient_name": patient_name,
        }
        if pdf_path.exists():
            result["pdf_path"] = str(pdf_path)

        return result

    def _render_json_report(self, stats: Dict[str, Any], output_path: Path):
        """Export structured non-clinical decision-support data as JSON."""
        import json
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)

    def _render_html_report(self, stats: Dict[str, Any], output_path: Path):
        # Determine badge color based on primary state
        state = stats['primary_state']
        if state == "RELAXED":
            badge_bg = "linear-gradient(135deg, rgba(34,197,94,0.15), rgba(34,197,94,0.08))"
            badge_color = "#22C55E"
            badge_border = "rgba(34,197,94,0.3)"
        elif state == "LOW_STRESS":
            badge_bg = "linear-gradient(135deg, rgba(6,182,212,0.15), rgba(6,182,212,0.08))"
            badge_color = "#06B6D4"
            badge_border = "rgba(6,182,212,0.3)"
        elif state == "MODERATE_STRESS":
            badge_bg = "linear-gradient(135deg, rgba(245,158,11,0.15), rgba(245,158,11,0.08))"
            badge_color = "#F59E0B"
            badge_border = "rgba(245,158,11,0.3)"
        else:
            badge_bg = "linear-gradient(135deg, rgba(239,68,68,0.15), rgba(239,68,68,0.08))"
            badge_color = "#EF4444"
            badge_border = "rgba(239,68,68,0.3)"

        disclaimer_text = "This report summarizes model-estimated physiological stress responses and signal characteristics during the recorded session. It is intended for research and non-clinical decision support and does not constitute a medical or psychiatric diagnosis."

        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MHS Report - {stats['session_id']}</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
    <style>
        *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
            background: linear-gradient(170deg, #06090F 0%, #070B12 40%, #0A0F1A 100%);
            color: #C9D1D9;
            min-height: 100vh;
            padding: 48px 24px;
            -webkit-font-smoothing: antialiased;
        }}
        .container {{
            max-width: 860px;
            margin: 0 auto;
            background: rgba(15, 23, 42, 0.65);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border: 1px solid rgba(56, 82, 130, 0.25);
            border-radius: 20px;
            padding: 40px;
            box-shadow: 0 4px 16px rgba(0,0,0,0.4), 0 16px 48px rgba(0,0,0,0.2);
        }}
        .header {{
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            padding-bottom: 28px;
            margin-bottom: 32px;
            border-bottom: 1px solid rgba(56, 82, 130, 0.2);
        }}
        .header-left {{
            display: flex;
            align-items: center;
            gap: 14px;
        }}
        .logo-mark {{
            width: 44px; height: 44px;
            border-radius: 12px;
            background: linear-gradient(135deg, #2563EB 0%, #06B6D4 100%);
            display: flex; align-items: center; justify-content: center;
            font-size: 22px;
            box-shadow: 0 4px 14px rgba(37, 99, 235, 0.3);
            flex-shrink: 0;
        }}
        .header h1 {{
            color: #F1F5F9;
            font-size: 22px;
            font-weight: 700;
            letter-spacing: -0.03em;
            line-height: 1.2;
        }}
        .header .meta {{
            color: #64748B;
            font-size: 12px;
            font-weight: 400;
            margin-top: 4px;
        }}
        .badge {{
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 6px 16px;
            border-radius: 20px;
            font-weight: 700;
            font-size: 12px;
            letter-spacing: 0.03em;
            text-transform: uppercase;
            background: {badge_bg};
            color: {badge_color};
            border: 1px solid {badge_border};
            white-space: nowrap;
        }}
        .badge::before {{
            content: '';
            width: 7px; height: 7px;
            border-radius: 50%;
            background: {badge_color};
        }}
        .grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(155px, 1fr));
            gap: 14px;
            margin-bottom: 28px;
        }}
        .card {{
            background: rgba(22, 33, 55, 0.7);
            border: 1px solid rgba(56, 82, 130, 0.2);
            border-radius: 14px;
            padding: 20px;
            position: relative;
            overflow: hidden;
            transition: border-color 0.3s, box-shadow 0.3s;
        }}
        .card:hover {{
            border-color: rgba(37, 99, 235, 0.35);
            box-shadow: 0 0 20px rgba(37, 99, 235, 0.1);
        }}
        .card::before {{
            content: '';
            position: absolute; top: 0; left: 0; right: 0;
            height: 2px;
            background: linear-gradient(90deg, #2563EB, #06B6D4);
            opacity: 0;
            transition: opacity 0.3s;
        }}
        .card:hover::before {{ opacity: 1; }}
        .card .title {{
            font-size: 11px;
            color: #64748B;
            text-transform: uppercase;
            letter-spacing: 0.07em;
            font-weight: 600;
        }}
        .card .value {{
            font-size: 26px;
            font-weight: 700;
            color: #F1F5F9;
            margin-top: 8px;
            letter-spacing: -0.02em;
        }}
        .card .sub {{
            font-size: 11px;
            color: #64748B;
            margin-top: 4px;
        }}
        .section {{
            margin-top: 28px;
            background: rgba(22, 33, 55, 0.5);
            border: 1px solid rgba(56, 82, 130, 0.18);
            border-radius: 16px;
            padding: 28px;
        }}
        .section h3 {{
            color: #94A3B8;
            font-size: 13px;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            margin: 0 0 20px;
        }}
        .bar-row {{
            display: flex;
            align-items: center;
            gap: 14px;
            margin-bottom: 14px;
        }}
        .bar-row:last-child {{ margin-bottom: 0; }}
        .bar-row .label {{
            width: 130px;
            font-size: 13px;
            font-weight: 500;
            color: #94A3B8;
            flex-shrink: 0;
        }}
        .bar-row .track {{
            flex: 1;
            height: 8px;
            border-radius: 4px;
            background: rgba(30, 41, 59, 0.6);
            overflow: hidden;
        }}
        .bar-row .fill {{
            height: 100%;
            border-radius: 4px;
            transition: width 0.6s cubic-bezier(0.4, 0, 0.2, 1);
        }}
        .bar-row .pct {{
            width: 48px;
            text-align: right;
            font-size: 13px;
            font-weight: 600;
            color: #F1F5F9;
            font-family: 'SF Mono', 'Fira Code', 'Cascadia Code', monospace;
        }}
        .disclaimer-box {{
            margin-top: 24px;
            padding: 16px;
            border-radius: 12px;
            background: rgba(100, 116, 139, 0.1);
            border: 1px solid rgba(100, 116, 139, 0.25);
            font-size: 12px;
            color: #94A3B8;
            line-height: 1.6;
        }}
        .footer {{
            margin-top: 36px;
            padding-top: 20px;
            border-top: 1px solid rgba(56, 82, 130, 0.15);
            text-align: center;
            font-size: 11px;
            color: #475569;
            letter-spacing: 0.02em;
        }}
        .footer strong {{
            color: #64748B;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <div class="header-left">
                <div class="logo-mark">&#9889;</div>
                <div>
                    <h1>Non-Clinical Decision Support MHS Report</h1>
                    <div class="meta">Patient: <strong style="color:#60A5FA;">{stats['patient_name']}</strong> &middot; Session: {stats['session_id']} &middot; {stats['date_str']}</div>
                </div>
            </div>
            <div class="badge">{stats['primary_state'].replace('_', ' ')}</div>
        </div>

        <div class="grid">
            <div class="card">
                <div class="title">Patient</div>
                <div class="value" style="font-size:18px; word-break:break-word;">{stats['patient_name']}</div>
                <div class="sub">Subject Name</div>
            </div>
            <div class="card">
                <div class="title">Duration</div>
                <div class="value">{stats['duration_sec']}s</div>
            </div>
            <div class="card">
                <div class="title">Total Samples</div>
                <div class="value">{stats['total_samples']}</div>
            </div>
            <div class="card">
                <div class="title">Sampling Rate</div>
                <div class="value">{stats['actual_rate_hz']} Hz</div>
            </div>
            <div class="card">
                <div class="title">Connection</div>
                <div class="value" style="font-size:16px;">{stats['connection_label']}</div>
            </div>
            <div class="card">
                <div class="title">Avg GSR</div>
                <div class="value">{stats['gsr_avg_uS']} &mu;S</div>
            </div>
            <div class="card">
                <div class="title">Motion Index</div>
                <div class="value">{stats['motion_avg_g']} g</div>
            </div>
        </div>

        <div class="section">
            <h3>Autonomic Stress State Distribution</h3>

            <div class="bar-row">
                <span class="label">Relaxed</span>
                <div class="track"><div class="fill" style="width:{stats['pct_relaxed']}%; background:linear-gradient(90deg,#22C55E,#4ADE80);"></div></div>
                <span class="pct">{stats['pct_relaxed']}%</span>
            </div>
            <div class="bar-row">
                <span class="label">Low Stress</span>
                <div class="track"><div class="fill" style="width:{stats['pct_low']}%; background:linear-gradient(90deg,#06B6D4,#22D3EE);"></div></div>
                <span class="pct">{stats['pct_low']}%</span>
            </div>
            <div class="bar-row">
                <span class="label">Moderate Stress</span>
                <div class="track"><div class="fill" style="width:{stats['pct_mod']}%; background:linear-gradient(90deg,#F59E0B,#FBBF24);"></div></div>
                <span class="pct">{stats['pct_mod']}%</span>
            </div>
            <div class="bar-row">
                <span class="label">High Stress</span>
                <div class="track"><div class="fill" style="width:{stats['pct_high']}%; background:linear-gradient(90deg,#EF4444,#F87171);"></div></div>
                <span class="pct">{stats['pct_high']}%</span>
            </div>
        </div>

        <div class="disclaimer-box">
            <strong>Non-Clinical Decision Support Disclaimer:</strong> {disclaimer_text}
        </div>

        <div class="footer">
            <strong>ESP32 Wearable Stress-Monitoring Research Prototype</strong> &middot; Non-Clinical Decision Support Platform
        </div>
    </div>
</body>
</html>
"""
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_content)

    def _render_markdown_report(self, stats: Dict[str, Any], output_path: Path):
        disclaimer_text = (
            "This report summarizes model-estimated physiological stress responses and signal characteristics "
            "during the recorded session. It is intended for research and non-clinical decision support and does not "
            "constitute a medical or psychiatric diagnosis."
        )

        md_content = f"""# Non-Clinical Decision Support MHS Report

**Patient Name**: **{stats['patient_name']}**  
**Session ID**: `{stats['session_id']}`  
**Generated At**: {stats['date_str']}  
**Primary Classified State**: **{stats['primary_state']}**  
**Connection Mode**: {stats['connection_label']}

---

### Session Overview
- **Total Duration**: {stats['duration_sec']} seconds
- **Total Samples Collected**: {stats['total_samples']} packets
- **Actual Sampling Rate**: {stats['actual_rate_hz']} Hz
- **Mean GSR Conductance**: {stats['gsr_avg_uS']} μS (Max: {stats['gsr_max_uS']} μS)
- **Mean Physical Acceleration**: {stats['motion_avg_g']} g

---

### Stress State Percentage Distribution
| Stress Level | Percentage |
| :--- | :--- |
| **RELAXED** | {stats['pct_relaxed']}% |
| **LOW STRESS** | {stats['pct_low']}% |
| **MODERATE STRESS** | {stats['pct_mod']}% |
| **HIGH STRESS** | {stats['pct_high']}% |

---

### Non-Clinical Decision Support Disclaimer
> **Notice**: {disclaimer_text}

---
*Non-Clinical Decision Support Report generated by ESP32 Wearable Stress-Monitoring Research Platform.*
"""
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(md_content)
