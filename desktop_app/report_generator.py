import os
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import config

class SessionReportGenerator:
    """Generates clinical research session report documents in HTML and Markdown."""

    def __init__(self, reports_dir: Path = config.REPORTS_DIR):
        self.reports_dir = Path(reports_dir)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    def generate_report_from_csv(self, csv_file_path: str) -> Dict[str, str]:
        """Read session CSV file and compile statistical report files."""
        if not os.path.exists(csv_file_path):
            raise FileNotFoundError(f"CSV file not found: {csv_file_path}")

        df = pd.read_csv(csv_file_path)
        if df.empty:
            raise ValueError("CSV session file is empty.")

        # Compute Session Metrics
        session_id = df["session_id"].iloc[0] if "session_id" in df.columns else "UNKNOWN_SESSION"
        total_samples = len(df)
        
        # Estimate duration from ISO timestamps if available
        duration_sec = total_samples / config.SAMPLING_RATE_HZ
        if "timestamp_iso" in df.columns and len(df) > 1:
            try:
                t_start = datetime.fromisoformat(str(df["timestamp_iso"].iloc[0]))
                t_end = datetime.fromisoformat(str(df["timestamp_iso"].iloc[-1]))
                duration_sec = (t_end - t_start).total_seconds()
            except Exception:
                pass

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
            "date_str": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "duration_sec": round(duration_sec, 1),
            "total_samples": total_samples,
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
        html_path = self.reports_dir / f"{session_id}_report.html"
        md_path = self.reports_dir / f"{session_id}_report.md"

        self._render_html_report(stats, html_path)
        self._render_markdown_report(stats, md_path)

        return {
            "html_path": str(html_path),
            "md_path": str(md_path),
            "session_id": session_id
        }

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

        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Stress Monitoring Session Report - {stats['session_id']}</title>
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
            grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
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
                    <h1>Bio-Signal Session Report</h1>
                    <div class="meta">Session: {stats['session_id']} &middot; {stats['date_str']}</div>
                </div>
            </div>
            <div class="badge">{stats['primary_state'].replace('_', ' ')}</div>
        </div>

        <div class="grid">
            <div class="card">
                <div class="title">Duration</div>
                <div class="value">{stats['duration_sec']}s</div>
            </div>
            <div class="card">
                <div class="title">Total Packets</div>
                <div class="value">{stats['total_samples']}</div>
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

        <div class="footer">
            <strong>ESP32 Wearable Stress-Monitoring Research Prototype</strong> &middot; Telemetry &amp; Analytics Platform
        </div>
    </div>
</body>
</html>
"""
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_content)

    def _render_markdown_report(self, stats: Dict[str, Any], output_path: Path):
        md_content = f"""# Bio-Signal Stress Monitoring Session Report

**Session ID**: `{stats['session_id']}`  
**Generated At**: {stats['date_str']}  
**Primary Classified State**: **{stats['primary_state']}**

---

### Session Overview
- **Total Duration**: {stats['duration_sec']} seconds
- **Total Samples Collected**: {stats['total_samples']} packets
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
*Report generated automatically by ESP32 Desktop Dashboard Receiver.*
"""
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(md_content)
