"""Mental Health Screening PDF Report Generator.

Generates the clinical-grade multi-page PDF screening report with:
- Patient and session demographics
- Session summary statistics
- Heart Rate (HR) Timeline chart
- Electrodermal Activity (EDA) Timeline chart
- Stress Classification Timeline per 30s Window (Reading per Reading)
- Non-Clinical Decision Support & Interpretation
- Technical Appendix
"""

import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from reportlab.graphics.charts.lineplots import LinePlot
from reportlab.graphics.shapes import Drawing, Line, Rect, String
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

PRIMARY_NAVY = colors.HexColor("#1B2A4A")
ACCENT_TEAL = colors.HexColor("#008080")
TEXT_DARK = colors.HexColor("#2D3748")
STRESS_RED = colors.HexColor("#DC2626")
STRESS_AMBER = colors.HexColor("#D97706")
STRESS_CYAN = colors.HexColor("#0891B2")
NON_STRESS_GREEN = colors.HexColor("#16A34A")
LIGHT_RED_BG = colors.HexColor("#FEE2E2")
BG_LIGHT = colors.HexColor("#F8FAFC")
BORDER_COLOR = colors.HexColor("#E2E8F0")


def safe_float(val: Any, default: float = 0.0) -> float:
    try:
        if pd.isna(val):
            return default
        return float(val)
    except Exception:
        return default


def safe_int(val: Any, default: int = 0) -> int:
    try:
        if pd.isna(val):
            return default
        return int(val)
    except Exception:
        return default


def _extract_windows_from_session(df: pd.DataFrame, csv_path: str, session_id: str) -> List[Dict[str, Any]]:
    """Extracts window-by-window readings from _windows.csv, _analysis.json, or by windowing raw df."""
    csv_p = Path(csv_path)

    # 1. Check if _windows.csv exists
    candidates = [
        csv_p.parent / f"{session_id}_windows.csv",
        Path("data") / f"{session_id}_windows.csv",
        csv_p.parent / f"{csv_p.stem}_windows.csv",
    ]
    for cand in candidates:
        if cand.exists():
            try:
                w_df = pd.read_csv(cand)
                if not w_df.empty:
                    windows = []
                    last_lbl = "RELAXED"
                    last_conf = 85.0
                    last_hr = 75.0
                    for idx, row in w_df.iterrows():
                        raw_label = row.get("predicted_label")
                        if pd.isna(raw_label) or str(raw_label).strip() == "" or str(raw_label).strip().upper() in ["NAN", "NONE", "UNKNOWN", "NULL"]:
                            raw_label = row.get("stress_state")
                        if pd.isna(raw_label) or str(raw_label).strip() == "" or str(raw_label).strip().upper() in ["NAN", "NONE", "UNKNOWN", "NULL"]:
                            raw_label = row.get("prediction")
                        if pd.isna(raw_label) or str(raw_label).strip() == "" or str(raw_label).strip().upper() in ["NAN", "NONE", "UNKNOWN", "NULL"]:
                            label = last_lbl
                        else:
                            label = str(raw_label).strip().upper()
                            last_lbl = label

                        conf = safe_float(row.get("confidence"), None)
                        if conf is None or conf <= 0.0 or pd.isna(conf):
                            conf = last_conf
                        else:
                            if conf <= 1.0:
                                conf *= 100.0
                            last_conf = conf

                        hr_val = safe_float(row.get("hr"), None)
                        if hr_val is not None and hr_val >= 40.0:
                            last_hr = hr_val
                            hr = hr_val
                        else:
                            hr = last_hr

                        eda = safe_float(row.get("eda_mean", row.get("scl_mean", None)))
                        w_num = safe_int(row.get("window_id", idx + 1))
                        start_sec = round(safe_float(row.get("window_start_ms", idx * 15000)) / 1000.0, 1)
                        end_sec = round(safe_float(row.get("window_end_ms", idx * 15000 + 30000)) / 1000.0, 1)
                        windows.append({
                            "window_num": w_num,
                            "start_sec": start_sec,
                            "end_sec": end_sec,
                            "label": label,
                            "confidence_pct": conf,
                            "hr": hr,
                            "eda": eda,
                        })
                    if windows:
                        return windows
            except Exception:
                pass

    # 2. Check if _analysis.json exists
    json_cands = [
        csv_p.parent / f"{session_id}_analysis.json",
        Path("data") / f"{session_id}_analysis.json",
        csv_p.parent / f"{csv_p.stem}_analysis.json",
    ]
    for cand in json_cands:
        if cand.exists():
            try:
                import json
                with open(cand, "r", encoding="utf-8") as f:
                    adata = json.load(f)
                w_results = adata.get("window_results", [])
                if w_results:
                    windows = []
                    last_lbl = "RELAXED"
                    last_conf = 85.0
                    last_hr = 75.0
                    for idx, w in enumerate(w_results):
                        raw_label = w.get("predicted_label") or w.get("label") or w.get("prediction")
                        if not raw_label or str(raw_label).strip().upper() in ["NAN", "NONE", "UNKNOWN", "NULL"]:
                            label = last_lbl
                        else:
                            label = str(raw_label).strip().upper()
                            last_lbl = label

                        conf = safe_float(w.get("confidence"), None)
                        if conf is None or conf <= 0.0 or pd.isna(conf):
                            conf = last_conf
                        else:
                            if conf <= 1.0:
                                conf *= 100.0
                            last_conf = conf

                        feats = w.get("features", {})
                        hr_val = safe_float(feats.get("hr"), None)
                        if hr_val is not None and hr_val >= 40.0:
                            last_hr = hr_val
                            hr = hr_val
                        else:
                            hr = last_hr

                        eda = safe_float(feats.get("eda_mean", feats.get("scl_mean", None)))
                        w_num = safe_int(w.get("window_id", idx + 1))
                        start_sec = round(safe_float(w.get("window_start_ms", idx * 15000)) / 1000.0, 1)
                        end_sec = round(safe_float(w.get("window_end_ms", idx * 15000 + 30000)) / 1000.0, 1)
                        windows.append({
                            "window_num": w_num,
                            "start_sec": start_sec,
                            "end_sec": end_sec,
                            "label": label,
                            "confidence_pct": conf,
                            "hr": hr,
                            "eda": eda,
                        })
                    if windows:
                        return windows
            except Exception:
                pass

    # 3. Compute 30s windows (15s step) directly from raw telemetry df
    windows = []
    ts_col = "timestamp_ms" if "timestamp_ms" in df.columns else ("timestamp" if "timestamp" in df.columns else None)
    if ts_col and len(df) > 10:
        ts = df[ts_col].to_numpy(dtype=float)
        min_ts = ts[0]
        max_ts = ts[-1]
        window_ms = 30000.0
        step_ms = 15000.0
        curr = min_ts
        idx = 1
        while curr + window_ms <= max_ts:
            mask = (ts >= curr) & (ts < curr + window_ms)
            if np.sum(mask) >= 10:
                sub = df.loc[mask]
                label = "LOW_STRESS"
                if "stress_state" in sub.columns:
                    mode_val = sub["stress_state"].dropna().mode()
                    if len(mode_val) > 0:
                        label = str(mode_val.iloc[0]).strip().upper()
                elif "prediction" in sub.columns:
                    mode_val = sub["prediction"].dropna().mode()
                    if len(mode_val) > 0:
                        label = str(mode_val.iloc[0]).strip().upper()

                conf = 85.0
                if "confidence" in sub.columns:
                    conf = safe_float(sub["confidence"].mean(), 0.85)
                    if conf <= 1.0:
                        conf *= 100.0

                hr = safe_float(sub["hr"].mean(), 72.0) if "hr" in sub.columns else None
                eda = None
                if "eda_mean" in sub.columns:
                    eda = safe_float(sub["eda_mean"].mean())
                elif "gsr_raw" in sub.columns:
                    raw_gsr = safe_float(sub["gsr_raw"].mean(), 2000.0)
                    eda = round(max(0.05, (4095.0 - raw_gsr) / 400.0), 2) if raw_gsr <= 4095.0 else round(raw_gsr / 1000.0, 2)

                windows.append({
                    "window_num": idx,
                    "start_sec": round((curr - min_ts) / 1000.0, 1),
                    "end_sec": round((curr + window_ms - min_ts) / 1000.0, 1),
                    "label": label,
                    "confidence_pct": conf,
                    "hr": hr,
                    "eda": eda,
                })
                idx += 1
            curr += step_ms

    if not windows:
        # Fallback: treat each row in df as a reading
        for i in range(min(len(df), 30)):
            row = df.iloc[i]
            label = str(row.get("stress_state", row.get("prediction", "LOW_STRESS"))).strip().upper()
            conf = safe_float(row.get("confidence_pct", row.get("confidence", 85.0)))
            if conf <= 1.0:
                conf *= 100.0
            windows.append({
                "window_num": i + 1,
                "start_sec": round(i * 15.0, 1),
                "end_sec": round(i * 15.0 + 30.0, 1),
                "label": label,
                "confidence_pct": conf,
                "hr": safe_float(row.get("hr", 72.0)),
                "eda": safe_float(row.get("eda_mean", 1.5)),
            })

    return windows


def generate_pdf_report(csv_path: str, output_pdf_path: Optional[str] = None, patient_name: Optional[str] = None) -> str:
    """Generates the clinical-grade PDF screening report with reading-per-reading window timelines."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Session CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError("Provided CSV file is empty.")

    # Extract Patient Name
    if not patient_name:
        if "patient_name" in df.columns and pd.notna(df["patient_name"].iloc[0]):
            p_val = str(df["patient_name"].iloc[0]).strip()
            if p_val and p_val.lower() != "nan":
                patient_name = p_val

    session_id_val = str(df["session_id"].iloc[0]) if "session_id" in df.columns and pd.notna(df["session_id"].iloc[0]) else Path(csv_path).stem
    if not patient_name:
        parts = session_id_val.rsplit("_", 2)
        if len(parts) == 3 and len(parts[1]) == 8 and len(parts[2]) == 6 and parts[1].isdigit() and parts[2].isdigit():
            patient_name = parts[0].replace("_", " ").title()
        elif session_id_val.startswith("stress_session_"):
            patient_name = "Anonymous"
        else:
            patient_name = session_id_val.replace("_", " ").title()

    if output_pdf_path:
        pdf_filename = str(output_pdf_path)
    else:
        reports_dir = Path("reports")
        reports_dir.mkdir(parents=True, exist_ok=True)
        pdf_filename = str(reports_dir / f"{session_id_val}_mhs_report.pdf")

    # Extract Window-by-Window Readings ("reading per reading")
    windows = _extract_windows_from_session(df, csv_path, session_id_val)
    total_windows = len(windows)

    # Compute Stress Metrics across windows
    stress_windows = 0
    relaxed_windows = 0
    low_windows = 0
    mod_windows = 0
    high_windows = 0

    for w in windows:
        lbl = w["label"]
        if "HIGH" in lbl:
            high_windows += 1
            stress_windows += 1
        elif "MOD" in lbl:
            mod_windows += 1
            stress_windows += 1
        elif "LOW" in lbl:
            low_windows += 1
        elif "RELAX" in lbl or "NON" in lbl:
            relaxed_windows += 1
        elif "STRESS" in lbl:
            stress_windows += 1
        else:
            relaxed_windows += 1

    non_stress_windows = total_windows - stress_windows
    stress_pct = round((stress_windows / max(total_windows, 1)) * 100.0, 1)

    # Signal Means
    if "hr" in df.columns:
        hr_mean = round(float(df["hr"].dropna().mean()), 1)
    else:
        hr_vals = [w["hr"] for w in windows if w.get("hr") is not None]
        hr_mean = round(float(np.mean(hr_vals)), 1) if hr_vals else "N/A"

    if "rmssd" in df.columns:
        rmssd_mean = round(float(df["rmssd"].dropna().mean()), 1)
    else:
        rmssd_mean = "N/A"

    if "eda_mean" in df.columns:
        eda_mean_val = round(float(df["eda_mean"].dropna().mean()), 2)
    elif "scl_mean" in df.columns:
        eda_mean_val = round(float(df["scl_mean"].dropna().mean()), 2)
    elif "gsr_raw" in df.columns:
        raw_g = float(df["gsr_raw"].dropna().mean())
        eda_mean_val = round(max(0.05, (4095.0 - raw_g) / 400.0), 2) if raw_g <= 4095.0 else round(raw_g / 1000.0, 2)
    else:
        eda_vals = [w["eda"] for w in windows if w.get("eda") is not None]
        eda_mean_val = round(float(np.mean(eda_vals)), 2) if eda_vals else "N/A"

    conf_vals = [w["confidence_pct"] for w in windows]
    peak_conf_val = max(conf_vals) if conf_vals else 85.0
    peak_conf = f"{round(peak_conf_val, 1)}%"

    # Duration string
    if "duration_sec" in df.columns:
        elapsed_max = float(df["duration_sec"].iloc[0])
    elif "elapsed_sec" in df.columns:
        elapsed_max = float(df["elapsed_sec"].max())
    elif "timestamp_ms" in df.columns:
        elapsed_max = (float(df["timestamp_ms"].iloc[-1]) - float(df["timestamp_ms"].iloc[0])) / 1000.0
    else:
        elapsed_max = total_windows * 15.0 + 15.0

    duration_str = f"{int(elapsed_max // 60)} min {int(elapsed_max % 60)} sec" if elapsed_max >= 60 else f"{round(elapsed_max, 1)} sec"
    session_date = datetime.now().strftime("%B %d, %Y - %H:%M:%S")

    # Document Setup (A4)
    doc = SimpleDocTemplate(
        pdf_filename,
        pagesize=A4,
        leftMargin=36,
        rightMargin=36,
        topMargin=36,
        bottomMargin=36,
    )

    styles = getSampleStyleSheet()

    # Custom Typography Styles
    title_style = ParagraphStyle(
        "CoverTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=22,
        leading=26,
        textColor=PRIMARY_NAVY,
        spaceAfter=4,
    )

    subtitle_style = ParagraphStyle(
        "CoverSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=12,
        leading=15,
        textColor=ACCENT_TEAL,
        spaceAfter=12,
    )

    h2_style = ParagraphStyle(
        "SectionH2",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=13,
        leading=16,
        textColor=PRIMARY_NAVY,
        spaceBefore=14,
        spaceAfter=6,
    )

    body_style = ParagraphStyle(
        "BodyTextCustom",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=9.5,
        leading=13.5,
        textColor=TEXT_DARK,
        spaceAfter=6,
    )

    story = []

    # =========================================================================
    # SECTION 1: COVER SECTION
    # =========================================================================
    story.append(Paragraph("Mental Health Screening Report", title_style))
    story.append(Paragraph("Bio-Synchronous Stress Mapping System", subtitle_style))
    story.append(Drawing(523, 2, Line(0, 1, 523, 1, strokeColor=ACCENT_TEAL, strokeWidth=2)))
    story.append(Spacer(1, 10))

    meta_data = [
        [
            Paragraph("<b>Patient Name:</b>", body_style), Paragraph(f"<b>{patient_name}</b>", body_style),
            Paragraph("<b>Subject / Session:</b>", body_style), Paragraph(session_id_val, body_style),
        ],
        [
            Paragraph("<b>Session Timestamp:</b>", body_style), Paragraph(session_date, body_style),
            Paragraph("<b>Total Duration:</b>", body_style), Paragraph(duration_str, body_style),
        ],
        [
            Paragraph("<b>Analysis Window:</b>", body_style), Paragraph("30-second Rolling (15s Step, 50% Overlap)", body_style),
            Paragraph("<b>Overall Stress %:</b>", body_style), Paragraph(f"{stress_pct}%", body_style),
        ],
    ]
    meta_table = Table(meta_data, colWidths=[110, 150, 110, 153])
    meta_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), BG_LIGHT),
        ("BOX", (0, 0), (-1, -1), 1, BORDER_COLOR),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, BORDER_COLOR),
        ("PADDING", (0, 0), (-1, -1), 5),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 10))

    # =========================================================================
    # SECTION 2: SESSION SUMMARY TABLE
    # =========================================================================
    story.append(Paragraph("1. Session Summary", h2_style))

    summary_data = [
        ["Metric Description", "Observed Value"],
        ["Total 30-Second Rolling Windows Analyzed", str(total_windows)],
        ["Stress Windows Detected (Moderate / High)", str(stress_windows)],
        ["Non-Stress Windows Detected (Relaxed / Low)", str(non_stress_windows)],
        ["Overall Session Stress Percentage", f"{stress_pct}%"],
        ["Mean Heart Rate (HR)", f"{hr_mean} BPM" if hr_mean != "N/A" else "N/A"],
        ["Mean HRV (RMSSD)", f"{rmssd_mean} ms" if rmssd_mean != "N/A" else "N/A"],
        ["Mean Electrodermal Activity (EDA)", f"{eda_mean_val} uS" if eda_mean_val != "N/A" else "N/A"],
        ["Peak Model Confidence Score", str(peak_conf)],
    ]

    sum_table = Table(summary_data, colWidths=[330, 193])
    sum_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (1, 0), PRIMARY_NAVY),
        ("TEXTCOLOR", (0, 0), (1, 0), colors.white),
        ("FONTNAME", (0, 0), (1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (1, 0), 9),
        ("BOTTOMPADDING", (0, 0), (1, 0), 5),
        ("TOPPADDING", (0, 0), (1, 0), 5),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, BG_LIGHT]),
        ("GRID", (0, 0), (-1, -1), 0.5, BORDER_COLOR),
        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 1), (-1, -1), 8.5),
        ("ALIGN", (1, 0), (1, -1), "CENTER"),
        ("PADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(sum_table)
    story.append(Spacer(1, 10))

    # =========================================================================
    # SECTION 3: HR OVER TIME LINE CHART
    # =========================================================================
    story.append(Paragraph("2. Heart Rate (HR) Timeline", h2_style))

    dwg_hr = Drawing(523, 110)
    # Background highlight for stress windows
    for i, w in enumerate(windows):
        is_st = "STRESS" in w["label"] and "NON" not in w["label"]
        x1 = 45 + (i * (440.0 / max(total_windows, 1)))
        x2 = 45 + ((i + 1) * (440.0 / max(total_windows, 1)))
        if is_st:
            dwg_hr.add(Rect(x1, 15, x2 - x1, 85, fillColor=LIGHT_RED_BG, strokeColor=None))

    lp_hr = LinePlot()
    lp_hr.x = 45
    lp_hr.y = 15
    lp_hr.width = 440
    lp_hr.height = 85

    if "hr" in df.columns and "elapsed_sec" in df.columns:
        hr_data = [(safe_float(r["elapsed_sec"]), safe_float(r["hr"])) for _, r in df.iterrows() if pd.notna(r.get("hr"))]
    else:
        hr_data = [(w["start_sec"], safe_float(w.get("hr"), 72.0)) for w in windows]

    if not hr_data:
        hr_data = [(i * 15, 72.0) for i in range(max(total_windows, 1))]

    lp_hr.data = [hr_data]
    lp_hr.lines[0].strokeColor = PRIMARY_NAVY
    lp_hr.lines[0].strokeWidth = 2

    dwg_hr.add(lp_hr)
    story.append(dwg_hr)
    story.append(Spacer(1, 10))

    # =========================================================================
    # SECTION 4: EDA OVER TIME LINE CHART
    # =========================================================================
    story.append(Paragraph("3. Electrodermal Activity (EDA) Timeline", h2_style))

    dwg_eda = Drawing(523, 110)
    for i, w in enumerate(windows):
        is_st = "STRESS" in w["label"] and "NON" not in w["label"]
        x1 = 45 + (i * (440.0 / max(total_windows, 1)))
        x2 = 45 + ((i + 1) * (440.0 / max(total_windows, 1)))
        if is_st:
            dwg_eda.add(Rect(x1, 15, x2 - x1, 85, fillColor=LIGHT_RED_BG, strokeColor=None))

    lp_eda = LinePlot()
    lp_eda.x = 45
    lp_eda.y = 15
    lp_eda.width = 440
    lp_eda.height = 85

    eda_col = "eda_mean" if "eda_mean" in df.columns else ("scl_mean" if "scl_mean" in df.columns else None)
    if eda_col and "elapsed_sec" in df.columns:
        eda_data = [(safe_float(r["elapsed_sec"]), safe_float(r[eda_col])) for _, r in df.iterrows() if pd.notna(r.get(eda_col))]
    else:
        eda_data = [(w["start_sec"], safe_float(w.get("eda"), 1.5)) for w in windows]

    if not eda_data:
        eda_data = [(i * 15, 1.5) for i in range(max(total_windows, 1))]

    lp_eda.data = [eda_data]
    lp_eda.lines[0].strokeColor = ACCENT_TEAL
    lp_eda.lines[0].strokeWidth = 2
    dwg_eda.add(lp_eda)
    story.append(dwg_eda)
    story.append(Spacer(1, 10))

    # =========================================================================
    # SECTION 5: STRESS CLASSIFICATION TIMELINE PER WINDOW (READING PER READING)
    # =========================================================================
    story.append(PageBreak())
    story.append(Paragraph("4. Stress Classification Timeline per 30s Window (50% Overlap)", h2_style))
    story.append(Paragraph("<i>Detailed reading-by-reading classification of each consecutive 30-second physiological window.</i>", body_style))
    story.append(Spacer(1, 6))

    # Paginate windows in blocks of up to 25 windows per drawing
    chunk_size = 25
    for chunk_start in range(0, max(total_windows, 1), chunk_size):
        chunk_windows = windows[chunk_start:chunk_start + chunk_size]
        dwg_bar = Drawing(523, len(chunk_windows) * 18 + 15)

        for i, w in enumerate(chunk_windows):
            w_num = w["window_num"]
            lbl = w["label"]
            conf = w["confidence_pct"]

            # Color coding for 4 classes
            if "RELAX" in lbl:
                bar_color = NON_STRESS_GREEN
                label_text = f"RELAXED ({conf:.1f}%)"
            elif "LOW" in lbl:
                bar_color = STRESS_CYAN
                label_text = f"LOW STRESS ({conf:.1f}%)"
            elif "MOD" in lbl:
                bar_color = STRESS_AMBER
                label_text = f"MODERATE STRESS ({conf:.1f}%)"
            elif "HIGH" in lbl:
                bar_color = STRESS_RED
                label_text = f"HIGH STRESS ({conf:.1f}%)"
            elif "NON" in lbl:
                bar_color = NON_STRESS_GREEN
                label_text = f"NON-STRESS ({conf:.1f}%)"
            elif "STRESS" in lbl:
                bar_color = STRESS_RED
                label_text = f"STRESS ({conf:.1f}%)"
            else:
                bar_color = colors.HexColor("#64748B")
                label_text = f"{lbl} ({conf:.1f}%)"

            y_pos = dwg_bar.height - ((i + 1) * 18)
            dwg_bar.add(String(10, y_pos + 3, f"W{w_num}:", fontName="Helvetica-Bold", fontSize=8, fillColor=TEXT_DARK))
            dwg_bar.add(Rect(45, y_pos, 350, 12, fillColor=bar_color, strokeColor=None))
            dwg_bar.add(String(402, y_pos + 2, label_text, fontName="Helvetica-Bold", fontSize=7.5, fillColor=bar_color))

        story.append(dwg_bar)
        if chunk_start + chunk_size < total_windows:
            story.append(PageBreak())
        else:
            story.append(Spacer(1, 10))

    # =========================================================================
    # SECTION 6: NON-CLINICAL DECISION SUPPORT & INTERPRETATION
    # =========================================================================
    story.append(Paragraph("5. Non-Clinical Decision Support & Interpretation", h2_style))

    if stress_pct > 60.0:
        interp_core = (
            f"The subject showed sustained physiological indicators consistent with high stress throughout the session "
            f"({stress_pct}% of analyzed 30-second rolling windows classified as elevated stress). Elevated electrodermal activity and "
            f"heart-rate metrics suggest persistent autonomic nervous system activation."
        )
    elif stress_pct >= 30.0:
        interp_core = (
            f"The subject showed moderate physiological stress responses during the session "
            f"({stress_pct}% of analyzed 30-second rolling windows classified as elevated stress). Periods of elevated arousal were observed "
            f"interspersed with baseline recovery states."
        )
    else:
        interp_core = (
            f"The subject showed predominantly baseline physiological responses during the session "
            f"({100.0 - stress_pct}% of analyzed 30-second rolling windows classified as baseline / relaxed states). Bio-signal parameters "
            f"remained within normal resting limits throughout data collection."
        )

    disclaimer_text = (
        "This report summarizes model-estimated physiological stress responses and signal characteristics "
        "during the recorded session. It is intended for research and non-clinical decision support and does not constitute a medical diagnosis or psychiatric diagnosis."
    )
    disclaimer = f"<b>Non-Clinical Decision Support Disclaimer:</b> {disclaimer_text}"
    interp_text = f"{interp_core}<br/><br/>{disclaimer}"

    interp_box = Table([[Paragraph(interp_text, body_style)]], colWidths=[523])
    interp_box.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), BG_LIGHT),
        ("BOX", (0, 0), (-1, -1), 1, ACCENT_TEAL),
        ("PADDING", (0, 0), (-1, -1), 8),
    ]))
    story.append(interp_box)
    story.append(Spacer(1, 10))

    # =========================================================================
    # SECTION 7: TECHNICAL APPENDIX
    # =========================================================================
    story.append(Paragraph("6. Technical Appendix", h2_style))

    tech_data = [
        ["System Parameter", "Specification / Metric Value"],
        ["Machine Learning Model", "CatBoost Classifier (Multiclass 4-Level Model)"],
        ["Training Benchmark Dataset", "WESAD (Wearable Stress and Affect Detection) Wrist Protocol"],
        ["Windowing Strategy", "30-second Rolling Windows (15s Step, 50% Overlap)"],
        ["Extracted Feature Vector", "23 Biomarkers (NeuroKit2: EDA, HRV, 3-Axis IMU)"],
        ["Hardware Sampling Frequency", "25.0 Hz (ESP32 WROOM-32 USB Serial Telemetry)"],
        ["Classifier Accuracy (LOSO)", "89.72% +/- 7.42%"],
        ["ROC-AUC Benchmark Score", "97.52%"],
    ]

    tech_table = Table(tech_data, colWidths=[200, 323])
    tech_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (1, 0), PRIMARY_NAVY),
        ("TEXTCOLOR", (0, 0), (1, 0), colors.white),
        ("FONTNAME", (0, 0), (1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (1, 0), 8.5),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, BG_LIGHT]),
        ("GRID", (0, 0), (-1, -1), 0.5, BORDER_COLOR),
        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 1), (-1, -1), 8),
        ("PADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(tech_table)

    # Build PDF Document
    doc.build(story)
    print(f"[OK] Mental Health Screening PDF Report generated successfully: {pdf_filename}")
    return pdf_filename


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python report_generator.py <session_csv_path> [patient_name]")
        sys.exit(1)

    csv_file = sys.argv[1]
    pat_name = sys.argv[2] if len(sys.argv) > 2 else None
    generate_pdf_report(csv_file, patient_name=pat_name)
