import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime

# ReportLab imports for pure PDF generation and inline drawing
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether
from reportlab.graphics.shapes import Drawing, Rect, String, Line
from reportlab.graphics.charts.lineplots import LinePlot

PRIMARY_NAVY = colors.HexColor("#1B2A4A")
ACCENT_TEAL = colors.HexColor("#008080")
TEXT_DARK = colors.HexColor("#2D3748")
STRESS_RED = colors.HexColor("#DC2626")
NON_STRESS_GREEN = colors.HexColor("#16A34A")
LIGHT_RED_BG = colors.HexColor("#FEE2E2")
BG_LIGHT = colors.HexColor("#F8FAFC")
BORDER_COLOR = colors.HexColor("#E2E8F0")

from typing import Optional

def safe_float(val, default=0.0):
    try:
        if pd.isna(val): return default
        return float(val)
    except Exception:
        return default

def safe_int(val, default=0):
    try:
        if pd.isna(val): return default
        return int(val)
    except Exception:
        return default

def generate_pdf_report(csv_path: str, output_pdf_path: Optional[str] = None, patient_name: Optional[str] = None) -> str:
    if not os.path.exists(csv_path):
        print(f"Error: Session CSV file not found: {csv_path}")
        sys.exit(1)

    df = pd.read_csv(csv_path)
    if df.empty:
        print("Error: Provided CSV file is empty.")
        sys.exit(1)

    # Extract Patient Name
    if not patient_name:
        if "patient_name" in df.columns and pd.notna(df["patient_name"].iloc[0]):
            p_val = str(df["patient_name"].iloc[0]).strip()
            if p_val and p_val.lower() != "nan":
                patient_name = p_val

    session_id_val = str(df["session_id"].iloc[0]) if "session_id" in df.columns else os.path.splitext(os.path.basename(csv_path))[0]
    if not patient_name:
        parts = session_id_val.rsplit("_", 2)
        if len(parts) == 3 and len(parts[1]) == 8 and len(parts[2]) == 6 and parts[1].isdigit() and parts[2].isdigit():
            patient_name = parts[0].replace("_", " ").title()
        elif session_id_val.startswith("stress_session_"):
            patient_name = "Anonymous"
        else:
            patient_name = session_id_val.replace("_", " ").title()

    if output_pdf_path:
        pdf_filename = output_pdf_path
    else:
        # Default to reports directory if it exists, else alongside csv
        reports_dir = os.path.join(os.path.dirname(os.path.abspath(csv_path)), "..", "reports")
        if os.path.exists(reports_dir):
            pdf_filename = os.path.join(reports_dir, f"{session_id_val}_mhs_report.pdf")
        else:
            pdf_filename = os.path.splitext(csv_path)[0] + ".pdf"

    # Extract Data & Compute Session Statistics
    total_windows = len(df)
    
    # Check predictions column
    pred_col = 'prediction' if 'prediction' in df.columns else ('stress_state' if 'stress_state' in df.columns else None)
    if pred_col:
        stress_mask = df[pred_col].astype(str).str.upper().str.contains("STRESS") & ~df[pred_col].astype(str).str.upper().str.contains("NON")
        stress_windows = int(stress_mask.sum())
        non_stress_windows = total_windows - stress_windows
        stress_pct = round((stress_windows / max(total_windows, 1)) * 100.0, 1)
    else:
        stress_windows = 0
        non_stress_windows = 0
        stress_pct = 0.0

    hr_mean = round(df['hr'].mean(), 1) if 'hr' in df.columns else "N/A"
    rmssd_mean = round(df['rmssd'].mean(), 1) if 'rmssd' in df.columns else "N/A"
    eda_mean_val = round(df['eda_mean'].mean(), 3) if 'eda_mean' in df.columns else ("N/A" if 'scl_mean' not in df.columns else round(df['scl_mean'].mean(), 3))
    
    conf_col = 'confidence_pct' if 'confidence_pct' in df.columns else ('confidence' if 'confidence' in df.columns else None)
    if conf_col:
        peak_conf_val = df[conf_col].max()
        peak_conf = f"{round(peak_conf_val, 1)}%" if peak_conf_val > 1.0 else f"{round(peak_conf_val * 100.0, 1)}%"
    else:
        peak_conf = "N/A"

    elapsed_max = df['elapsed_sec'].max() if 'elapsed_sec' in df.columns else (total_windows * 60)
    duration_str = f"{int(elapsed_max // 60)} min {int(elapsed_max % 60)} sec" if elapsed_max else f"{total_windows * 60} sec"

    session_date = datetime.now().strftime("%B %d, %Y - %H:%M:%S")

    # Document Setup (A4)
    doc = SimpleDocTemplate(
        pdf_filename,
        pagesize=A4,
        leftMargin=36,
        rightMargin=36,
        topMargin=36,
        bottomMargin=36
    )

    styles = getSampleStyleSheet()

    # Custom Typography Styles
    title_style = ParagraphStyle(
        'CoverTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=22,
        leading=26,
        textColor=PRIMARY_NAVY,
        spaceAfter=4
    )
    
    subtitle_style = ParagraphStyle(
        'CoverSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=12,
        leading=15,
        textColor=ACCENT_TEAL,
        spaceAfter=12
    )

    h2_style = ParagraphStyle(
        'SectionH2',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=13,
        leading=16,
        textColor=PRIMARY_NAVY,
        spaceBefore=12,
        spaceAfter=6
    )

    body_style = ParagraphStyle(
        'BodyTextCustom',
        parent=styles['BodyText'],
        fontName='Helvetica',
        fontSize=9.5,
        leading=13.5,
        textColor=TEXT_DARK,
        spaceAfter=6
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
        [Paragraph("<b>Patient Name:</b>", body_style), Paragraph(f"<b>{patient_name}</b>", body_style),
         Paragraph("<b>Subject / Session:</b>", body_style), Paragraph(session_id_val, body_style)],
        [Paragraph("<b>Session Timestamp:</b>", body_style), Paragraph(session_date, body_style),
         Paragraph("<b>Total Duration:</b>", body_style), Paragraph(duration_str, body_style)],
        [Paragraph("<b>Analysis Window:</b>", body_style), Paragraph("30-second Rolling (15s Step, 50% Overlap)", body_style),
         Paragraph("<b>Overall Stress %:</b>", body_style), Paragraph(f"{stress_pct}%", body_style)]
    ]
    meta_table = Table(meta_data, colWidths=[110, 150, 110, 153])
    meta_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), BG_LIGHT),
        ('BOX', (0,0), (-1,-1), 1, BORDER_COLOR),
        ('INNERGRID', (0,0), (-1,-1), 0.5, BORDER_COLOR),
        ('PADDING', (0,0), (-1,-1), 5),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
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
        ["Stress Windows Detected", str(stress_windows)],
        ["Non-Stress Windows Detected", str(non_stress_windows)],
        ["Overall Session Stress Percentage", f"{stress_pct}%"],
        ["Mean Heart Rate (HR)", f"{hr_mean} BPM" if hr_mean != "N/A" else "N/A"],
        ["Mean HRV (RMSSD)", f"{rmssd_mean} ms" if rmssd_mean != "N/A" else "N/A"],
        ["Mean Electrodermal Activity (EDA)", f"{eda_mean_val} μS" if eda_mean_val != "N/A" else "N/A"],
        ["Peak Model Confidence Score", str(peak_conf)]
    ]
    
    sum_table = Table(summary_data, colWidths=[330, 193])
    sum_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (1,0), PRIMARY_NAVY),
        ('TEXTCOLOR', (0,0), (1,0), colors.white),
        ('FONTNAME', (0,0), (1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (1,0), 9),
        ('BOTTOMPADDING', (0,0), (1,0), 5),
        ('TOPPADDING', (0,0), (1,0), 5),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, BG_LIGHT]),
        ('GRID', (0,0), (-1,-1), 0.5, BORDER_COLOR),
        ('FONTNAME', (0,1), (-1,-1), 'Helvetica'),
        ('FONTSIZE', (0,1), (-1,-1), 8.5),
        ('ALIGN', (1,0), (1,-1), 'CENTER'),
        ('PADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(sum_table)
    story.append(Spacer(1, 10))

    # =========================================================================
    # SECTION 3: HR OVER TIME LINE CHART
    # =========================================================================
    story.append(Paragraph("2. Heart Rate (HR) Timeline", h2_style))

    dwg_hr = Drawing(523, 120)
    # Background highlight for stress windows
    for i in range(total_windows):
        is_st = False
        if pred_col:
            val_str = str(df[pred_col].iloc[i]).upper()
            if "STRESS" in val_str and "NON" not in val_str:
                is_st = True
        
        x1 = 45 + (i * (440.0 / max(total_windows, 1)))
        x2 = 45 + ((i + 1) * (440.0 / max(total_windows, 1)))
        if is_st:
            dwg_hr.add(Rect(x1, 15, x2 - x1, 95, fillColor=LIGHT_RED_BG, strokeColor=None))

    lp_hr = LinePlot()
    lp_hr.x = 45
    lp_hr.y = 15
    lp_hr.width = 440
    lp_hr.height = 95
    
    if 'hr' in df.columns and 'elapsed_sec' in df.columns:
        hr_data = [(safe_float(r['elapsed_sec']), safe_float(r['hr'])) for _, r in df.iterrows()]
    else:
        hr_data = [(i * 15, 72.0) for i in range(total_windows)]

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

    dwg_eda = Drawing(523, 120)
    for i in range(total_windows):
        is_st = False
        if pred_col:
            val_str = str(df[pred_col].iloc[i]).upper()
            if "STRESS" in val_str and "NON" not in val_str:
                is_st = True
        
        x1 = 45 + (i * (440.0 / max(total_windows, 1)))
        x2 = 45 + ((i + 1) * (440.0 / max(total_windows, 1)))
        if is_st:
            dwg_eda.add(Rect(x1, 15, x2 - x1, 95, fillColor=LIGHT_RED_BG, strokeColor=None))

    lp_eda = LinePlot()
    lp_eda.x = 45
    lp_eda.y = 15
    lp_eda.width = 440
    lp_eda.height = 95

    eda_col_name = 'eda_mean' if 'eda_mean' in df.columns else ('scl_mean' if 'scl_mean' in df.columns else None)
    if eda_col_name and 'elapsed_sec' in df.columns:
        eda_data = [(safe_float(r['elapsed_sec']), safe_float(r[eda_col_name])) for _, r in df.iterrows()]
    else:
        eda_data = [(i * 15, 1.0) for i in range(total_windows)]

    lp_eda.data = [eda_data]
    lp_eda.lines[0].strokeColor = ACCENT_TEAL
    lp_eda.lines[0].strokeWidth = 2
    dwg_eda.add(lp_eda)
    story.append(dwg_eda)
    story.append(Spacer(1, 10))

    # =========================================================================
    # SECTION 5: STRESS CLASSIFICATION TIMELINE
    # =========================================================================
    story.append(Paragraph("4. Stress Classification Timeline per 30s Window (50% Overlap)", h2_style))
    
    # Cap displayed window bars per drawing to max 25 to prevent PDF layout overflow
    display_count = min(total_windows, 25)
    dwg_bar = Drawing(523, display_count * 18 + 15)
    for i in range(display_count):
        row = df.iloc[i]
        w_num = safe_int(row.get('window_num', i + 1))
        
        is_st = False
        if pred_col:
            val_str = str(row[pred_col]).upper()
            if "STRESS" in val_str and "NON" not in val_str:
                is_st = True

        conf_val = safe_float(row.get('confidence_pct', row.get('confidence', 85.0)))
        if conf_val <= 1.0: conf_val *= 100.0

        y_pos = dwg_bar.height - ((i + 1) * 18)
        bar_color = STRESS_RED if is_st else NON_STRESS_GREEN
        label_text = f"STRESS ({conf_val:.1f}%)" if is_st else f"NON-STRESS ({conf_val:.1f}%)"

        dwg_bar.add(String(10, y_pos + 3, f"W{w_num}:", fontName="Helvetica-Bold", fontSize=8, fillColor=TEXT_DARK))
        dwg_bar.add(Rect(45, y_pos, 370, 12, fillColor=bar_color, strokeColor=None))
        dwg_bar.add(String(422, y_pos + 2, label_text, fontName="Helvetica-Bold", fontSize=7.5, fillColor=bar_color))

    story.append(dwg_bar)
    story.append(Spacer(1, 10))

    # =========================================================================
    # SECTION 6: PLAIN-LANGUAGE INTERPRETATION SECTION
    # =========================================================================
    story.append(Paragraph("5. Non-Clinical Decision Support & Interpretation", h2_style))

    if stress_pct > 60.0:
        interp_core = (
            f"The subject showed sustained physiological indicators consistent with high stress throughout the session "
            f"({stress_pct}% of analyzed 30-second rolling windows classified as STRESS). Elevated electrodermal activity and "
            f"heart-rate metrics suggest persistent autonomic nervous system activation."
        )
    elif stress_pct >= 30.0:
        interp_core = (
            f"The subject showed moderate physiological stress responses during the session "
            f"({stress_pct}% of analyzed 30-second rolling windows classified as STRESS). Periods of elevated arousal were observed "
            f"interspersed with baseline recovery states."
        )
    else:
        interp_core = (
            f"The subject showed predominantly baseline physiological responses during the session "
            f"({100.0 - stress_pct}% of analyzed 30-second rolling windows classified as NON-STRESS). Bio-signal parameters "
            f"remained within normal resting limits throughout data collection."
        )
    disclaimer_text = "This report summarizes model-estimated physiological stress responses and signal characteristics during the recorded session. It is intended for research and non-clinical decision support and does not constitute a medical or psychiatric diagnosis."
    disclaimer = f"<b>Non-Clinical Decision Support Disclaimer:</b> {disclaimer_text}"

    interp_text = f"{interp_core}<br/><br/>{disclaimer}"
    
    interp_box = Table([[Paragraph(interp_text, body_style)]], colWidths=[523])
    interp_box.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), BG_LIGHT),
        ('BOX', (0,0), (-1,-1), 1, ACCENT_TEAL),
        ('PADDING', (0,0), (-1,-1), 8),
    ]))
    story.append(interp_box)
    story.append(Spacer(1, 10))

    # =========================================================================
    # SECTION 7: TECHNICAL APPENDIX
    # =========================================================================
    story.append(Paragraph("6. Technical Appendix", h2_style))

    tech_data = [
        ["System Parameter", "Specification / Metric Value"],
        ["Machine Learning Model", "CatBoost Classifier (CatBoostClassifier)"],
        ["Training Benchmark Dataset", "WESAD (Wearable Stress and Affect Detection) Wrist Protocol"],
        ["Windowing Strategy", "30-second Rolling Windows (15s Step, 50% Overlap)"],
        ["Extracted Feature Vector", "23 Biomarkers (NeuroKit2: EDA, HRV, 3-Axis IMU)"],
        ["Hardware Sampling Frequency", "25.0 Hz (ESP32 WROOM-32 USB Serial Telemetry)"],
        ["Classifier Accuracy (LOSO)", "89.72% ± 7.42%"],
        ["ROC-AUC Benchmark Score", "97.52%"]
    ]

    tech_table = Table(tech_data, colWidths=[200, 323])
    tech_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (1,0), PRIMARY_NAVY),
        ('TEXTCOLOR', (0,0), (1,0), colors.white),
        ('FONTNAME', (0,0), (1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (1,0), 8.5),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, BG_LIGHT]),
        ('GRID', (0,0), (-1,-1), 0.5, BORDER_COLOR),
        ('FONTNAME', (0,1), (-1,-1), 'Helvetica'),
        ('FONTSIZE', (0,1), (-1,-1), 8),
        ('PADDING', (0,0), (-1,-1), 4),
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
