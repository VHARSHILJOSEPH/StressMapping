"""
ESP32 Stress Monitoring System — Interactive Launcher
=====================================================
Menu-driven launcher for common tasks: dashboard, reports, port scan.
"""

import os
import sys
import subprocess
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent


def check_model_artifacts():
    """Verify CatBoost model and scaler files exist."""
    model_exists = (BASE_DIR / "wesad_model.cbm").exists()
    scaler_exists = (BASE_DIR / "scaler.pkl").exists()
    return model_exists, scaler_exists


def scan_serial_ports():
    """Scan for serial ports using pyserial."""
    try:
        import serial.tools.list_ports
        return [(p.device, p.description) for p in serial.tools.list_ports.comports()]
    except ImportError:
        return []


def main():
    print("=================================================================")
    print(" Bio-Synchronous Stress Mapping System — Interactive Launcher")
    print("=================================================================")
    print(f" Working Directory: {BASE_DIR}")

    # Artifact check
    model_ok, scaler_ok = check_model_artifacts()
    print("\n[1/2] Model Artifacts:")
    print(f"  - wesad_model.cbm : {'[OK]' if model_ok else '[MISSING]'}")
    print(f"  - scaler.pkl      : {'[OK]' if scaler_ok else '[MISSING]'}")
    if not (model_ok and scaler_ok):
        print("  → Heuristic fallback will be used until model files are placed in project root.\n")

    # Serial port scan
    print("[2/2] USB Serial Ports:")
    ports = scan_serial_ports()
    if ports:
        for dev, desc in ports:
            print(f"  → {dev} ({desc})")
    else:
        print("  → No serial ports detected.")

    print("\n-----------------------------------------------------------------")
    print(" Select Action:")
    print("  1. Launch Streamlit Dashboard")
    print("  2. Generate Report from Session CSV")
    print("  3. Exit")
    print("-----------------------------------------------------------------")

    choice = input("Enter choice (1-3): ").strip()

    if choice == "1":
        print("\n📊 Launching Streamlit Dashboard...")
        dash_path = BASE_DIR / "dashboard" / "streamlit_app.py"
        subprocess.run([sys.executable, "-m", "streamlit", "run", str(dash_path)], cwd=str(BASE_DIR))

    elif choice == "2":
        csv_files = list((BASE_DIR / "data").glob("*.csv")) + list((BASE_DIR / "sessions").glob("*.csv"))
        if not csv_files:
            print("\n[!] No session CSV files found.")
            return

        print("\nAvailable Session CSV files:")
        for idx, f in enumerate(csv_files, 1):
            print(f"  {idx}. {f.name}")

        try:
            sel = int(input("\nSelect file number: ").strip()) - 1
            if 0 <= sel < len(csv_files):
                selected_csv = str(csv_files[sel])
                pat_name = input("Enter patient name (leave blank to auto-detect): ").strip()
                print(f"\nGenerating report for {selected_csv}...")
                venv_py = BASE_DIR / "venv" / "Scripts" / "python.exe"
                py_bin = str(venv_py) if venv_py.exists() else sys.executable
                cmd = [py_bin, "report_generator.py", selected_csv]
                if pat_name:
                    cmd.append(pat_name)
                subprocess.run(cmd, cwd=str(BASE_DIR))
            else:
                print("Invalid selection.")
        except Exception as exc:
            print(f"Error: {exc}")

    elif choice == "3":
        print("Exiting.")


if __name__ == "__main__":
    main()
