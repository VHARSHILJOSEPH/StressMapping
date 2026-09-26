"""
ESP32 Stress Monitoring System — Launcher
==========================================
Auto-detects ESP32 serial port and launches the Streamlit dashboard.
"""

import sys
import subprocess
from pathlib import Path

import config
from desktop_app.receiver import detect_esp32_port, list_serial_ports


def main():
    print("=================================================================")
    print(" ESP32 Bio-Signal Stress Monitor — System Launcher")
    print("=================================================================")
    print(f"Project Directory : {config.BASE_DIR}")
    print(f"Data Directory    : {config.DATA_DIR}")
    print(f"Baud Rate         : {config.BAUD_RATE}")
    print(f"Sampling Rate     : {config.SAMPLING_RATE_HZ} Hz")
    print("-----------------------------------------------------------------")

    # USB Serial Port Detection
    print("\n[USB] Scanning for ESP32 USB-Serial devices...")
    ports = list_serial_ports()
    if ports:
        print("   [OK] Serial port(s) detected:")
        for p in ports:
            esp_tag = " [ESP32]" if p["is_esp32"] else ""
            print(f"      [{p['device']}] {p['description']}{esp_tag}")

        detected = detect_esp32_port()
        if detected:
            print(f"\n   -> Auto-detected ESP32: {detected}")
        else:
            print(f"\n   -> No ESP32 chipset found. Fallback: {config.SERIAL_PORT}")
    else:
        print("   [!!] No serial ports detected.")
        print("      Make sure the USB-C cable is plugged in and drivers installed.")
        print("      CP210x: https://www.silabs.com/developers/usb-to-uart-bridge-vcp-drivers")

    print("-----------------------------------------------------------------\n")

    # Launch Streamlit
    streamlit_app = config.BASE_DIR / "dashboard" / "streamlit_app.py"
    print(f"Launching Dashboard: {streamlit_app}")

    cmd = [sys.executable, "-m", "streamlit", "run", str(streamlit_app)]
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\n[Launcher] Shutdown requested. Exiting cleanly.")
    except Exception as exc:
        print(f"\n[Launcher] Error: {exc}")


if __name__ == "__main__":
    main()
