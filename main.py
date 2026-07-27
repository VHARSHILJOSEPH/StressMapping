import sys
import socket
import subprocess
from pathlib import Path
import config

def get_network_ips():
    """Discover local IPv4 addresses to assist ESP32 configuration."""
    ip_list = []
    try:
        hostname = socket.gethostname()
        for ip in socket.gethostbyname_ex(hostname)[2]:
            if not ip.startswith("127."):
                ip_list.append(ip)
    except Exception:
        pass
    
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        default_ip = s.getsockname()[0]
        s.close()
        if default_ip not in ip_list:
            ip_list.insert(0, default_ip)
    except Exception:
        pass

    return ip_list if ip_list else ["127.0.0.1"]

def main():
    print("=================================================================")
    print(" ⚡ ESP32 Sensor Connectivity & Desktop Dashboard Host Launcher  ")
    print("=================================================================")
    print(f"Project Base Directory: {config.BASE_DIR}")
    print(f"Data Storage Dir     : {config.DATA_DIR}")
    print(f"Reports Storage Dir  : {config.REPORTS_DIR}")
    print(f"Telemetry Port (UDP) : {config.UDP_PORT}")
    print("-----------------------------------------------------------------")
    
    ips = get_network_ips()
    print("🌐 Active Computer IP Addresses for ESP32 target configuration:")
    for ip in ips:
        print(f"   -> {ip}")
    print(f"\n👉 In esp32_stress_monitor.ino, set UDP_DEST_IP = \"{ips[0]}\"")
    print("-----------------------------------------------------------------\n")

    streamlit_app_path = config.BASE_DIR / "dashboard" / "streamlit_app.py"

    print(f"Launching Desktop Dashboard via Streamlit...")
    cmd = [sys.executable, "-m", "streamlit", "run", str(streamlit_app_path)]

    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\n[Launcher] System shutdown requested. Exiting cleanly.")
    except Exception as e:
        print(f"\n[Launcher] Error executing Streamlit app: {e}")

if __name__ == "__main__":
    main()
