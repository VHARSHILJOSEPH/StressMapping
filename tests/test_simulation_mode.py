import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest
from desktop_app.receiver import SerialDataReceiver


def test_simulation_mode_lifecycle_and_rate():
    receiver = SerialDataReceiver()
    assert not receiver.running
    assert receiver.status == "DISCONNECTED"

    # Start in simulation mode
    receiver.start(port="SIMULATED", simulation_mode=True)
    assert receiver.running
    time.sleep(0.5)

    assert receiver.status == "CONNECTED (SIMULATED)"
    status = receiver.get_status_summary()
    assert status["connection_mode"] == "SIMULATED"
    assert status["serial_port"] == "SIMULATED"
    assert "Simulated Hardware" in status["last_remote_ip"]

    # Verify packets are being generated
    data = receiver.get_latest_data(count=10)
    assert len(data) > 0
    pkt = data[-1]
    assert "ppg_raw" in pkt
    assert "gsr_raw" in pkt
    assert "imu_ax" in pkt

    # Stop simulation mode
    receiver.stop()
    assert not receiver.running
    assert receiver.status == "DISCONNECTED"


def test_switch_from_reconnecting_to_simulation():
    """Verify that toggling simulation mode while stuck in serial reconnecting transitions cleanly."""
    receiver = SerialDataReceiver()
    # Start on non-existent port without auto-detect
    import config
    orig_autodetect = config.AUTO_DETECT_SERIAL
    config.AUTO_DETECT_SERIAL = False
    try:
        receiver.start(port="NON_EXISTENT_PORT_999", simulation_mode=False)
        time.sleep(0.3)
        assert receiver.status in ["RECONNECTING", "DISCONNECTED"]

        # Now simulate user toggling the sidebar switch:
        receiver.stop()
        receiver.set_simulation_mode(True)
        receiver.start(port="SIMULATED", simulation_mode=True)
        time.sleep(0.5)

        assert receiver.running
        assert receiver.status == "CONNECTED (SIMULATED)"
        data = receiver.get_latest_data(count=5)
        assert len(data) > 0

        receiver.stop()
        assert not receiver.running
    finally:
        config.AUTO_DETECT_SERIAL = orig_autodetect
