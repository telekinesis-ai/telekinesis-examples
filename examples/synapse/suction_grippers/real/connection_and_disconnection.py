"""
Demonstrates connecting and disconnecting a suction gripper.

Usage:
    python connection_and_disconnection.py --ip <ROBOT_IP>
    python connection_and_disconnection.py --protocol MODBUS_RTU --serial-port COM3
"""

import argparse
import time
from loguru import logger

from telekinesis.synapse.tools.suction_grippers import piab


def main(ip: str | None, serial_port: str, protocol: str) -> None:
    """Connects and disconnects a Piab gripper."""

    #===================== Create Gripper ======================================
    gripper = piab.PiabPiCobotElectric()

    # ==================== Run Skill ===========================================
    try:
        gripper.connect(ip=ip, serial_port=serial_port, protocol=protocol)
    except (ConnectionError, OSError) as e:
        logger.error(f"Error occurred: {e}")
    finally:
        gripper.disconnect()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Piab gripper connect/disconnect")
    p.add_argument("--protocol",
                   choices=["URCAP", "MODBUS_RTU"],
                   default="URCAP")
    p.add_argument("--ip", default="192.168.2.2", help="IP for Robot Controller")
    p.add_argument("--serial-port", dest="serial_port", default="COM3",
                   help="Serial port for MODBUS_RTU")
    args = p.parse_args()

    main(ip=args.ip,
         serial_port=args.serial_port,
         protocol=args.protocol)
