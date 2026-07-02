"""Robotiq 2F-85 connect and disconnect example for the Synapse SDK.

Supported only for real hardware.

Usage:
    python connection_and_disconnection.py --ip <ROBOT_IP>
    python connection_and_disconnection.py --protocol MODBUS_RTU --serial-port COM4
"""

import argparse
import time
from typing import Optional

from loguru import logger

from telekinesis.synapse.tools.parallel_grippers import robotiq


def main(ip: Optional[str], serial_port: str, protocol: str) -> None:
    gripper = robotiq.Robotiq2F85()
    logger.info(f"Connecting Robotiq via {protocol}...")
    gripper.connect(ip=ip, serial_port=serial_port, protocol=protocol)
    logger.success("Connected.")

    time.sleep(2.0)

    gripper.disconnect()
    logger.success("Disconnected.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Robotiq gripper connect/disconnect example")
    p.add_argument("--protocol", choices=["RTDE", "MODBUS_RTU"], default="RTDE")
    p.add_argument("--ip", default=None, help="IP for RTDE")
    p.add_argument("--serial-port", dest="serial_port", default="COM4",
                   help="Serial port for MODBUS_RTU")
    args = p.parse_args()

    main(ip=args.ip, serial_port=args.serial_port, protocol=args.protocol)
