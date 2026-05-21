"""Robotiq 2F-85 read current position example for the Synapse SDK.

Reads the gripper's current position in the configured unit.

Usage:
    python get_current_position.py --ip <ROBOT_IP>
    python get_current_position.py --protocol MODBUS_RTU --serial-port COM4
"""

import argparse
from typing import Optional

from loguru import logger

from telekinesis.synapse.tools.parallel_grippers import robotiq


def main(ip: Optional[str], serial_port: str, protocol: str) -> None:
    gripper = robotiq.Robotiq2F85()
    gripper.connect(ip=ip, serial_port=serial_port, protocol=protocol)
    try:
        logger.success(f"Current position: {gripper.get_current_position()}")
    finally:
        gripper.disconnect()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Robotiq gripper get_current_position example")
    p.add_argument("--protocol", choices=["RTDE", "MODBUS_RTU"], default="RTDE")
    p.add_argument("--ip", default=None, help="IP for RTDE")
    p.add_argument("--serial-port", dest="serial_port", default="COM4",
                   help="Serial port for MODBUS_RTU")
    args = p.parse_args()

    main(ip=args.ip, serial_port=args.serial_port, protocol=args.protocol)
