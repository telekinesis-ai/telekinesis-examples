"""Robotiq 2F-85 open example for the Synapse SDK.

Opens the gripper fully at 100% speed and 50% force, synchronously.
Supported only for real hardware.

Usage:
    python open.py --ip <ROBOT_IP>
    python open.py --protocol MODBUS_RTU --serial-port COM4
"""

import argparse
from typing import Optional

from loguru import logger

from telekinesis.synapse.tools.parallel_grippers import robotiq


def main(ip: Optional[str], serial_port: str, protocol: str) -> None:
    gripper = robotiq.Robotiq2F85()
    gripper.connect(ip=ip, serial_port=serial_port, protocol=protocol)
    try:
        status = gripper.open(speed=100.0, force=50.0, asynchronous=False)
        logger.success(f"open() status: {status}, position: {gripper.get_current_position():.2f}")
    finally:
        gripper.disconnect()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Robotiq gripper open example")
    p.add_argument("--protocol", choices=["RTDE", "MODBUS_RTU"], default="RTDE")
    p.add_argument("--ip", default=None, help="IP for RTDE")
    p.add_argument("--serial-port", dest="serial_port", default="COM4",
                   help="Serial port for MODBUS_RTU")
    args = p.parse_args()

    main(ip=args.ip, serial_port=args.serial_port, protocol=args.protocol)
