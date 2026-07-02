"""Robotiq 2F-85 set position range example for the Synapse SDK.

Sets the gripper stroke range. 85 mm is the 2F-85 maximum.
Supported only for real hardware.

Usage:
    python set_position_range.py --ip <ROBOT_IP>
    python set_position_range.py --protocol MODBUS_RTU --serial-port COM4
"""

import argparse
from typing import Optional

from loguru import logger

from telekinesis.synapse.tools.parallel_grippers import robotiq


def main(ip: Optional[str], serial_port: str, protocol: str) -> None:
    gripper = robotiq.Robotiq2F85()
    gripper.connect(ip=ip, serial_port=serial_port, protocol=protocol)
    try:
        gripper.set_position_range_mm(position_range_mm=85.0)
        logger.success("Position range set to 85 mm.")
    finally:
        gripper.disconnect()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Robotiq gripper set_position_range example")
    p.add_argument("--protocol", choices=["RTDE", "MODBUS_RTU"], default="RTDE")
    p.add_argument("--ip", default=None, help="IP for RTDE")
    p.add_argument("--serial-port", dest="serial_port", default="COM4",
                   help="Serial port for MODBUS_RTU")
    args = p.parse_args()

    main(ip=args.ip, serial_port=args.serial_port, protocol=args.protocol)
