"""Robotiq 2F-85 move-to-position example for the Synapse SDK.

Configures mm units and 85 mm stroke range, then moves to 20 mm at 100% speed
and 50% force.
Supported only for real hardware.

Usage:
    python move.py --ip <ROBOT_IP>
    python move.py --protocol MODBUS_RTU --serial-port COM4
"""

import argparse
from typing import Optional

from loguru import logger

from telekinesis.synapse.tools.parallel_grippers import robotiq


def main(ip: Optional[str], serial_port: str, protocol: str) -> None:
    gripper = robotiq.Robotiq2F85()
    gripper.connect(ip=ip, serial_port=serial_port, protocol=protocol)
    try:
        gripper.set_unit(parameter="position", unit="mm")
        gripper.set_position_range_mm(position_range_mm=85.0)
        status = gripper.move(position=20.0, speed=100.0, force=50.0, asynchronous=False)
        logger.success(f"move() status: {status}, position: {gripper.get_current_position():.2f}")
    finally:
        gripper.disconnect()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Robotiq gripper move example")
    p.add_argument("--protocol", choices=["URCAP", "MODBUS_RTU"], default="URCAP")
    p.add_argument("--ip", default=None, help="IP for Robot Controller")
    p.add_argument("--serial-port", dest="serial_port", default="COM4",
                   help="Serial port for MODBUS_RTU")
    args = p.parse_args()

    main(ip=args.ip, serial_port=args.serial_port, protocol=args.protocol)
