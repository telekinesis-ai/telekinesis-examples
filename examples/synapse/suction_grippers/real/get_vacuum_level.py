"""
Demonstrates reading the last commanded vacuum level of a suction gripper.

Supports Piab grippers.

Usage:
    python get_vacuum_level.py --ip <ROBOT_IP>
    python get_vacuum_level.py --protocol MODBUS_RTU --serial-port COM3
"""

import argparse
from loguru import logger

from telekinesis.synapse.tools.suction_grippers import piab


def main(ip: str | None, serial_port: str, protocol: str) -> None:
    """Reads the last commanded vacuum level of a Piab gripper in both units."""

    #===================== Create Gripper ======================================
    gripper = piab.PiabPiCobotElectric()

    # ==================== Run Skill ===========================================
    try:
        gripper.connect(ip=ip, serial_port=serial_port, protocol=protocol)
        logger.success(f"Vacuum level: "
                       f"{gripper.get_vacuum_level(unit='percentage')}%, "
                       f"{gripper.get_vacuum_level(unit='kPa')} kPa")
    except (ConnectionError, OSError) as e:
        logger.error(f"Error occurred: {e}")
    finally:
        gripper.disconnect()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Piab gripper get vacuum level")
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
