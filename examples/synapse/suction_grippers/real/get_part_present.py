"""
Demonstrates reading whether a suction gripper currently holds an object.

Supports Piab grippers on MODBUS_RTU protocol.

Usage:
    python get_part_present.py --serial-port COM3
"""

import argparse
from loguru import logger

from telekinesis.synapse.tools.suction_grippers import piab


def main(serial_port: str) -> None:
    """Reads whether a Piab gripper currently holds an object."""

    #===================== Create Gripper ======================================
    gripper = piab.PiabPiCobotElectric()

    # ==================== Run Skill ===========================================
    try:
        gripper.connect(serial_port=serial_port, protocol="MODBUS_RTU")
        logger.success(f"Part present: {gripper.get_part_present()}")
    except (ConnectionError, OSError) as e:
        logger.error(f"Error occurred: {e}")
    finally:
        gripper.disconnect()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Piab gripper get part present")
    p.add_argument("--serial-port", dest="serial_port", default="COM3",
                   help="Serial port for MODBUS_RTU")
    args = p.parse_args()

    main(serial_port=args.serial_port)
