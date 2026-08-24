"""
Demonstrates setting the maximum vacuum-pump speed of a suction gripper.

Supports Piab grippers only on the URCAP protocol.

Usage:
    python set_max_pump_speed.py --ip <ROBOT_IP>
"""

import argparse
from loguru import logger

from telekinesis.synapse.tools.suction_grippers import piab


def main(ip: str) -> None:
    """Sets the maximum pump speed of a Piab gripper to 80%."""

    #===================== Create Gripper ======================================
    gripper = piab.PiabPiCobotElectric()

    # ==================== Run Skill ===========================================
    try:
        gripper.connect(ip=ip, protocol="URCAP")
        gripper.set_max_pump_speed(max_speed=80)
        logger.success("Maximum pump speed set to 80%.")
    except (ConnectionError, OSError) as e:
        logger.error(f"Error occurred: {e}")
    finally:
        gripper.disconnect()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Piab gripper set max pump speed")
    p.add_argument("--ip", default="192.168.2.2", help="IP for Robot Controller")
    args = p.parse_args()

    main(ip=args.ip)