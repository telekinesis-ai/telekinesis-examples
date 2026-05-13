"""
Robotiq 2F-85 set position unit example for the Synapse SDK.

Switches the gripper position unit between normalized (0..1) and mm.

Usage:
    python set_unit.py --ip <ROBOT_IP>
"""

import argparse
from loguru import logger

from telekinesis.synapse.tools.parallel_grippers import robotiq


def main(ip: str):
    """Switch the gripper position unit to normalized (0..1)."""

    # Create and connect to the gripper
    gripper = robotiq.Robotiq2F85()
    gripper.connect(ip=ip)

    try:
        gripper.set_unit(parameter='position', unit='normalized')
        logger.success("Position unit set to 'normalized'.")
    finally:
        gripper.disconnect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Robotiq gripper set_unit example")
    parser.add_argument("--ip", type=str, required=True, help="UR robot IP address")
    args = parser.parse_args()

    main(ip=args.ip)
