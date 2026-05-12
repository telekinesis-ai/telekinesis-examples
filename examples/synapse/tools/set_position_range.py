"""
Robotiq 2F-85 set position range example for the Synapse SDK.

Sets the gripper stroke range. 85 mm is the 2F-85 maximum.

Usage:
    python set_position_range.py --ip <ROBOT_IP>
"""

import argparse
from loguru import logger

from telekinesis.synapse.tools.parallel_grippers import robotiq


def main(ip: str):
    """Set the gripper stroke range to 85 mm (2F-85 max)."""

    # Create and connect to the gripper
    gripper = robotiq.Robotiq2F85()
    gripper.connect(ip=ip)

    try:
        gripper.set_position_range_mm(position_range_mm=85.0)
        logger.success("Position range set to 85 mm.")
    finally:
        gripper.disconnect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Robotiq gripper set_position_range example")
    parser.add_argument("--ip", type=str, required=True, help="UR robot IP address")
    args = parser.parse_args()

    main(ip=args.ip)
