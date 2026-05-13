"""
Robotiq 2F-85 set default speed example for the Synapse SDK.

Sets the default speed used by subsequent open/close/move calls.
Value is a percentage of the gripper's max speed.

Usage:
    python set_speed.py --ip <ROBOT_IP>
"""

import argparse
from loguru import logger

from telekinesis.synapse.tools.parallel_grippers import robotiq


def main(ip: str):
    """Set the gripper default speed to 50%."""

    # Create and connect to the gripper
    gripper = robotiq.Robotiq2F85()
    gripper.connect(ip=ip)

    try:
        actual = gripper.set_speed(speed=50.0)
        logger.success(f"Default speed set; effective: {actual}")
    finally:
        gripper.disconnect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Robotiq gripper set_speed example")
    parser.add_argument("--ip", type=str, required=True, help="UR robot IP address")
    args = parser.parse_args()

    main(ip=args.ip)
