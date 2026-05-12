"""
Robotiq 2F-85 set default grip force example for the Synapse SDK.

Sets the default grip force used by subsequent open/close/move calls.
Value is a percentage of the gripper's max force.

Usage:
    python set_force.py --ip <ROBOT_IP>
"""

import argparse
from loguru import logger

from telekinesis.synapse.tools.parallel_grippers import robotiq


def main(ip: str):
    """Set the gripper default grip force to 50%."""

    # Create and connect to the gripper
    gripper = robotiq.Robotiq2F85()
    gripper.connect(ip=ip)

    try:
        actual = gripper.set_force(force=50.0)
        logger.success(f"Default force set; effective: {actual}")
    finally:
        gripper.disconnect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Robotiq gripper set_force example")
    parser.add_argument("--ip", type=str, required=True, help="UR robot IP address")
    args = parser.parse_args()

    main(ip=args.ip)
