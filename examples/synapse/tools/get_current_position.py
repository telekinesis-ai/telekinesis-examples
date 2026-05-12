"""
Robotiq 2F-85 read current position example for the Synapse SDK.

Reads the gripper's current position in the configured unit (normalized
or mm).

Usage:
    python get_current_position.py --ip <ROBOT_IP>
"""

import argparse
from loguru import logger

from telekinesis.synapse.tools.parallel_grippers import robotiq


def main(ip: str):
    """Read the gripper's current position in the configured unit."""

    # Create and connect to the gripper
    gripper = robotiq.Robotiq2F85()

    # Connect to the gripper
    gripper.connect(ip=ip)

    try:

        # Read and report the current position
        logger.success(f"Current position: {gripper.get_current_position()}")

    finally:
        # Disconnect cleanly even if there was an error
        gripper.disconnect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Robotiq gripper get_current_position example")
    parser.add_argument("--ip", type=str, required=True, help="UR robot IP address")
    args = parser.parse_args()

    main(ip=args.ip)
