"""
Robotiq 2F-85 close example for the Synapse SDK.

Closes the gripper fully at 100% speed and 50% force, synchronously.

Usage:
    python close.py --ip <ROBOT_IP>
"""

import argparse
from loguru import logger

from telekinesis.synapse.tools.parallel_grippers import robotiq


def main(ip: str):
    """Close the gripper fully at 100% speed and 50% force."""

    # Create and connect to the gripper
    gripper = robotiq.Robotiq2F85()
    gripper.connect(ip=ip)

    try:
        status = gripper.close(speed=100.0, force=50.0, asynchronous=False)
        logger.success(f"close() status: {status}, position: {gripper.get_current_position():.2f}")
    finally:
        gripper.disconnect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Robotiq gripper close example")
    parser.add_argument("--ip", type=str, required=True, help="UR robot IP address")
    args = parser.parse_args()

    main(ip=args.ip)
