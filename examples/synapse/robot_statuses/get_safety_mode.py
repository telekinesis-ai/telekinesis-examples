"""
Read safety mode example for the Synapse SDK.

Returns the controller's safety mode (e.g. ``"NORMAL"``, ``"REDUCED"``,
``"PROTECTIVE_STOP"``, ``"SAFEGUARD_STOP"``).

Currently supported only for Universal Robots (UR10e).

Usage:
    python get_safety_mode.py --ip <ROBOT_IP>
"""

import argparse
from loguru import logger

from telekinesis.synapse.robots.manipulators import universal_robots


def main(ip: str):
    """Log the current safety mode."""

    # Create and connect to the robot
    robot = universal_robots.UniversalRobotsUR10E()
    robot.connect(ip=ip)

    try:
        logger.success(f"Safety mode: {robot.get_safety_mode()}")
    finally:
        robot.disconnect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Read safety mode Synapse example")
    parser.add_argument("--ip", type=str, required=True, help="UR robot IP address")
    args = parser.parse_args()

    main(ip=args.ip)
