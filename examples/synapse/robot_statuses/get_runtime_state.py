"""
Read runtime state example for the Synapse SDK.

Returns the controller's program runtime state (e.g. ``"PLAYING"``,
``"PAUSED"``, ``"STOPPED"``).

Currently supported only for Universal Robots (UR10e).

Usage:
    python get_runtime_state.py --ip <ROBOT_IP>
"""

import argparse
from loguru import logger

from telekinesis.synapse.robots.manipulators import universal_robots


def main(ip: str):
    """Log the current runtime state."""

    # Create and connect to the robot
    robot = universal_robots.UniversalRobotsUR10E()
    robot.connect(ip=ip)

    try:
        logger.success(f"Runtime state: {robot.get_runtime_state()}")
    finally:
        robot.disconnect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Read runtime state Synapse example")
    parser.add_argument("--ip", type=str, required=True, help="UR robot IP address")
    args = parser.parse_args()

    main(ip=args.ip)
