"""
Read joint positions example for the Synapse SDK.

Returns the manipulator's joint positions [deg]. Connects to ``--ip`` (default ``192.168.1.100``) and reads the live state.

Illustrated using Universal Robots (UR10e), supported on all robots.

Usage:
    python get_joint_positions.py [--ip <ROBOT_IP>]
"""

import argparse
from loguru import logger

from telekinesis.synapse.robots.manipulators import universal_robots


def main(ip: str | None = None):
    """Log the current joint positions [deg]."""

    # Create the robot instance
    robot = universal_robots.UniversalRobotsUR10E()

    robot.connect(ip=ip)

    try:
        logger.success(f"joint_positions [deg]: {robot.get_joint_positions()}")
    finally:
        robot.disconnect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Read joint positions Synapse example")
    parser.add_argument("--ip", type=str, default="192.168.1.100", help="UR robot IP address (default: 192.168.1.100)")
    args = parser.parse_args()

    main(ip=args.ip)
