"""
Read publisher names and types example for the Synapse SDK.

Returns the babyros topics this robot publishes on and their message types
— empty when the robot was constructed without a ``name``.

Currently supported only for real hardware. Works on Universal Robots (UR) and Epson.

Usage:
    python get_publisher_names_and_types.py [--ip <ROBOT_IP>]
"""

import argparse

from loguru import logger

from telekinesis.synapse.robots.manipulators import universal_robots


def main(ip: str):
    """Log the babyros topics published by a named robot."""

    #===================== Create Robot ==========================================
    robot = universal_robots.UniversalRobotsUR10E(name="manipulator1")
    robot.connect(ip=ip)

    # ==================== Run Skill ============================================
    try:
        logger.success(f"publisher_names_and_types: {robot.get_publisher_names_and_types()}")
    finally:
        robot.disconnect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Read publisher names and types Synapse example")
    parser.add_argument("--ip", type=str, default="192.168.1.100", help="UR robot IP address (default: 192.168.1.100)")
    args = parser.parse_args()

    main(ip=args.ip)
