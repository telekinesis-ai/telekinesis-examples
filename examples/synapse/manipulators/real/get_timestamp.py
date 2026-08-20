"""
Read state timestamp example for the Synapse SDK.

Returns the time the most recent state update was captured, in seconds
since the Unix epoch.

Currently supported only for real hardware. Works on Universal Robots (UR) and Epson.

Usage:
    python get_timestamp.py [--ip <ROBOT_IP>]
"""

import argparse

from loguru import logger

from telekinesis.synapse.robots.manipulators import universal_robots


def main(ip: str):
    """Log the timestamp of the most recent state update."""

    #===================== Create Robot ==========================================
    robot = universal_robots.UniversalRobotsUR10E()
    robot.connect(ip=ip)

    # ==================== Run Skill ============================================
    try:
        logger.success(f"timestamp [s since epoch]: {robot.get_timestamp()}")
    finally:
        robot.disconnect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Read state timestamp Synapse example")
    parser.add_argument("--ip", type=str, default="192.168.1.100", help="UR robot IP address (default: 192.168.1.100)")
    args = parser.parse_args()

    main(ip=args.ip)
