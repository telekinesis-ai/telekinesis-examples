"""
Check connection status example for the Synapse SDK.

``is_connected`` reports whether the manipulator state is being driven by
live hardware. Once ``connect(ip=...)`` succeeds it reports ``True``; after
``disconnect()`` it reports ``False`` again.

Currently supported only for real hardware. Works on Universal Robots (UR) and Epson.

Usage:
    python is_connected.py [--ip <ROBOT_IP>]
"""

import argparse

from loguru import logger

from telekinesis.synapse.robots.manipulators import universal_robots


def main(ip: str):
    """Log ``is_connected`` before and after connecting to real hardware."""

    #===================== Create Robot ==========================================
    robot = universal_robots.UniversalRobotsUR10E()
    logger.info(f"is_connected before connect(): {robot.is_connected()}")

    # ==================== Run Skill ============================================
    robot.connect(ip=ip)
    try:
        logger.success(f"is_connected while connected: {robot.is_connected()}")
    finally:
        robot.disconnect()
    logger.info(f"is_connected after disconnect(): {robot.is_connected()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check connection status Synapse example")
    parser.add_argument("--ip", type=str, default="192.168.1.100", help="UR robot IP address (default: 192.168.1.100)")
    args = parser.parse_args()

    main(ip=args.ip)
