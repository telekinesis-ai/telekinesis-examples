"""
Logs whether the manipulator state is being driven by live hardware, before and after connecting.

Supports Universal Robots (UR), Epson, and virtual.

Usage:
    python is_connected.py [--ip <ROBOT_IP>]
"""

import argparse

from loguru import logger

from telekinesis.synapse.robots.manipulators import universal_robots


def main(ip: str) -> None:
    """Log is_connected before and after connecting to real hardware."""

    #===================== Create Robot ==========================================
    robot = universal_robots.UniversalRobotsUR10E(name='UR10e')
    logger.info(f"is_connected before connect(): {robot.is_connected()}")

    try:
        #===================== Connect Robot ==========================================
        robot.connect(ip=ip)

        # ==================== Run Skill ============================================
        logger.success(f"is_connected while connected: {robot.is_connected()}")
    except (ConnectionError, OSError) as e:
        logger.error(f"Error occurred: {e}")
    finally:
        robot.disconnect()
    logger.info(f"is_connected after disconnect(): {robot.is_connected()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check connection status Synapse example")
    parser.add_argument("--ip", type=str, default="192.168.1.100", help="UR robot IP address (default: 192.168.1.100)")
    args = parser.parse_args()

    main(ip=args.ip)
