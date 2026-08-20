"""
Check steady status example for the Synapse SDK.

``is_steady`` reports whether the robot is fully at rest and ready to
accept external forces. Always ``False`` in force mode or teach mode.

Currently supported only for real hardware, and only Universal Robots (UR).

Usage:
    python is_steady.py [--ip <ROBOT_IP>]
"""

import argparse

from loguru import logger

from telekinesis.synapse.robots.manipulators import universal_robots


def main(ip: str):
    """Log whether the robot is fully at rest."""

    #===================== Create Robot ==========================================
    robot = universal_robots.UniversalRobotsUR10E()
    robot.connect(ip=ip)

    # ==================== Run Skill ============================================
    try:
        logger.success(f"is_steady: {robot.is_steady()}")
    finally:
        robot.disconnect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check steady status Synapse example")
    parser.add_argument("--ip", type=str, default="192.168.1.100", help="UR robot IP address (default: 192.168.1.100)")
    args = parser.parse_args()

    main(ip=args.ip)
