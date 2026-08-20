"""
Check controller program status example for the Synapse SDK.

``is_program_running_on_controller`` reports whether a PolyScope program is
currently running on the controller.

Currently supported only for real hardware, and only Universal Robots (UR).

Usage:
    python is_program_running_on_controller.py [--ip <ROBOT_IP>]
"""

import argparse

from loguru import logger

from telekinesis.synapse.robots.manipulators import universal_robots


def main(ip: str):
    """Log whether a PolyScope program is currently running."""

    #===================== Create Robot ==========================================
    robot = universal_robots.UniversalRobotsUR10E()
    robot.connect(ip=ip)

    # ==================== Run Skill ============================================
    try:
        logger.success(f"is_program_running_on_controller: {robot.is_program_running_on_controller()}")
    finally:
        robot.disconnect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check controller program status Synapse example")
    parser.add_argument("--ip", type=str, default="192.168.1.100", help="UR robot IP address (default: 192.168.1.100)")
    args = parser.parse_args()

    main(ip=args.ip)
