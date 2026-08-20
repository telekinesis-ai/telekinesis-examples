"""
Freedrive mode example for the Synapse SDK.

``start_freedrive_mode`` puts the robot into hand-guiding mode — the operator
can physically push the arm and it complies. ``free_axes`` is a 6-element
mask ``[x, y, z, rx, ry, rz]`` where ``1`` means the axis is free and ``0``
means it is locked. ``stop_freedrive_mode`` returns the controller to normal
motion control.

Currently supported only for real hardware, and only Universal Robots (UR).

Usage:
    python start_and_stop_freedrive_mode.py [--ip <ROBOT_IP>]
"""

import argparse
import time
from loguru import logger

from telekinesis.synapse.robots.manipulators import universal_robots


def main(ip: str):
    """Enter freedrive for 10 seconds, then exit."""

    #===================== Create Robot ==========================================
    robot = universal_robots.UniversalRobotsUR10E()
    robot.connect(ip=ip)

    # ==================== Run Skill ============================================
    # Enter freedrive with all axes free
    free_axes = [1, 1, 1, 1, 1, 1]
    logger.info(f"Starting freedrive - free axes: {free_axes}")
    robot.start_freedrive_mode(free_axes=free_axes)

    # Hold freedrive open for hand-guiding
    time.sleep(10)

    # Exit freedrive
    robot.stop_freedrive_mode()
    logger.success("Freedrive mode stopped.")

    # ==================== Visualization (Optional) =============================
    robot.visualize_rerun(live=False)

    # Disconnect
    robot.disconnect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Start freedrive (hand-guiding) mode, then stop it")
    parser.add_argument("--ip", type=str, default="192.168.1.100", help="UR robot IP address (default: 192.168.1.100)")
    args = parser.parse_args()

    main(ip=args.ip)

