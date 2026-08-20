"""
Read target TCP pose example for the Synapse SDK.

Returns the controller-commanded target TCP pose ``[x, y, z, rx, ry, rz]``
(m, deg). Reads from ``self.state``, which the control loop keeps up to date
from the connected backend.

Currently supported only for real hardware. Works on Universal Robots (UR) and Epson.

Usage:
    python get_target_tcp_pose.py [--ip <ROBOT_IP>]
"""

import argparse

from loguru import logger

from telekinesis.synapse.robots.manipulators import universal_robots


def main(ip: str):
    """Log the controller-commanded target TCP pose."""

    #===================== Create Robot ==========================================
    robot = universal_robots.UniversalRobotsUR10E()
    robot.connect(ip=ip)

    # ==================== Run Skill ============================================
    try:
        logger.success(f"target_tcp_pose [m, deg]: {robot.get_target_tcp_pose()}")
    finally:
        robot.disconnect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Read target TCP pose Synapse example")
    parser.add_argument("--ip", type=str, default="192.168.1.100", help="UR robot IP address (default: 192.168.1.100)")
    args = parser.parse_args()

    main(ip=args.ip)
