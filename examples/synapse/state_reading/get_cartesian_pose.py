"""
Read TCP Cartesian pose example for the Synapse SDK.

Returns the TCP pose ``[x, y, z (m), rx, ry, rz (deg)]``.
Connects to ``--ip`` (default ``192.168.1.100``) and reads the live state.

Currently supported only for real hardware from Universal Robots.

For offline, refer to set_cartesian_pose in state_reading/offline/

Usage:
    python get_cartesian_pose.py [--ip <ROBOT_IP>]
"""

import argparse
from loguru import logger

from telekinesis.synapse.robots.manipulators import universal_robots


def main(ip: str):
    """Log the current TCP pose [m, deg]."""

    robot = universal_robots.UniversalRobotsUR10E()
    robot.connect(ip=ip)

    try:
        logger.success(f"tcp_pose [m, deg]: {robot.get_cartesian_pose()}")
    finally:
        robot.disconnect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Read TCP Cartesian pose Synapse example")
    parser.add_argument("--ip", type=str, default="192.168.1.100", help="UR robot IP address (default: 192.168.1.100)")
    args = parser.parse_args()

    main(ip=args.ip)
