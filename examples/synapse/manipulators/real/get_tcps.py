"""
Example: Get all registered TCPs from the robot

Usage:
    python get_tcps.py [--ip <ROBOT_IP>]

Demonstrates:
- get_tcps()                  — retrieve all registered TCP frames from the controller
- get_active_tcp_transform()  — read the active TCP offset (metres, Euler-XYZ degrees)
- active_tcp                  — check which frame is currently active

Currently supported only for real hardware. Works on Universal Robots (UR) and Epson.

For an offline version, refer to tcp/offline/get_tcps.py
"""

import argparse
from loguru import logger

from telekinesis.synapse.robots.manipulators.universal_robots import UniversalRobotsUR10E

def main():
    """
    Get all registered TCPs from the robot.
    """

    # Parse command-line arguments for the UR controller IP address
    parser = argparse.ArgumentParser(description="List all registered TCPs on a real UR10E.")
    parser.add_argument("--ip", type=str, default="192.168.1.100", help="UR controller IP address (default: 192.168.1.100)")
    args = parser.parse_args()

    #===================== Create Robot ==========================================
    robot = UniversalRobotsUR10E()
    robot.connect(ip=args.ip)

    # ==================== Run Skill ============================================
    try:
        # Get all registered TCPs
        tcps = robot.get_tcps()
        logger.info(f"Registered TCPs: {tcps}")

        # Current Active TCP, transform w.r.t default tcp, and current TCP pose
        logger.info(f"Active TCP: {robot.active_tcp}"
                    f" \nActive TCP transform: {robot.get_active_tcp_transform()}"
                    f" \n TCP pose: {robot.get_cartesian_pose()}")


    finally:
        # Disconnect
        robot.disconnect()


if __name__ == "__main__":
    main()
