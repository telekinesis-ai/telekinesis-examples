"""
Example: Set controller interface TCP as active

Usage:
    python set_controller_interface_tcp_as_active.py [--ip <ROBOT_IP>]

Demonstrates:
- get_tcps()                  — retrieve all registered TCP frames from the controller
- use_controller_interface_tcp()  — set the controller interface TCP as the active TCP
"""

import argparse
from loguru import logger

from telekinesis.synapse.robots.manipulators.universal_robots import UniversalRobotsUR10E

def main():
    """Set the controller-interface TCP as active."""

    # Parse command-line arguments for the UR controller IP address
    parser = argparse.ArgumentParser(description="Set and get TCP on a real UR10E.")
    parser.add_argument("--ip", type=str, default="192.168.1.100", help="UR controller IP address (default: 192.168.1.100)")
    args = parser.parse_args()

    # Create a UniversalRobotsUR10E instance and connect to the robot
    robot = UniversalRobotsUR10E()
    robot.connect(ip=args.ip)

    try:
        # Get all registered TCPs
        tcps = robot.get_tcps()
        logger.info(f"Registered TCPs: {tcps}")

        # Current Active TCP, transform w.r.t default tcp, and current TCP pose
        logger.info(f"Active TCP: {robot.active_tcp}"
                    f" \nActive TCP transform: {robot.get_active_tcp_transform()}"
                    f" \n TCP pose: {robot.get_cartesian_pose()}")

        # Set the controller-interface TCP as active
        robot.active_tcp = "controller_interface_tcp"

        # Get updated Active TCP, transform w.r.t default tcp, and TCP pose
        logger.info(f"Active TCP after setting controller_interface_tcp: {robot.active_tcp}"
                    f" \nActive TCP transform: {robot.get_active_tcp_transform()}"
                    f" \n TCP pose: {robot.get_cartesian_pose()}")

    finally:
        # Disconnect
        robot.disconnect()


if __name__ == "__main__":
    main()
