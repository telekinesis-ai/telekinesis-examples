"""
Example: Demonstrates adding and deleting TCP

Usage:
    python delete_tcp.py [--ip <ROBOT_IP>]

Demonstrates:
- add_tcp()                   — register a custom TCP frame and push it to the controller
- delete_tcp()                — remove a custom TCP frame from the controller
"""

import argparse
from loguru import logger

from telekinesis.synapse.robots.manipulators.universal_robots import UniversalRobotsUR10E

def main():
    """
    Add and delete TCP
    """

    # Parse command-line arguments for the UR controller IP address
    parser = argparse.ArgumentParser(description="Set and get TCP on a real UR10E.")
    parser.add_argument("--ip", type=str, default="192.168.1.100", help="UR controller IP address (default: 192.168.1.100)")
    args = parser.parse_args()

    # Create a UniversalRobotsUR10E instance and connect to the robot
    robot = UniversalRobotsUR10E()
    robot.connect(ip=args.ip)

    try:
        new_tcp_pose_in_default_tcp_frame = [0.0, 0.0, 0.1, 0.0, 0.0, 0.0]  # 100 mm along Z-axis
        robot.add_tcp(name="new_tool",
                      transform=new_tcp_pose_in_default_tcp_frame,
                      set_active=True)

        # Get updated Active TCP, transform w.r.t default tcp, and TCP pose
        logger.info(f"Active TCP after add_tcp(): {robot.active_tcp}"
                    f" \nActive TCP transform: {robot.get_active_tcp_transform()}"
                    f" \n TCP pose: {robot.get_cartesian_pose()}")

        # Delete the TCP
        robot.delete_tcp(name="new_tool")

        # Get updated Active TCP, transform w.r.t default tcp, and TCP pose
        logger.info(f"Active TCP after delete_tcp(): {robot.active_tcp}"
                    f" \nActive TCP transform: {robot.get_active_tcp_transform()}"
                    f" \n TCP pose: {robot.get_cartesian_pose()}")

    finally:
        # Disconnect
        robot.disconnect()


if __name__ == "__main__":
    main()
