"""
Example: Demonstrates how to change the active TCP

Usage:
    python change_active_tcp.py [--ip <ROBOT_IP>]

Demonstrates:
- add_tcp()                   — register a custom TCP frame and push it to the controller
- active_tcp                  — change which frame is currently active

Currently supported only for real hardware from Universal Robots.

For offline, refer to set_cartesian_pose in state_reading/offline/
"""

import argparse
from loguru import logger

from telekinesis.synapse.robots.manipulators.universal_robots import UniversalRobotsUR10E

def main():
    """Change the active TCP and observe it before and after each change."""

    # Parse command-line arguments for the UR controller IP address
    parser = argparse.ArgumentParser(description="Set and get TCP on a real UR10E.")
    parser.add_argument("--ip", type=str, default="192.168.1.100", help="UR controller IP address (default: 192.168.1.100)")
    args = parser.parse_args()

    # Create a UniversalRobotsUR10E instance and connect to the robot
    robot = UniversalRobotsUR10E()
    robot.connect(ip=args.ip)

    try:
        new_tcp_pose_in_default_tcp_frame = [0.0, 0.0, 0.1, 0.0, 0.0, 0.0]  # 100 mm along Z-axis
        robot.add_tcp(name="camera_tip",
                      transform=new_tcp_pose_in_default_tcp_frame,
                      set_active=True)
        robot.add_tcp(name="gripper_tip",
                      transform=[0.0, 0.0, 0.2, 0.0, 0.0, 0.0],
                      set_active=False)
        robot.add_tcp(name="laser_tip",
                      transform=[0.0, 0.0, 0.3, 0.0, 0.0, 0.0],
                      set_active=False)
        

        # Get updated Active TCP, transform w.r.t default tcp, and TCP pose
        logger.info(f"Active TCP after add_tcp(): {robot.active_tcp}"
                    f" \nActive TCP transform: {robot.get_active_tcp_transform()}"
                    f" \n TCP pose: {robot.get_cartesian_pose()}")

        # Change the active TCP
        robot.active_tcp = "gripper_tip"

        # Get updated Active TCP, transform w.r.t default tcp, and TCP pose
        logger.info(f"Active TCP after changing active TCP: {robot.active_tcp}"
                    f" \nActive TCP transform: {robot.get_active_tcp_transform()}"
                    f" \n TCP pose: {robot.get_cartesian_pose()}")

        # Change the active TCP again
        robot.active_tcp = "laser_tip"

        # Get updated Active TCP, transform w.r.t default tcp, and TCP pose
        logger.info(f"Active TCP after changing active TCP again: {robot.active_tcp}"
                    f" \nActive TCP transform: {robot.get_active_tcp_transform()}"
                    f" \n TCP pose: {robot.get_cartesian_pose()}")

    finally:
        # Disconnect
        robot.disconnect()


if __name__ == "__main__":
    main()
