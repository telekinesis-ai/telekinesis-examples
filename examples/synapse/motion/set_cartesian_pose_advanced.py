"""
Set Cartesian Pose example for the Synapse SDK.

Drives a real robot to the relative pose from current pose.
Currently supported only for Universal Robots (UR10e).

For offline, refer to quick start examples.

Usage:
    python set_cartesian_pose.py --ip <ROBOT_IP>
"""

import argparse
import time

from loguru import logger

from telekinesis.synapse.robots.manipulators import universal_robots


def main(robot_ip: str):
    """
    Main function to demonstrate how to create an instance of a robot using the Universal Robots module in Python.

    Args:
        robot_ip (str): The IP address of the UR robot
    Returns:
        None
    Raises:
        None
    """

    # Create robot instance
    robot = universal_robots.UniversalRobotsUR10E()

    # Connect to the robot
    robot.connect(ip=robot_ip)

    # Example 1
    # Get initial Cartesian pose
    initial_tcp_pose = robot.get_cartesian_pose()

    # Move to target Cartesian pose
    new_tcp_pose = initial_tcp_pose[:]
    new_tcp_pose[2] -= 0.2
    tcp_speed = 0.25
    tcp_acceleration = 0.25
    asynchronous = False

    # Move to target Cartesian pose
    robot.set_cartesian_pose(
        cartesian_pose=new_tcp_pose,
        speed=tcp_speed,
        acceleration=tcp_acceleration,
        asynchronous=asynchronous,
    )
    logger.info(f"Moved to target Cartesian pose: {new_tcp_pose}")

    # Example 2: Async
    # Get initial Cartesian pose
    actual_tcp_pose = robot.get_cartesian_pose()

    # Move to target Cartesian pose
    new_tcp_pose = actual_tcp_pose[:]
    new_tcp_pose[2] += 0.2
    tcp_speed = 0.25
    tcp_acceleration = 0.25
    stopping_speed = 0.25
    asynchronous = False

    # Move to target Cartesian pose
    robot.set_cartesian_pose(
        cartesian_pose=new_tcp_pose,
        speed=tcp_speed,
        acceleration=tcp_acceleration,
        asynchronous=asynchronous,
    )
    time.sleep(0.5)
    robot.stop_cartesian_motion(stopping_speed=stopping_speed)
    logger.info(f"Stopped Cartesian motion before reaching target Cartesian pose: {new_tcp_pose}")

    # Disconnect
    robot.disconnect()


if __name__ == "__main__":
    # args parser to get ip
    parser = argparse.ArgumentParser(description="UR10e robot movel example")
    parser.add_argument("--ip", type=str, default="192.168.1.2", help="IP address of the UR robot")
    args = parser.parse_args()

    main(args.ip)
