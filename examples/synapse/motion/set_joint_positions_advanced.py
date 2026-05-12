"""
Set Joint Positions example for the Synapse SDK.

Drives a real robot to the relative joint positions from current positions in synchronous mode
and back in asynchronous mode.

Currently supported only for Universal Robots (UR10e).

Usage:
    python set_cartesian_pose.py --ip <ROBOT_IP>
"""

import argparse
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
    robot = universal_robots.UniversalRobotsUR5()

    # Connect to the robot
    robot.connect(ip=robot_ip)

    # Get initial joint positions
    initial_joint_positions = robot.get_joint_positions()
    logger.info(f"Initial joint positions: {initial_joint_positions}")

    # Move to target joint positions
    new_joint_positions = initial_joint_positions[:]
    new_joint_positions[0] -= 5
    speed = 20
    acceleration = 20
    asynchronous = False

    # Move to target joint positions
    robot.set_joint_positions(
        joint_positions=new_joint_positions,
        speed=speed,
        acceleration=acceleration,
        asynchronous=asynchronous,
    )
    logger.info(f"Moved to target joint positions: {new_joint_positions}")

    # Get current joint positions
    actual_joint_positions = robot.get_joint_positions()

    # Move to target joint positions
    new_joint_positions = actual_joint_positions[:]
    new_joint_positions[0] += 5
    speed = 20
    acceleration = 20
    asynchronous = True

    # Move to target joint positions
    robot.set_joint_positions(
        joint_positions=new_joint_positions,
        speed=speed,
        acceleration=acceleration,
        asynchronous=asynchronous,
    )

    robot.stop_joint_motion(stopping_speed=20)
    logger.info(f"Stopped joint motion before reaching target joint positions: {new_joint_positions}")

    # Disconnect
    robot.disconnect()


if __name__ == "__main__":
    # args parser to get ip
    parser = argparse.ArgumentParser(description="UR5cb robot set joint positions example")
    parser.add_argument("--ip", type=str, required=True, help="IP address of the UR robot")
    args = parser.parse_args()

    main(args.ip)
