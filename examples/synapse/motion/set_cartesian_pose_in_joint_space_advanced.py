"""
Set Cartesian Pose in Joint Space example for the Synapse SDK.

Drives a real robot to a Cartesian pose using joint-space motion in synchronous
mode, then commands the same move asynchronously and interrupts it mid-trajectory
with ``stop_joint_motion``.

Currently supported only for Universal Robots (UR10e).

Usage:
    python set_cartesian_pose_in_joint_space_advanced.py --ip <ROBOT_IP>
"""

import argparse
import time
from loguru import logger

from telekinesis.synapse.robots.manipulators import universal_robots


def main(robot_ip: str):
    """Demonstrate synchronous and interrupted asynchronous Cartesian-in-joint-space motion."""

    # Create robot instance
    robot = universal_robots.UniversalRobotsUR10E()

    # Connect to the robot
    robot.connect(ip=robot_ip)

    # Get initial Cartesian pose [x, y, z, rx, ry, rz] (m, deg)
    initial_pose = robot.get_cartesian_pose()
    logger.info(f"Initial Cartesian pose: {initial_pose}")

    # Synchronous move: raise the TCP by 10 cm using joint-space motion
    target_pose = initial_pose[:]
    target_pose[2] += 0.1
    robot.set_cartesian_pose_in_joint_space(
        cartesian_pose=target_pose,
        speed=20,
        acceleration=40,
        asynchronous=False,
    )
    logger.info(f"Moved to target Cartesian pose: {target_pose}")

    # Asynchronous move back, interrupted mid-trajectory
    actual_pose = robot.get_cartesian_pose()
    target_pose = actual_pose[:]
    target_pose[2] -= 0.1
    robot.set_cartesian_pose_in_joint_space(
        cartesian_pose=target_pose,
        speed=20,
        acceleration=40,
        asynchronous=True,
    )
    time.sleep(0.5)
    robot.stop_joint_motion(stopping_speed=20)
    logger.info(f"Stopped joint motion before reaching: {target_pose}")

    # Disconnect
    robot.disconnect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="UR robot set cartesian pose in joint space example"
    )
    parser.add_argument("--ip", type=str, required=True, help="IP address of the UR robot")
    args = parser.parse_args()

    main(args.ip)
