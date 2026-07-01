"""
Set Cartesian Pose (advanced) example for the Synapse SDK.

Moves the TCP down 20 cm synchronously, then commands a 20 cm move back up
and interrupts it mid-trajectory with ``stop_cartesian_motion``.

Currently supported only for Universal Robots (UR10e).

For offline, refer to quick start examples.

Usage:
    python set_cartesian_pose_advanced.py --ip <ROBOT_IP>
"""

import argparse
import time

from loguru import logger

from telekinesis.synapse.robots.manipulators import universal_robots


def main(robot_ip: str):
    """Run a synchronous Cartesian move, then a move interrupted mid-trajectory."""

    # Create robot instance
    robot = universal_robots.UniversalRobotsUR10E()

    # Connect to the robot
    robot.connect(ip=robot_ip)

    # Example 1: synchronous move down 20 cm in Z
    initial_tcp_pose = robot.get_cartesian_pose()

    # Build the target pose
    new_tcp_pose = initial_tcp_pose[:]
    new_tcp_pose[2] -= 0.2
    tcp_speed = 0.25
    tcp_acceleration = 0.25
    asynchronous = False

    # Command the move
    robot.set_cartesian_pose(
        cartesian_pose=new_tcp_pose,
        speed=tcp_speed,
        acceleration=tcp_acceleration,
        asynchronous=asynchronous,
    )
    logger.info(f"Moved to target Cartesian pose: {new_tcp_pose}")

    # Example 2: move back up, then interrupt with stop_cartesian_motion
    # Get current Cartesian pose
    actual_tcp_pose = robot.get_cartesian_pose()

    # Build the target pose
    new_tcp_pose = actual_tcp_pose[:]
    new_tcp_pose[2] += 0.2
    tcp_speed = 0.25
    tcp_acceleration = 0.25
    stopping_speed = 0.25
    asynchronous = False

    # Command the move, then stop it mid-trajectory
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
