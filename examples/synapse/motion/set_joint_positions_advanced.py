"""
Set Joint Positions (advanced) example for the Synapse SDK.

Moves joint 0 by -5 deg synchronously, then commands a +5 deg move back and
interrupts it mid-trajectory with ``stop_joint_motion``.

Currently supported only for Universal Robots.

For offline, refer to quick start examples.

Usage:
    python set_joint_positions_advanced.py [--ip <ROBOT_IP>]
"""

import argparse
from loguru import logger

from telekinesis.synapse.robots.manipulators import universal_robots

def main(robot_ip: str):
    """Run a synchronous joint move, then a move interrupted with stop_joint_motion."""

    # Create robot instance
    robot = universal_robots.UniversalRobotsUR5()

    # Connect to the robot
    robot.connect(ip=robot_ip)

    # Get initial joint positions
    initial_joint_positions = robot.get_joint_positions()
    logger.info(f"Initial joint positions: {initial_joint_positions}")

    # Synchronous move: joint 0 by -5 deg
    new_joint_positions = initial_joint_positions[:]
    new_joint_positions[0] -= 5
    speed = 20
    acceleration = 20
    asynchronous = False

    robot.set_joint_positions(
        joint_positions=new_joint_positions,
        speed=speed,
        acceleration=acceleration,
        asynchronous=asynchronous,
    )
    logger.info(f"Moved to target joint positions: {new_joint_positions}")

    # Get current joint positions
    actual_joint_positions = robot.get_joint_positions()

    # Asynchronous move back +5 deg, interrupted with stop_joint_motion
    new_joint_positions = actual_joint_positions[:]
    new_joint_positions[0] += 5
    speed = 20
    acceleration = 20
    asynchronous = True

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
    parser.add_argument("--ip", type=str, default="192.168.1.100", help="IP address of the UR robot (default: 192.168.1.100)")
    args = parser.parse_args()

    main(args.ip)
