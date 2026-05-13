"""
Stop Joint Motion example for the Synapse SDK.

Commands an asynchronous joint move and interrupts it mid-trajectory
with ``stop_joint_motion``. ``stopping_speed`` controls the deceleration
profile (deg/s).

Currently supported only for Universal Robots (UR10e).

Usage:
    python stop_joint_motion.py --ip <ROBOT_IP>
"""

import argparse
import time
from loguru import logger

from telekinesis.synapse.robots.manipulators import universal_robots


def main(robot_ip: str):
    """Start an async joint move and interrupt it with stop_joint_motion."""

    # Create robot instance
    robot = universal_robots.UniversalRobotsUR10E()

    # Connect to the robot
    robot.connect(ip=robot_ip)

    # Get initial joint positions [deg]
    initial_joint_positions = robot.get_joint_positions()

    # Asynchronous +20 deg move on joint 0
    target_joint_positions = initial_joint_positions[:]
    target_joint_positions[0] += 20
    robot.set_joint_positions(
        joint_positions=target_joint_positions,
        speed=60,
        acceleration=80,
        asynchronous=True,
    )

    # Let the move run briefly, then interrupt it
    time.sleep(0.3)
    robot.stop_joint_motion(stopping_speed=30)
    logger.info("Stopped joint motion.")

    # Disconnect
    robot.disconnect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="UR robot stop joint motion example")
    parser.add_argument("--ip", type=str, required=True, help="IP address of the UR robot")
    args = parser.parse_args()

    main(args.ip)
