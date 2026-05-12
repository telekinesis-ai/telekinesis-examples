"""
Stop Cartesian Motion example for the Synapse SDK.

Commands an asynchronous Cartesian move and interrupts it mid-trajectory
with ``stop_cartesian_motion``. ``stopping_speed`` controls the
deceleration profile (m/s).

Currently supported only for Universal Robots (UR10e).

Usage:
    python stop_cartesian_motion.py --ip <ROBOT_IP>
"""

import argparse
import time
from loguru import logger

from telekinesis.synapse.robots.manipulators import universal_robots


def main(robot_ip: str):
    """Start an async Cartesian move and interrupt it with stop_cartesian_motion."""

    # Create robot instance
    robot = universal_robots.UniversalRobotsUR10E()

    # Connect to the robot
    robot.connect(ip=robot_ip)

    # Get initial Cartesian pose [x, y, z, rx, ry, rz] (m, deg)
    actual_pose = robot.get_cartesian_pose()

    # Asynchronous +15 cm move along Z
    target_pose = actual_pose[:]
    target_pose[2] += 0.15
    robot.set_cartesian_pose(
        cartesian_pose=target_pose,
        speed=0.25,
        acceleration=0.5,
        asynchronous=True,
    )

    # Let the move run briefly, then interrupt it
    time.sleep(0.3)
    robot.stop_cartesian_motion(stopping_speed=0.25)
    logger.info("Stopped Cartesian motion.")

    # Disconnect
    robot.disconnect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="UR robot stop cartesian motion example")
    parser.add_argument("--ip", type=str, required=True, help="IP address of the UR robot")
    args = parser.parse_args()

    main(args.ip)
