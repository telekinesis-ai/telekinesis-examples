"""
Set Cartesian Pose example for the Synapse SDK.

Drives a real UR10e to the target Cartesian pose. Currently supported
only for Universal Robots (UR10e).

Usage:
    python set_cartesian_pose.py --ip <ROBOT_IP>
"""

import argparse

from loguru import logger

from telekinesis.synapse.robots.manipulators import universal_robots


def set_cartesian_pose(robot_ip: str):
    """Move the TCP to a target Cartesian pose on a connected UR10e."""

    # Create robot instance
    robot = universal_robots.UniversalRobotsUR10E()

    # Define target Cartesian pose [x, y, z (m), rx, ry, rz (deg)]
    target_pose = [0, 0.65, 0.85, 180, 0, 90]

    # Safety warning before commanding real motion
    logger.warning(
        f"About to move real robot to target pose {target_pose}. "
        "Make sure it's safe to move there, otherwise use the advanced example."
    )

    # Connect to the robot
    robot.connect(ip=robot_ip)

    # Command the move, then disconnect cleanly
    try:
        robot.set_cartesian_pose(
            cartesian_pose=target_pose,
            speed=0.1,
            acceleration=0.1,
        )
        logger.info(f"Moved to target Cartesian pose: {target_pose}")
    finally:
        robot.disconnect()


def main():
    """Run the set_cartesian_pose Synapse example."""
    parser = argparse.ArgumentParser(description="UR10e set_cartesian_pose example")
    parser.add_argument("--ip", type=str, required=True, help="IP address of the UR robot")
    args = parser.parse_args()

    # Run the example
    set_cartesian_pose(robot_ip=args.ip)


if __name__ == "__main__":
    main()
