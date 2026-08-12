"""
Set Cartesian Pose with Rerun visualization example for the Synapse SDK -- offline.

Moves the TCP to a target Cartesian pose on the kinematic model; no hardware
connection is made. The robot is drawn in Rerun before and after the move.

Supports all robots.

Usage:
    python set_cartesian_pose_with_rerun.py
"""

import time

from loguru import logger

from telekinesis.synapse.robots.manipulators import universal_robots


def main():
    """Move the TCP to a target pose on the kinematic model, drawing it in Rerun before and after the move."""

    # Create robot instance
    robot = universal_robots.UniversalRobotsUR10E(name="manipulator1")

    # Visualize the robot in Rerun
    robot.visualize_rerun()
    time.sleep(2.0)  # Wait for Rerun to initialize

    # Define target Cartesian pose
    target_cartesian_pose = [0.5, 0.0, 0.5, 180.0, 0.0, 0.0]

    # Command the move
    robot.set_cartesian_pose(
        cartesian_pose=target_cartesian_pose,
        speed=0.1,
        acceleration=0.1,
    )
    logger.info(f"Moved to target Cartesian pose: {target_cartesian_pose}")
    robot.visualize_rerun()


if __name__ == "__main__":
    main()
