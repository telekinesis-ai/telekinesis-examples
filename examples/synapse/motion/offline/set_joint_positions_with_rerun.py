"""
Set Joint Positions with Rerun visualization example for the Synapse SDK -- offline.

Moves the robot to a target joint configuration on the kinematic model; no
hardware connection is made. The robot is drawn in Rerun before and after the move.

Supports all robots.

Usage:
    python set_joint_positions_with_rerun.py
"""

import time

from loguru import logger

from telekinesis.synapse.robots.manipulators import universal_robots


def main():
    """Move the robot to a target joint configuration on the kinematic model, drawing it in Rerun before and after the move."""

    # Create robot instance with a name so its state publisher starts
    # (the subscriber below needs a non-empty state_publisher_topic).
    robot = universal_robots.UniversalRobotsUR10E(name="manipulator1")

    # Visualize the robot in Rerun
    robot.visualize_rerun()
    time.sleep(2.0)  # Wait for Rerun to initialize

    # Target: current joint configuration with the base joint rotated +30 deg
    target_joint_positions = robot.get_joint_positions().copy()
    target_joint_positions[0] += 30

    # Command the move
    robot.set_joint_positions(
        joint_positions=target_joint_positions,
        speed=60,
        acceleration=80,
    )
    robot.visualize_rerun()
    logger.info(f"Moved to target joint positions: {target_joint_positions}")


if __name__ == "__main__":
    main()
