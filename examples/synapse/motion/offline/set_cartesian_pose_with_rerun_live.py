"""
Set Cartesian Pose with live Rerun logging example for the Synapse SDK -- offline.

Moves the TCP to a target Cartesian pose on the kinematic model; no hardware
connection is made. A babyros subscriber logs each state message and redraws
the robot in Rerun continuously as the move runs.

Supports all robots.

Usage:
    python set_cartesian_pose_with_rerun_live.py
"""

import time
from functools import partial

from loguru import logger
from babyros import node

from telekinesis.synapse.robots.manipulators import universal_robots


def on_state(msg, robot):
    """Log each state message and redraw the robot in Rerun.

    ``robot`` is bound via functools.partial so the callback keeps babyros's
    single-argument (msg) signature.
    """
    logger.info(f"Received robot state: {msg}")
    robot.visualize_rerun()


def main():
    """Move the TCP to a target pose on the kinematic model, logging live state to Rerun."""

    # Create robot instance with a name so its state publisher starts
    # (the subscriber below needs a non-empty state_publisher_topic).
    robot = universal_robots.UniversalRobotsUR10E(name="manipulator1")

    # Visualize the robot in Rerun
    robot.visualize_rerun()
    time.sleep(2.0)  # Wait for Rerun to initialize

    # Define target Cartesian pose
    current_cartesian_pose = robot.get_cartesian_pose()
    target_cartesian_pose = current_cartesian_pose.copy()
    target_cartesian_pose[2] -= 0.2  # Move 20 cm down in Z

    # Subscriber to states
    sub = node.Subscriber(topic=robot.state_publisher_topic,
                          callback=partial(on_state, robot=robot))

    # Command the move, then remove the subscriber cleanly
    try:
        robot.set_cartesian_pose(
            cartesian_pose=target_cartesian_pose,
            speed=0.1,
            acceleration=0.1,
        )
        logger.info(f"Moved to target Cartesian pose: {target_cartesian_pose}")
    finally:
        sub.delete()

if __name__ == "__main__":
    main()
