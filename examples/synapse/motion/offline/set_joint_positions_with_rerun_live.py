"""
Set Joint Positions with live Rerun logging example for the Synapse SDK -- offline.

Moves the robot to a target joint configuration on the kinematic model; no
hardware connection is made. A babyros subscriber logs each state message and
redraws the robot in Rerun continuously as the move runs.

Supports all robots.

Usage:
    python set_joint_positions_with_rerun_live.py
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
    """Move the robot to a target joint configuration on the kinematic model, logging live state to Rerun."""

    # Create robot instance with a name so its state publisher starts
    # (the subscriber below needs a non-empty state_publisher_topic).
    robot = universal_robots.UniversalRobotsUR10E(name="manipulator1")

    # Visualize the robot in Rerun
    robot.visualize_rerun()
    time.sleep(2.0)  # Wait for Rerun to initialize

    # Target: current joint configuration with the base joint rotated +30 deg
    target_joint_positions = robot.get_joint_positions().copy()
    target_joint_positions[0] += 30

    # Subscriber to states
    sub = node.Subscriber(topic=robot.state_publisher_topic,
                          callback=partial(on_state, robot=robot))

    # Command the move, then remove the subscriber cleanly
    try:
        robot.set_joint_positions(
            joint_positions=target_joint_positions,
            speed=60,
            acceleration=80,
        )
        logger.info(f"Moved to target joint positions: {target_joint_positions}")
    finally:
        sub.delete()

if __name__ == "__main__":
    main()
