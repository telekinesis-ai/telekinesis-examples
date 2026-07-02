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

import rerun as rr
from loguru import logger
from babyros import node

from telekinesis.synapse.robots.manipulators import universal_robots


def on_state(msg, robot, recording):
    """Redraw the robot in Rerun on each state message.

    ``robot`` and ``recording`` are bound via functools.partial so the callback
    keeps babyros's single-argument (msg) signature. Passing ``recording``
    explicitly is required because this callback runs on a babyros worker thread,
    where Rerun's thread-local active recording is not set -- without it,
    ``visualize_rerun`` would spawn a new viewer per message.
    """
    robot.visualize_rerun(recording_stream=recording)


def main():
    """Move the TCP to a target pose on the kinematic model, logging live state to Rerun."""

    # Create robot instance with a name so its state publisher starts
    # (the subscriber below needs a non-empty state_publisher_topic).
    robot = universal_robots.UniversalRobotsUR10E(name="manipulator1")

    # Visualize the robot in Rerun, and capture the recording stream so the
    # subscriber callback can draw into it from its worker thread.
    robot.visualize_rerun()
    recording = rr.get_global_data_recording()
    time.sleep(2.0)  # Wait for Rerun to initialize

    # Define target Cartesian pose
    current_cartesian_pose = robot.get_cartesian_pose()
    target_cartesian_pose = current_cartesian_pose.copy()
    target_cartesian_pose[2] -= 0.2  # Move 20 cm down in Z

    # Subscriber to states
    sub = node.Subscriber(topic=robot.state_publisher_topic,
                          callback=partial(on_state, robot=robot, recording=recording))

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
