"""
Read target (commanded) joint positions example for the Synapse SDK — offline.

Returns the manipulator's target/commanded joint positions [deg]. Reads from the internal commanded-cache
state; no hardware connection is made.

Illustrated using Universal Robots (UR10e), supported on all robots.

Usage:
    python get_target_joint_positions_offline.py
"""

from loguru import logger

from telekinesis.synapse.robots.manipulators import universal_robots


def main():
    """Log the commanded-cache target joint positions [deg]."""

    robot = universal_robots.UniversalRobotsUR10E()
    logger.success(f"target_joint_positions [deg]: {robot.get_target_joint_positions()}")


if __name__ == "__main__":
    main()
