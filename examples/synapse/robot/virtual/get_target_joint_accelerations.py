"""
Read target (commanded) joint accelerations example for the Synapse SDK — offline.

Returns the manipulator's target/commanded joint accelerations [deg/s²]. Reads from the internal commanded-cache
state; no hardware connection is made.

Illustrated using Universal Robots (UR10e), supported on all robots.

Usage:
    python get_target_joint_accelerations_offline.py
"""

from loguru import logger

from telekinesis.synapse.robots.manipulators import universal_robots


def main():
    """Log the commanded-cache target joint accelerations [deg/s²]."""

    robot = universal_robots.UniversalRobotsUR10E()
    logger.success(f"target_joint_accelerations [deg/s²]: {robot.get_target_joint_accelerations()}")


if __name__ == "__main__":
    main()
