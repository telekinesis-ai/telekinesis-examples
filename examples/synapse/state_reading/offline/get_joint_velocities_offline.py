"""
Read joint velocities example for the Synapse SDK — offline.

Returns the manipulator's joint velocities [deg/s]. Reads from the internal commanded-cache
state; no hardware connection is made.

Illustrated using Universal Robots (UR10e), supported on all robots.

Usage:
    python get_joint_velocities_offline.py
"""

from loguru import logger

from telekinesis.synapse.robots.manipulators import universal_robots


def main():
    """Log the commanded-cache joint velocities [deg/s]."""

    robot = universal_robots.UniversalRobotsUR10E()
    logger.success(f"joint_velocities [deg/s]: {robot.get_joint_velocities()}")


if __name__ == "__main__":
    main()
