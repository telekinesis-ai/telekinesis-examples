"""
Read joint torques example for the Synapse SDK — offline.

Returns the manipulator's joint torques [N·m]. Reads from the internal commanded-cache
state; no hardware connection is made.

Illustrated using Universal Robots (UR10e), supported on all robots.

Usage:
    python get_joint_torques.py
"""

from loguru import logger

from telekinesis.synapse.robots.manipulators import universal_robots


def main():
    """Log the commanded-cache joint torques [N·m]."""

    robot = universal_robots.UniversalRobotsUR10E()
    logger.success(f"joint_torques [N·m]: {robot.get_joint_torques()}")


if __name__ == "__main__":
    main()
