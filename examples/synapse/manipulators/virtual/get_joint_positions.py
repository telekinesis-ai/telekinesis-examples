"""
Read joint positions example for the Synapse SDK — offline.

Returns the manipulator's joint positions [deg]. Reads from the internal commanded-cache
state; no hardware connection is made.

Illustrated using Universal Robots (UR10e), supported on all robots.

Usage:
    python get_joint_positions.py
"""

from loguru import logger

from telekinesis.synapse.robots.manipulators import universal_robots


def main():
    """Log the commanded-cache joint positions [deg]."""

    #===================== Create Robot ==========================================
    robot = universal_robots.UniversalRobotsUR10E()

    # ==================== Run Skill ============================================
    logger.success(f"joint_positions [deg]: {robot.get_joint_positions()}")


if __name__ == "__main__":
    main()
