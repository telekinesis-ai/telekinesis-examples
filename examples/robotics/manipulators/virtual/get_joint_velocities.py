"""
Read the manipulator's joint velocities.

Supports Universal Robots (UR), Epson, and virtual.

Usage:
    python get_joint_velocities.py
"""

from loguru import logger

from telekinesis.synapse.robots.manipulators import universal_robots


def main():
    """Log the commanded-cache joint velocities [deg/s]."""

    #===================== Create Robot ==========================================
    robot = universal_robots.UniversalRobotsUR10E(name='UR10e')

    # ==================== Run Skill ============================================
    logger.success(f"joint_velocities [deg/s]: {robot.get_joint_velocities()}")


if __name__ == "__main__":
    main()
