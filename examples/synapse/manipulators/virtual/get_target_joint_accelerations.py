"""
Read the manipulator's target joint accelerations.

Supports Universal Robots (UR), Epson, and virtual/sim.

Usage:
    python get_target_joint_accelerations.py
"""

from loguru import logger

from telekinesis.synapse.robots.manipulators import universal_robots


def main():
    """Log the commanded-cache target joint accelerations [deg/s²]."""

    #===================== Create Robot ==========================================
    robot = universal_robots.UniversalRobotsUR10E(name='UR10e')

    # ==================== Run Skill ============================================
    logger.success(f"target_joint_accelerations [deg/s²]: {robot.get_target_joint_accelerations()}")


if __name__ == "__main__":
    main()
