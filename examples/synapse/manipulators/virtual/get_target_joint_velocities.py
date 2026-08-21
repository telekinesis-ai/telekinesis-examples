"""
Read the manipulator's target joint velocities.

Supports Universal Robots (UR), Epson, and virtual/sim.

Usage:
    python get_target_joint_velocities.py
"""

from loguru import logger

from telekinesis.synapse.robots.manipulators import universal_robots


def main():
    """Log the commanded-cache target joint velocities [deg/s]."""

    #===================== Create Robot ==========================================
    robot = universal_robots.UniversalRobotsUR10E(name='UR10e')

    # ==================== Run Skill ============================================
    logger.success(f"target_joint_velocities [deg/s]: {robot.get_target_joint_velocities()}")


if __name__ == "__main__":
    main()
