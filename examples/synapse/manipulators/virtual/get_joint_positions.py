"""
Read the manipulator's joint positions.

Supports Universal Robots (UR), Epson, and virtual/sim.

Usage:
    python get_joint_positions.py
"""

from loguru import logger

from telekinesis.synapse.robots.manipulators import universal_robots


def main():
    """Log the commanded-cache joint positions [deg]."""

    #===================== Create Robot ==========================================
    robot = universal_robots.UniversalRobotsUR10E(name='UR10e')

    # ==================== Run Skill ============================================
    logger.success(f"joint_positions [deg]: {robot.get_joint_positions()}")


if __name__ == "__main__":
    main()
