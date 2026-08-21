"""
Read the manipulator's joint torques.

Supports Universal Robots (UR), Epson, and virtual/sim.

Usage:
    python get_joint_torques.py
"""

from loguru import logger

from telekinesis.synapse.robots.manipulators import universal_robots


def main():
    """Log the commanded-cache joint torques [N·m]."""

    #===================== Create Robot ==========================================
    robot = universal_robots.UniversalRobotsUR10E(name='UR10e')

    # ==================== Run Skill ============================================
    logger.success(f"joint_torques [N·m]: {robot.get_joint_torques()}")


if __name__ == "__main__":
    main()
