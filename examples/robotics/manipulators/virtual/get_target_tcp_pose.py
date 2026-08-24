"""
Read the manipulator's target TCP pose.

Supports Universal Robots (UR), Epson, and virtual.

Usage:
    python get_target_tcp_pose.py
"""

from loguru import logger

from telekinesis.synapse.robots.manipulators import universal_robots


def main():
    """Log the commanded-cache target TCP pose [m, deg]."""

    #===================== Create Robot ==========================================
    robot = universal_robots.UniversalRobotsUR10E(name='UR10e')

    # ==================== Run Skill ============================================
    logger.success(f"target_tcp_pose [m, deg]: {robot.get_target_tcp_pose()}")


if __name__ == "__main__":
    main()
