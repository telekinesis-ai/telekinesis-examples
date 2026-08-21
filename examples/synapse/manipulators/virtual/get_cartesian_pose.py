"""
Read the manipulator's TCP Cartesian pose.

Supports Universal Robots (UR), Epson, and virtual/sim.

Usage:
    python get_cartesian_pose.py
"""

from loguru import logger

from telekinesis.synapse.robots.manipulators import universal_robots


def main():
    """Log the commanded-cache TCP pose [m, deg]."""

    #===================== Create Robot ==========================================
    robot = universal_robots.UniversalRobotsUR10E(name='UR10e')

    # ==================== Run Skill ============================================
    logger.success(f"tcp_pose [m, deg]: {robot.get_cartesian_pose()}")


if __name__ == "__main__":
    main()
