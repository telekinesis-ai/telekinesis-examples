"""
Read the state update timestamp.

Supports Universal Robots (UR), Epson, and virtual.

Usage:
    python get_timestamp.py
"""

from loguru import logger

from telekinesis.synapse.robots.manipulators import universal_robots


def main():
    """Log the commanded-cache state timestamp [s]."""

    #===================== Create Robot ==========================================
    robot = universal_robots.UniversalRobotsUR10E(name='UR10e')

    # ==================== Run Skill ============================================
    logger.success(f"timestamp [s]: {robot.get_timestamp()}")


if __name__ == "__main__":
    main()
