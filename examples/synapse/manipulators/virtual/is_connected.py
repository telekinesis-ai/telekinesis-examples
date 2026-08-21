"""
Check the manipulator's connection status.

Supports Universal Robots (UR), Epson, and virtual.

Usage:
    python is_connected.py
"""

from loguru import logger

from telekinesis.synapse.robots.manipulators import universal_robots


def main():
    """Log ``is_connected`` for an unconnected robot (always False offline)."""

    #===================== Create Robot ==========================================
    robot = universal_robots.UniversalRobotsUR10E(name='UR10e')

    # ==================== Run Skill ============================================
    logger.info(f"is_connected: {robot.is_connected()}")


if __name__ == "__main__":
    main()
