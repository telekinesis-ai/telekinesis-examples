"""
Log the full robot state dictionary.

Supports Universal Robots (UR), Epson, and virtual.

Usage:
    python get_state.py
"""

from loguru import logger

from telekinesis.synapse.robots.manipulators import universal_robots


def main():
    """Log the full commanded-cache robot state dictionary."""

    #===================== Create Robot ==========================================
    robot = universal_robots.UniversalRobotsUR10E(name='UR10e')

    # ==================== Run Skill ============================================
    state = robot.get_state()
    for key, value in state.items():
        logger.success(f"{key}: {value}")


if __name__ == "__main__":
    main()
