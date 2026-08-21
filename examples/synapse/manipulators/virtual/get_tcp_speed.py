"""
Read the TCP velocity (twist).

Supports Universal Robots (UR), Epson, and virtual/sim.

Usage:
    python get_tcp_speed.py
"""

from loguru import logger

from telekinesis.synapse.robots.manipulators import universal_robots


def main():
    """Log the commanded-cache TCP velocity [m/s, deg/s]."""

    #===================== Create Robot ==========================================
    robot = universal_robots.UniversalRobotsUR10E(name='UR10e')

    # ==================== Run Skill ============================================
    logger.success(f"tcp_speed [m/s, deg/s]: {robot.get_tcp_speed()}")


if __name__ == "__main__":
    main()
