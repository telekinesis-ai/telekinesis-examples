"""
Read the TCP wrench (force/torque).

Supports Universal Robots (UR), Epson, and virtual/sim.

Usage:
    python get_tcp_force.py
"""

from loguru import logger

from telekinesis.synapse.robots.manipulators import universal_robots


def main():
    """Log the commanded-cache TCP wrench [N, N·m]."""

    #===================== Create Robot ==========================================
    robot = universal_robots.UniversalRobotsUR10E(name='UR10e')

    # ==================== Run Skill ============================================
    logger.success(f"tcp_force [N, N·m]: {robot.get_tcp_force()}")


if __name__ == "__main__":
    main()
