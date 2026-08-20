"""
Read TCP wrench (force/torque) example for the Synapse SDK — offline.

Returns the TCP wrench ``[Fx, Fy, Fz (N), Tx, Ty, Tz (N·m)]``. Reads from the internal commanded-cache
state; no hardware connection is made.

Illustrated using Universal Robots (UR10e), supported on all robots.

Usage:
    python get_tcp_force_offline.py
"""

from loguru import logger

from telekinesis.synapse.robots.manipulators import universal_robots


def main():
    """Log the commanded-cache TCP wrench [N, N·m]."""

    #===================== Create Robot ==========================================
    robot = universal_robots.UniversalRobotsUR10E()

    # ==================== Run Skill ============================================
    logger.success(f"tcp_force [N, N·m]: {robot.get_tcp_force()}")


if __name__ == "__main__":
    main()
