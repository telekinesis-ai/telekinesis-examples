"""
List all registered TCP frames and the active one.

Supports Universal Robots (UR), Epson, and virtual/sim.

Usage:
    python get_tcps.py
"""

from loguru import logger

from telekinesis.synapse.robots.manipulators.universal_robots import UniversalRobotsUR10E


def main():
    """List all registered TCPs and the currently active one."""

    #===================== Create Robot ==========================================
    robot = UniversalRobotsUR10E(name='UR10e')

    # ==================== Run Skill ============================================
    # Get all registered TCPs
    tcps = robot.get_tcps()
    logger.info(f"Registered TCPs: {tcps}")

    # Current Active TCP, transform w.r.t default tcp, and current TCP pose
    logger.info(f"Active TCP: {robot.active_tcp}"
                f" \nActive TCP transform: {robot.get_active_tcp_transform()}"
                f" \n TCP pose: {robot.get_cartesian_pose()}")


if __name__ == "__main__":
    main()
