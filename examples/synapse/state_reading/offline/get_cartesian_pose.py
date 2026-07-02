"""
Read TCP Cartesian pose example for the Synapse SDK — offline.

Returns the TCP pose ``[x, y, z (m), rx, ry, rz (deg)]``. Reads from the internal commanded-cache
state; no hardware connection is made.

Illustrated using Universal Robots (UR10e), supported on all robots.

Usage:
    python get_cartesian_pose.py
"""

from loguru import logger

from telekinesis.synapse.robots.manipulators import universal_robots


def main():
    """Log the commanded-cache TCP pose [m, deg]."""

    robot = universal_robots.UniversalRobotsUR10E()
    logger.success(f"tcp_pose [m, deg]: {robot.get_cartesian_pose()}")


if __name__ == "__main__":
    main()
