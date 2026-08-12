"""
Read TCP velocity (twist) example for the Synapse SDK — offline.

Returns the TCP twist ``[vx, vy, vz (m/s), ωx, ωy, ωz (deg/s)]``. Reads from the internal commanded-cache
state; no hardware connection is made.

Illustrated using Universal Robots (UR10e), supported on all robots.

Usage:
    python get_tcp_speed_offline.py
"""

from loguru import logger

from telekinesis.synapse.robots.manipulators import universal_robots


def main():
    """Log the commanded-cache TCP velocity [m/s, deg/s]."""

    robot = universal_robots.UniversalRobotsUR10E()
    logger.success(f"tcp_speed [m/s, deg/s]: {robot.get_tcp_speed()}")


if __name__ == "__main__":
    main()
