"""
Check connection status example for the Synapse SDK — offline.

``is_connected`` reports whether the manipulator state is being driven by live
hardware. Offline (no hardware connection), it always reports ``False``.

Illustrated using Universal Robots (UR10e), supported on all robots.

Usage:
    python is_connected_offline.py
"""

from loguru import logger

from telekinesis.synapse.robots.manipulators import universal_robots


def main():
    """Log ``is_connected`` for an unconnected robot (always False offline)."""

    robot = universal_robots.UniversalRobotsUR10E()
    logger.info(f"is_connected: {robot.is_connected()}")


if __name__ == "__main__":
    main()
