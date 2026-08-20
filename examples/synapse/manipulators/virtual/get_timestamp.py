"""
Read state timestamp example for the Synapse SDK — offline.

Returns the timestamp of the most recent state update [s since epoch]. Reads from the internal commanded-cache
state; no hardware connection is made.

Illustrated using Universal Robots (UR10e), supported on all robots.

Usage:
    python get_timestamp_offline.py
"""

from loguru import logger

from telekinesis.synapse.robots.manipulators import universal_robots


def main():
    """Log the commanded-cache state timestamp [s]."""

    #===================== Create Robot ==========================================
    robot = universal_robots.UniversalRobotsUR10E()

    # ==================== Run Skill ============================================
    logger.success(f"timestamp [s]: {robot.get_timestamp()}")


if __name__ == "__main__":
    main()
