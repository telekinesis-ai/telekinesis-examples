"""
Read the full robot state dictionary example for the Synapse SDK — offline.

Returns the same state dict broadcast over the robot's state topic, with keys
such as ``joint_positions``, ``joint_velocities``, ``tcp_pose``,
``target_joint_positions``, ``target_tcp_pose`` and ``timestamp`` (plus
hardware-dependent optional fields). Reads from the internal commanded-cache
state; no hardware connection is made.

``get_state()`` requires an initialized state publisher, so the robot is created
with a ``name`` (which starts the publisher at construction).

Illustrated using Universal Robots (UR10e), supported on all robots.

Usage:
    python get_state_offline.py
"""

from loguru import logger

from telekinesis.synapse.robots.manipulators import universal_robots


def main():
    """Log the full commanded-cache robot state dictionary."""

    #===================== Create Robot ==========================================
    robot = universal_robots.UniversalRobotsUR10E(name="manipulator1")

    # ==================== Run Skill ============================================
    state = robot.get_state()
    for key, value in state.items():
        logger.success(f"{key}: {value}")


if __name__ == "__main__":
    main()
